import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import GPT, BigramLanguageModel
from tokenization.character import vocab_size
from tokenization.bpe import BPETokenizer, WordPieceTokenizer
from tokenization.byte_bpe import ByteBPETokenizer
from tokenization.regex_bpe import RegexBPETokenizer
from tokenization.gpt_tokenizer import GPTTokenizer
from attention.mha import MultiHeadAttention
from attention.gqa import GroupedQueryAttention
from attention.mqa import MultiQueryAttention
from normalization.rms_norm import RMSNorm
from normalization.layernorm import LayerNorm
from feedforward.swiglu import SwiGlu
from feedforward.geglu import GEGLU


class TestCausalMasking(unittest.TestCase):
    def test_causal_mask_invariance(self):
        device = torch.device("cpu")
        model = GPT(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            hidden_dim=512,
            num_layers=2,
            attention_type="mha",
            normalization_type="rms",
            feedforward_type="swiglu",
            position_encoding="sinusoidal"
        ).to(device)
        model.eval()

        seq_len = 8
        x1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
        x2 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 85]], dtype=torch.long)

        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)

        diff = torch.abs(logits1[:, :-1, :] - logits2[:, :-1, :]).max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)


class TestAttentionVariants(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_mha_shape(self):
        attn = MultiHeadAttention(self.d_model, self.num_heads, position_encoding="sinusoidal")
        out = attn(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_mqa_shape(self):
        attn = MultiQueryAttention(self.d_model, self.num_heads, position_encoding="sinusoidal")
        out = attn(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_gqa_shapes(self):
        for num_kv_heads in [1, 2, 4, 8]:
            attn = GroupedQueryAttention(self.d_model, self.num_heads, num_kv_heads, position_encoding="sinusoidal")
            out = attn(self.x)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))


class TestNormalizationAndFFN(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.hidden_dim = 512
        self.x = torch.randn(2, 16, self.d_model)

    def test_rms_norm(self):
        norm = RMSNorm(self.d_model)
        out = norm(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_layer_norm(self):
        norm = LayerNorm(self.d_model)
        out = norm(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_swiglu(self):
        ffn = SwiGlu(self.d_model, self.hidden_dim)
        out = ffn(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_geglu(self):
        ffn = GEGLU(self.d_model, self.hidden_dim)
        out = ffn(self.x)
        self.assertEqual(out.shape, self.x.shape)


class TestSubwordTokenizers(unittest.TestCase):
    def setUp(self):
        self.sample_text = "Once upon a time in a small forest, there was a tiny frog who loved to sing."

    def test_bpe_tokenizer(self):
        tok = BPETokenizer(vocab_size=128)
        tok.fit(self.sample_text)
        encoded = tok.encode("small forest")
        self.assertTrue(len(encoded) > 0)

    def test_wordpiece_tokenizer(self):
        tok = WordPieceTokenizer(vocab_size=128)
        tok.fit(self.sample_text)
        encoded = tok.encode("small forest")
        self.assertTrue(len(encoded) > 0)

    def test_byte_bpe_tokenizer(self):
        tok = ByteBPETokenizer(vocab_size=128)
        tok.fit(self.sample_text)
        encoded = tok.encode("small forest")
        self.assertTrue(len(encoded) > 0)

    def test_regex_bpe_tokenizer(self):
        tok = RegexBPETokenizer(vocab_size=128)
        tok.fit(self.sample_text)
        encoded = tok.encode("small forest")
        self.assertTrue(len(encoded) > 0)

    def test_gpt_tokenizer(self):
        tok = GPTTokenizer(vocab_size=128)
        tok.fit(self.sample_text)
        encoded = tok.encode("small forest <|endoftext|>")
        self.assertTrue(len(encoded) > 0)


class Test13ArchitecturePresets(unittest.TestCase):
    def setUp(self):
        self.input_ids = torch.randint(0, vocab_size, (2, 16))
        self.targets = torch.randint(0, vocab_size, (2, 16))

    def test_bigram_model(self):
        model = BigramLanguageModel()
        logits, loss = model(self.input_ids, self.targets)
        self.assertEqual(logits.shape, (2 * 16, vocab_size))
        self.assertGreater(loss.item(), 0.0)

    def test_all_gpt_presets(self):
        presets = [
            ("mha", "sinusoidal", "swiglu", "rms", 4),
            ("gqa", "sinusoidal", "swiglu", "rms", 2),
            ("mqa", "sinusoidal", "swiglu", "layer", None),
            ("mha", "learned", "geglu", "layer", None),
            ("mha", "sinusoidal", "geglu", "layer", None),
            ("mha", "alibi", "swiglu", "rms", None),
            ("mha", "rope", "swiglu", "rms", None),
            ("mqa", "rope", "swiglu", "rms", None),
            ("gqa", "sinusoidal", "geglu", "rms", 4),
            ("mha", "absolute", "swiglu", "layer", None),
        ]

        for attn, pos, ffn, norm, kv_heads in presets:
            with self.subTest(attn=attn, pos=pos, ffn=ffn, norm=norm):
                model = GPT(
                    vocab_size=vocab_size,
                    d_model=128,
                    num_heads=4,
                    hidden_dim=512,
                    num_layers=2,
                    attention_type=attn,
                    normalization_type=norm,
                    feedforward_type=ffn,
                    position_encoding=pos,
                    num_kv_heads=kv_heads
                )
                logits, loss = model(self.input_ids, self.targets)
                self.assertEqual(logits.shape, (2, 16, vocab_size))
                self.assertGreater(loss.item(), 0.0)
                self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
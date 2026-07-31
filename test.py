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
from lora_qlora_scratch import (
    LoRALinear,
    INT8Quantizer,
    INT8Linear,
    INT4Quantizer,
    BlockwiseINT4Quantizer,
    PackedINT4Storage,
    NF4Quantizer,
    DoubleQuantizer,
    QLoRALinear,
    inject_lora_to_model,
    mark_only_lora_as_trainable,
)


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

    def test_mha(self):
        attn = MultiHeadAttention(self.d_model, self.num_heads)
        out = attn(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_gqa(self):
        attn = GroupedQueryAttention(self.d_model, self.num_heads, num_kv_heads=2)
        out = attn(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_mqa(self):
        attn = MultiQueryAttention(self.d_model, self.num_heads)
        out = attn(self.x)
        self.assertEqual(out.shape, self.x.shape)


class TestTokenizers(unittest.TestCase):
    def setUp(self):
        self.sample_text = "The quick brown fox jumps over 13 lazy dogs! #AI #NLP\n"

    def test_bpe_tokenizer(self):
        tok = BPETokenizer(vocab_size=128)
        tok.fit(self.sample_text)
        encoded = tok.encode(self.sample_text)
        decoded = tok.decode(encoded)
        self.assertEqual(self.sample_text, decoded)

    def test_byte_bpe_tokenizer(self):
        tok = ByteBPETokenizer(vocab_size=300)
        tok.fit(self.sample_text)
        encoded = tok.encode("hello world 123")
        decoded = tok.decode(encoded)
        self.assertEqual("hello world 123", decoded)

    def test_regex_bpe_tokenizer(self):
        tok = RegexBPETokenizer(vocab_size=300)
        tok.fit(self.sample_text)
        encoded = tok.encode("hello world 123")
        decoded = tok.decode(encoded)
        self.assertEqual("hello world 123", decoded)

    def test_wordpiece_tokenizer(self):
        tok = WordPieceTokenizer(vocab_size=300)
        tok.fit(self.sample_text)
        encoded = tok.encode("quick brown")
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
        model = BigramLanguageModel(vocab_size=vocab_size)
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


class TestLoRAandQLoRA(unittest.TestCase):
    def test_lora_linear_forward_and_grad(self):
        lora_layer = LoRALinear(in_dim=32, out_dim=64, rank=4, alpha=16)
        x = torch.randn(2, 10, 32)
        out = lora_layer(x)
        self.assertEqual(out.shape, (2, 10, 64))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(lora_layer.A.grad)
        self.assertIsNotNone(lora_layer.B.grad)
        self.assertIsNone(lora_layer.linear.weight.grad)

    def test_int8_quantization(self):
        w = torch.randn(128, 128)
        w_int8, scale = INT8Quantizer.quantize(w)
        w_dequant = INT8Quantizer.dequantize(w_int8, scale)
        diff = torch.abs(w - w_dequant).mean().item()
        self.assertLess(diff, 0.05)

    def test_int8_linear(self):
        layer = INT8Linear(32, 64)
        w = torch.randn(64, 32)
        layer.load_fp32_weight(w)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_int4_quantization(self):
        w = torch.randn(64, 64)
        w_int4, scale = INT4Quantizer.quantize(w)
        w_dequant = INT4Quantizer.dequantize(w_int4, scale)
        diff = torch.abs(w - w_dequant).mean().item()
        self.assertLess(diff, 0.3)

    def test_blockwise_int4(self):
        w = torch.randn(128, 128)
        q_blocks, scales, shape, pad_len = BlockwiseINT4Quantizer.quantize(w, block_size=64)
        w_dequant = BlockwiseINT4Quantizer.dequantize(q_blocks, scales, shape, pad_len)
        self.assertEqual(w_dequant.shape, (128, 128))

    def test_packed_int4_storage(self):
        int4_tensor = torch.tensor([-8, -4, 0, 3, 7, -1, 2, 5], dtype=torch.int8)
        packed = PackedINT4Storage.pack(int4_tensor)
        unpacked = PackedINT4Storage.unpack(packed, len(int4_tensor))
        self.assertTrue(torch.equal(int4_tensor, unpacked))

    def test_nf4_quantization(self):
        w = torch.randn(128, 128)
        packed, scales, shape, pad_len = NF4Quantizer.quantize(w, block_size=64)
        w_dequant = NF4Quantizer.dequantize(packed, scales, shape, pad_len)
        self.assertEqual(w_dequant.shape, (128, 128))
        diff = torch.abs(w - w_dequant).mean().item()
        self.assertLess(diff, 0.25)

    def test_double_quantization(self):
        scales = torch.rand(256) * 0.1
        int8_s, s_of_s = DoubleQuantizer.quantize(scales)
        dequant_s = DoubleQuantizer.dequantize(int8_s, s_of_s)
        diff = torch.abs(scales - dequant_s).mean().item()
        self.assertLess(diff, 0.01)

    def test_qlora_linear(self):
        qlora_layer = QLoRALinear(in_dim=32, out_dim=64, rank=4, alpha=16)
        w_fp32 = torch.randn(64, 32)
        qlora_layer.load_fp32_weight(w_fp32)
        x = torch.randn(2, 10, 32)
        out = qlora_layer(x)
        self.assertEqual(out.shape, (2, 10, 64))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(qlora_layer.A.grad)
        self.assertIsNotNone(qlora_layer.B.grad)

    def test_model_lora_injection(self):
        model = GPT(vocab_size=100, d_model=128, num_heads=4, hidden_dim=512, num_layers=2, attention_type="mha", normalization_type="rms", feedforward_type="swiglu", position_encoding="rope")
        model = inject_lora_to_model(model, rank=4, alpha=16, use_qlora=False)
        trainable, total = mark_only_lora_as_trainable(model)
        self.assertLess(trainable, total)

        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x)
        self.assertEqual(logits.shape, (2, 16, 100))


if __name__ == "__main__":
    unittest.main()
import unittest
import torch
import torch.nn as nn
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
from gpt import GPT

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

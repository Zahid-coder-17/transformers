import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16, bias=True, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (self.dropout(x) @ self.A.t()) @ self.B.t()
        return base_out + (self.scaling * lora_out)


class INT8Quantizer:
    @staticmethod
    def quantize(weight_fp32):
        scale = torch.max(torch.abs(weight_fp32)) / 127.0
        scale = torch.clamp(scale, min=1e-8)
        weight_int8 = torch.clamp(torch.round(weight_fp32 / scale), -128, 127).to(torch.int8)
        return weight_int8, scale

    @staticmethod
    def dequantize(weight_int8, scale):
        return weight_int8.to(scale.dtype) * scale


class INT8Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.register_buffer("weight_int8", torch.zeros(out_dim, in_dim, dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(1.0))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def load_fp32_weight(self, weight_fp32):
        w_int8, scale = INT8Quantizer.quantize(weight_fp32)
        self.weight_int8.copy_(w_int8)
        self.scale.copy_(scale)

    def forward(self, x):
        w_dequant = INT8Quantizer.dequantize(self.weight_int8, self.scale)
        return F.linear(x, w_dequant, self.bias)


class INT4Quantizer:
    @staticmethod
    def quantize(weight_fp32):
        scale = torch.max(torch.abs(weight_fp32)) / 7.0
        scale = torch.clamp(scale, min=1e-8)
        weight_int4 = torch.clamp(torch.round(weight_fp32 / scale), -8, 7)
        return weight_int4, scale

    @staticmethod
    def dequantize(weight_int4, scale):
        return weight_int4.float() * scale


class BlockwiseINT4Quantizer:
    @staticmethod
    def quantize(weight_fp32, block_size=64):
        shape = weight_fp32.shape
        flat = weight_fp32.flatten()
        pad_len = (block_size - (flat.numel() % block_size)) % block_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        blocks = flat.view(-1, block_size)
        scales = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values / 7.0
        scales = torch.clamp(scales, min=1e-8)

        quant_blocks = torch.clamp(torch.round(blocks / scales), -8, 7).to(torch.int8)
        return quant_blocks, scales, shape, pad_len

    @staticmethod
    def dequantize(quant_blocks, scales, shape, pad_len=0):
        dequant = (quant_blocks.float() * scales).flatten()
        if pad_len > 0:
            dequant = dequant[:-pad_len]
        return dequant.reshape(shape)


class PackedINT4Storage:
    @staticmethod
    def pack(int4_tensor):
        flat = int4_tensor.flatten()
        if flat.numel() % 2 != 0:
            flat = F.pad(flat, (0, 1))

        u4 = (flat + 8).to(torch.uint8)
        even = u4[0::2] & 0x0F
        odd = u4[1::2] & 0x0F
        packed = even | (odd << 4)
        return packed

    @staticmethod
    def unpack(packed_uint8, original_numel):
        even = (packed_uint8 & 0x0F).to(torch.int8) - 8
        odd = ((packed_uint8 >> 4) & 0x0F).to(torch.int8) - 8
        interleaved = torch.stack([even, odd], dim=-1).flatten()
        return interleaved[:original_numel]


class NF4Codebook:
    CODEBOOK = torch.tensor([
        -1.0, -0.6961928010010719, -0.5250929007428059, -0.39491749554872527,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124878, 0.24612408757209778, 0.33791524171829224,
        0.4407098288536072, 0.5626170291946411, 0.7229568362236023, 1.0
    ])

class NF4Quantizer:
    @staticmethod
    def quantize(weight_fp32, block_size=64):
        shape = weight_fp32.shape
        flat = weight_fp32.flatten()
        pad_len = (block_size - (flat.numel() % block_size)) % block_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        blocks = flat.view(-1, block_size)
        scales = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values
        scales = torch.clamp(scales, min=1e-8)

        norm_blocks = blocks / scales
        codebook = NF4Codebook.CODEBOOK.to(weight_fp32.device)

        dists = torch.abs(norm_blocks.unsqueeze(-1) - codebook)
        indices = torch.argmin(dists, dim=-1).to(torch.uint8)

        flat_indices = indices.flatten()
        even = flat_indices[0::2] & 0x0F
        odd = flat_indices[1::2] & 0x0F
        packed_bytes = even | (odd << 4)

        return packed_bytes, scales.squeeze(-1), shape, pad_len

    @staticmethod
    def dequantize(packed_bytes, scales, shape, pad_len=0, device="cpu"):
        even = (packed_bytes & 0x0F).long()
        odd = ((packed_bytes >> 4) & 0x0F).long()
        indices = torch.stack([even, odd], dim=-1).flatten()

        codebook = NF4Codebook.CODEBOOK.to(device)
        q_vals = codebook[indices]

        block_size = 64
        num_blocks = scales.numel()
        q_blocks = q_vals[:num_blocks * block_size].view(num_blocks, block_size)

        dequant = (q_blocks * scales.unsqueeze(-1)).flatten()
        if pad_len > 0:
            dequant = dequant[:-pad_len]
        return dequant.reshape(shape)


class DoubleQuantizer:
    @staticmethod
    def quantize(fp32_scales, block_size_scales=256):
        scale_of_scales = torch.max(torch.abs(fp32_scales)) / 127.0
        scale_of_scales = torch.clamp(scale_of_scales, min=1e-8)
        int8_scales = torch.clamp(torch.round(fp32_scales / scale_of_scales), -128, 127).to(torch.int8)
        return int8_scales, scale_of_scales

    @staticmethod
    def dequantize(int8_scales, scale_of_scales):
        return int8_scales.float() * scale_of_scales


class QLoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16, block_size=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.block_size = block_size

        self.register_buffer("packed_bytes", torch.zeros(math.ceil((in_dim * out_dim) / 2), dtype=torch.uint8))
        self.register_buffer("int8_scales", torch.zeros(math.ceil((in_dim * out_dim) / block_size), dtype=torch.int8))
        self.register_buffer("scale_of_scales", torch.tensor(1.0))
        self.shape = (out_dim, in_dim)
        self.pad_len = 0

        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def load_fp32_weight(self, weight_fp32):
        packed, scales, shape, pad_len = NF4Quantizer.quantize(weight_fp32, self.block_size)
        int8_s, s_of_s = DoubleQuantizer.quantize(scales)

        self.packed_bytes = packed
        self.int8_scales = int8_s
        self.scale_of_scales = s_of_s
        self.shape = shape
        self.pad_len = pad_len

    def forward(self, x):
        fp32_scales = DoubleQuantizer.dequantize(self.int8_scales, self.scale_of_scales)
        w_base = NF4Quantizer.dequantize(self.packed_bytes, fp32_scales, self.shape, self.pad_len, device=x.device)

        base_out = F.linear(x, w_base)
        lora_out = (x @ self.A.t()) @ self.B.t()
        return base_out + (self.scaling * lora_out)


def inject_lora_to_model(model, target_module_names=["c_attn", "q_proj", "v_proj"], rank=4, alpha=16, use_qlora=False):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and any(tgt in name for tgt in target_module_names):
            in_dim, out_dim = child.in_features, child.out_features
            bias = child.bias is not None
            if use_qlora:
                adapter = QLoRALinear(in_dim, out_dim, rank=rank, alpha=alpha)
                adapter.load_fp32_weight(child.weight.data)
            else:
                adapter = LoRALinear(in_dim, out_dim, rank=rank, alpha=alpha, bias=bias)
                adapter.linear.weight.data.copy_(child.weight.data)
                if bias:
                    adapter.linear.bias.data.copy_(child.bias.data)
            setattr(model, name, adapter)
        else:
            inject_lora_to_model(child, target_module_names, rank, alpha, use_qlora)
    return model


def mark_only_lora_as_trainable(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "A" in name or "B" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    return trainable_params, total_params


def train_lora(model, train_dataloader, epochs=3, lr=3e-4, save_adapter_path="lora_adapters.pth"):
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dataloader:
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    adapter_state = {k: v for k, v in model.state_dict().items() if "A" in k or "B" in k}
    torch.save(adapter_state, save_adapter_path)
    return adapter_state

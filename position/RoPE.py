import torch

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def get_rope_sin_cos(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    cos = emb.cos().unsqueeze(0).unsqueeze(1)
    return sin, cos

def apply_rotary_pos_emb(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)
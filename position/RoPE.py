import torch

def rotate_half(x):
    x1 = x[...,:x.shape[-1] // 2]
    x2 = x[...,x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)
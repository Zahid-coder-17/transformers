import torch

def get_slopes(num_heads):
    slope = []
    for i in range(num_heads):
        slope.append(1.0 / (2.0 ** (i / num_heads)))
    return torch.tensor(slope).unsqueeze(0).unsqueeze(0)

def build_alibi_bias(seq_len, num_heads):
    slopes = get_slopes(num_heads).view(1, num_heads, 1, 1)
    positions = torch.arange(seq_len)
    bias = -torch.abs(positions[:, None] - positions[None, :  ])
    bias = bias.view(1,1, seq_len, seq_len)
    return bias * slopes

def apply_alibi_bias(attention_scores, alibi_bias):
    return attention_scores + alibi_bias
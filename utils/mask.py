import torch


def build_causal_mask(seq_len):

    mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
    ).bool()

    return mask.view(1, 1, seq_len, seq_len)


def apply_causal_mask(attention_scores, causal_mask):
    

    return attention_scores.masked_fill(
        causal_mask,
        float("-inf")
    )
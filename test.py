import torch

from gpt import GPT

model = GPT(
    vocab_size=1000,
    d_model=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=4,

    attention_type="mha",
    normalization_type="rms",
    feedforward_type="swiglu",
    position_encoding="sinusoidal"
)

print(model)

input_ids = torch.randint(
    0,
    1000,
    (2, 16)
)

print("Input Shape :", input_ids.shape)

logits = model(input_ids)

print("Output Shape:", logits.shape)
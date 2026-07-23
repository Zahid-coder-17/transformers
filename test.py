import torch
from tokenization.character import get_batch,vocab_size
from gpt import GPT

model = GPT(
    vocab_size=vocab_size,
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

input_ids , targets = get_batch("train")

print("Input Shape :", input_ids.shape)
print("Target Shape :",targets.shape)

logits,loss = model(input_ids,targets)

print("Output Shape:", logits.shape)

assert logits.shape == (
    input_ids.shape[0],input_ids.shape[1],vocab_size
)
print("\nGPT FORWARD PASS successful")
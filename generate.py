import torch
import sys
import argparse
from gpt import GPT
from tokenization.character import vocab_size, decode, encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(prompt="Once upon a time", max_new_tokens=300, temperature=0.8, top_k=40, top_p=0.9):
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
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/gpt_character.pth", map_location=device))
    model.eval()

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    text = decode(generated[0].tolist())
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained GPT model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial prompt text")
    parser.add_argument("--max_tokens", type=int, default=300, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) filtering")
    args = parser.parse_args()

    print(f"\nPrompt: '{args.prompt}'")
    print(f"Sampling Parameters: Temp={args.temp}, Top-k={args.top_k}, Top-p={args.top_p}")
    print("=" * 60)
    result = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print(result)
    print("=" * 60)
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from gpt import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

english_text = open("data/input.txt", encoding="utf-8").read()
arabic_text = open("data/arabic_input.txt", encoding="utf-8").read()
code_text = open("data/code_input.txt", encoding="utf-8").read()

combined_text = english_text + "\n" + arabic_text + "\n" + code_text

chars = sorted(list(set(combined_text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text if c in stoi]

def decode(ids):
    return "".join([itos[i] for i in ids])

print(f"Combined vocab size: {vocab_size}", flush=True)
print(f"Total corpus length: {len(combined_text):,} chars", flush=True)
print(f"  English : {len(english_text):,} chars", flush=True)
print(f"  Arabic  : {len(arabic_text):,} chars", flush=True)
print(f"  Code    : {len(code_text):,} chars", flush=True)

data = torch.tensor(encode(combined_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n].to(device)
val_data = data[n:].to(device)

batch_size = 64
block_size = 256
offsets_gpu = torch.arange(block_size, device=device).unsqueeze(0)

def fast_get_batch(split):
    d = train_data if split == "train" else val_data
    max_idx = len(d) - block_size - 1
    ix = torch.randint(0, max_idx, (batch_size, 1), device=device)
    indices = ix + offsets_gpu
    x = d[indices]
    y = d[indices + 1]
    return x, y

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

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
max_iter = 15000
losses = []

print("Starting Multi-Corpus Transformer Training for 15000 Epochs...", flush=True)
model.train()

for epoch in range(max_iter):
    xb, yb = fast_get_batch("train")
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    loss_val = loss.item()
    losses.append(loss_val)

    if epoch % 1000 == 0 or epoch == max_iter - 1:
        print(f"Epoch {epoch:4d} / {max_iter} | Cross-Entropy Loss: {loss_val:.4f}", flush=True)

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/gpt_multicorpus.pth")
torch.save({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, "checkpoints/multicorpus_vocab.pth")
print("Saved checkpoint to checkpoints/gpt_multicorpus.pth", flush=True)
print("Saved vocab to checkpoints/multicorpus_vocab.pth", flush=True)

os.makedirs("assets", exist_ok=True)
plt.style.use("dark_background")
plt.figure(figsize=(10, 5), dpi=300)

epochs = list(range(max_iter))
plt.plot(epochs, losses, alpha=0.35, color="#F59E0B", label="Raw Iteration Loss")

window_size = 50
moving_avg = [sum(losses[max(0, i-window_size):i+1])/len(losses[max(0, i-window_size):i+1]) for i in range(len(losses))]
plt.plot(epochs, moving_avg, color="#FBBF24", linewidth=2.5, label="Moving Average (Window=50)")

plt.title("Multi-Corpus Transformer Training Loss (English + Arabic + Code)", fontsize=13, fontweight="bold", pad=15, color="white")
plt.xlabel("Training Epoch", fontsize=12, labelpad=10)
plt.ylabel("Cross-Entropy Loss", fontsize=12, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(frameon=True, facecolor="#1E1E1E", edgecolor="none")

plt.annotate(f"Start: {losses[0]:.2f}", xy=(0, losses[0]), xytext=(500, losses[0] + 0.3),
             arrowprops=dict(facecolor="#FF6B6B", shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight="bold", color="#FF6B6B")

plt.annotate(f"Final: {losses[-1]:.2f}", xy=(max_iter-1, losses[-1]), xytext=(max_iter-2500, losses[-1] + 0.8),
             arrowprops=dict(facecolor="#4EBD40", shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight="bold", color="#4EBD40")

plt.tight_layout()
plt.savefig("assets/multicorpus_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved training plot to assets/multicorpus_loss_curve.png", flush=True)

print("\n--- Multi-Corpus Text Generation Samples ---", flush=True)
model.eval()

prompts = ["Once upon a time", "def fibonacci(", "\u0645\u0631\u062d\u0628\u0627"]
for p in prompts:
    ctx = torch.tensor([encode(p)], dtype=torch.long, device=device)
    out = decode(model.generate(ctx, max_new_tokens=150)[0].tolist())
    print(f"\n[Prompt: '{p}']\n{out}", flush=True)

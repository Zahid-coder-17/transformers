import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gpt import GPT
import os
import matplotlib.pyplot as plt
from tokenization.character import train_data, val_data, decode, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

train_data_gpu = train_data.to(device)
val_data_gpu = val_data.to(device)
batch_size = 32
block_size = 128
offsets_gpu = torch.arange(block_size, device=device).unsqueeze(0)

def fast_get_batch(split):
    data = train_data_gpu if split == "train" else val_data_gpu
    max_idx = len(data) - block_size - 1
    ix = torch.randint(0, max_idx, (batch_size, 1), device=device)
    indices = ix + offsets_gpu
    x = data[indices]
    y = data[indices + 1]
    return x, y

model = GPT(
    vocab_size=vocab_size,
    d_model=256,
    num_heads=4,
    hidden_dim=1024,
    num_layers=2,
    attention_type="mha",
    normalization_type="rms",
    feedforward_type="swiglu",
    position_encoding="sinusoidal"
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
max_iter = 2000

losses = []

print("Starting Fast Modular Transformer Training for 2000 Epochs...", flush=True)
model.train()
for epoch in range(max_iter):
    xb, yb = fast_get_batch("train")

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    loss_val = loss.item()
    losses.append(loss_val)

    if epoch % 200 == 0 or epoch == max_iter - 1:
        print(f"Epoch {epoch:4d} / {max_iter} | Loss: {loss_val:.4f}", flush=True)

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/gpt_character.pth")
print("Saved trained checkpoint to checkpoints/gpt_character.pth", flush=True)

os.makedirs("assets", exist_ok=True)
plt.style.use("dark_background")
plt.figure(figsize=(10, 5), dpi=300)

epochs = list(range(max_iter))
plt.plot(epochs, losses, alpha=0.35, color="#4A90E2", label="Raw Iteration Loss")

window_size = 50
moving_avg = [sum(losses[max(0, i-window_size):i+1])/len(losses[max(0, i-window_size):i+1]) for i in range(len(losses))]
plt.plot(epochs, moving_avg, color="#61DAFB", linewidth=2.5, label="Moving Average (Window=50)")

plt.title("Modular Transformer Training Loss Curve (2000 Epochs)", fontsize=14, fontweight="bold", pad=15, color="white")
plt.xlabel("Training Iteration / Epoch", fontsize=12, labelpad=10)
plt.ylabel("Cross-Entropy Loss", fontsize=12, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(frameon=True, facecolor="#1E1E1E", edgecolor="none")

initial_loss = losses[0]
final_loss = losses[-1]
plt.annotate(f"Start: {initial_loss:.2f}", xy=(0, initial_loss), xytext=(150, initial_loss + 0.3),
             arrowprops=dict(facecolor="#FF6B6B", shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight="bold", color="#FF6B6B")

plt.annotate(f"Final: {final_loss:.2f}", xy=(max_iter-1, final_loss), xytext=(max_iter-500, final_loss + 0.8),
             arrowprops=dict(facecolor="#4EBD40", shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight="bold", color="#4EBD40")

plt.tight_layout()
plot_path = "assets/loss_curve.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved training plot to {plot_path}", flush=True)

print("\n--- Trained Transformer Text Generation Sample ---", flush=True)
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(ctx, 150)[0].tolist())
print(generated_text, flush=True)
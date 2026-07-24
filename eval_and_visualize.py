import torch
import torch.nn.functional as F
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from gpt import GPT
from tokenization.character import get_batch, vocab_size, decode, encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing evaluation on device: {device}", flush=True)

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

checkpoint_path = "checkpoints/gpt_character.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.", flush=True)

model.eval()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Total Parameters: {num_params:,} ({num_params/1e6:.2f}M)", flush=True)

@torch.no_grad()
def evaluate_performance(eval_iters=50):
    metrics = {"train": {"loss": [], "accuracy": []}, "val": {"loss": [], "accuracy": []}}

    for split in ["train", "val"]:
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)

            metrics[split]["loss"].append(loss.item())

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == yb).float().mean().item()
            metrics[split]["accuracy"].append(acc)

    train_loss = float(np.mean(metrics["train"]["loss"]))
    val_loss = float(np.mean(metrics["val"]["loss"]))
    train_acc = float(np.mean(metrics["train"]["accuracy"]))
    val_acc = float(np.mean(metrics["val"]["accuracy"]))
    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "raw_metrics": metrics
    }

print("\n--- Evaluating Model Metrics ---", flush=True)
results = evaluate_performance(eval_iters=50)

print(f"Train Loss      : {results['train_loss']:.4f}", flush=True)
print(f"Validation Loss : {results['val_loss']:.4f}", flush=True)
print(f"Train Perplexity: {results['train_ppl']:.2f}", flush=True)
print(f"Val Perplexity  : {results['val_ppl']:.2f}", flush=True)
print(f"Train Accuracy  : {results['train_acc']*100:.2f}%", flush=True)
print(f"Val Accuracy    : {results['val_acc']*100:.2f}%", flush=True)

print("\n--- Measuring Generation Speed ---", flush=True)
prompt_text = "Once upon a time"
context = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)
num_tokens_gen = 200

start_time = time.time()
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=num_tokens_gen, temperature=0.8, top_k=40)
elapsed = time.time() - start_time
throughput = num_tokens_gen / elapsed
ms_per_token = (elapsed / num_tokens_gen) * 1000

print(f"Generated {num_tokens_gen} tokens in {elapsed:.2f} seconds", flush=True)
print(f"Throughput       : {throughput:.2f} tokens/sec", flush=True)
print(f"Latency          : {ms_per_token:.2f} ms/token", flush=True)

os.makedirs("assets", exist_ok=True)
plt.style.use("dark_background")

fig = plt.figure(figsize=(16, 10), dpi=300)
fig.patch.set_facecolor("#0B0F19")

ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor("#111827")
train_losses = results["raw_metrics"]["train"]["loss"]
val_losses = results["raw_metrics"]["val"]["loss"]

ax1.hist(train_losses, bins=15, alpha=0.6, color="#3B82F6", label=f"Train Loss (Mean: {results['train_loss']:.3f})")
ax1.hist(val_losses, bins=15, alpha=0.6, color="#10B981", label=f"Val Loss (Mean: {results['val_loss']:.3f})")
ax1.axvline(results["train_loss"], color="#3B82F6", linestyle="--", linewidth=2)
ax1.axvline(results["val_loss"], color="#10B981", linestyle="--", linewidth=2)
ax1.set_title("Cross-Entropy Loss Distribution Across Batches", fontsize=12, fontweight="bold", color="white", pad=10)
ax1.set_xlabel("Loss Value", color="#9CA3AF")
ax1.set_ylabel("Batch Count", color="#9CA3AF")
ax1.grid(True, linestyle=":", alpha=0.3)
ax1.legend(facecolor="#1F2937", edgecolor="none", labelcolor="white")

ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor("#111827")
train_accs = [a * 100 for a in results["raw_metrics"]["train"]["accuracy"]]
val_accs = [a * 100 for a in results["raw_metrics"]["val"]["accuracy"]]

batches = np.arange(1, len(train_accs) + 1)
ax2.plot(batches, train_accs, color="#60A5FA", linewidth=1.8, alpha=0.8, label=f"Train Acc (Avg: {results['train_acc']*100:.1f}%)")
ax2.plot(batches, val_accs, color="#34D399", linewidth=1.8, alpha=0.8, label=f"Val Acc (Avg: {results['val_acc']*100:.1f}%)")
ax2.axhline(results["train_acc"]*100, color="#60A5FA", linestyle=":", alpha=0.7)
ax2.axhline(results["val_acc"]*100, color="#34D399", linestyle=":", alpha=0.7)
ax2.set_title("Character Prediction Accuracy per Batch (%)", fontsize=12, fontweight="bold", color="white", pad=10)
ax2.set_xlabel("Evaluation Batch", color="#9CA3AF")
ax2.set_ylabel("Accuracy (%)", color="#9CA3AF")
ax2.grid(True, linestyle=":", alpha=0.3)
ax2.legend(facecolor="#1F2937", edgecolor="none", labelcolor="white")

ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor("#111827")
categories = ["Train Perplexity", "Val Perplexity"]
values = [results["train_ppl"], results["val_ppl"]]
colors = ["#8B5CF6", "#EC4899"]

bars = ax3.bar(categories, values, color=colors, width=0.45, edgecolor="white", linewidth=0.5)
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom", color="white", fontweight="bold", fontsize=11)

ax3.set_title("Model Perplexity (PPL = e^Loss)", fontsize=12, fontweight="bold", color="white", pad=10)
ax3.set_ylabel("Perplexity (Lower is better)", color="#9CA3AF")
ax3.set_ylim(0, max(values) * 1.25)
ax3.grid(True, axis="y", linestyle=":", alpha=0.3)

ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor("#111827")
ax4.axis("off")

summary_text = (
    "  GPT Transformer Performance Summary\n"
    "  -------------------------------------\n"
    f"  • Architecture      : Decoder-Only GPT\n"
    f"  • Model Parameters  : {num_params/1e6:.2f} Million ({num_params:,})\n"
    f"  • Embedding Dim     : 512 | Heads: 8 | Layers: 4\n"
    f"  • FeedForward Dim   : 2048 (SwiGLU)\n"
    f"  • Normalization     : RMSNorm\n"
    f"  • Positional Enc.   : Sinusoidal\n"
    "  -------------------------------------\n"
    f"  • Evaluation Device : {device.type.upper()}\n"
    f"  • Train / Val Loss  : {results['train_loss']:.4f} / {results['val_loss']:.4f}\n"
    f"  • Train / Val PPL   : {results['train_ppl']:.2f} / {results['val_ppl']:.2f}\n"
    f"  • Next-Token Acc.   : {results['val_acc']*100:.2f}%\n"
    f"  • Inference Speed   : {throughput:.1f} tokens/sec ({ms_per_token:.1f} ms/token)\n"
)

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         fontfamily="monospace", verticalalignment="top", color="#F3F4F6",
         bbox=dict(boxstyle="round,pad=1", facecolor="#1F2937", edgecolor="#374151", alpha=0.9))

plt.suptitle("TinyStories GPT LLM - Evaluation & Performance Analytics", fontsize=16, fontweight="bold", color="white", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

dashboard_path = "assets/performance_dashboard.png"
plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved evaluation dashboard to {dashboard_path}", flush=True)

print("\n" + "="*70, flush=True)
print("             LLM TEXT GENERATION SAMPLES (Trained Model)", flush=True)
print("="*70, flush=True)

test_prompts = [
    ("Once upon a time", 0.7, 40, 0.9),
    ("Lily was a little girl who loved", 0.8, 50, 0.95),
    ("One day, a friendly dog saw", 0.6, 30, 0.85),
]

generated_samples = []

for prompt, temp, top_k, top_p in test_prompts:
    ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ctx, max_new_tokens=220, temperature=temp, top_k=top_k, top_p=top_p)
    story = decode(out[0].tolist())
    generated_samples.append((prompt, temp, story))

    print(f"\n[Prompt: '{prompt}'] (Temp={temp}, Top-K={top_k}, Top-P={top_p})", flush=True)
    print("-" * 65, flush=True)
    print(story, flush=True)
    print("-" * 65, flush=True)

print("\nEvaluation and generation complete!", flush=True)

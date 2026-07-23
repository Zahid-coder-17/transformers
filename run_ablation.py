import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import time
import math
import numpy as np

from gpt import GPT, BigramLanguageModel
from tokenization.character import get_batch, vocab_size, decode, encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRESETS = [
    {
        "id": 1,
        "name": "LLaMA-3 Style (Default)",
        "attention": "mha",
        "position": "sinusoidal",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": 4,
        "is_bigram": False
    },
    {
        "id": 2,
        "name": "Mistral-Style (GQA - 2 KV Heads)",
        "attention": "gqa",
        "position": "sinusoidal",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": 2,
        "is_bigram": False
    },
    {
        "id": 3,
        "name": "Falcon / PaLM (MQA - 1 KV Head)",
        "attention": "mqa",
        "position": "sinusoidal",
        "feedforward": "swiglu",
        "norm": "layer",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": 1,
        "is_bigram": False
    },
    {
        "id": 4,
        "name": "Classic GPT-2",
        "attention": "mha",
        "position": "learned",
        "feedforward": "geglu",
        "norm": "layer",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 5,
        "name": "Vaswani Standard (2017)",
        "attention": "mha",
        "position": "sinusoidal",
        "feedforward": "geglu",
        "norm": "layer",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 6,
        "name": "ALiBi Relative Transformer",
        "attention": "mha",
        "position": "alibi",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 7,
        "name": "RoPE Rotary Embedding",
        "attention": "mha",
        "position": "rope",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 8,
        "name": "Deep RMSNorm + MQA",
        "attention": "mqa",
        "position": "rope",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": 1,
        "is_bigram": False
    },
    {
        "id": 9,
        "name": "GQA Balanced (4 KV Heads)",
        "attention": "gqa",
        "position": "sinusoidal",
        "feedforward": "geglu",
        "norm": "rms",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": 4,
        "is_bigram": False
    },
    {
        "id": 10,
        "name": "Absolute Position GPT",
        "attention": "mha",
        "position": "absolute",
        "feedforward": "swiglu",
        "norm": "layer",
        "d_model": 512,
        "heads": 8,
        "layers": 4,
        "hidden_dim": 2048,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 11,
        "name": "Lightweight Compact (2 Layers)",
        "attention": "mha",
        "position": "sinusoidal",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 256,
        "heads": 4,
        "layers": 2,
        "hidden_dim": 1024,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 12,
        "name": "Heavy Deep GPT (6 Layers)",
        "attention": "mha",
        "position": "sinusoidal",
        "feedforward": "swiglu",
        "norm": "rms",
        "d_model": 768,
        "heads": 12,
        "layers": 6,
        "hidden_dim": 3072,
        "num_kv_heads": None,
        "is_bigram": False
    },
    {
        "id": 13,
        "name": "Bigram Baseline Model",
        "attention": "none",
        "position": "none",
        "feedforward": "none",
        "norm": "none",
        "d_model": vocab_size,
        "heads": 0,
        "layers": 0,
        "hidden_dim": 0,
        "num_kv_heads": None,
        "is_bigram": True
    }
]

@torch.no_grad()
def evaluate_preset(cfg, num_batches=25):
    torch.manual_seed(42)
    if cfg["is_bigram"]:
        model = BigramLanguageModel().to(device)
    else:
        model = GPT(
            vocab_size=vocab_size,
            d_model=cfg["d_model"],
            num_heads=cfg["heads"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["layers"],
            attention_type=cfg["attention"],
            normalization_type=cfg["norm"],
            feedforward_type=cfg["feedforward"],
            position_encoding=cfg["position"],
            num_kv_heads=cfg["num_kv_heads"]
        ).to(device)
        
        # Load trained state dict if default architecture config matches
        checkpoint_path = "checkpoints/gpt_character.pth"
        if cfg["id"] == 1 and os.path.exists(checkpoint_path):
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            except Exception:
                pass

    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    val_losses = []
    for _ in range(num_batches):
        xb, yb = get_batch("val")
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        val_losses.append(loss.item())

    avg_val_loss = float(np.mean(val_losses))
    perplexity = math.exp(avg_val_loss)

    # Benchmark Throughput
    prompt = torch.tensor([encode("Once upon a time")], dtype=torch.long, device=device)
    gen_tokens = 100
    start = time.time()
    _ = model.generate(prompt, max_new_tokens=gen_tokens)
    elapsed = time.time() - start
    throughput = gen_tokens / elapsed if elapsed > 0 else 0

    return {
        "id": cfg["id"],
        "name": cfg["name"],
        "attention": cfg["attention"].upper(),
        "position": cfg["position"],
        "feedforward": cfg["feedforward"].upper(),
        "norm": cfg["norm"].upper(),
        "params_m": num_params / 1e6,
        "val_loss": avg_val_loss,
        "val_ppl": perplexity,
        "tokens_sec": throughput
    }

def main():
    print(f"Running Ablation Benchmark across 13 Presets on device: {device}...")
    results = []

    for cfg in PRESETS:
        res = evaluate_preset(cfg)
        results.append(res)
        print(f"Preset {res['id']:2d} | {res['name']:<35} | Params: {res['params_m']:6.2f}M | Val Loss: {res['val_loss']:.4f} | PPL: {res['val_ppl']:6.2f} | Speed: {res['tokens_sec']:6.1f} tok/s")

    # Write CSV
    csv_path = "ablation_results.csv"
    fieldnames = ["id", "name", "attention", "position", "feedforward", "norm", "params_m", "val_loss", "val_ppl", "tokens_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV ablation results to {csv_path}")

    # Output Markdown Table
    print("\n" + "="*80)
    print("                      ABLATION RESULTS MARKDOWN TABLE")
    print("="*80)
    print("| Preset # | Architecture Preset Name | Attention | Position | FFN | Norm | Params (M) | Val Loss | Val Perplexity | Speed (tok/s) |")
    print("| :-: | :--- | :---: | :---: | :---: | :---: | :-: | :-: | :-: | :-: |")
    for r in results:
        print(f"| **{r['id']}** | **{r['name']}** | `{r['attention']}` | `{r['position']}` | `{r['feedforward']}` | `{r['norm']}` | `{r['params_m']:.2f}M` | `{r['val_loss']:.4f}` | `{r['val_ppl']:.2f}` | `{r['tokens_sec']:.1f}` |")

if __name__ == "__main__":
    main()

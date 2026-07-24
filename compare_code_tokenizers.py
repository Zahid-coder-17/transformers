import os
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from gpt import GPT
from tokenization.bpe import BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer
from tokenization.byte_bpe import ByteBPETokenizer
from tokenization.regex_bpe import RegexBPETokenizer
from tokenization.gpt_tokenizer import GPTTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeCharacterTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.inverse_vocab = {i: c for c, i in self.vocab.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.vocab.get(c, 0) for c in text if c in self.vocab]

    def decode(self, ids):
        return "".join([self.inverse_vocab.get(i, "") for i in ids])


def train_and_eval_model(vocab_size, get_batch_fn, max_iters=50):
    model = GPT(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=2,
        hidden_dim=256,
        num_layers=2,
        attention_type="mha",
        normalization_type="rms",
        feedforward_type="swiglu",
        position_encoding="sinusoidal"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for i in range(max_iters):
        xb, yb = get_batch_fn("train")
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(10):
            xb, yb = get_batch_fn("val")
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            val_losses.append(loss.item())

    avg_val_loss = float(np.mean(val_losses))
    perplexity = math.exp(avg_val_loss)

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    start_t = time.time()
    with torch.no_grad():
        out = model.generate(ctx, max_new_tokens=50)
    elapsed = time.time() - start_t
    speed = 50 / elapsed if elapsed > 0 else 0

    return avg_val_loss, perplexity, speed, model, out[0].tolist()


def main():
    print("Running Code Corpus Tokenizer Benchmarking and Training Comparison...")

    code_path = "data/code_input.txt"
    if not os.path.exists(code_path):
        sample_text = "def forward(self, x):\n    return self.w2(F.silu(self.w1(x)))"
    else:
        with open(code_path, "r", encoding="utf-8") as f:
            sample_text = f.read(15000)

    short_sample = "class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.q_proj = nn.Linear(d_model, d_model)"

    tokenizers = {}

    print("Fitting Code Character Tokenizer...")
    char_tok = CodeCharacterTokenizer(sample_text)

    print("Fitting Standard BPE on Code Corpus...")
    bpe_tok = BPETokenizer(vocab_size=128)
    bpe_tok.fit(sample_text)
    tokenizers["Standard BPE"] = bpe_tok

    print("Fitting WordPiece on Code Corpus...")
    wp_tok = WordPieceTokenizer(vocab_size=128)
    wp_tok.fit(sample_text)
    tokenizers["WordPiece"] = wp_tok

    print("Fitting SentencePiece on Code Corpus...")
    sp_tok = SentencePieceTokenizer(vocab_size=128, model_prefix="spm_code")
    sp_tok.fit(sample_text, input_file=code_path)
    tokenizers["SentencePiece"] = sp_tok

    print("Fitting Byte-BPE on Code Corpus...")
    byte_bpe_tok = ByteBPETokenizer(vocab_size=128)
    byte_bpe_tok.fit(sample_text)
    tokenizers["Byte-Level BPE"] = byte_bpe_tok

    print("Fitting Regex-BPE on Code Corpus...")
    code_pattern = r"""[a-zA-Z_]\w*|\d+|==|!=|<=|>=|\+\+|--|->|[:;{}()\[\]=+\-*/&|^%!<>,.]|\s+"""
    regex_tok = RegexBPETokenizer(vocab_size=128, pattern=code_pattern)
    regex_tok.fit(sample_text)
    tokenizers["Regex BPE"] = regex_tok

    print("Fitting GPT Tokenizer on Code Corpus...")
    gpt_tok = GPTTokenizer(vocab_size=128)
    gpt_tok.tokenizer.pattern = code_pattern
    gpt_tok.tokenizer.compiled_pattern = re.compile(code_pattern) if hasattr(gpt_tok.tokenizer, "compiled_pattern") else None
    gpt_tok.fit(sample_text)
    tokenizers["GPT Tokenizer"] = gpt_tok

    raw_bytes = len(short_sample.encode("utf-8"))

    results = []

    char_ids = char_tok.encode(short_sample)
    char_comp = raw_bytes / len(char_ids) if len(char_ids) > 0 else 1.0

    def make_char_batch(split):
        encoded_full = char_tok.encode(sample_text)
        data = torch.tensor(encoded_full, dtype=torch.long)
        n = int(0.9 * len(data))
        split_d = data[:n] if split == "train" else data[n:]
        max_i = max(1, len(split_d) - 32 - 1)
        ix = torch.randint(0, max_i, (8,))
        x = torch.stack([split_d[i:i+32] for i in ix])
        y = torch.stack([split_d[i+1:i+32+1] for i in ix])
        return x, y

    val_loss, ppl, speed, _, gen_ids = train_and_eval_model(char_tok.vocab_size, make_char_batch, max_iters=50)
    gen_text = char_tok.decode(gen_ids)

    results.append({
        "name": "Character",
        "vocab_size": char_tok.vocab_size,
        "encoded_tokens": len(char_ids),
        "compression_ratio": char_comp,
        "val_loss": val_loss,
        "perplexity": ppl,
        "speed": speed,
        "generated_sample": gen_text[:120]
    })

    for name, tok in tokenizers.items():
        encoded = tok.encode(short_sample)
        num_toks = len(encoded) if len(encoded) > 0 else 1
        comp_ratio = raw_bytes / num_toks
        v_size = len(tok.vocab) if hasattr(tok, "vocab") and tok.vocab else 128

        def make_code_batch(t):
            def batch_fn(split):
                if hasattr(t, "get_batch"):
                    return t.get_batch(split, data_path=code_path)
                encoded_full = t.encode(sample_text)
                data = torch.tensor(encoded_full, dtype=torch.long)
                n = int(0.9 * len(data))
                split_d = data[:n] if split == "train" else data[n:]
                max_i = max(1, len(split_d) - 32 - 1)
                ix = torch.randint(0, max_i, (8,))
                x = torch.stack([split_d[i:i+32] for i in ix])
                y = torch.stack([split_d[i+1:i+32+1] for i in ix])
                return x, y
            return batch_fn

        val_l, p_pl, spd, _, g_ids = train_and_eval_model(v_size, make_code_batch(tok), max_iters=50)
        dec_text = tok.decode(g_ids) if hasattr(tok, "decode") else str(g_ids)

        results.append({
            "name": name,
            "vocab_size": v_size,
            "encoded_tokens": num_toks,
            "compression_ratio": comp_ratio,
            "val_loss": val_l,
            "perplexity": p_pl,
            "speed": spd,
            "generated_sample": dec_text[:120]
        })

    os.makedirs("assets", exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor("#0B0F19")

    names = [r["name"] for r in results]
    comp_ratios = [r["compression_ratio"] for r in results]
    losses = [r["val_loss"] for r in results]
    ppls = [r["perplexity"] for r in results]
    speeds = [r["speed"] for r in results]

    ax1 = axes[0, 0]
    ax1.set_facecolor("#111827")
    bars1 = ax1.bar(names, comp_ratios, color="#3B82F6", edgecolor="white", linewidth=0.5)
    ax1.set_title("Code Compression Ratio (Bytes / Token)", color="white", fontweight="bold")
    ax1.set_ylabel("Compression Ratio", color="#9CA3AF")
    ax1.tick_params(axis="x", rotation=30)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", color="white", fontsize=9)

    ax2 = axes[0, 1]
    ax2.set_facecolor("#111827")
    bars2 = ax2.bar(names, losses, color="#10B981", edgecolor="white", linewidth=0.5)
    ax2.set_title("Code GPT Validation Cross-Entropy Loss", color="white", fontweight="bold")
    ax2.set_ylabel("Loss", color="#9CA3AF")
    ax2.tick_params(axis="x", rotation=30)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", color="white", fontsize=9)

    ax3 = axes[1, 0]
    ax3.set_facecolor("#111827")
    bars3 = ax3.bar(names, ppls, color="#8B5CF6", edgecolor="white", linewidth=0.5)
    ax3.set_title("Code Perplexity (Lower is Better)", color="white", fontweight="bold")
    ax3.set_ylabel("Perplexity", color="#9CA3AF")
    ax3.tick_params(axis="x", rotation=30)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.1f}", ha="center", va="bottom", color="white", fontsize=9)

    ax4 = axes[1, 1]
    ax4.set_facecolor("#111827")
    bars4 = ax4.bar(names, speeds, color="#F59E0B", edgecolor="white", linewidth=0.5)
    ax4.set_title("Code Generation Speed (Tokens / Sec)", color="white", fontweight="bold")
    ax4.set_ylabel("Tokens / Sec", color="#9CA3AF")
    ax4.tick_params(axis="x", rotation=30)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}", ha="center", va="bottom", color="white", fontsize=9)

    plt.suptitle("Source Code Tokenizer Benchmarks & Model Performance", color="white", fontweight="bold", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("assets/code_tokenizer_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Code tokenizer comparison plot to assets/code_tokenizer_comparison.png")

    print("\n--- Summary of Code Tokenizer Benchmarking Results ---")
    for r in results:
        print(f"[{r['name']}] Vocab: {r['vocab_size']} | Compression: {r['compression_ratio']:.2f} B/tok | Loss: {r['val_loss']:.4f} | PPL: {r['perplexity']:.2f} | Speed: {r['speed']:.1f} tok/s")


if __name__ == "__main__":
    main()

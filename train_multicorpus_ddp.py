import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT


def setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def load_corpora():
    corpora = {
        "english": open("data/input.txt",         encoding="utf-8").read(),
        "arabic":  open("data/arabic_input.txt",   encoding="utf-8").read(),
        "code":    open("data/code_input.txt",     encoding="utf-8").read(),
    }
    all_chars  = sorted(set("".join(corpora.values())))
    stoi       = {c: i for i, c in enumerate(all_chars)}
    itos       = {i: c for i, c in enumerate(all_chars)}
    vocab_size = len(all_chars)
    return corpora, stoi, itos, vocab_size


def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


class BalancedCorpusSampler:
    def __init__(self, corpora_tensors, block_size, batch_size, device):
        n_corpora = len(corpora_tensors)
        assert batch_size % n_corpora == 0
        self.corpora    = corpora_tensors
        self.block_size = block_size
        self.per_corpus = batch_size // n_corpora
        self.device     = device
        self.offsets    = torch.arange(block_size, device=device).unsqueeze(0)

    def get_batch(self, split="train"):
        xs, ys = [], []
        for corpus in self.corpora.values():
            data    = corpus[split]
            max_idx = len(data) - self.block_size - 1
            ix      = torch.randint(0, max_idx, (self.per_corpus, 1), device=self.device)
            indices = ix + self.offsets
            xs.append(data[indices])
            ys.append(data[indices + 1])
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def get_lr(step, warmup_steps, max_iter, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_iter - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(rank, world_size):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        print(f"[DDP] World size : {world_size} process(es)", flush=True)
        print(f"[DDP] Backend    : {'nccl' if torch.cuda.is_available() else 'gloo'}", flush=True)

    corpora, stoi, itos, vocab_size = load_corpora()

    if is_main(rank):
        print(f"\nVocab size       : {vocab_size}", flush=True)
        for name, text in corpora.items():
            print(f"  {name:8s}     : {len(text):>12,} chars", flush=True)

    block_size = 256
    batch_size = 63

    corpora_tensors = {}
    for name, text in corpora.items():
        encoded = torch.tensor(encode(text, stoi), dtype=torch.long)
        n = int(0.9 * len(encoded))
        corpora_tensors[name] = {
            "train": encoded[:n].to(device),
            "val":   encoded[n:].to(device),
        }

    sampler = BalancedCorpusSampler(corpora_tensors, block_size, batch_size, device)

    model = GPT(
        vocab_size         = vocab_size,
        d_model            = 512,
        num_heads          = 8,
        hidden_dim         = 2048,
        num_layers         = 4,
        attention_type     = "mha",
        normalization_type = "rms",
        feedforward_type   = "swiglu",
        position_encoding  = "sinusoidal",
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    if is_main(rank):
        raw = model.module if world_size > 1 else model
        n_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"\nModel parameters : {n_params:,} ({n_params / 1e6:.2f}M)", flush=True)
        print(f"Effective batch  : {batch_size * world_size} ({batch_size} x {world_size} GPU(s))\n", flush=True)

    max_iter     = 10_000
    warmup_steps = 200
    max_lr       = 5e-4
    min_lr       = 1e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    losses = []

    if is_main(rank):
        print("Starting DDP Multi-Corpus Training ...", flush=True)
        print("Optimisations: DDP + Balanced Sampling + AMP + Cosine LR + Grad Clip\n", flush=True)

    model.train()
    for step in range(max_iter):
        lr = get_lr(step, warmup_steps, max_iter, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = sampler.get_batch("train")
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if is_main(rank):
            loss_val = loss.item()
            losses.append(loss_val)
            if step % 1000 == 0 or step == max_iter - 1:
                print(f"Step {step:5d}/{max_iter}  |  LR: {lr:.6f}  |  Loss: {loss_val:.4f}", flush=True)

    if is_main(rank):
        os.makedirs("checkpoints", exist_ok=True)
        raw = model.module if world_size > 1 else model
        torch.save(raw.state_dict(), "checkpoints/gpt_multicorpus.pth")
        torch.save({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, "checkpoints/multicorpus_vocab.pth")
        print("\nSaved checkpoint  ->  checkpoints/gpt_multicorpus.pth",    flush=True)
        print("Saved vocab map   ->  checkpoints/multicorpus_vocab.pth\n", flush=True)

        os.makedirs("assets", exist_ok=True)
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 5), dpi=300)

        steps_range = list(range(len(losses)))
        plt.plot(steps_range, losses, alpha=0.35, color="#F59E0B", label="Raw Step Loss")

        window = 50
        moving_avg = [
            sum(losses[max(0, i - window): i + 1]) / len(losses[max(0, i - window): i + 1])
            for i in range(len(losses))
        ]
        plt.plot(steps_range, moving_avg, color="#FBBF24", linewidth=2.5,
                 label=f"Moving Average (window={window})")

        plt.title("Multi-Corpus DDP Training Loss  (English + Arabic + Code)",
                  fontsize=13, fontweight="bold", pad=15, color="white")
        plt.xlabel("Training Step",      fontsize=12, labelpad=10)
        plt.ylabel("Cross-Entropy Loss", fontsize=12, labelpad=10)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(frameon=True, facecolor="#1E1E1E", edgecolor="none")

        plt.annotate(f"Start: {losses[0]:.2f}", xy=(0, losses[0]),
                     xytext=(500, losses[0] + 0.3),
                     arrowprops=dict(facecolor="#FF6B6B", shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, fontweight="bold", color="#FF6B6B")
        plt.annotate(f"Final: {losses[-1]:.2f}", xy=(len(losses) - 1, losses[-1]),
                     xytext=(len(losses) - 2500, losses[-1] + 0.8),
                     arrowprops=dict(facecolor="#4EBD40", shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, fontweight="bold", color="#4EBD40")

        plt.tight_layout()
        plt.savefig("assets/multicorpus_ddp_loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved loss curve  ->  assets/multicorpus_ddp_loss_curve.png", flush=True)

        print("\n--- Multi-Corpus DDP Generation Samples ---", flush=True)
        raw.eval()
        for prompt in ["Once upon a time", "def fibonacci(", "\u0645\u0631\u062d\u0628\u0627"]:
            ctx = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
            out = decode(raw.generate(ctx, max_new_tokens=150)[0].tolist(), itos)
            print(f"\n[Prompt: {repr(prompt)}]")
            print(out.encode("utf-8", errors="replace").decode("utf-8"), flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        train(local_rank, world_size)
    else:
        n_gpus     = torch.cuda.device_count()
        world_size = max(n_gpus, 1)
        if world_size == 1:
            print("[INFO] Single GPU / CPU detected — running as single DDP process.")
            train(0, 1)
        else:
            print(f"[INFO] Spawning {world_size} DDP processes across {n_gpus} GPU(s).")
            mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

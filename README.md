---
title: Modular GPT Playground
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Modular Transformer Architectures & Multi-Corpus Subword Tokenizer Engine

[![CI](https://github.com/Zahid-coder-17/transformers/actions/workflows/test.yml/badge.svg)](https://github.com/Zahid-coder-17/transformers/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A comprehensive PyTorch implementation of modular Transformer components, attention variants (Multi-Head, Multi-Query, Grouped-Query), positional encodings (Sinusoidal, Learned, RoPE, ALiBi), normalization layers (RMSNorm, LayerNorm), feed-forward networks (SwiGLU, GEGLU), and **7 Subword Tokenization Algorithms** across English, Arabic, and Python Code corpora.

---

## 🚀 Key Features & Modules

1. **Modular Transformer Architecture**:
   - **Attention Mechanisms**: Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Self-Attention.
   - **Positional Encodings**: Rotary Position Embeddings (RoPE), Attention with Linear Biases (ALiBi), Sinusoidal Encodings, Learned Embeddings, and Absolute Lookups.
   - **Feed-Forward Networks**: SwiGLU (LLaMA/Mistral) and GEGLU (GPT-2/BERT).
   - **Normalization Layers**: RMSNorm and LayerNorm.

2. **`zahidgpt` Python Library**:
   - Easily importable Python package (`pip install git+https://github.com/Zahid-coder-17/transformers`) for text generation and fine-tuning.

3. **FastAPI REST API Server**:
   - Production REST API endpoint (`python api_server.py`) serving `/generate` and `/health` for app integrations.

4. **Hugging Face Hub Model Repository**:
   - Pre-trained 17.45M parameter DDP multi-corpus model hosted live at [`Zahid2005/modular-gpt-multicorpus`](https://huggingface.co/Zahid2005/modular-gpt-multicorpus).

2. **Subword Tokenizer Engine (From Scratch & Integrated)**:
   - **Character Tokenizer**: Fine-grained byte/character representation.
   - **Standard BPE Tokenizer**: Byte Pair Encoding built from scratch.
   - **WordPiece Tokenizer**: WordPiece algorithm built from scratch (BERT style with `##` subwords).
   - **SentencePiece Tokenizer**: Unigram/BPE SentencePiece subword integration.
   - **Byte-Level BPE Tokenizer**: GPT-2 byte-to-unicode byte BPE from scratch.
   - **Regex-BPE Tokenizer**: GPT-4 style regex splitting pre-tokenization with BPE pair merging.
   - **GPT Tokenizer**: Full GPT-style Regex Byte-BPE with special token handling (`<|endoftext|>`).

3. **Multi-Corpus Evaluation (English, Arabic & Code)**:
   - **English Corpus**: TinyStories narrative text (`roneneldan/TinyStories`).
   - **Arabic Corpus**: Rich Arabic literature and prose corpus.
   - **Python Code Corpus**: Hugging Face Python Code dataset (`flytech/python-codes-25k` - 403,441 bytes).

---

## 📊 Subword Tokenizer Multi-Corpus Benchmarks

| Corpus | Tokenizer | Vocab Size | Compression Ratio (Bytes / Token) | Cross-Entropy Loss | Perplexity (PPL) | Speed (Tokens / Sec) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Python Code** | **Character** | 91 | 1.00 B/tok | 3.1470 | 23.27 | 261.0 tok/s |
| | **Standard BPE** | 128 | 1.27 B/tok | 3.9970 | 54.44 | 328.6 tok/s |
| | **WordPiece** | 145 | 1.24 B/tok | 3.5335 | 34.24 | 396.3 tok/s |
| | **SentencePiece** | 128 | 14.27 B/tok | 1.2155 | 3.37 | 433.4 tok/s |
| | **Byte-Level BPE** | 256 | 1.00 B/tok | 3.2765 | 26.48 | 349.6 tok/s |
| | **Regex BPE** | 256 | 1.00 B/tok | 3.1542 | 23.43 | 411.2 tok/s |
| | **GPT Tokenizer** | 256 | 1.00 B/tok | 3.3534 | 28.60 | 411.1 tok/s |
| **Arabic** | **Character** | 43 | 1.81 B/tok | 2.6630 | 14.34 | 220.0 tok/s |
| | **Standard BPE** | 128 | 3.00 B/tok | 4.3841 | 80.17 | 347.2 tok/s |
| | **WordPiece** | 128 | 2.19 B/tok | 3.2429 | 25.61 | 293.4 tok/s |
| | **SentencePiece** | 128 | 9.80 B/tok | 2.2254 | 9.26 | 445.1 tok/s |
| | **Byte-Level BPE** | 256 | 1.00 B/tok | 1.9115 | 6.76 | 336.3 tok/s |
| | **Regex BPE** | 256 | 1.00 B/tok | 1.8030 | 6.07 | 406.0 tok/s |
| | **GPT Tokenizer** | 256 | 1.00 B/tok | 1.8100 | 6.11 | 424.6 tok/s |

---

## 🏛️ Model Architecture Matrix (13 Presets)

Configurations supported via the `GPT` model interface:

| # | Preset / Target Architecture | Attention | Position Encoding | Normalization | Feedforward | Description |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **LLaMA-3 Style** | `"mha"` / `"gqa"` | `"rope"` / `"sinusoidal"` | `"rms"` | `"swiglu"` | Default architecture variant |
| **2** | **Mistral / Mixtral** | `"gqa"` | `"rope"` | `"rms"` | `"swiglu"` | Grouped-Query Attention ($N_{\text{kv}}=2$) |
| **3** | **Falcon / PaLM** | `"mqa"` | `"rope"` / `"sinusoidal"` | `"layer"` | `"swiglu"` | Multi-Query Attention ($N_{\text{kv}}=1$) |
| **4** | **GPT-2** | `"mha"` | `"learned"` | `"layer"` | `"geglu"` | Standard GPT-2 decoder layout |
| **5** | **Vaswani Standard (2017)** | `"mha"` | `"sinusoidal"` | `"layer"` | `"geglu"` | Original Transformer decoder |
| **6** | **ALiBi Transformer** | `"mha"` | `"alibi"` | `"rms"` | `"swiglu"` | Linear bias relative positioning |
| **7** | **RoPE Transformer** | `"mha"` | `"rope"` | `"rms"` | `"swiglu"` | Rotary positional embeddings |
| **8** | **Deep RMSNorm + MQA** | `"mqa"` | `"rope"` | `"rms"` | `"swiglu"` | Reduced KV cache memory layout |
| **9** | **GQA Balanced** | `"gqa"` | `"sinusoidal"` | `"rms"` | `"geglu"` | Grouped-Query Attention ($N_{\text{kv}}=4$) |
| **10** | **Absolute Position GPT** | `"mha"` | `"absolute"` | `"layer"` | `"swiglu"` | Absolute lookup embeddings |
| **11** | **Lightweight Compact** | `"mha"` | `"sinusoidal"` | `"rms"` | `"swiglu"` | 2-layer compact variant ($d_{\text{model}}=256$) |
| **12** | **Heavy Deep GPT** | `"mha"` | `"sinusoidal"` | `"rms"` | `"swiglu"` | 6-layer high-capacity variant ($d_{\text{model}}=768$) |
| **13** | **Bigram Baseline** | None | None | None | None | Non-transformer bigram table lookup |

---

## 💻 Quickstart Code Example

```python
import torch
from gpt import GPT
from tokenization.gpt_tokenizer import GPTTokenizer

# Instantiate GPT Tokenizer
tokenizer = GPTTokenizer(vocab_size=256)
tokenizer.fit("def train_gpt_model(text): return GPT(text)")

# Instantiate LLaMA-3 Style Model (MHA + SwiGLU + RMSNorm + Sinusoidal)
model = GPT(
    vocab_size=256,
    d_model=256,
    num_heads=4,
    hidden_dim=1024,
    num_layers=2,
    attention_type="mha",
    normalization_type="rms",
    feedforward_type="swiglu",
    position_encoding="sinusoidal"
)

tokens = torch.tensor([tokenizer.encode("def train_gpt_model(text):")], dtype=torch.long)
logits, loss = model(tokens)
print("Output Logits Shape:", logits.shape)
```

---

## 🛠️ Open-Source API & Fine-Tuning Guide

The `zahidgpt` library provides complete open-source APIs for **Inference**, **Fine-Tuning**, **Custom Tokenization**, and **Hugging Face Hub Integration**.

```bash
pip install git+https://github.com/Zahid-coder-17/transformers
```

### 1. Zero-Setup Inference
```python
from zahidgpt import generate

# Generates text using 17.45M parameter pre-trained weights (auto-downloaded from HF Hub)
print(generate("def fibonacci(", model_type="multicorpus"))
print(generate("Once upon a time", model_type="multicorpus"))
print(generate("مرحبا بك", model_type="multicorpus"))
```

### 2. Fine-Tune on Your Custom Text Dataset
```python
from zahidgpt import finetune

my_dataset = "def custom_function(): return 'Hello World'\n" * 100

# Fine-tune pre-trained model on your dataset in 1 line
model, losses = finetune(
    text_data=my_dataset,
    model_type="multicorpus",
    epochs=10,
    save_path="my_finetuned_model.pth"
)
```

### 3. Build Custom Transformer Architectures
```python
from zahidgpt import GPT, TransformerBlock, GPTTokenizer

# Custom 6-layer GQA SwiGLU RoPE transformer
custom_model = GPT(
    vocab_size=512,
    d_model=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=6,
    attention_type="gqa",
    position_encoding="rope",
    normalization_type="rms",
    feedforward_type="swiglu",
    num_kv_heads=2
)
```

### 4. Push & Share Models on Hugging Face Hub
```python
from zahidgpt import push_to_hub, load_from_hub

# Push fine-tuned model weights to Hugging Face
push_to_hub(repo_id="username/my-custom-gpt", checkpoint_path="my_finetuned_model.pth")

# Load model weights from any Hugging Face Hub repository
weight_path = load_from_hub(repo_id="Zahid2005/modular-gpt-multicorpus", filename="gpt_multicorpus.pth")
```


---

## 📈 Trained Model Evaluation Metrics

Evaluation metrics for the trained Modular Transformer model (`checkpoints/gpt_character.pth`):

| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Model Parameters** | `2.15M` | Total trainable parameters |
| **Training Cross-Entropy Loss** | `0.9375` | Final epoch training loss |
| **Validation Cross-Entropy Loss** | `2.3653` | Held-out validation set loss |
| **Validation Perplexity** | `10.65` | $\exp(\text{Validation Loss})$ |
| **Next-Token Character Accuracy** | `43.94%` | Correct character prediction rate |
| **Inference Latency** | `4.44 ms/token` | GPU token generation latency |
| **Inference Throughput** | `225.44 tokens/sec` | GPU generation throughput |

---

## 📁 Repository Structure

```
transformers/
├── assets/
│   ├── arabic_tokenizer_comparison.png # Arabic tokenizer benchmark chart
│   ├── code_tokenizer_comparison.png   # Python code tokenizer benchmark chart
│   ├── loss_curve.png                  # Training loss curve chart
│   ├── performance_dashboard.png       # Metrics & performance dashboard
│   └── tokenizer_comparison.png        # English tokenizer benchmark chart
├── attention/
│   ├── gqa.py                          # Grouped-Query Attention
│   ├── mha.py                          # Multi-Head Attention
│   ├── mqa.py                          # Multi-Query Attention
│   └── self_attention.py               # Single-Head Self Attention
├── data/
│   ├── arabic_input.txt                # Arabic corpus dataset
│   ├── code_input.txt                  # Python Code dataset (403,441 bytes from HF)
│   ├── download.py                     # TinyStories download script
│   └── input.txt                       # TinyStories English corpus
├── feedforward/
│   ├── geglu.py                        # GELU-Gated Linear Unit
│   └── swiglu.py                       # Swish-Gated Linear Unit
├── normalization/
│   ├── layernorm.py                    # Standard Layer Normalization
│   └── rms_norm.py                     # Root Mean Square Normalization
├── position/
│   ├── alibi.py                        # Attention with Linear Biases
│   ├── learnedpe.py                    # Learned Positional Embedding
│   ├── rope.py                         # Rotary Position Embedding
│   └── sinusodal.py                    # Fixed Sinusoidal Positional Encoding
├── tokenization/
│   ├── bpe.py                          # BPE, WordPiece (from scratch), SentencePiece
│   ├── byte_bpe.py                     # Byte-Level BPE Tokenizer
│   ├── character.py                    # Character Tokenizer
│   ├── gpt_tokenizer.py                # GPT Tokenizer with special tokens
│   └── regex_bpe.py                    # Regex BPE Tokenizer
├── utils/
│   ├── kv_cache.py                     # Key-Value Cache mechanism
│   └── mask.py                         # Causal triangular attention masking
├── app.py                              # Gradio interactive Web Application
├── compare_arabic_tokenizers.py        # Arabic tokenizer comparison script
├── compare_code_tokenizers.py          # Code tokenizer comparison script
├── compare_tokenizers.py               # English tokenizer comparison script
├── download_expanded_datasets.py       # Hugging Face dataset downloader
├── eval_and_visualize.py               # Evaluation dashboard generator
├── generate.py                         # Text generation CLI script
├── gpt.py                              # Top-level GPT model definition
├── run_ablation.py                     # Automated 13-preset ablation benchmark
├── test.py                             # Unit test suite
├── train.py                            # Model training script
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

---

## ⚡ Distributed Training with DDP (DistributedDataParallel)

The project includes a full **PyTorch DDP** training implementation (`train_multicorpus_ddp.py`) that replaces the original single-process `train_multicorpus.py`.

### What is DDP?

`DistributedDataParallel` (DDP) is PyTorch's recommended strategy for scaling model training across multiple GPUs or machines.  Each GPU runs an independent copy of the model (a *replica*) and processes a different slice of the batch.  After every backward pass, DDP uses **ring-AllReduce** to average the gradients across all replicas so every GPU's weights stay in sync.

```
┌──────────────────────────────────────────────────────────┐
│               DistributedDataParallel (DDP)              │
│                                                          │
│  GPU 0 (rank 0)    GPU 1 (rank 1)    GPU 2 (rank 2)      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │  GPT model  │   │  GPT model  │   │  GPT model  │    │
│  │  replica 0  │   │  replica 1  │   │  replica 2  │    │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │
│         │  forward + loss  │                 │           │
│         ↓                 ↓                 ↓           │
│  ┌────────────────────────────────────────────────────┐  │
│  │     AllReduce  (average gradients across GPUs)     │  │
│  └────────────────────────────────────────────────────┘  │
│         ↓                 ↓                 ↓           │
│    optimizer.step    optimizer.step    optimizer.step    │
│         │                                               │
│  Checkpoint saved by rank-0 only (no duplicate writes)  │
└──────────────────────────────────────────────────────────┘
```

### Key Improvements over `train_multicorpus.py`

| Feature | `train_multicorpus.py` | `train_multicorpus_ddp.py` |
|:---|:---:|:---:|
| Multi-GPU support | ❌ | ✅ DDP |
| Gradient sync | ❌ | ✅ AllReduce |
| Balanced corpus sampling | ❌ | ✅ Equal 1/3 per language |
| Duplicate checkpoint writes | ❌ | ✅ Rank-0 only |
| UTF-8 safe Arabic printing | ❌ | ✅ |
| Mixed-precision AMP | ✅ | ✅ |
| Cosine LR warmup | ✅ | ✅ |
| Gradient clipping | ✅ | ✅ |

### Balanced Corpus Sampling (Critical Fix)

The old script concatenated all corpora and sampled randomly, which means the model saw English ~95% of the time due to the size imbalance:

```
data/input.txt        →  8,797,812 bytes  (English)
data/code_input.txt   →    415,332 bytes  (Code)
data/arabic_input.txt →      7,686 bytes  (Arabic)  ← 1145x smaller than English
```

The new `BalancedCorpusSampler` keeps each corpus as a **separate tensor** and draws exactly `batch_size / 3` samples from each language per step — regardless of corpus size.

### How to Run

**Single GPU / CPU** (auto-detected):
```bash
python train_multicorpus_ddp.py
```

**Multi-GPU on one machine** (recommended — uses `torchrun`):
```bash
torchrun --nproc_per_node=2 train_multicorpus_ddp.py   # 2 GPUs
torchrun --nproc_per_node=4 train_multicorpus_ddp.py   # 4 GPUs
```

**Multi-Node** (e.g., 2 machines × 4 GPUs = 8 GPUs total):
```bash
# On node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 --master_addr=<NODE_0_IP> --master_port=29500 \
         train_multicorpus_ddp.py

# On node 1 (worker)
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=1 --master_addr=<NODE_0_IP> --master_port=29500 \
         train_multicorpus_ddp.py
```

### Expected Speedup

| Setup | Effective Batch | Approx. Speedup |
|:---|:---:|:---:|
| 1 GPU  (baseline) | 63  | 1× |
| 2 GPUs | 126 | ~1.8× |
| 4 GPUs | 252 | ~3.5× |
| 8 GPUs | 504 | ~6.5× |

> Speedup is sub-linear due to AllReduce communication overhead, which decreases as a fraction of total time as model size grows.

---

## 📊 Dataset Size Requirements

For the model to learn all three languages properly, the corpora need to be **roughly equal in size** and large enough for the model to learn patterns.  Recommended minimum sizes:

| Corpus | Current Size | Recommended Minimum | Good Target |
|:---|:---:|:---:|:---:|
| **English** (TinyStories) | 8.7 MB | 8 MB ✅ | 8–20 MB |
| **Code** (Python) | 415 KB | **5 MB** | 8–15 MB |
| **Arabic** | 7 KB | **5 MB** | 8–15 MB |

**Arabic** needs the most attention — 7 KB is only ~3,500 words, far too little for a character-level model to learn Arabic morphology.

#### Recommended Arabic Datasets
- [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) — `ar` subset (~800 MB, use a slice of 5–10 MB)
- [`oscar-corpus/OSCAR-2301`](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) — `ar` subset (very large, sample freely)
- [`cc100`](https://huggingface.co/datasets/cc100) — `ar` subset (~6 GB, sample 10 MB)

#### Recommended Code Datasets
- [`flytech/python-codes-25k`](https://huggingface.co/datasets/flytech/python-codes-25k) — already used, expand to full download
- [`codeparrot/github-code`](https://huggingface.co/datasets/codeparrot/github-code) — filter for Python, sample 10–15 MB
- [`bigcode/the-stack`](https://huggingface.co/datasets/bigcode/the-stack) — Python subset

---

## 🛠️ Usage Instructions

```bash
# Run unit tests
python -m unittest test.py

# Download expanded Hugging Face datasets (Arabic & Python Code)
python download_expanded_datasets.py

# Benchmark tokenizers on English, Arabic, and Code corpora
python compare_tokenizers.py
python compare_arabic_tokenizers.py
python compare_code_tokenizers.py

# Train single-GPU model (original)
python train.py

# Train multi-corpus model with DDP (single GPU / auto-detect)
python train_multicorpus_ddp.py

# Train multi-corpus model with DDP (multi-GPU via torchrun)
torchrun --nproc_per_node=<N> train_multicorpus_ddp.py

# Evaluate model metrics & generate visual dashboard
python eval_and_visualize.py

# Generate text from trained model
python generate.py --prompt "Once upon a time" --max_tokens 300

# Launch Gradio Interactive Web Application
python app.py
```

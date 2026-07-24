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

# Train Modular Transformer Model
python train.py

# Evaluate model metrics & generate visual dashboard
python eval_and_visualize.py

# Launch Gradio Interactive Web Application
python app.py
```

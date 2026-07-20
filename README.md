# Transformers Components & Algorithms 🚀

A modular collection of modern Transformer architecture components, attention mechanisms, and related algorithms. This repository provides clean and understandable implementations of various building blocks used in state-of-the-art Large Language Models (LLMs).

> **Note:** 🚧 *This repository is under active development and will be updated soon! Expect new experiments, optimizations, and more cutting-edge algorithms to be added in the near future.*

## 📑 Index of Algorithms & Components

Here is a list of the currently implemented algorithms and modules:

### 🧠 Attention Mechanisms
- **Self-Attention** (`attention/self_attention.py`)
- **Multi-Head Attention** (`attention/multi_head_attention.py`)
- **Grouped-Query Attention (GQA)** (`attention/gqa.py`)
- **Multi-Query Attention (MQA)** (`attention/mqa.py`)

### 📍 Positional Encodings
- **RoPE (Rotary Position Embedding)** (`position/RoPE.py`)
- **ALiBi (Attention with Linear Biases)** (`position/alibi.py`)

### ⚖️ Normalization
- **RMSNorm (Root Mean Square Normalization)** (`normalization/rms_norm.py`)

### ⚡ Feedforward Networks
- **SwiGLU** (`feedforward/swiglu.py`)

### 🛠️ Utilities & Architectures
- **KV Cache** (`utils/kv_cache.py`) - For efficient autoregressive generation
- **Attention Masking** (`utils/mask.py`) - Utilities for causal and padding masks
- **Transformer Block** (`transformer.py`) - Standard Transformer implementation
- **GPT** (`gpt.py`) - A full GPT-style model architecture assembly

---

*More components coming soon!*

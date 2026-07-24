---
language:
- en
- ar
- code
license: mit
tags:
- pytorch
- transformer
- text-generation
- multi-corpus
- custom-gpt
metrics:
- perplexity
---

# Modular GPT Multi-Corpus (17.45M Parameters)

This is a custom PyTorch-native **Modular GPT** model trained on a balanced multi-corpus spanning **English (TinyStories)**, **Arabic (Wikipedia/Literature)**, and **Python Code**.

## Architecture & Hyperparameters
- **Model Parameters**: 17,451,520 (17.45 Million)
- **Vocabulary Size**: 628 characters (shared multilingual/code character vocab)
- **Embedding Dimension ($d_{\text{model}}$)**: 512
- **Attention**: Multi-Head Attention (8 heads)
- **Positional Encoding**: Sinusoidal
- **Feed-Forward**: SwiGLU (Hidden Dimension: 2048)
- **Normalization**: RMSNorm
- **Layers**: 4 Transformer blocks

## Training Optimizations
- **Distributed Data Parallel (DDP)**
- **Balanced Corpus Sampler**: Equal 1/3 sampling weight across English, Arabic, and Code to prevent language imbalance
- **Automatic Mixed Precision (AMP)**
- **Cosine Learning Rate Schedule with Warmup** (Max LR: 5e-4, Min LR: 1e-5)
- **Gradient Clipping**: 1.0 norm

## Usage with `zahidgpt` Python Library

```bash
pip install git+https://github.com/Zahid-coder-17/transformers
```

```python
from zahidgpt import generate

# Generate Code
print(generate("def fibonacci(", model_type="multicorpus"))

# Generate English
print(generate("Once upon a time", model_type="multicorpus"))

# Generate Arabic
print(generate("مرحبا", model_type="multicorpus"))
```

## Quick CLI Usage
```bash
python generate.py --prompt "def train_model(" --max_tokens 200
```

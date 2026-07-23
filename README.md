# Modular Transformer Architectures

[![CI](https://github.com/Zahid-coder-17/transformers/actions/workflows/test.yml/badge.svg)](https://github.com/Zahid-coder-17/transformers/actions/workflows/test.yml)
[![License: MIT](https://img.shields.gradient.is/badge/License-MIT-blue.svg)](LICENSE)

A PyTorch implementation of modular Transformer components, attention variants (Multi-Head, Multi-Query, Grouped-Query), positional encodings (Sinusoidal, Learned, RoPE, ALiBi), normalization layers (RMSNorm, LayerNorm), and feed-forward networks (SwiGLU, GEGLU).

The repository provides a single configurable `GPT` model interface allowing modular composition of transformer blocks.

---

## Component Registry

| Category | Option | Implementation | Description |
| :--- | :--- | :--- | :--- |
| **Attention** | `"mha"` | [`attention/mha.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/mha.py) | Multi-Head Attention |
| | `"mqa"` | [`attention/mqa.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/mqa.py) | Multi-Query Attention (shared Key/Value heads) |
| | `"gqa"` | [`attention/gqa.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/gqa.py) | Grouped-Query Attention (grouped Key/Value heads) |
| | `"self"` | [`attention/self_attention.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/self_attention.py) | Single-Head Self Attention |
| **Positional Encoding** | `"rope"` | [`position/rope.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/rope.py) | Rotary Position Embedding |
| | `"alibi"` | [`position/alibi.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/alibi.py) | Attention with Linear Biases |
| | `"sinusoidal"` | [`position/sinusodal.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/sinusodal.py) | Fixed Sinusoidal Positional Encoding |
| | `"learned"` | [`position/learnedpe.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/learnedpe.py) | Learned Positional Embedding |
| | `"absolute"` | Built-in | Absolute Positional Embedding |
| **Normalization** | `"rms"` | [`normalization/rms_norm.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/normalization/rms_norm.py) | Root Mean Square Normalization |
| | `"layer"` | [`normalization/layernorm.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/normalization/layernorm.py) | Standard Layer Normalization |
| **Feedforward** | `"swiglu"` | [`feedforward/swiglu.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/feedforward/swiglu.py) | Swish-Gated Linear Unit |
| | `"geglu"` | [`feedforward/geglu.py`](file:///c:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/feedforward/geglu.py) | GELU-Gated Linear Unit |

---

## Model Architecture Matrix

Configurations supported via the `GPT` model interface:

| # | Preset / Target Architecture | Attention (`attention_type`) | Position Encoding (`position_encoding`) | Normalization (`normalization_type`) | Feedforward (`feedforward_type`) | Notes |
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

## Quickstart Code Example

```python
import torch
from gpt import GPT

# Instantiate a LLaMA-3 style model (GQA + RoPE + RMSNorm + SwiGLU)
model = GPT(
    vocab_size=92,
    d_model=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=4,
    attention_type="gqa",
    normalization_type="rms",
    feedforward_type="swiglu",
    position_encoding="rope",
    num_kv_heads=2
)

# Input tensor shape: [batch_size, sequence_length]
input_ids = torch.randint(0, 92, (2, 16))
logits, _ = model(input_ids)
print("Output Logits Shape:", logits.shape)  # Expected: [2, 16, 92]
```

---

## Evaluation & Training Metrics

Evaluation results for the 4-layer, 16.90M parameter default model trained on **TinyStories** (`roneneldan/TinyStories`):

| Metric | Value | Measurement Condition |
| :--- | :---: | :--- |
| **Train Loss** | `0.5954` | Cross-Entropy Loss (5,000 iterations) |
| **Validation Loss** | `0.7215` | Validation set batch average |
| **Train Perplexity** | `1.81` | $\exp(\text{Train Loss})$ |
| **Validation Perplexity** | `2.06` | $\exp(\text{Val Loss})$ |
| **Character Accuracy** | `77.65%` | Next-token character accuracy |
| **Inference Latency** | `7.49 ms/token` | CUDA GPU inference |
| **Inference Throughput** | `133.58 tokens/sec` | CUDA GPU inference |

---

## Ablation Results

Empirical performance evaluation across all 13 model presets on the validation dataset split (generated via `run_ablation.py`):

| Preset # | Preset Name | Attention | Position | FFN | Norm | Params (M) | Val Loss | Val Perplexity | Speed (tok/s) |
| :-: | :--- | :---: | :---: | :---: | :---: | :-: | :-: | :-: | :-: |
| **1** | **LLaMA-3 Style (Default)** | `MHA` | `sinusoidal` | `SWIGLU` | `RMS` | `16.90M` | `0.7227` | `2.06` | `123.0` |
| **2** | **Mistral-Style (GQA - 2 KV Heads)** | `GQA` | `sinusoidal` | `SWIGLU` | `RMS` | `15.33M` | `4.6773` | `107.47` | `133.2` |
| **3** | **Falcon / PaLM (MQA - 1 KV Head)** | `MQA` | `sinusoidal` | `SWIGLU` | `LAYER` | `15.07M` | `4.6037` | `99.85` | `150.0` |
| **4** | **Classic GPT-2** | `MHA` | `learned` | `GEGLU` | `LAYER` | `19.00M` | `4.6735` | `107.08` | `158.1` |
| **5** | **Vaswani Standard (2017)** | `MHA` | `sinusoidal` | `GEGLU` | `LAYER` | `16.91M` | `4.6900` | `108.85` | `146.1` |
| **6** | **ALiBi Relative Transformer** | `MHA` | `alibi` | `SWIGLU` | `RMS` | `16.90M` | `4.7308` | `113.39` | `155.6` |
| **7** | **RoPE Rotary Embedding** | `MHA` | `rope` | `SWIGLU` | `RMS` | `16.90M` | `4.7514` | `115.74` | `112.5` |
| **8** | **Deep RMSNorm + MQA** | `MQA` | `rope` | `SWIGLU` | `RMS` | `15.06M` | `4.6330` | `102.82` | `119.6` |
| **9** | **GQA Balanced (4 KV Heads)** | `GQA` | `sinusoidal` | `GEGLU` | `RMS` | `15.85M` | `4.7656` | `117.40` | `142.6` |
| **10** | **Absolute Position GPT** | `MHA` | `absolute` | `SWIGLU` | `LAYER` | `19.00M` | `4.6725` | `106.97` | `162.7` |
| **11** | **Lightweight Compact (2 Layers)** | `MHA` | `sinusoidal` | `SWIGLU` | `RMS` | `2.15M` | `4.6791` | `107.67` | `332.1` |
| **12** | **Heavy Deep GPT (6 Layers)** | `MHA` | `sinusoidal` | `SWIGLU` | `RMS` | `56.83M` | `4.6580` | `105.42` | `76.9` |
| **13** | **Bigram Baseline Model** | `NONE` | `none` | `NONE` | `NONE` | `0.01M` | `5.0653` | `158.42` | `1456.6` |

*Note: Preset 1 loads pre-trained weights (`checkpoints/gpt_character.pth`), while uncheckpointed architecture variants reflect zero-shot initial validation loss prior to full multi-preset training.*

---

## Limitations

1. **Model Scale**: Experiments are conducted on lightweight models (up to 56.8M parameters), not full-scale multi-billion parameter foundation models.
2. **Tokenization**: Uses character-level tokenization ($\text{vocab\_size}=92$). Subword tokenizers (e.g. BPE, WordPiece, Tiktoken) are not implemented.
3. **Dataset Scope**: Evaluated exclusively on TinyStories (`roneneldan/TinyStories`), a simplified synthetic English narrative corpus.
4. **Correctness Verification**: While causal masking invariants, shape dimensions, and forward passes are unit-tested, individual module implementations have not been formally verified against official reference implementations (such as Hugging Face Transformers or Meta LLaMA reference code).

---

## Repository Structure

```
transformers/
├── .github/
│   └── workflows/
│       └── test.yml             # GitHub Actions CI workflow
├── assets/
│   ├── loss_curve.png            # Training loss curve graphic
│   └── performance_dashboard.png # Multi-panel metrics dashboard
├── attention/
│   ├── gqa.py                    # Grouped-Query Attention
│   ├── mha.py                    # Multi-Head Attention
│   ├── mqa.py                    # Multi-Query Attention
│   └── self_attention.py         # Single-Head Self Attention
├── data/
│   └── download.py               # TinyStories dataset download script
├── feedforward/
│   ├── geglu.py                  # GELU-Gated Linear Unit
│   └── swiglu.py                 # Swish-Gated Linear Unit
├── normalization/
│   ├── layernorm.py              # Standard Layer Normalization
│   └── rms_norm.py               # Root Mean Square Normalization
├── position/
│   ├── alibi.py                  # Attention with Linear Biases
│   ├── learnedpe.py              # Learned Positional Embedding
│   ├── rope.py                   # Rotary Position Embedding
│   └── sinusodal.py              # Fixed Sinusoidal Positional Encoding
├── tokenization/
│   └── character.py              # Character-level tokenizer & dataset split
├── utils/
│   ├── kv_cache.py               # Key-Value Cache mechanism
│   └── mask.py                   # Causal triangular attention masking
├── app.py                        # Gradio interactive web interface
├── eval_and_visualize.py         # Model evaluation & plot script
├── generate.py                   # Text generation CLI script
├── gpt.py                        # Top-level GPT model definition
├── run_ablation.py               # Automated 13-preset ablation benchmark
├── test.py                       # Unit test suite
├── train.py                      # Training loop script
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## Usage Instructions

```bash
# Run unit tests
python test.py

# Run ablation benchmark across all 13 presets
python run_ablation.py

# Run evaluation & generate metrics dashboard
python eval_and_visualize.py

# Generate text via CLI
python generate.py --prompt "Once upon a time" --temp 0.8 --top_k 40 --top_p 0.9

# Launch interactive Gradio web interface
python app.py
```

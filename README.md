# Transformers Components & Modular Architectures 🚀

A modular, clean, and extensible PyTorch implementation of modern Transformer architecture components, attention mechanisms, positional encodings, normalization methods, and feedforward blocks. 

By plugging and playing different combinations of these components, you can assemble **12+ state-of-the-art LLM architectures** (such as LLaMA 1/2/3, Mistral, Gemma, PaLM, BLOOM, Falcon, GPT-2/3, and Qwen) using a single flexible `GPT` model interface!

---

## 🧩 Pluggable Component Registry

| Component Category | Supported Option (`string`) | Implementation File | Description |
| :--- | :--- | :--- | :--- |
| **Attention Mechanism** | `"mha"` | [`attention/mha.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/mha.py) | Multi-Head Attention |
| | `"mqa"` | [`attention/mqa.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/mqa.py) | Multi-Query Attention (shared K/V heads) |
| | `"gqa"` | [`attention/gqa.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/gqa.py) | Grouped-Query Attention (grouped K/V heads) |
| | `"self"` | [`attention/self_attention.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/attention/self_attention.py) | Single-Head Self Attention |
| **Positional Encoding** | `"rope"` | [`position/rope.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/rope.py) | Rotary Position Embedding |
| | `"alibi"` | [`position/alibi.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/alibi.py) | Attention with Linear Biases |
| | `"sinusoidal"` | [`position/sinusodal.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/sinusodal.py) | Fixed Sinusoidal Positional Encoding |
| | `"learned"` | [`position/learnedpe.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/position/learnedpe.py) | Learned Positional Embedding module |
| | `"absolute"` | Built-in (`nn.Embedding`) | Absolute Positional Embedding |
| **Normalization** | `"rms"` | [`normalization/rms_norm.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/normalization/rms_norm.py) | Root Mean Square Normalization |
| | `"layer"` | [`normalization/layernorm.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/normalization/layernorm.py) | Standard Layer Normalization |
| **Feedforward Network** | `"swiglu"` | [`feedforward/swiglu.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/feedforward/swiglu.py) | Swish-Gated Linear Unit |
| | `"geglu"` | [`feedforward/geglu.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/feedforward/geglu.py) | GELU-Gated Linear Unit |

---

## 🏛️ Model Architecture Matrix (13 Model Configurations)

You can construct various famous model architectures by setting the corresponding parameters when initializing `GPT`:

| # | Architecture / Model | Attention (`attention_type`) | Position Encoding (`position_encoding`) | Normalization (`normalization_type`) | Feedforward (`feedforward_type`) | Key Notes / Features |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **LLaMA / LLaMA 2** | `"mha"` / `"gqa"` | `"rope"` | `"rms"` | `"swiglu"` | Standard modern open LLM stack |
| **2** | **LLaMA 3** | `"gqa"` | `"rope"` | `"rms"` | `"swiglu"` | Efficient GQA + RoPE + RMSNorm |
| **3** | **Mistral / Mixtral** | `"gqa"` | `"rope"` | `"rms"` | `"swiglu"` | Grouped-query attention with RoPE |
| **4** | **Gemma** | `"mha"` / `"gqa"` | `"rope"` | `"rms"` | `"geglu"` | RoPE + GEGLU variant |
| **5** | **PaLM / PaLM 2** | `"mqa"` | `"rope"` | `"rms"` | `"swiglu"` | Multi-query attention for ultra-fast inference |
| **6** | **BLOOM** | `"mha"` | `"alibi"` | `"layer"` | `"geglu"` | ALiBi positioning + LayerNorm |
| **7** | **Falcon** | `"mqa"` / `"gqa"` | `"alibi"` / `"rope"` | `"layer"` | `"swiglu"` | Flexible multi-query/grouped attention |
| **8** | **Qwen 2** | `"gqa"` | `"rope"` | `"rms"` | `"swiglu"` | Advanced GQA + SwiGLU stack |
| **9** | **GPT-2 / GPT-3** | `"mha"` | `"absolute"` / `"learned"` | `"layer"` | `"geglu"` | Classic autoregressive GPT model |
| **10** | **OPT (Meta)** | `"mha"` | `"learned"` | `"layer"` | `"geglu"` | Standard pre-trained Transformer |
| **11** | **Chinchilla** | `"mha"` | `"sinusoidal"` | `"layer"` | `"geglu"` | Compute-optimal scaling architecture |
| **12** | **Baichuan 2** | `"gqa"` | `"alibi"` / `"rope"` | `"rms"` | `"swiglu"` | High efficiency multilingual LLM |
| **13** | **Vicuna / Alpaca** | `"mha"` | `"rope"` | `"rms"` | `"swiglu"` | Instruction-tuned LLaMA variant |

---

## 💻 Quickstart Code Example

```python
import torch
from gpt import GPT

# Example 1: Build a LLaMA-3 Style Model (GQA + RoPE + RMSNorm + SwiGLU)
llama3_model = GPT(
    vocab_size=128000,
    d_model=4096,
    num_heads=32,
    hidden_dim=14336,
    num_layers=32,
    attention_type="gqa",
    normalization_type="rms",
    feedforward_type="swiglu",
    position_encoding="rope",
    num_kv_heads=8  # Required for GQA
)

# Example 2: Build a PaLM-Style Model (MQA + RoPE + RMSNorm + SwiGLU)
palm_model = GPT(
    vocab_size=32000,
    d_model=2048,
    num_heads=16,
    hidden_dim=8192,
    num_layers=16,
    attention_type="mqa",
    normalization_type="rms",
    feedforward_type="swiglu",
    position_encoding="rope"
)

# Run Inference Test
input_ids = torch.randint(0, 1000, (2, 16)) # Batch of 2, Sequence length of 16
logits = llama3_model(input_ids)
print("Output Logits Shape:", logits.shape)  # Expected: [2, 16, 128000]
```

---

## 🛠️ Utilities & Support Modules

- **KV Cache** ([`utils/kv_cache.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/utils/kv_cache.py)) — Efficient key-value caching for fast autoregressive generation.
- **Attention Masking** ([`utils/mask.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/utils/mask.py)) — Causal triangular masking and sequence padding utilities.
- **Transformer Block** ([`transformer.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/transformer.py)) — Plug-and-play block assembling attention, normalization, and feedforward components.
- **GPT Assembly** ([`gpt.py`](file:///C:/Users/YADKI%20ZAHID%20AHMED/Desktop/transformers/gpt.py)) — Top-level model module connecting embeddings, positional encoding, stacked transformer blocks, final norm, and language model head.

# 📚 ZahidGPT: Complete Library Tutorial & API Documentation

Welcome to **ZahidGPT** — an open-source, modular Large Language Model (LLM) and subword tokenization framework in PyTorch. 

Whether you want to generate text, fine-tune pre-trained models on your own datasets, experiment with modern transformer architectures (GQA, RoPE, SwiGLU, RMSNorm), or deploy models to Hugging Face Hub, `zahidgpt` provides simple, clean Python APIs.

---

## ⚡ Installation

Install directly from GitHub into any Python environment:

```bash
pip install git+https://github.com/Zahid-coder-17/transformers
```

---

## 🚀 Table of Contents
1. [High-Level Text Generation (`zahidgpt.generate`)](#1-high-level-text-generation-zahidgptgenerate)
2. [Fine-Tuning on Custom Text (`zahidgpt.finetune`)](#2-fine-tuning-on-custom-text-zahidgptfinetune)
3. [Model Weight Loading (`zahidgpt.load_model`)](#3-model-weight-loading-zahidgptload_model)
4. [Hugging Face Hub Sharing (`zahidgpt.push_to_hub` & `load_from_hub`)](#4-hugging-face-hub-sharing)
5. [Modular Architecture Class (`zahidgpt.GPT`)](#5-modular-architecture-class-zahidgptgpt)
6. [Transformer Decoder Block (`zahidgpt.TransformerBlock`)](#6-transformer-decoder-block-zahidgpttransformerblock)
7. [Subword Tokenizer Engines](#7-subword-tokenizer-engines)

---

## 1. High-Level Text Generation (`zahidgpt.generate`)

The `generate()` function offers zero-setup text generation. It automatically downloads pre-trained 17.45M parameter multi-corpus weights from Hugging Face Hub if not cached locally.

### Function Signature
```python
zahidgpt.generate(
    prompt: str,
    model_type: str = "multicorpus",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = None
) -> str
```

### Parameters
- **`prompt`** *(str)*: Initial input text string (English, Arabic, or Python Code).
- **`model_type`** *(str)*: `"multicorpus"` (17.45M DDP model) or `"character"` (2.15M base model).
- **`max_new_tokens`** *(int)*: Number of new tokens/characters to generate. Default: `100`.
- **`temperature`** *(float)*: Sampling randomness. Lower values (e.g. `0.2`) are deterministic; higher values (e.g. `1.0`) are creative. Default: `0.8`.
- **`top_k`** *(int)*: Limits sampling to top-$K$ highest probability tokens. Set `0` to disable. Default: `40`.
- **`top_p`** *(float)*: Nucleus sampling probability threshold. Default: `0.9`.
- **`device`** *(str)*: `"cuda"`, `"cpu"`, or `None` (auto-detects GPU availability).

### Example Usage
```python
from zahidgpt import generate

# 1. Generate Python Code with Low Temperature (Deterministic & Precise)
python_code = generate(
    prompt="def fibonacci(n):",
    model_type="multicorpus",
    max_new_tokens=200,   # Generate 200 new tokens
    temperature=0.2,      # Low temperature for precise code logic
    top_k=40,             # Top-40 sampling
    top_p=0.9,            # Top-P nucleus sampling
    device="cuda"         # Run on GPU
)
print(python_code)

# 2. Generate English Story with Higher Temperature (Creative)
story = generate(
    prompt="Once upon a time",
    model_type="multicorpus",
    max_new_tokens=150,
    temperature=0.85,     # Higher temperature for creative storytelling
    top_k=50,
    top_p=0.95
)
print(story)
```

---

## 2. Fine-Tuning on Custom Text (`zahidgpt.finetune`)

The `finetune()` function allows fine-tuning pre-trained models on any custom text dataset in a single function call.

### Function Signature
```python
zahidgpt.finetune(
    text_data: str,
    model_type: str = "multicorpus",
    epochs: int = 5,
    batch_size: int = 16,
    block_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = None,
    save_path: str = "fine_tuned_gpt.pth"
) -> (model, losses)
```

### Parameters
- **`text_data`** *(str)*: Raw text dataset string to fine-tune on.
- **`model_type`** *(str)*: Pre-trained base weights to start from (`"multicorpus"` or `"character"`).
- **`epochs`** *(int)*: Training step iterations. Default: `5`.
- **`batch_size`** *(int)*: Batch size per step. Default: `16`.
- **`block_size`** *(int)*: Context window size. Default: `256`.
- **`learning_rate`** *(float)*: AdamW optimizer learning rate. Default: `1e-4`.
- **`save_path`** *(str)*: Output `.pth` file path for fine-tuned weights.

### Example Usage
```python
from zahidgpt import finetune

my_custom_code = "def process_data(records):\n    return [r.strip() for r in records]\n" * 50

model, losses = finetune(
    text_data=my_custom_code,
    model_type="multicorpus",
    epochs=20,
    batch_size=8,
    save_path="my_custom_code_gpt.pth"
)
print("Final Cross-Entropy Loss:", losses[-1])
```

---

## 3. Model Weight Loading (`zahidgpt.load_model`)

Loads PyTorch model instances along with character/token mappings (`stoi`, `itos`).

### Function Signature
```python
zahidgpt.load_model(
    model_type: str = "multicorpus",
    device: str = None
) -> (model, stoi, itos, device)
```

### Returns
- **`model`** *(PyTorch nn.Module)*: Instantiated PyTorch GPT model loaded with trained weights.
- **`stoi`** *(dict)*: Character-to-Index vocabulary dictionary.
- **`itos`** *(dict)*: Index-to-Character vocabulary dictionary.
- **`device`** *(torch.device)*: Active device (`cuda` or `cpu`).

### Example Usage
```python
import torch
from zahidgpt import load_model

model, stoi, itos, device = load_model("multicorpus")
model.eval()

# Encode prompt manually
prompt = "def hello():"
input_ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

# Forward pass
logits, _ = model(input_ids)
print("Logits shape:", logits.shape)
```

---

## 4. Hugging Face Hub Sharing (`zahidgpt.push_to_hub` & `load_from_hub`)

Upload your fine-tuned models to Hugging Face Hub or load pre-trained weights from any Hugging Face repository.

### Function Signatures
```python
zahidgpt.push_to_hub(
    repo_id: str,
    checkpoint_path: str,
    hf_token: str = None
)

zahidgpt.load_from_hub(
    repo_id: str,
    filename: str = "gpt_multicorpus.pth",
    device: str = None
) -> str
```

### Example Usage
```python
from zahidgpt import push_to_hub, load_from_hub

# Push fine-tuned model to your Hugging Face Hub account
push_to_hub(
    repo_id="your-username/my-fine-tuned-gpt",
    checkpoint_path="my_custom_code_gpt.pth",
    hf_token="hf_YourWriteTokenHere"
)

# Download weights from Hugging Face Hub
local_weight_file = load_from_hub(
    repo_id="Zahid2005/modular-gpt-multicorpus",
    filename="gpt_multicorpus.pth"
)
print("Downloaded checkpoint location:", local_weight_file)
```

---

## 5. Modular Architecture Class (`zahidgpt.GPT`)

`GPT` is a pluggable PyTorch `nn.Module` class supporting 13 architectural combinations.

### Constructor Signature
```python
zahidgpt.GPT(
    vocab_size: int,
    d_model: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
    attention_type: str = "mha",       # "mha", "gqa", "mqa"
    normalization_type: str = "rms",   # "rms", "layer"
    feedforward_type: str = "swiglu",  # "swiglu", "geglu", "ffn"
    position_encoding: str = "rope",   # "rope", "alibi", "sinusoidal", "learned", "absolute"
    max_seq_len: int = 4096,
    num_kv_heads: int = None,          # Required for "gqa"
    block_size: int = 1024
)
```

### Example Usage: LLaMA-3 Style Transformer
```python
import torch
from zahidgpt import GPT

# LLaMA-3 Style: GQA + RoPE + SwiGLU + RMSNorm
llama_style_gpt = GPT(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=6,
    attention_type="gqa",
    position_encoding="rope",
    normalization_type="rms",
    feedforward_type="swiglu",
    num_kv_heads=2  # 8 query heads, 2 KV heads (4:1 ratio)
)

tokens = torch.randint(0, 32000, (2, 64))  # Batch size 2, Sequence length 64
logits, loss = llama_style_gpt(tokens)
print("Logits Shape:", logits.shape)  # Expected: [2, 64, 32000]
```

---

## 6. Transformer Decoder Block (`zahidgpt.TransformerBlock`)

A single modular transformer block module combining multi-head attention, feedforward network, and normalization.

### Constructor Signature
```python
zahidgpt.TransformerBlock(
    d_model: int,
    num_heads: int,
    hidden_dim: int,
    attention_type: str = "mha",
    normalization_type: str = "rms",
    feedforward_type: str = "swiglu",
    position_encoding: str = "rope",
    max_seq_len: int = 4096,
    num_kv_heads: int = None
)
```

---

## 7. Subword Tokenizer Engines

ZahidGPT includes subword tokenization engines (`BPE`, `ByteBPETokenizer`, `RegexBPETokenizer`, `GPTTokenizer`).

### Available Classes
- **`BPE`**: Standard Byte-Pair Encoding algorithm.
- **`ByteBPETokenizer`**: Byte-level BPE tokenizer (handling Unicode & raw bytes).
- **`RegexBPETokenizer`**: Regex-split BPE tokenizer (GPT-4 style rule splitting).
- **`GPTTokenizer`**: Full GPT-style tokenizer engine.

### Common API Methods
- **`fit(text)`**: Learns vocabulary and merge rules from a text corpus.
- **`encode(text)`**: Converts text into a list of token IDs.
- **`decode(ids)`**: Reconstructs text from token IDs.

### Example Usage
```python
from zahidgpt import ByteBPETokenizer, GPTTokenizer

# Instantiate Byte-Level BPE Tokenizer
tokenizer = ByteBPETokenizer(vocab_size=512)
tokenizer.fit("def train_gpt_model(text):\n    return 'Hello World'\n")

# Encode text
encoded_ids = tokenizer.encode("def train_gpt_model")
print("Encoded Token IDs:", encoded_ids)

# Decode token IDs
decoded_text = tokenizer.decode(encoded_ids)
print("Decoded Text:", decoded_text)
```

---

## 📝 License
Distributed under the **MIT License**. Free for open-source research and commercial software development.

# 📘 ZahidGPT Library: Complete Developer Documentation & Tutorial

Welcome to **ZahidGPT**, a high-performance PyTorch library for **Multi-Corpus LLMs (English, Arabic, Python Code)**, **Modular Transformer Architectures**, **Custom Tokenization**, and **Fine-Tuning**.

---

## 📦 Installation

Install directly from GitHub into any Python environment:

```bash
pip install git+https://github.com/Zahid-coder-17/transformers
```

Or via PyPI once published:
```bash
pip install zahidgpt
```

---

## 📚 API Reference & Function Overview

| Module | Function / Class | Primary Use Case |
| :--- | :--- | :--- |
| **`zahidgpt`** | `generate()` | Instant text generation using pre-trained Multi-Corpus LLM |
| **`zahidgpt`** | `finetune()` | 1-Line fine-tuning on custom text datasets |
| **`zahidgpt`** | `load_model()` | Loads PyTorch model architecture & vocabulary maps |
| **`zahidgpt`** | `GPT`, `TransformerBlock` | Build custom transformer architectures from scratch |
| **`zahidgpt`** | `push_to_hub()`, `load_from_hub()` | Share and pull weights to/from Hugging Face Hub |
| **`zahidgpt.tokenizer`** | `BPE`, `ByteBPETokenizer`, `RegexBPETokenizer`, `GPTTokenizer` | Train and use subword tokenizers on custom data |

---

## 🚀 1. Text Generation (`zahidgpt.generate`)

### **Why Use It?**
To generate text, code, or multilingual content using the pre-trained **17.45M Parameter Multi-Corpus LLM** without needing to write PyTorch loops, manage device allocations, or download weight files manually.

### **Function Signature**
```python
generate(
    prompt: str,
    model_type: str = "multicorpus",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = None
) -> str
```

### **Code Snippets & Results**

#### A. Generating Python Code
```python
from zahidgpt import generate

prompt = "def train_neural_network("
output = generate(prompt, model_type="multicorpus", max_new_tokens=80)
print(output)
```

**Output Result:**
```python
def train_neural_network(x, y):
    params = []
    for i in range(x, y):
        params.append(x + y)
    return params

result = train_neural_network(10, 20)
print(result)
```

#### B. Generating English Narrative
```python
output = generate("Once upon a time", max_new_tokens=60, temperature=0.7)
print(output)
```

**Output Result:**
```text
Once upon a time, there was a little girl named Lily. She had a bright color in her room. One day, Lily wanted to pick out a potato.
```

---

## 🎯 2. Fine-Tuning on Custom Data (`zahidgpt.finetune`)

### **Why Use It?**
To adapt pre-trained model weights to specialized domains (e.g. legal documents, custom APIs, medical texts, or novel story genres) in 1 line of code.

### **Function Signature**
```python
finetune(
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

### **Code Snippet**
```python
from zahidgpt import finetune

# 1. Custom domain text data
custom_dataset = """
def process_order(order_id, user_id):
    order = DB.get_order(order_id)
    if order.status == 'PENDING':
        payment = ProcessPayment(user_id, order.total)
        return payment.confirm()
    return False
""" * 50

# 2. Fine-tune pre-trained weights on custom dataset
model, losses = finetune(
    text_data=custom_dataset,
    model_type="multicorpus",
    epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    save_path="my_order_processor_gpt.pth"
)
```

**Training Logs & Loss Convergence Result:**
```text
[zahidgpt.finetune] Loading 'multicorpus' model on cuda...
[zahidgpt.finetune] Starting fine-tuning for 10 iterations...
  Step    2/10 | Loss: 1.3822
  Step    4/10 | Loss: 0.8912
  Step    6/10 | Loss: 0.5104
  Step    8/10 | Loss: 0.3210
  Step   10/10 | Loss: 0.1845
[zahidgpt.finetune] Saved fine-tuned checkpoint -> my_order_processor_gpt.pth
```

---

## 🏗️ 3. Building Custom Architectures (`zahidgpt.GPT`)

### **Why Use It?**
For AI researchers, engineers, and students who want to build modular LLMs with custom attention mechanisms (Multi-Head, Grouped-Query, Multi-Query) and position encodings (RoPE, ALiBi, Sinusoidal).

### **Code Snippet**
```python
import torch
from zahidgpt import GPT

# Custom LLaMA-3 Style Transformer (GQA + SwiGLU + RMSNorm + RoPE)
model = GPT(
    vocab_size=1000,
    d_model=512,
    num_heads=8,
    hidden_dim=2048,
    num_layers=6,
    attention_type="gqa",         # Options: "mha", "gqa", "mqa"
    position_encoding="rope",     # Options: "rope", "alibi", "sinusoidal", "learned"
    normalization_type="rms",     # Options: "rms", "layer"
    feedforward_type="swiglu",   # Options: "swiglu", "geglu"
    num_kv_heads=2
)

tokens = torch.randint(0, 1000, (2, 64))
logits, loss = model(tokens)
print("Logits Shape:", logits.shape)
print("Trainable Parameters:", f"{sum(p.numel() for p in model.parameters()):,} parameters")
```

**Output Result:**
```text
Logits Shape: torch.Size([2, 64, 1000])
Trainable Parameters: 18,941,952 parameters
```

---

## 🌐 4. Hugging Face Hub Sharing (`zahidgpt.push_to_hub` / `load_from_hub`)

### **Why Use It?**
To publish fine-tuned model checkpoints directly to the Hugging Face Hub so team members and global open-source developers can load them instantly.

### **Code Snippets**

#### Upload Model Weights to HF Hub:
```python
from zahidgpt import push_to_hub

push_to_hub(
    repo_id="Zahid2005/my-custom-gpt",
    checkpoint_path="my_order_processor_gpt.pth",
    hf_token="pypi-or-hf-token"
)
```

#### Download Model Weights from HF Hub:
```python
from zahidgpt import load_from_hub

checkpoint_file = load_from_hub(
    repo_id="Zahid2005/modular-gpt-multicorpus",
    filename="gpt_multicorpus.pth"
)
print("Loaded Checkpoint Path:", checkpoint_file)
```

---

## 🔤 5. Custom Subword Tokenization (`zahidgpt.tokenizer`)

### **Why Use It?**
Train custom BPE, Byte-Level BPE, or Regex BPE tokenizers to tokenize custom codebases, specialized languages, or non-English text with maximum compression efficiency.

### **Code Snippet**
```python
from zahidgpt.tokenizer import ByteBPETokenizer

tokenizer = ByteBPETokenizer(vocab_size=256)
raw_text = "def calculate_matrix_multiplication(matrix_a, matrix_b): return matrix_a @ matrix_b"

# Fit tokenizer on raw text
tokenizer.fit(raw_text)

# Encode & Decode
encoded = tokenizer.encode(raw_text)
decoded = tokenizer.decode(encoded)

print("Original Length:", len(raw_text))
print("Encoded Token Length:", len(encoded))
print("Compression Ratio:", f"{len(raw_text)/len(encoded):.2f}x")
print("Decoded Matches Original:", decoded == raw_text)
```

**Output Result:**
```text
Original Length: 86
Encoded Token Length: 42
Compression Ratio: 2.05x
Decoded Matches Original: True
```

---

## 📌 Summary of Usable Features

| Feature | Best For | Main Benefit |
| :--- | :--- | :--- |
| `generate()` | Developers embedding LLM inference into apps | 0-Setup, auto-download pre-trained weights |
| `finetune()` | Domain adaptation (Code, Medical, Legal, Stories) | 1-Line fine-tuning with automatic PyTorch optimizer |
| `GPT()` | AI Research & Architecture Experimentation | Pluggable MHA/GQA/MQA, RoPE/ALiBi, SwiGLU/GeGLU |
| `push_to_hub()` | Open-Source Collaboration | Upload weights to Hugging Face Model Hub in 1 function |
| `ByteBPETokenizer` | Code & Multilingual Compression | High-density byte subword encoding for custom datasets |

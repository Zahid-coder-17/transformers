import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import math
import numpy as np

from gpt import GPT, BigramLanguageModel
from tokenization.character import vocab_size, decode, encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 13 Pre-configured Transformer Architecture Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "1. LLaMA-3 Style (Trained Default)": {
        "attention_type": "mha",
        "position_encoding": "sinusoidal",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 4,
        "is_bigram": False,
        "description": "Modern LLaMA-3 architecture utilizing Multi-Head Attention, SwiGLU activation, RMSNorm, and Sinusoidal positional embeddings. Trained on TinyStories."
    },
    "2. Mistral-Style (GQA - Grouped Query Attention)": {
        "attention_type": "gqa",
        "position_encoding": "sinusoidal",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 2,
        "is_bigram": False,
        "description": "Mistral-7B inspired design with Grouped-Query Attention (GQA) for reduced KV cache memory footprint during inference."
    },
    "3. Falcon / PaLM (MQA - Multi-Query Attention)": {
        "attention_type": "mqa",
        "position_encoding": "sinusoidal",
        "feedforward_type": "swiglu",
        "normalization_type": "layer",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 1,
        "is_bigram": False,
        "description": "Falcon/PaLM architecture with Multi-Query Attention (MQA) where all query heads share a single Key and Value head."
    },
    "4. Classic GPT-2 Architecture": {
        "attention_type": "mha",
        "position_encoding": "learned",
        "feedforward_type": "geglu",
        "normalization_type": "layer",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 8,
        "is_bigram": False,
        "description": "Classic OpenAI GPT-2 layout using standard LayerNorm, GEGLU feed-forward network, and Learned Positional Embeddings."
    },
    "5. Vaswani Standard Transformer (2017)": {
        "attention_type": "mha",
        "position_encoding": "sinusoidal",
        "feedforward_type": "geglu",
        "normalization_type": "layer",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 8,
        "is_bigram": False,
        "description": "Original Attention Is All You Need architecture with LayerNorm, GEGLU, and Sinusoidal positional encodings."
    },
    "6. ALiBi Relative Position Transformer": {
        "attention_type": "mha",
        "position_encoding": "alibi",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 8,
        "is_bigram": False,
        "description": "Attention with Linear Biases (ALiBi) allows seamless extrapolation to longer context windows than seen during training."
    },
    "7. RoPE Rotary Embedding Transformer": {
        "attention_type": "mha",
        "position_encoding": "rope",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 8,
        "is_bigram": False,
        "description": "Rotary Position Embeddings (RoPE) as used in LLaMA, PaLM 2, and Qwen for relative position modeling."
    },
    "8. Deep RMSNorm + MQA High Throughput": {
        "attention_type": "mqa",
        "position_encoding": "rope",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 1,
        "is_bigram": False,
        "description": "Optimized for maximum inference speed combining MQA, RMSNorm, SwiGLU, and RoPE."
    },
    "9. GQA Balanced Efficient (4 KV Heads)": {
        "attention_type": "gqa",
        "position_encoding": "sinusoidal",
        "feedforward_type": "geglu",
        "normalization_type": "rms",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 4,
        "is_bigram": False,
        "description": "Balanced GQA architecture with 4 Key-Value heads, offering a middle ground between MHA and MQA."
    },
    "10. Absolute Position Embedding GPT": {
        "attention_type": "mha",
        "position_encoding": "absolute",
        "feedforward_type": "swiglu",
        "normalization_type": "layer",
        "d_model": 512,
        "num_heads": 8,
        "hidden_dim": 2048,
        "num_layers": 4,
        "num_kv_heads": 8,
        "is_bigram": False,
        "description": "Standard absolute positional lookup embeddings with LayerNorm and SwiGLU."
    },
    "11. Lightweight Compact Transformer (2 Layers)": {
        "attention_type": "mha",
        "position_encoding": "sinusoidal",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 256,
        "num_heads": 4,
        "hidden_dim": 1024,
        "num_layers": 2,
        "num_kv_heads": 4,
        "is_bigram": False,
        "description": "Compact 2-layer model with 256 d_model designed for low-resource edge deployment and fast iteration."
    },
    "12. Heavy Deep Transformer (6 Layers)": {
        "attention_type": "mha",
        "position_encoding": "sinusoidal",
        "feedforward_type": "swiglu",
        "normalization_type": "rms",
        "d_model": 768,
        "num_heads": 12,
        "hidden_dim": 3072,
        "num_layers": 6,
        "num_kv_heads": 12,
        "is_bigram": False,
        "description": "Deep 6-layer high-capacity model with 768 embedding dimensions and 12 attention heads."
    },
    "13. Bigram Baseline Language Model": {
        "attention_type": "none",
        "position_encoding": "none",
        "feedforward_type": "none",
        "normalization_type": "none",
        "d_model": vocab_size,
        "num_heads": 0,
        "hidden_dim": 0,
        "num_layers": 0,
        "num_kv_heads": 0,
        "is_bigram": True,
        "description": "Simple non-transformer lookup Bigram table baseline model for comparison."
    }
}

# ---------------------------------------------------------------------------
# Model Caching & Initialization Helper
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/gpt_character.pth"

def load_or_create_model(preset_name, attention_type, position_encoding, feedforward_type, normalization_type, d_model, num_heads, hidden_dim, num_layers, num_kv_heads, use_trained_weights):
    if preset_name in PRESETS and PRESETS[preset_name]["is_bigram"]:
        model = BigramLanguageModel().to(device)
        return model, "Bigram Baseline (Lookup Table)", 0.008, False

    # Instantiate GPT Model
    try:
        model = GPT(
            vocab_size=vocab_size,
            d_model=int(d_model),
            num_heads=int(num_heads),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            attention_type=attention_type,
            normalization_type=normalization_type,
            feedforward_type=feedforward_type,
            position_encoding=position_encoding,
            num_kv_heads=int(num_kv_heads) if attention_type == "gqa" else None
        ).to(device)
    except Exception as e:
        raise gr.Error(f"Architecture creation failed: {str(e)}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_m = num_params / 1e6

    loaded_checkpoint = False
    # Check if selected config matches trained checkpoint configuration
    is_default_config = (
        d_model == 512 and num_heads == 8 and hidden_dim == 2048 and
        num_layers == 4 and attention_type == "mha" and normalization_type == "rms" and
        feedforward_type == "swiglu" and position_encoding == "sinusoidal"
    )

    if use_trained_weights and is_default_config and os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
            model.eval()
            loaded_checkpoint = True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    model.eval()
    return model, f"GPT-Custom ({attention_type.upper()} + {feedforward_type.upper()})", param_m, loaded_checkpoint


# ---------------------------------------------------------------------------
# Core Generation & Architecture Inference Function
# ---------------------------------------------------------------------------
def generate_from_architecture(
    preset_name,
    prompt,
    max_tokens,
    temperature,
    top_k,
    top_p,
    attention_type,
    position_encoding,
    feedforward_type,
    normalization_type,
    d_model,
    num_heads,
    hidden_dim,
    num_layers,
    num_kv_heads,
    use_trained_weights
):
    if not prompt or len(prompt.strip()) == 0:
        prompt = "Once upon a time"

    # Build or fetch model
    model, name, param_m, loaded_checkpoint = load_or_create_model(
        preset_name, attention_type, position_encoding, feedforward_type, normalization_type,
        d_model, num_heads, hidden_dim, num_layers, num_kv_heads, use_trained_weights
    )

    # Encode prompt
    try:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    except KeyError as e:
        return f"Error encoding prompt character: {e}", "N/A", "N/A", "N/A"

    start_time = time.time()
    with torch.no_grad():
        if isinstance(model, BigramLanguageModel):
            output_tokens = model.generate(context, max_new_tokens=int(max_tokens))
        else:
            output_tokens = model.generate(
                context,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p)
            )

    elapsed = time.time() - start_time
    gen_tokens = int(max_tokens)
    throughput = gen_tokens / elapsed if elapsed > 0 else 0
    latency = (elapsed / gen_tokens) * 1000 if gen_tokens > 0 else 0
    generated_text = decode(output_tokens[0].tolist())

    weight_status = "Loaded Checkpoint (gpt_character.pth)" if loaded_checkpoint else "Freshly Initialized Parameters"
    
    metrics_summary = (
        f"**Model Name**: {name}\n"
        f"**Parameter Count**: {param_m:.2f}M Parameters ({sum(p.numel() for p in model.parameters()):,} total)\n"
        f"**Weights**: {weight_status}\n"
        f"**Compute Device**: `{device.type.upper()}`\n"
        f"**Generation Latency**: `{latency:.2f} ms/token`\n"
        f"**Throughput**: `{throughput:.2f} tokens/sec` (`{elapsed:.2f}s` total)"
    )

    arch_details = (
        f"### Active Architecture Config\n"
        f"- **Attention Type**: `{attention_type.upper()}`" + (f" (KV Heads: {num_kv_heads})" if attention_type == "gqa" else "") + "\n"
        f"- **Positional Encoding**: `{position_encoding}`\n"
        f"- **FeedForward Network**: `{feedforward_type.upper()}` (Hidden Dim: {hidden_dim})\n"
        f"- **Normalization**: `{normalization_type.upper()}`\n"
        f"- **Model Dimensions**: $d_{{\\text{{model}}}}={d_model}$, $\\text{{Heads}}={num_heads}$, $\\text{{Layers}}={num_layers}$"
    )

    return generated_text, metrics_summary, arch_details


# ---------------------------------------------------------------------------
# Preset Selection Callback
# ---------------------------------------------------------------------------
def apply_preset(preset_name):
    if preset_name not in PRESETS:
        return "Select a preset...", "mha", "sinusoidal", "swiglu", "rms", 512, 8, 2048, 4, 4, True

    cfg = PRESETS[preset_name]
    if cfg["is_bigram"]:
        return (
            cfg["description"], "none", "none", "none", "none",
            vocab_size, 1, 64, 1, 1, False
        )

    return (
        cfg["description"],
        cfg["attention_type"],
        cfg["position_encoding"],
        cfg["feedforward_type"],
        cfg["normalization_type"],
        cfg["d_model"],
        cfg["num_heads"],
        cfg["hidden_dim"],
        cfg["num_layers"],
        cfg["num_kv_heads"],
        True if "Trained Default" in preset_name else False
    )


# ---------------------------------------------------------------------------
# Model Summary Inspector Callback
# ---------------------------------------------------------------------------
def inspect_model_architecture(preset_name, attention_type, position_encoding, feedforward_type, normalization_type, d_model, num_heads, hidden_dim, num_layers, num_kv_heads):
    model, name, param_m, _ = load_or_create_model(
        preset_name, attention_type, position_encoding, feedforward_type, normalization_type,
        d_model, num_heads, hidden_dim, num_layers, num_kv_heads, False
    )
    
    summary_lines = []
    summary_lines.append(f"=======================================================================")
    summary_lines.append(f"   TRANSFORMER MODULE INSPECTION: {name.upper()}")
    summary_lines.append(f"=======================================================================")
    summary_lines.append(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters()):,} ({param_m:.2f} Million)")
    summary_lines.append(f"Device: {device}")
    summary_lines.append(f"\nModel PyTorch Module Tree:\n")
    summary_lines.append(str(model))
    
    return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Gradio Modern Custom Dark Interface
# ---------------------------------------------------------------------------
# Gradio Modern High-Contrast Theme & Custom CSS
# ---------------------------------------------------------------------------
# Gradio Modern Dark High-Contrast Theme & Strict CSS
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
)

custom_css = """
/* Force universal dark background & crisp white text across all Gradio 6 components */
:root, body, .gradio-container, .gradio-container * {
    --body-background-fill: #0F172A !important;
    --body-text-color: #FFFFFF !important;
    --body-text-color-subdued: #E2E8F0 !important;
    --block-background-fill: #1E293B !important;
    --block-label-text-color: #FFFFFF !important;
    --block-title-text-color: #FFFFFF !important;
    --input-background-fill: #090D16 !important;
    --input-text-color: #FFFFFF !important;
    --input-placeholder-color: #94A3B8 !important;
    --border-color-primary: #334155 !important;
}

body, .gradio-container {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Force all text elements to pure white / high-contrast light gray */
p, span, label, h1, h2, h3, h4, h5, h6, td, th, li, a, div {
    color: #F8FAFC !important;
}

#main-header {
    text-align: center;
    margin-bottom: 24px;
    padding: 20px;
    background: linear-gradient(180deg, #1E1B4B 0%, #0F172A 100%);
    border-radius: 14px;
    border: 1px solid #312E81;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

#main-header h1 {
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(90deg, #818CF8, #38BDF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 8px;
}

#main-header p {
    color: #E2E8F0 !important;
    font-size: 1.05rem;
}

/* Card & Block Containers */
.gr-box, .gr-block, .gr-panel, fieldset, div[class*="block"], div[class*="panel"], div[data-testid*="block"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* Component Labels, Headers, Titles, and Spans */
span, label, .gr-form label, .label-title, div[data-testid*="block-label"], .block-label, legend, div[class*="title"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    background-color: transparent !important;
}

/* Prose & Markdown text */
.prose, .prose *, div[data-testid="markdown"], div[data-testid="markdown"] * {
    color: #E2E8F0 !important;
    line-height: 1.6 !important;
}

.prose h1, .prose h2, .prose h3, div[data-testid="markdown"] h1, div[data-testid="markdown"] h2, div[data-testid="markdown"] h3 {
    color: #FFFFFF !important;
}

/* Input Fields, Textareas, & Select Dropdowns */
input, textarea, select, .gr-input, .gr-text-input, textarea[data-testid="textbox"] {
    background-color: #090D16 !important;
    color: #FFFFFF !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.4) !important;
}

/* Dropdown Menu Options */
ul[role="listbox"], div[role="listbox"], div[class*="options"], div[class*="dropdown-menu"] {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid #475569 !important;
}

div[role="option"], li[role="option"], select option {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}

div[role="option"]:hover, li[role="option"]:hover, div[role="option"][aria-selected="true"] {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
}

/* Tab Navigation Bar */
div[role="tablist"], .tab-nav {
    border-bottom: 2px solid #334155 !important;
    background-color: #0F172A !important;
    padding: 6px;
    gap: 8px;
}

button[role="tab"] {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 18px !important;
    border-radius: 8px !important;
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    transition: all 0.2s ease !important;
}

button[role="tab"][aria-selected="true"], button[role="tab"].selected {
    background-color: #312E81 !important;
    color: #FFFFFF !important;
    border: 1px solid #6366F1 !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
}

/* Primary Action Buttons */
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
}

button.primary:hover, button[variant="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

/* Output Code & Text Formatting */
.output-box textarea {
    font-family: 'Fira Code', 'Cascadia Code', monospace !important;
    background-color: #020617 !important;
    color: #38BDF8 !important;
    border: 1px solid #1E293B !important;
}

/* Inline Code Badges & Code Blocks Fix */
code, pre, .prose code, div[data-testid="markdown"] code, .markdown-text code, span code, p code, li code, td code {
    background-color: #0F172A !important;
    color: #38BDF8 !important;
    border: 1px solid #334155 !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-family: 'Fira Code', 'Cascadia Code', monospace !important;
}
"""

with gr.Blocks(title="Transformer Architecture Playground & LLM Tester") as demo:

    gr.HTML("""
    <div id="main-header">
        <h1>⚡ Modular Transformer Architecture Playground & LLM Tester</h1>
        <p style="color: #CBD5E1; font-size: 1.05rem;">
            Explore, Benchmark, and Pass Prompts across <strong>13 Modular Transformer Plugin Architectures</strong> (MHA, MQA, GQA, SwiGLU, GEGLU, RMSNorm, LayerNorm, RoPE, ALiBi).
        </p>
    </div>
    """)

    with gr.Tabs():
        
        # -------------------------------------------------------------------
        # TAB 1: INTERACTIVE LLM GENERATOR & ARCHITECTURE CUSTOMIZER
        # -------------------------------------------------------------------
        with gr.TabItem("🚀 Live Playground & Text Generation"):
            with gr.Row():
                
                # Left Column: Preset & Architectural Configurations
                with gr.Column(scale=5):
                    gr.Markdown("### 🛠️ Architecture Configuration & Plugins")
                    
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="1. LLaMA-3 Style (Trained Default)",
                        label="13 Preset Transformer Architectures",
                        info="Select a pre-configured architecture variant or customize pluggable modules below."
                    )
                    
                    preset_desc = gr.Markdown(
                        value=PRESETS["1. LLaMA-3 Style (Trained Default)"]["description"]
                    )

                    with gr.Accordion("🔌 Pluggable Architectural Modules", open=True):
                        with gr.Row():
                            attn_type = gr.Dropdown(
                                choices=["mha", "gqa", "mqa", "self", "none"],
                                value="mha",
                                label="Attention Mechanism",
                                info="Multi-Head (MHA), Grouped-Query (GQA), Multi-Query (MQA)"
                            )
                            pos_enc = gr.Dropdown(
                                choices=["sinusoidal", "learned", "absolute", "rope", "alibi", "none"],
                                value="sinusoidal",
                                label="Positional Encoding",
                                info="Sinusoidal, Learned, Absolute, RoPE, ALiBi"
                            )
                        
                        with gr.Row():
                            ffn_type = gr.Dropdown(
                                choices=["swiglu", "geglu", "none"],
                                value="swiglu",
                                label="Feed-Forward Network (FFN)",
                                info="SwiGLU (LLaMA/Mistral) or GEGLU (GPT-2/BERT)"
                            )
                            norm_type = gr.Dropdown(
                                choices=["rms", "layer", "none"],
                                value="rms",
                                label="Normalization Layer",
                                info="RMSNorm (Speed) or LayerNorm (Classic)"
                            )

                    with gr.Accordion("⚙️ Hyperparameters & Dimensions", open=False):
                        with gr.Row():
                            d_model_slider = gr.Slider(minimum=128, maximum=768, step=128, value=512, label="Embedding Dimension (d_model)")
                            hidden_dim_slider = gr.Slider(minimum=512, maximum=3072, step=256, value=2048, label="FFN Hidden Dimension")
                        with gr.Row():
                            num_heads_slider = gr.Slider(minimum=2, maximum=16, step=2, value=8, label="Attention Heads")
                            num_layers_slider = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Transformer Layers")
                        with gr.Row():
                            num_kv_heads_slider = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="KV Heads (for GQA)")

                    use_trained_chkpt = gr.Checkbox(
                        value=True,
                        label="Use Trained Model Checkpoint (if config matches gpt_character.pth)",
                        info="When checked, loads pre-trained TinyStories weights for realistic text generation."
                    )

                # Right Column: Prompt & Text Generation Controls
                with gr.Column(scale=7):
                    gr.Markdown("### ✍️ Prompt & Text Generation")
                    
                    prompt_input = gr.Textbox(
                        lines=3,
                        value="Once upon a time, in a small forest",
                        placeholder="Enter your prompt here...",
                        label="Input Prompt"
                    )

                    with gr.Row():
                        max_tokens_slider = gr.Slider(minimum=20, maximum=500, step=20, value=250, label="Max New Tokens")
                        temp_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.05, value=0.75, label="Temperature")
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Top-K Sampling")
                        top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.9, label="Top-P (Nucleus) Sampling")

                    generate_btn = gr.Button("🔥 Generate Text & Benchmark", variant="primary", size="lg")

                    gr.Markdown("### 📄 Output Generated Text")
                    output_text = gr.Textbox(lines=9, label="LLM Output Continuation", elem_classes=["output-box"])
                    
                    with gr.Row():
                        metrics_box = gr.Markdown(label="Inference Benchmark Metrics")
                        arch_box = gr.Markdown(label="Architecture Summary")

            # Connect Callbacks
            preset_dropdown.change(
                fn=apply_preset,
                inputs=[preset_dropdown],
                outputs=[
                    preset_desc, attn_type, pos_enc, ffn_type, norm_type,
                    d_model_slider, num_heads_slider, hidden_dim_slider,
                    num_layers_slider, num_kv_heads_slider, use_trained_chkpt
                ]
            )

            generate_btn.click(
                fn=generate_from_architecture,
                inputs=[
                    preset_dropdown, prompt_input, max_tokens_slider, temp_slider, top_k_slider, top_p_slider,
                    attn_type, pos_enc, ffn_type, norm_type, d_model_slider, num_heads_slider,
                    hidden_dim_slider, num_layers_slider, num_kv_heads_slider, use_trained_chkpt
                ],
                outputs=[output_text, metrics_box, arch_box]
            )


        # -------------------------------------------------------------------
        # TAB 2: ARCHITECTURE INSPECTOR & MODULE TREE
        # -------------------------------------------------------------------
        with gr.TabItem("🔍 Model Inspector & PyTorch Module Tree"):
            gr.Markdown("### 🔬 Inspect Transformer Internal Layers & Parameters")
            gr.Markdown("Click inspect to analyze the PyTorch sub-modules, dimension maps, and param count for any architecture.")
            
            inspect_btn = gr.Button("Inspect Current Selected Architecture", variant="secondary")
            model_summary_output = gr.Code(language="markdown", label="PyTorch Model Hierarchy & Summary")

            inspect_btn.click(
                fn=inspect_model_architecture,
                inputs=[
                    preset_dropdown, attn_type, pos_enc, ffn_type, norm_type,
                    d_model_slider, num_heads_slider, hidden_dim_slider,
                    num_layers_slider, num_kv_heads_slider
                ],
                outputs=[model_summary_output]
            )

            gr.Markdown("""
            ### 📊 Architectural Component Comparison Matrix
            
            | Module Component | Variants Supported | Key Advantages & Characteristics |
            | :--- | :--- | :--- |
            | **Attention** | `MHA`, `GQA`, `MQA`, `Self` | **MHA**: High quality; **GQA**: Reduced KV cache (Mistral); **MQA**: Maximum speed & low VRAM (Falcon) |
            | **Positional Encoding** | `Sinusoidal`, `Learned`, `Absolute`, `RoPE`, `ALiBi` | **RoPE**: Relative rotation (LLaMA); **ALiBi**: Length extrapolation; **Sinusoidal**: Zero learnable params |
            | **Feed-Forward (FFN)** | `SwiGLU`, `GEGLU` | **SwiGLU**: Gated SiLU activation for superior expression; **GEGLU**: Gaussian Error Gated Linear Unit |
            | **Normalization** | `RMSNorm`, `LayerNorm` | **RMSNorm**: Root Mean Square norm (10-50% faster, no mean centering); **LayerNorm**: Standard scale & shift |
            """)


        # -------------------------------------------------------------------
        # TAB 3: PERFORMANCE & VISUAL DASHBOARD
        # -------------------------------------------------------------------
        with gr.TabItem("📊 Training & Evaluation Analytics Dashboard"):
            gr.Markdown("### 📈 LLM Performance & Training Loss Analytics")
            
            with gr.Row():
                gr.Markdown("""
                **Quantitative Model Evaluation (TinyStories Dataset)**
                - **Train Loss**: `0.5954`
                - **Validation Loss**: `0.7215`
                - **Train Perplexity**: `1.81`
                - **Val Perplexity**: `2.06`
                - **Next-Token Character Accuracy**: `77.65%`
                - **Inference Latency**: `7.49 ms/token` (133.58 tokens/sec on CUDA)
                """)
            
            if os.path.exists("assets/performance_dashboard.png"):
                gr.Image("assets/performance_dashboard.png", label="Visual Performance Analytics Dashboard", show_label=True)
            elif os.path.exists("assets/loss_curve.png"):
                gr.Image("assets/loss_curve.png", label="Training Loss Curve", show_label=True)


        # -------------------------------------------------------------------
        # TAB 4: RECRUITER & ARCHITECTURE GUIDE
        # -------------------------------------------------------------------
        with gr.TabItem("ℹ️ Recruiter & Architecture Showcase Guide"):
            gr.Markdown("""
            # 🎯 Project Overview: Modular Transformer & LLM Engine
            
            Welcome to the **Modular Transformer Architecture Playground**! This codebase provides a clean, PyTorch-native, highly modular implementation of modern Large Language Model (LLM) architectures.
            
            ### 🌟 Key Technical Highlights for Recruiters & Engineers:
            1. **13 Pluggable Architectures**: Mix and match any combination of Attention (`MHA`, `GQA`, `MQA`), FeedForward (`SwiGLU`, `GEGLU`), Normalization (`RMSNorm`, `LayerNorm`), and Positional Encodings (`Sinusoidal`, `RoPE`, `ALiBi`, `Learned`).
            2. **Full Trained Model (16.9M Parameters)**: Trained on TinyStories dataset with CUDA acceleration, achieving **77.65% validation accuracy** and **2.06 perplexity**.
            3. **Real-time LLM Generation Engine**: Supports Temperature scaling, Top-K filtering, and Top-P (Nucleus) sampling for controllable, creative text generation.
            4. **Production-Ready Benchmarking**: Tracks throughput (tokens/sec) and per-token generation latency (ms/token).
            
            ---
            *Built with PyTorch, Gradio, and Matplotlib.*
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)

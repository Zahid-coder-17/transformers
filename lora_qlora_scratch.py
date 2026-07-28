import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# BLOCK 1: LoRA (Low-Rank Adaptation) Linear Layer
# =====================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16, bias=True, dropout=0.0):
        super().__init__()
        # TODO: Initialize base linear layer (self.linear)
        # TODO: Freeze base linear layer weights (requires_grad = False)
        # TODO: Initialize rank, alpha, and scaling (scaling = alpha / rank)
        # TODO: Initialize LoRA A matrix (Parameter of shape [rank, in_dim]) using kaiming_uniform_
        # TODO: Initialize LoRA B matrix (Parameter of shape [out_dim, rank]) using zeros
        # TODO: Add optional dropout layer
        pass

    def forward(self, x):
        # TODO: Compute original base linear output
        # TODO: Compute low-rank adaptation update: (x @ A.T) @ B.T
        # TODO: Return original + (scaling * update)
        pass


# =====================================================================
# BLOCK 2: QLoRA (Quantized 4-bit NF4 + LoRA) Layer
# =====================================================================
class NF4Quantizer:
    """Helper for NormalFloat4 (NF4) Quantization & Dequantization"""
    @staticmethod
    def quantize_nf4(weight_fp32):
        # TODO: Quantize FP32 weight tensor into 4-bit NF4 representation & quantization scales
        pass

    @staticmethod
    def dequantize_nf4(weight_quant, scale, shape):
        # TODO: Dequantize 4-bit NF4 weight back into FP16/FP32 for forward computation
        pass


class QLoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=16, double_quant=True):
        super().__init__()
        # TODO: Store quantized base weight in 4-bit representation (uint8/packed tensor)
        # TODO: Store quantization scale (with optional double quantization)
        # TODO: Initialize LoRA A matrix (FP32/FP16 trainable parameter)
        # TODO: Initialize LoRA B matrix (FP32/FP16 trainable parameter)
        # TODO: Store scaling factor (alpha / rank)
        pass

    def forward(self, x):
        # TODO: Dequantize 4-bit base weights on the fly -> W_dequant
        # TODO: Compute base linear output: F.linear(x, W_dequant)
        # TODO: Compute LoRA update: (x @ A.T) @ B.T
        # TODO: Return base_output + (scaling * lora_update)
        pass


# =====================================================================
# BLOCK 3: Model Target Injector (Replace Linear -> LoRA / QLoRA)
# =====================================================================
def inject_lora_to_model(model, target_module_names=["c_attn", "q_proj", "v_proj"], rank=4, alpha=16, use_qlora=False):
    """
    Recursively scans model modules and replaces target nn.Linear layers
    with LoRALinear or QLoRALinear adapters.
    """
    # TODO: Traverse named_children of the model
    # TODO: Check if module name matches target_module_names
    # TODO: Instantiate LoRALinear or QLoRALinear with matching in_features & out_features
    # TODO: Copy trained base weights into adapter's base linear layer
    # TODO: Replace child module in model
    pass


# =====================================================================
# BLOCK 4: Parameter Freeze & Count Helper
# =====================================================================
def mark_only_lora_as_trainable(model):
    """
    Freezes all base model parameters (requires_grad = False)
    and enables gradients ONLY for LoRA adapter matrices (A and B).
    """
    # TODO: Iterate over all parameters in model
    # TODO: Set param.requires_grad = False for base weights
    # TODO: Set param.requires_grad = True for parameters containing 'A' or 'B' (LoRA weights)
    # TODO: Print total vs trainable parameter count & memory savings percentage
    pass


# =====================================================================
# BLOCK 5: LoRA / QLoRA Fine-Tuning & Saving Pipeline
# =====================================================================
def train_lora(model, train_dataloader, epochs=3, lr=3e-4, save_adapter_path="lora_adapters.pth"):
    # TODO: Call mark_only_lora_as_trainable(model)
    # TODO: Create AdamW optimizer targeting ONLY trainable parameters (p for p in model.parameters() if p.requires_grad)
    # TODO: Run training loop (forward pass, loss calculation, backward pass, optimizer step)
    # TODO: Save ONLY LoRA adapter weights (A and B matrices) to save_adapter_path
    pass


if __name__ == "__main__":
    print("LoRA & QLoRA Scratch Starter Code Template Created!")
    print("Fill in the TODO blocks above to train your model with LoRA/QLoRA.")

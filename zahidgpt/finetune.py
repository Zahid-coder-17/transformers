import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from zahidgpt.model import load_model

def finetune(
    text_data: str,
    model_type: str = "multicorpus",
    epochs: int = 5,
    batch_size: int = 16,
    block_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = None,
    save_path: str = "fine_tuned_gpt.pth"
):
    """
    Fine-tunes a pre-trained ZahidGPT model on custom text data.

    Parameters
    ----------
    text_data     : str         – Raw text string to fine-tune on
    model_type    : str         – "multicorpus" or "character"
    epochs        : int         – Number of training iterations/epochs
    batch_size    : int         – Training batch size
    block_size    : int         – Context window length
    learning_rate : float       – Learning rate for AdamW optimizer
    device        : str         – 'cuda' or 'cpu'
    save_path     : str         – Filepath to save fine-tuned checkpoint

    Returns
    -------
    model         : PyTorch model object
    losses        : list of loss values per epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"[zahidgpt.finetune] Loading '{model_type}' model on {dev}...")
    model, stoi, itos, dev = load_model(model_type=model_type, device=dev)
    model.train()

    # Encode user text
    encoded = torch.tensor([stoi[c] for c in text_data if c in stoi], dtype=torch.long, device=dev)
    if len(encoded) <= block_size + 1:
        raise ValueError(f"Text data length ({len(encoded)}) must be greater than block_size+1 ({block_size+1}).")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    losses = []

    print(f"[zahidgpt.finetune] Starting fine-tuning for {epochs} iterations...")

    for epoch in range(epochs):
        max_idx = len(encoded) - block_size - 1
        ix = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([encoded[i : i + block_size] for i in ix])
        y = torch.stack([encoded[i + 1 : i + block_size + 1] for i in ix])

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  Step {epoch + 1:4d}/{epochs} | Loss: {loss_val:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[zahidgpt.finetune] Saved fine-tuned checkpoint -> {save_path}")

    model.eval()
    return model, losses

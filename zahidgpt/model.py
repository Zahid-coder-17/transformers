import os
import torch
from huggingface_hub import hf_hub_download

# Ensure internal modules can be imported relative to zahidgpt package
import sys
pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from gpt import GPT

HF_REPO_ID = "Zahid2005/modular-gpt-multicorpus"

def load_model(model_type="multicorpus", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chkpt_dir = os.path.join(root_dir, "checkpoints")

    if model_type == "multicorpus":
        vocab_path = os.path.join(chkpt_dir, "multicorpus_vocab.pth")
        model_path = os.path.join(chkpt_dir, "gpt_multicorpus.pth")

        if not os.path.exists(vocab_path):
            print(f"[zahidgpt] Downloading vocab from Hugging Face Hub ({HF_REPO_ID})...")
            vocab_path = hf_hub_download(repo_id=HF_REPO_ID, filename="multicorpus_vocab.pth")

        if not os.path.exists(model_path):
            print(f"[zahidgpt] Downloading 17.45M model weights from Hugging Face Hub ({HF_REPO_ID})...")
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="gpt_multicorpus.pth")

        vocab_data = torch.load(vocab_path, map_location=device)
        stoi = vocab_data["stoi"]
        itos = vocab_data["itos"]
        vocab_size = vocab_data["vocab_size"]

        model = GPT(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            hidden_dim=2048,
            num_layers=4,
            attention_type="mha",
            normalization_type="rms",
            feedforward_type="swiglu",
            position_encoding="sinusoidal"
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, stoi, itos, device

    elif model_type in ["character", "tinystories", "english"]:
        model_path = os.path.join(chkpt_dir, "gpt_character.pth")
        if not os.path.exists(model_path):
            print(f"[zahidgpt] Downloading character baseline weights from Hugging Face Hub...")
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="gpt_character.pth")

        from tokenization.character import vocab_size, stoi, itos

        model = GPT(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            hidden_dim=2048,
            num_layers=4,
            attention_type="mha",
            normalization_type="rms",
            feedforward_type="swiglu",
            position_encoding="sinusoidal"
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, stoi, itos, device

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'multicorpus' or 'character'.")

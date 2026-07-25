import os
import torch
from huggingface_hub import hf_hub_download
from .gpt import GPT

HF_REPO_ID = "Zahid2005/modular-gpt-multicorpus"

def load_model(model_type="multicorpus", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(pkg_dir)

    # Search locations: 1. Installed package checkpoints, 2. Root workspace checkpoints
    possible_chkpt_dirs = [
        os.path.join(pkg_dir, "checkpoints"),
        os.path.join(workspace_dir, "checkpoints")
    ]

    chkpt_dir = possible_chkpt_dirs[0]
    for d in possible_chkpt_dirs:
        if os.path.exists(d):
            chkpt_dir = d
            break

    if model_type == "multicorpus":
        vocab_path = os.path.join(chkpt_dir, "multicorpus_vocab.pth")
        model_path = os.path.join(chkpt_dir, "gpt_multicorpus.pth")

        if not os.path.exists(vocab_path):
            try:
                print(f"[zahidgpt] Downloading vocab from Hugging Face Hub ({HF_REPO_ID})...")
                vocab_path = hf_hub_download(repo_id=HF_REPO_ID, filename="multicorpus_vocab.pth")
            except Exception as e:
                print(f"[zahidgpt] Note: Remote vocab download fallback ({e})")

        if not os.path.exists(model_path):
            try:
                print(f"[zahidgpt] Downloading 17.45M model weights from Hugging Face Hub ({HF_REPO_ID})...")
                model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="gpt_multicorpus.pth")
            except Exception as e:
                print(f"[zahidgpt] Note: Remote model download fallback ({e})")

        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path, map_location=device)
            stoi = vocab_data["stoi"]
            itos = vocab_data["itos"]
            vocab_size = vocab_data["vocab_size"]
        else:
            from .tokenization.character import vocab_size, stoi, itos

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

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, stoi, itos, device

    elif model_type in ["character", "tinystories", "english"]:
        model_path = os.path.join(chkpt_dir, "gpt_character.pth")
        if not os.path.exists(model_path):
            try:
                print(f"[zahidgpt] Downloading character baseline weights from Hugging Face Hub...")
                model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="gpt_character.pth")
            except Exception as e:
                print(f"[zahidgpt] Note: Remote model download fallback ({e})")

        from .tokenization.character import vocab_size, stoi, itos

        model = GPT(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            hidden_dim=2048,
            num_layers=2,
            attention_type="mha",
            normalization_type="rms",
            feedforward_type="swiglu",
            position_encoding="sinusoidal"
        ).to(device)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, stoi, itos, device

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'multicorpus' or 'character'.")

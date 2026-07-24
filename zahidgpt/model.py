import os
import torch
from gpt import GPT

def load_model(model_type="multicorpus", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chkpt_dir = os.path.join(root_dir, "checkpoints")
    
    if model_type == "multicorpus":
        vocab_path = os.path.join(chkpt_dir, "multicorpus_vocab.pth")
        model_path = os.path.join(chkpt_dir, "gpt_multicorpus.pth")
        
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path, map_location=device)
            stoi = vocab_data["stoi"]
            itos = vocab_data["itos"]
            vocab_size = vocab_data["vocab_size"]
        else:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

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
        from tokenization.character import vocab_size, stoi, itos
        model_path = os.path.join(chkpt_dir, "gpt_character.pth")
        
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

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'multicorpus' or 'character'.")

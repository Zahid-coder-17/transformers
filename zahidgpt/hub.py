import os
import torch
from huggingface_hub import HfApi, login, hf_hub_download

def push_to_hub(repo_id: str, checkpoint_path: str, hf_token: str = None):
    """
    Pushes a custom checkpoint to Hugging Face Hub.

    Parameters
    ----------
    repo_id         : str – e.g. "username/my-custom-gpt"
    checkpoint_path : str – Local path to checkpoint file
    hf_token        : str – Hugging Face API write token
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF Token required. Provide hf_token parameter or set HF_TOKEN env var.")

    login(token=token)
    api = HfApi()

    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
    filename = os.path.basename(checkpoint_path)
    print(f"[zahidgpt.push_to_hub] Uploading {checkpoint_path} to https://huggingface.co/{repo_id}...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print(f"🎉 Model uploaded successfully to https://huggingface.co/{repo_id}")

def load_from_hub(repo_id: str, filename: str = "gpt_multicorpus.pth", device: str = None):
    """
    Downloads model weights from any Hugging Face model repository.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[zahidgpt.load_from_hub] Downloading {filename} from https://huggingface.co/{repo_id}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return local_path

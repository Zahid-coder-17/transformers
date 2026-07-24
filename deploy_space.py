import os
import sys
from huggingface_hub import HfApi, login

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HF_TOKEN = os.environ.get("HF_TOKEN")

def deploy_to_hf_space():
    token = HF_TOKEN
    if not token:
        print("Error: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    print("Logging into Hugging Face Hub ...")
    login(token=token)
    api = HfApi()

    user_info = api.whoami(token=token)
    username = user_info["name"]
    print(f"Authenticated as Hugging Face User: '{username}'")

    space_id = f"{username}/modular-gpt-playground"

    files_to_upload = [
        "app.py",
        "gpt.py",
        "transformer.py",
        "generate.py",
        "requirements.txt",
        "README.md",
        "checkpoints/gpt_multicorpus.pth",
        "checkpoints/multicorpus_vocab.pth",
        "checkpoints/gpt_character.pth",
        "assets/multicorpus_ddp_loss_curve.png",
        "assets/arabic_tokenizer_comparison.png",
        "assets/code_tokenizer_comparison.png",
        "assets/tokenizer_comparison.png",
    ]

    subdirs = ["attention", "normalization", "feedforward", "position", "tokenization", "utils", "zahidgpt"]
    for sdir in subdirs:
        if os.path.exists(sdir):
            for root, _, files in os.walk(sdir):
                for f in files:
                    if not f.endswith(".pyc") and "__pycache__" not in root:
                        files_to_upload.append(os.path.join(root, f).replace("\\", "/"))

    print(f"\nUploading repository files to Space https://huggingface.co/spaces/{space_id} ...")
    success_count = 0
    for fpath in files_to_upload:
        if os.path.exists(fpath):
            try:
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=fpath,
                    repo_id=space_id,
                    repo_type="space",
                    token=token,
                )
                print(f"  [+] Uploaded {fpath}")
                success_count += 1
            except Exception as e:
                print(f"  [-] Error uploading {fpath}: {e}")

    if success_count > 0:
        print(f"\n🎉 Gradio Space Deployment Complete!")
        print(f"👉 Live Space URL: https://huggingface.co/spaces/{space_id}\n")
    else:
        print("\nNotice: Please ensure the Space is created on https://huggingface.co/new-space first.")

if __name__ == "__main__":
    deploy_to_hf_space()

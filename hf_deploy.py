import os
import sys
from huggingface_hub import HfApi, login, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")

def deploy_to_hf():
    print("Logging into Hugging Face Hub ...")
    login(token=HF_TOKEN)
    api = HfApi()

    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]
    print(f"Authenticated as Hugging Face User: '{username}'")

    repo_id = f"{username}/modular-gpt-multicorpus"
    print(f"Creating repository '{repo_id}' on HF Hub (if not exists)...")
    try:
        create_repo(repo_id=repo_id, token=HF_TOKEN, exist_ok=True, repo_type="model")
        print(f"Repository ready at https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo setup notice: {e}")

    files_to_upload = [
        ("checkpoints/gpt_multicorpus.pth", "gpt_multicorpus.pth"),
        ("checkpoints/multicorpus_vocab.pth", "multicorpus_vocab.pth"),
        ("checkpoints/gpt_character.pth", "gpt_character.pth"),
        ("model_card.md", "README.md"),
        ("assets/multicorpus_ddp_loss_curve.png", "assets/multicorpus_ddp_loss_curve.png"),
        ("gpt.py", "gpt.py"),
        ("transformer.py", "transformer.py"),
        ("generate.py", "generate.py"),
    ]

    print(f"\nUploading files to https://huggingface.co/{repo_id} ...")
    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            print(f" -> Uploading {local_path} as {repo_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=HF_TOKEN,
                )
                print(f"    ✔ Successfully uploaded {repo_path}")
            except Exception as e:
                print(f"    ❌ Error uploading {local_path}: {e}")
        else:
            print(f"    ⚠️ File {local_path} not found locally, skipping.")

    print(f"\n🎉 Deployment completed! Your model is live on Hugging Face Hub at:")
    print(f"👉 https://huggingface.co/{repo_id}\n")

if __name__ == "__main__":
    deploy_to_hf()

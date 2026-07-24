import os
import sys
from datasets import load_dataset

TARGET_BYTES = 8 * 1024 * 1024  # 8 MB


def download_code(target_bytes=TARGET_BYTES):
    print(f"[Code] Target: {target_bytes / 1e6:.1f} MB", flush=True)
    print("[Code] Loading flytech/python-codes-25k ...", flush=True)

    chunks = []
    total  = 0

    try:
        ds = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
        for item in ds:
            text = item.get("output") or item.get("code") or item.get("text") or ""
            if not text.strip():
                continue
            encoded = text.encode("utf-8")
            chunks.append(text)
            total += len(encoded)
            if total % (512 * 1024) < len(encoded):
                print(f"  [Code] {total / 1e6:.2f} MB collected ...", flush=True)
            if total >= target_bytes:
                break
    except Exception as e:
        print(f"  [Code] flytech failed: {e}", flush=True)

    if total < target_bytes:
        print("[Code] Supplementing with codeparrot/github-code (Python) ...", flush=True)
        try:
            ds2 = load_dataset(
                "codeparrot/github-code",
                streaming=True,
                split="train",
            )
            for item in ds2:
                if item.get("language", "") != "Python":
                    continue
                text = item.get("code", "")
                if not text.strip():
                    continue
                encoded = text.encode("utf-8")
                chunks.append(text)
                total += len(encoded)
                if total % (512 * 1024) < len(encoded):
                    print(f"  [Code] {total / 1e6:.2f} MB collected ...", flush=True)
                if total >= target_bytes:
                    break
        except Exception as e:
            print(f"  [Code] codeparrot fallback failed: {e}", flush=True)

    full_text = "\n\n".join(chunks)
    actual    = len(full_text.encode("utf-8"))
    os.makedirs("data", exist_ok=True)
    with open("data/code_input.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[Code] Saved -> data/code_input.txt  ({actual / 1e6:.2f} MB)\n", flush=True)


def download_arabic(target_bytes=TARGET_BYTES):
    print(f"[Arabic] Target: {target_bytes / 1e6:.1f} MB", flush=True)

    chunks = []
    total  = 0

    print("[Arabic] Loading wikimedia/wikipedia ar ...", flush=True)
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ar",
            split="train",
            streaming=True,
        )
        for item in ds:
            text = item.get("text", "")
            if not text.strip():
                continue
            encoded = text.encode("utf-8")
            chunks.append(text)
            total += len(encoded)
            if total % (512 * 1024) < len(encoded):
                print(f"  [Arabic] {total / 1e6:.2f} MB collected ...", flush=True)
            if total >= target_bytes:
                break
    except Exception as e:
        print(f"  [Arabic] Wikipedia failed: {e}", flush=True)

    if total < target_bytes:
        print("[Arabic] Supplementing with cc100 ar ...", flush=True)
        try:
            ds2 = load_dataset("cc100", lang="ar", split="train", streaming=True)
            for item in ds2:
                text = item.get("text", "")
                if not text.strip():
                    continue
                encoded = text.encode("utf-8")
                chunks.append(text)
                total += len(encoded)
                if total % (512 * 1024) < len(encoded):
                    print(f"  [Arabic] {total / 1e6:.2f} MB collected ...", flush=True)
                if total >= target_bytes:
                    break
        except Exception as e:
            print(f"  [Arabic] cc100 fallback failed: {e}", flush=True)

    full_text = "\n\n".join(chunks)
    actual    = len(full_text.encode("utf-8"))
    os.makedirs("data", exist_ok=True)
    with open("data/arabic_input.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[Arabic] Saved -> data/arabic_input.txt  ({actual / 1e6:.2f} MB)\n", flush=True)


if __name__ == "__main__":
    print("=" * 55)
    print("  Multi-Corpus Dataset Downloader  (8 MB per corpus)")
    print("=" * 55)
    download_code()
    download_arabic()
    print("All done. Run train_multicorpus_ddp.py when ready.")

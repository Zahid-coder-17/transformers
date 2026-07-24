import os
import urllib.request
from datasets import load_dataset


def download_arabic_dataset():
    print("Downloading Arabic Dataset...")
    urls = [
        "https://raw.githubusercontent.com/1B-Arabic-Words/arabic-text-corpus/master/arabic_corpus.txt",
        "https://raw.githubusercontent.com/tashkeel/arabic-corpus/master/corpus.txt"
    ]
    downloaded = False
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8")
                if len(text) > 2000:
                    with open("data/arabic_input.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Successfully downloaded Arabic corpus! Size: {len(text):,} characters ({len(text.encode('utf-8')):,} bytes)")
                    downloaded = True
                    break
        except Exception:
            continue

    if not downloaded:
        try:
            ds = load_dataset("arabic_billion_words", split="train", streaming=True)
            text_data = []
            for i, item in enumerate(ds):
                if i >= 500:
                    break
                if "text" in item:
                    text_data.append(item["text"])
            if text_data:
                full_text = "\n\n".join(text_data)
                with open("data/arabic_input.txt", "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"Successfully downloaded HF Arabic billion words! Size: {len(full_text):,} characters")
        except Exception as e:
            print(f"Preserving rich local Arabic corpus: {e}")


def download_code_dataset():
    print("Downloading Python Code Dataset from Hugging Face...")
    try:
        ds = load_dataset("flytech/python-codes-25k", split="train")
        text_data = []
        for i, item in enumerate(ds):
            if i >= 1500:
                break
            if "code" in item and item["code"]:
                text_data.append(item["code"])
            elif "text" in item and item["text"]:
                text_data.append(item["text"])

        full_code = "\n\n".join(text_data)
        with open("data/code_input.txt", "w", encoding="utf-8") as f:
            f.write(full_code)
        print(f"Successfully downloaded Python Code dataset! Size: {len(full_code):,} characters ({len(full_code.encode('utf-8')):,} bytes)")
    except Exception as e:
        print(f"Code dataset load failed: {e}.")


if __name__ == "__main__":
    download_arabic_dataset()
    download_code_dataset()

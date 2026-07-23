from datasets import load_dataset

print("Downloading TinyStories...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train"
)

with open("data/input.txt","w",encoding="utf-8") as f:
    for i , story in enumerate(dataset):
        if i > 10000:
            break
        f.write(story["text"] +"\n")
import torch

d_model = 512
sequence_length = 256

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {
    character: index
    for index, character in enumerate(chars)
}

itos = {
    index: character
    for index, character in enumerate(chars)
}

def encode(text):
    return [stoi[c] for c in text]

def decode(ids):
    return "".join([itos[i] for i in ids])

data = torch.tensor(encode(text),dtype= torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 64
block_size = 256

def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = torch.randint(0,len(data)- block_size,(batch_size,) )
 
    x = torch.stack([data[i:i+block_size ] for i in ix])

    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y




if __name__ == "__main__":

    xb, yb = get_batch("train")

    print("\nInput Shape :", xb.shape)
    print("Target Shape:", yb.shape)

    print("\nFirst Input Sequence:")
    print(xb[0])

    print("\nFirst Target Sequence:")
    print(yb[0])

    print("\nDecoded Input:")
    print(decode(xb[0].tolist()))

    print("\nDecoded Target:")
    print(decode(yb[0].tolist()))




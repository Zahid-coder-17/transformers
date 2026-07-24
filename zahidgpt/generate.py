import torch
from zahidgpt.model import load_model

def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]

def decode(ids, itos):
    return "".join([itos[i] for i in ids if i in itos])

def generate(prompt="Once upon a time", model_type="multicorpus", max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.9, device=None):
    model, stoi, itos, dev = load_model(model_type=model_type, device=device)
    
    tokens = encode(prompt, stoi)
    if not tokens:
        tokens = [0]
        
    context = torch.tensor([tokens], dtype=torch.long, device=dev)
    
    with torch.no_grad():
        out = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
    generated_text = decode(out[0].tolist(), itos)
    return generated_text

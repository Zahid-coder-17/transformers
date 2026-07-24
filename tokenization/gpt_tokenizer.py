import torch
import os
import json
from tokenization.regex_bpe import RegexBPETokenizer


class GPTTokenizer:
    SPECIAL_TOKENS = {"<|endoftext|>": 256}

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.tokenizer = RegexBPETokenizer(vocab_size=vocab_size)
        self.special_tokens = dict(self.SPECIAL_TOKENS)

    def fit(self, text):
        self.tokenizer.fit(text)
        self.vocab = self.tokenizer.vocab
        self.inverse_vocab = self.tokenizer.inverse_vocab

    def encode(self, text):
        for token in self.special_tokens:
            if token in text:
                parts = text.split(token)
                res = []
                for i, p in enumerate(parts):
                    if p:
                        res.extend(self.tokenizer.encode(p))
                    if i < len(parts) - 1:
                        res.append(self.special_tokens[token])
                return res
        return self.tokenizer.encode(text)

    def decode(self, ids):
        clean_ids = []
        for i in ids:
            if i in self.special_tokens.values():
                continue
            clean_ids.append(i)
        return self.tokenizer.decode(clean_ids)

    def get_batch(self, split, data_path="data/input.txt", batch_size=64, block_size=256):
        if not hasattr(self, "_cached_data"):
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    text = f.read()
                tokens = self.encode(text)
                self._cached_data = torch.tensor(tokens, dtype=torch.long)
            else:
                self._cached_data = torch.zeros(1000, dtype=torch.long)

        data = self._cached_data
        n = int(0.9 * len(data))
        split_data = data[:n] if split == "train" else data[n:]
        
        max_idx = len(split_data) - block_size - 1
        if max_idx <= 0:
            ix = torch.zeros(batch_size, dtype=torch.long)
        else:
            ix = torch.randint(0, max_idx, (batch_size,))

        x = torch.stack([split_data[i : i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer.load(path)
        self.vocab = self.tokenizer.vocab
        self.inverse_vocab = self.tokenizer.inverse_vocab

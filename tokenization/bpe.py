from collections import Counter
import re
import json
import torch
import torch.nn as nn

class BPE:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.word_freqs = {}

    def fit(self, text):
        chars = sorted(list(set(text)))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.inverse_vocab = {c: i for i, c in enumerate(chars)}

    def encode(self, text):
        if not hasattr(self, "inverse_vocab"):
            self.fit(text)
        return [self.inverse_vocab.get(c, 0) for c in text]

    def decode(self, ids):
        if not hasattr(self, "vocab"):
            return str(ids)
        return "".join([self.vocab.get(i, "") for i in ids])


class BPETokenizer(BPE):
    def __init__(self, vocab_size=256):
        super().__init__(vocab_size=vocab_size)


class WordPieceTokenizer(BPE):
    def __init__(self, vocab_size=256):
        super().__init__(vocab_size=vocab_size)


class SentencePieceTokenizer(BPE):
    def __init__(self, vocab_size=256, model_prefix="spm_app"):
        super().__init__(vocab_size=vocab_size)
        self.model_prefix = model_prefix

    def fit(self, text, input_file=None):
        super().fit(text)
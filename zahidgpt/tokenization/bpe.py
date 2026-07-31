from collections import Counter
import re
from .wordpiece import WordPieceTokenizer
from .sentencepiece_tokenizer import SentencePieceTokenizer

class BPE:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []

    def fit(self, text):
        chars = sorted(list(set(text)))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.inverse_vocab = {c: i for i, c in enumerate(chars)}

    def encode(self, text):
        if not self.inverse_vocab:
            self.fit(text)
        return [self.inverse_vocab.get(c, 0) for c in text]

    def decode(self, ids):
        if not self.vocab:
            return str(ids)
        return "".join([self.vocab.get(i, "") for i in ids])


class BPETokenizer(BPE):
    def __init__(self, vocab_size=256):
        super().__init__(vocab_size=vocab_size)

__all__ = ["BPE", "BPETokenizer", "WordPieceTokenizer", "SentencePieceTokenizer"]
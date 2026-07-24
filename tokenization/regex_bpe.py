import re
import json
from collections import defaultdict, Counter
from tokenization.byte_bpe import bytes_to_unicode


class RegexBPETokenizer:
    GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""

    def __init__(self, vocab_size=256, pattern=None):
        self.vocab_size = vocab_size
        self.pattern = pattern if pattern else self.GPT2_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        self._init_base_vocab()

    def _init_base_vocab(self):
        for b, c in self.byte_encoder.items():
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.inverse_vocab[idx] = c

    def fit(self, text):
        chunks = re.findall(self.compiled_pattern, text)
        chunk_freqs = Counter(chunks)

        splits = {}
        for chunk, freq in chunk_freqs.items():
            raw_bytes = chunk.encode("utf-8")
            splits[chunk] = [self.byte_encoder[b] for b in raw_bytes]

        self.merges = []
        while len(self.vocab) < self.vocab_size:
            pairs = defaultdict(int)
            for chunk, freq in chunk_freqs.items():
                symbols = splits[chunk]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)

            new_splits = {}
            for chunk in splits:
                symbols = splits[chunk]
                i = 0
                new_syms = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_syms.append("".join(best_pair))
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                new_splits[chunk] = new_syms
            splits = new_splits

            self.merges.append(best_pair)
            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token

    def encode(self, text):
        chunks = re.findall(self.compiled_pattern, text)
        encoded_ids = []

        for chunk in chunks:
            raw_bytes = chunk.encode("utf-8")
            symbols = [self.byte_encoder[b] for b in raw_bytes]

            for pair in self.merges:
                i = 0
                new_syms = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                        new_syms.append("".join(pair))
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                symbols = new_syms

            for sym in symbols:
                if sym in self.vocab:
                    encoded_ids.append(self.vocab[sym])
                else:
                    for b_char in sym:
                        if b_char in self.vocab:
                            encoded_ids.append(self.vocab[b_char])

        return encoded_ids

    def decode(self, ids):
        chars = [self.inverse_vocab.get(i, "") for i in ids]
        combined = "".join(chars)
        raw_bytes = bytearray([self.byte_decoder[c] for c in combined if c in self.byte_decoder])
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "pattern": self.pattern
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in data["merges"]]
        self.pattern = data.get("pattern", self.GPT2_SPLIT_PATTERN)
        self.compiled_pattern = re.compile(self.pattern)

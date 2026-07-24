import json
from collections import defaultdict, Counter


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class ByteBPETokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
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

    def _get_stats(self, splits, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def fit(self, text):
        raw_bytes = text.encode("utf-8")
        byte_symbols = [self.byte_encoder[b] for b in raw_bytes]
        
        words = text.split(" ")
        word_freqs = Counter(words)
        splits = {}
        
        for word in word_freqs:
            w_bytes = word.encode("utf-8")
            splits[word] = [self.byte_encoder[b] for b in w_bytes]

        self.merges = []
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(splits, word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            
            new_splits = {}
            for word in splits:
                symbols = splits[word]
                i = 0
                new_syms = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_syms.append("".join(best_pair))
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                new_splits[word] = new_syms
            splits = new_splits
            self.merges.append(best_pair)
            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token

    def encode(self, text):
        raw_bytes = text.encode("utf-8")
        symbols = [self.byte_encoder[b] for b in raw_bytes]

        for pair in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        ids = []
        for sym in symbols:
            if sym in self.vocab:
                ids.append(self.vocab[sym])
            else:
                for b_char in sym:
                    if b_char in self.vocab:
                        ids.append(self.vocab[b_char])
        return ids

    def decode(self, ids):
        chars = [self.inverse_vocab.get(i, "") for i in ids]
        combined = "".join(chars)
        raw_bytes = bytearray([self.byte_decoder[c] for c in combined if c in self.byte_decoder])
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in data["merges"]]

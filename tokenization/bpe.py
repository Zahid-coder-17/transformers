import re
import os
import json
from collections import Counter, defaultdict

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class BPETokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        self.word_freqs = defaultdict(int)

    def get_stats(self, splits):
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_splits(self, splits, pair):
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_splits = {}
        for word in splits:
            w_str = " ".join(splits[word])
            w_str = pattern.sub("".join(pair), w_str)
            new_splits[word] = w_str.split()
        return new_splits

    def fit(self, text):
        words = re.findall(r"\S+", text)
        self.word_freqs = Counter(words)
        splits = {}
        chars = set()

        for word in self.word_freqs:
            split = list(word) + ["</w>"]
            splits[word] = split
            chars.update(split)

        sorted_chars = sorted(list(chars))
        self.vocab = {token: i for i, token in enumerate(sorted_chars)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(splits)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            splits = self.merge_splits(splits, best_pair)
            self.merges.append(best_pair)
            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token

    def encode(self, text):
        words = re.findall(r"\S+", text)
        encoded_ids = []
        for word in words:
            tokens = list(word) + ["</w>"]
            for pair in self.merges:
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                        new_tokens.append("".join(pair))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            for t in tokens:
                if t in self.vocab:
                    encoded_ids.append(self.vocab[t])
                else:
                    for char in t:
                        if char in self.vocab:
                            encoded_ids.append(self.vocab[char])
        return encoded_ids

    def decode(self, ids):
        tokens = [self.inverse_vocab.get(i, "") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

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


class WordPieceTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        self.inverse_vocab = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]", 4: "[MASK]"}
        self.word_freqs = defaultdict(int)

    def fit(self, text):
        words = re.findall(r"\w+|\S", text)
        self.word_freqs = Counter(words)
        splits = {}
        alphabet = set()

        for word in self.word_freqs:
            split = [word[0]] + ["##" + char for char in word[1:]]
            splits[word] = split
            for symbol in split:
                alphabet.add(symbol)

        for token in sorted(list(alphabet)):
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token

        while len(self.vocab) < self.vocab_size:
            symbol_freqs = defaultdict(int)
            pair_freqs = defaultdict(int)

            for word, freq in self.word_freqs.items():
                symbols = splits[word]
                for symbol in symbols:
                    symbol_freqs[symbol] += freq
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = None
            best_score = -1
            for pair, p_freq in pair_freqs.items():
                s1, s2 = pair
                score = p_freq / (symbol_freqs[s1] * symbol_freqs[s2])
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if not best_pair:
                break

            s1, s2 = best_pair
            new_symbol = s1 + s2[2:] if s2.startswith("##") else s1 + s2
            idx = len(self.vocab)
            self.vocab[new_symbol] = idx
            self.inverse_vocab[idx] = new_symbol

            for word in self.word_freqs:
                symbols = splits[word]
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_symbols.append(new_symbol)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

    def encode(self, text):
        words = re.findall(r"\w+|\S", text)
        encoded_ids = []
        for word in words:
            start = 0
            while start < len(word):
                end = len(word)
                cur_substr = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    encoded_ids.append(self.vocab["[UNK]"])
                    start += 1
                else:
                    encoded_ids.append(self.vocab[cur_substr])
                    start = end
        return encoded_ids

    def decode(self, ids):
        tokens = [self.inverse_vocab.get(i, "[UNK]") for i in ids]
        res = []
        for token in tokens:
            if token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
                continue
            if token.startswith("##"):
                res.append(token[2:])
            else:
                if res:
                    res.append(" ")
                res.append(token)
        return "".join(res)


class SentencePieceTokenizer:
    def __init__(self, vocab_size=256, model_prefix="spm_model"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = None
        self.vocab = {}
        self.inverse_vocab = {}

    def fit(self, text, input_file="data/input.txt"):
        if spm is None:
            words = list(set(re.findall(r"\S+", text)))
            tokens = ["<unk>", "<s>", "</s>"] + words[:self.vocab_size - 3]
            self.vocab = {t: i for i, t in enumerate(tokens)}
            self.inverse_vocab = {i: t for i, t in enumerate(tokens)}
            return

        temp_path = input_file
        if not os.path.exists(temp_path):
            temp_path = "temp_spm_input.txt"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(text)

        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type="unigram",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{self.model_prefix}.model")

    def encode(self, text):
        if self.sp is not None:
            return self.sp.encode_as_ids(text)
        tokens = re.findall(r"\S+", text)
        return [self.vocab.get(t, 1) for t in tokens]

    def decode(self, ids):
        if self.sp is not None:
            return self.sp.decode_ids(ids)
        tokens = [self.inverse_vocab.get(i, "<unk>") for i in ids]
        return " ".join(tokens)
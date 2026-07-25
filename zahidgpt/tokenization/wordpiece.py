import re
from collections import Counter, defaultdict

class WordPieceTokenizer:
    """
    WordPiece Tokenizer implementation (BERT / WordPiece style).
    Uses '##' subword prefixes for continuation sub-tokens.
    """
    def __init__(self, vocab_size=256, unk_token="[UNK]", pad_token="[PAD]"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab = {}
        self.inv_vocab = {}

    def fit(self, text):
        words = re.findall(r"\w+|[^\w\s]", text)
        word_counts = Counter(words)

        # Base characters
        chars = set()
        for word in word_counts:
            for i, c in enumerate(word):
                if i == 0:
                    chars.add(c)
                else:
                    chars.add("##" + c)

        vocab_list = [self.pad_token, self.unk_token] + sorted(list(chars))
        
        # Build frequency table for word piece candidates
        subword_counts = Counter()
        for word, count in word_counts.items():
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    sub = word[i:j]
                    if i > 0:
                        sub = "##" + sub
                    subword_counts[sub] += count

        most_common = [s for s, _ in subword_counts.most_common(self.vocab_size)]
        for sub in most_common:
            if sub not in vocab_list and len(vocab_list) < self.vocab_size:
                vocab_list.append(sub)

        self.vocab = {s: i for i, s in enumerate(vocab_list)}
        self.inv_vocab = {i: s for i, s in enumerate(vocab_list)}

    def encode(self, text):
        if not self.vocab:
            self.fit(text)

        words = re.findall(r"\w+|[^\w\s]", text)
        token_ids = []

        for word in words:
            start = 0
            word_tokens = []
            is_bad = False

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
                    is_bad = True
                    break

                word_tokens.append(self.vocab[cur_substr])
                start = end

            if is_bad:
                token_ids.append(self.vocab.get(self.unk_token, 1))
            else:
                token_ids.extend(word_tokens)

        return token_ids

    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, self.unk_token) for i in ids]
        text = ""
        for token in tokens:
            if token in (self.pad_token, self.unk_token):
                continue
            if token.startswith("##"):
                text += token[2:]
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        return text.strip()

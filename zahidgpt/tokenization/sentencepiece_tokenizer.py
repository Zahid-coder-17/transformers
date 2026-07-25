import os
import tempfile

class SentencePieceTokenizer:
    """
    SentencePiece Tokenizer (Google SPM style).
    Uses whitespace meta-character ' ' (U+2581) to preserve exact word boundaries.
    Integrates native sentencepiece library with pure Python fallback.
    """
    def __init__(self, vocab_size=256, model_prefix="spm_model"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp_processor = None
        self.vocab = {}
        self.inv_vocab = {}

    def fit(self, text, input_file=None):
        try:
            import sentencepiece as spm
            if input_file is None or not os.path.exists(input_file):
                with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
                    f.write(text)
                    input_file = f.name

            spm.SentencePieceTrainer.train(
                input=input_file,
                model_prefix=self.model_prefix,
                vocab_size=min(self.vocab_size, 1000),
                character_coverage=1.0,
                model_type="unigram"
            )
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(f"{self.model_prefix}.model")
            return
        except Exception:
            # Pure Python fallback using ' ' whitespace meta-symbol
            pass

        # Fallback whitespace meta-symbol tokenization
        formatted_text = text.replace(" ", " ")
        chars = sorted(list(set(formatted_text)))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.inv_vocab = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        if self.sp_processor is not None:
            return self.sp_processor.encode_as_ids(text)

        if not self.vocab:
            self.fit(text)

        formatted_text = text.replace(" ", " ")
        return [self.vocab.get(c, 0) for c in formatted_text]

    def decode(self, ids):
        if self.sp_processor is not None:
            return self.sp_processor.decode_ids(ids)

        raw = "".join([self.inv_vocab.get(i, "") for i in ids])
        return raw.replace(" ", " ")

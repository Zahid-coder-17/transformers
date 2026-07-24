import sys
import os

pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from gpt import GPT, BigramLanguageModel
from transformer import TransformerBlock
from zahidgpt.generate import generate, encode, decode
from zahidgpt.model import load_model
from zahidgpt.finetune import finetune
from zahidgpt.hub import push_to_hub, load_from_hub
from zahidgpt.tokenizer import (
    BPE,
    ByteBPETokenizer,
    RegexBPETokenizer,
    GPTTokenizer,
)

__version__ = "0.1.0"
__all__ = [
    "GPT",
    "TransformerBlock",
    "BigramLanguageModel",
    "generate",
    "finetune",
    "load_model",
    "push_to_hub",
    "load_from_hub",
    "encode",
    "decode",
    "BPE",
    "ByteBPETokenizer",
    "RegexBPETokenizer",
    "GPTTokenizer",
]

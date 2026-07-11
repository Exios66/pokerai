"""Shared device / tokenizer helpers for training scripts."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from pokerai.config import TOKENIZER_DIR


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(path: Path = TOKENIZER_DIR) -> PreTrainedTokenizerFast:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tokenizer at {path}. Run: python scripts/train_tokenizer.py"
        )
    return PreTrainedTokenizerFast.from_pretrained(str(path))

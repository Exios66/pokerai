"""Shared device / tokenizer / experiment-tracking helpers for training scripts."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from pokerai.config import TOKENIZER_DIR, WANDB_PROJECT


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


def wandb_is_configured() -> bool:
    """True when W&B should run (explicit mode or logged-in API key)."""
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode == "disabled":
        return False
    if mode in ("offline", "online", "shared"):
        return True
    if os.environ.get("WANDB_API_KEY"):
        return True
    # Fall back to a saved login without forcing an interactive prompt.
    try:
        import wandb

        return bool(wandb.api.api_key)
    except Exception:
        return False


def wandb_report_to() -> str:
    """Value for HF TrainingArguments.report_to — never force wandb."""
    return "wandb" if wandb_is_configured() else "none"


def ensure_wandb_project() -> None:
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

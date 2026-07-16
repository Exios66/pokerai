"""Shared paths and hyperparameters for the poker AI pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# Repo root: src/pokerai/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
HANDS_RAW = DATA_DIR / "hands.txt"
HANDS_CLEAN = DATA_DIR / "hands_clean.txt"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
MODEL_GPT2_DIR = ARTIFACTS_DIR / "models" / "gpt2"
MODEL_TRL_DIR = ARTIFACTS_DIR / "models" / "gpt2_trl"
MODEL_BIGRAM_DIR = ARTIFACTS_DIR / "models" / "bigram"

HF_DATASET = "SoelMgd/Poker_Dataset"
# https://wandb.ai/mooslin-university-of-wisconsin-madison/poker-ai
WANDB_ENTITY = "mooslin-university-of-wisconsin-madison"
WANDB_PROJECT = "poker-ai"

# Special tokens — BOS and EOS must be distinct for clean generation stopping.
BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

VOCAB_SIZE = 4000
N_POSITIONS = 384
N_EMBD = 256
N_LAYER = 6
N_HEAD = 8

TEST_SIZE = 0.05
SPLIT_SEED = 42

GPT2_LR = 3e-4
GPT2_BATCH_SIZE = 32
GPT2_EPOCHS = 3

BIGRAM_LR = 1e-2
BIGRAM_BATCH_SIZE = 4096
BIGRAM_STEPS = 2000


@dataclass(frozen=True)
class GPT2Hyperparams:
    n_positions: int = N_POSITIONS
    n_embd: int = N_EMBD
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    learning_rate: float = GPT2_LR
    batch_size: int = GPT2_BATCH_SIZE
    num_epochs: int = GPT2_EPOCHS

    def as_dict(self) -> dict:
        return asdict(self)

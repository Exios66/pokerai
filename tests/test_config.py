"""Config and training-helper sanity checks."""

import os

from pokerai.config import (
    BOS_TOKEN,
    EOS_TOKEN,
    N_POSITIONS,
    PAD_TOKEN,
    REPO_ROOT,
)
from pokerai.training import wandb_is_configured, wandb_report_to


def test_special_tokens_are_distinct():
    assert BOS_TOKEN != EOS_TOKEN
    assert PAD_TOKEN not in (BOS_TOKEN, EOS_TOKEN)


def test_context_length():
    assert N_POSITIONS >= 384


def test_repo_root_exists():
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "src" / "pokerai").is_dir()


def test_wandb_disabled_by_env(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_is_configured() is False
    assert wandb_report_to() == "none"


def test_wandb_offline_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert wandb_is_configured() is True
    assert wandb_report_to() == "wandb"

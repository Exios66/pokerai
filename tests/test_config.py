"""Config and training-helper sanity checks."""

import os

from pokerai.config import (
    BOS_TOKEN,
    EOS_TOKEN,
    N_POSITIONS,
    PAD_TOKEN,
    REPO_ROOT,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from pokerai.training import ensure_wandb_project, wandb_is_configured, wandb_report_to


def test_special_tokens_are_distinct():
    assert BOS_TOKEN != EOS_TOKEN
    assert PAD_TOKEN not in (BOS_TOKEN, EOS_TOKEN)


def test_context_length():
    assert N_POSITIONS >= 384


def test_repo_root_exists():
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "src" / "pokerai").is_dir()


def test_wandb_team_project():
    assert WANDB_ENTITY == "mooslin-university-of-wisconsin-madison"
    assert WANDB_PROJECT == "poker-ai"


def test_ensure_wandb_project_sets_env(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    ensure_wandb_project()
    assert os.environ["WANDB_ENTITY"] == WANDB_ENTITY
    assert os.environ["WANDB_PROJECT"] == WANDB_PROJECT


def test_wandb_disabled_by_env(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_is_configured() is False
    assert wandb_report_to() == "none"


def test_wandb_offline_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert wandb_is_configured() is True
    assert wandb_report_to() == "wandb"

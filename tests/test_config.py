"""Config sanity checks."""

from pokerai.config import (
    BOS_TOKEN,
    EOS_TOKEN,
    N_POSITIONS,
    PAD_TOKEN,
    REPO_ROOT,
)


def test_special_tokens_are_distinct():
    assert BOS_TOKEN != EOS_TOKEN
    assert PAD_TOKEN not in (BOS_TOKEN, EOS_TOKEN)


def test_context_length():
    assert N_POSITIONS >= 384


def test_repo_root_exists():
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "src" / "pokerai").is_dir()

"""Unit tests for data cleaning and action-mask encoding."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from pokerai.data import (
    ACTION_SEP,
    clean_line,
    encode_with_mask,
    load_text_split,
    split_prompt_completion,
    to_line,
)


class _FakeTok:
    """Minimal tokenizer stub: one char -> one id, specials at high ids."""

    bos_token_id = 100
    eos_token_id = 101
    pad_token_id = 102

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) % 97 for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def _train_tiny_bpe(tmpdir: Path, lines: list[str]) -> PreTrainedTokenizerFast:
    corpus = tmpdir / "hands.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[str(corpus)],
        vocab_size=200,
        min_frequency=1,
        special_tokens=["<|startoftext|>", "<|endoftext|>", "<|pad|>"],
    )
    tok_json = tmpdir / "tokenizer.json"
    bpe.save(str(tok_json))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_json))
    tok.add_special_tokens(
        {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
        }
    )
    return tok


def test_to_line_flattens_whitespace():
    assert to_line("a\nb  c", "FOLD") == "a b c,FOLD"


def test_action_sep_is_comma():
    assert ACTION_SEP == ","


def test_clean_call_zero_bb():
    line = "[PREFLOP] P1:,CALL 0BB"
    assert clean_line(line) == "[PREFLOP] P1:,CALL"


def test_clean_fold_with_amount():
    assert clean_line("state,FOLD0BB") == "state,FOLD"
    assert clean_line("state,FOLD") == "state,FOLD"


def test_split_prompt_completion():
    prompt, completion = split_prompt_completion("ctx,a,b,FOLD")
    assert prompt == "ctx,a,b,"
    assert completion == "FOLD"


def test_split_requires_comma():
    with pytest.raises(ValueError):
        split_prompt_completion("no-comma-here")


def test_split_rejects_arrow_separator():
    """Legacy ' => ' format must not be treated as a valid hand line."""
    with pytest.raises(ValueError):
        split_prompt_completion("state => FOLD")


def test_encode_with_mask_supervises_action_only():
    tok = _FakeTok()
    text = "ABC,XY"
    ids, labels = encode_with_mask(text, tok, max_length=64)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    # Prompt "ABC," -> 4 tokens after BOS are masked; "XY" + EOS supervised
    prompt_len = 1 + len(tok("ABC,", add_special_tokens=False)["input_ids"])
    assert all(t == -100 for t in labels[:prompt_len])
    assert all(t != -100 for t in labels[prompt_len:])
    assert labels[prompt_len:] == ids[prompt_len:]


def test_encode_truncates_from_front():
    tok = _FakeTok()
    text = "ABCDEFGHIJ,Z"
    ids, labels = encode_with_mask(text, tok, max_length=6)
    assert len(ids) == 6
    assert ids[-1] == tok.eos_token_id
    # Action region should still be present at the end
    assert labels[-1] == tok.eos_token_id


def test_encode_with_real_bpe_matches_offsets(tmp_path: Path):
    lines = [
        "[PREFLOP] P3: RAISE 2BB P1:,FOLD",
        "[STACKS] P1: 44.2BB [Qh 9h] P2: 103.4BB,CALL",
        "BTN SB BB POT=1.5BB [PREFLOP] P3: RAISE 2BB P1:,RAISE 6BB",
    ] * 10
    tok = _train_tiny_bpe(tmp_path, lines)
    text = lines[0]
    ids, labels = encode_with_mask(text, tok, max_length=128)
    cut = text.rfind(",") + 1
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_body = next(
        (i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= cut),
        len(enc["input_ids"]),
    )
    prompt_len = 1 + prompt_body
    assert ids == [tok.bos_token_id] + list(enc["input_ids"]) + [tok.eos_token_id]
    assert all(t == -100 for t in labels[:prompt_len])
    assert labels[prompt_len:] == ids[prompt_len:]


def test_load_text_split_rejects_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_text_split(empty)


def test_load_text_split_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "hands.txt"
    path.write_text("a,FOLD\n\n\nb,CALL\n", encoding="utf-8")
    split = load_text_split(path, test_size=0.5, seed=0)
    assert len(split["train"]) + len(split["test"]) == 2

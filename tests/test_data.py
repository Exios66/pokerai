"""Unit tests for data cleaning and action-mask encoding."""

from __future__ import annotations

import pytest

from pokerai.data import clean_line, encode_with_mask, split_prompt_completion, to_line


class _FakeTok:
    """Minimal tokenizer stub: one char -> one id, specials at high ids."""

    bos_token_id = 100
    eos_token_id = 101
    pad_token_id = 102

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 97 for c in text]}


def test_to_line_flattens_whitespace():
    assert to_line("a\nb  c", "FOLD") == "a b c,FOLD"


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

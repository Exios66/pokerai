"""Data loading, cleaning, and dataset helpers."""

from __future__ import annotations

import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from pokerai.config import HANDS_CLEAN, HANDS_RAW, HF_DATASET, SPLIT_SEED, TEST_SIZE

# CALL 0BB is noise (no chips to put in); FOLD never carries an amount in this dataset.
_CALL_ZERO_BB = re.compile(r",CALL 0BB$")
_FOLD_WITH_AMOUNT = re.compile(r",FOLD\d+(\.\d+)?BB$")

# Prompt / action separator used by every train and inference path.
ACTION_SEP = ","


def to_line(question: str, answer: str) -> str:
    """Flatten multiline state into one line; join with comma for prompt/completion split."""
    context = " ".join(str(question).split())
    return f"{context}{ACTION_SEP}{str(answer).strip()}"


def clean_line(line: str) -> str:
    """Normalize redundant action suffixes in serialized hands."""
    line = _FOLD_WITH_AMOUNT.sub(",FOLD", line)
    line = _CALL_ZERO_BB.sub(",CALL", line)
    return line


def fetch_hands(output_path: Path = HANDS_RAW) -> list[str]:
    """Download SoelMgd/Poker_Dataset and write one hand per line."""
    ds = load_dataset(HF_DATASET)
    frames = []
    for split in ds:
        frames.append(ds[split])
    combined = frames[0]
    for frame in frames[1:]:
        combined = concatenate_datasets_safe(combined, frame)

    lines = [to_line(q, a) for q, a in zip(combined["question"], combined["answer"])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def concatenate_datasets_safe(a: Dataset, b: Dataset) -> Dataset:
    from datasets import concatenate_datasets

    return concatenate_datasets([a, b])


def write_cleaned(lines: list[str], output_path: Path = HANDS_CLEAN) -> list[str]:
    cleaned = [clean_line(line) for line in lines]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return cleaned


def load_text_split(
    path: Path = HANDS_CLEAN,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> DatasetDict:
    """Load cleaned hands and split into train/test.

    Skips blank lines. Raises a clear error if the file is missing or empty
    after filtering.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/prepare_data.py"
        )
    if path.stat().st_size == 0:
        raise ValueError(
            f"{path} is empty. Re-run: python scripts/prepare_data.py"
        )

    raw = load_dataset("text", data_files={"train": str(path)})["train"]
    raw = raw.filter(lambda ex: bool(ex["text"] and ex["text"].strip()))
    if len(raw) == 0:
        raise ValueError(
            f"{path} has no non-empty lines. Re-run: python scripts/prepare_data.py"
        )
    return raw.train_test_split(test_size=test_size, seed=seed)


def split_prompt_completion(text: str) -> tuple[str, str]:
    """Split at the last comma into (prompt including comma, completion)."""
    cut = text.rfind(ACTION_SEP) + 1
    if cut <= 0:
        raise ValueError(f"Hand line has no comma separator: {text[:80]!r}")
    return text[:cut], text[cut:]


def encode_with_mask(
    text: str,
    tokenizer,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenize the full hand once and mask labels up to the action boundary.

    Locates the prompt/completion cut from character offsets on a single
    tokenization pass. Tokenizing the prompt alone can disagree with joint
    BPE merges, which silently shifts the supervised region.

    Context tokens (and BOS) get label -100; action tokens and EOS are supervised.
    Long sequences keep the end (action) by dropping tokens from the front.
    """
    prompt_text, _completion_text = split_prompt_completion(text)
    cut = len(prompt_text)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("Tokenizer must define bos_token_id and eos_token_id")

    # Prefer offset mapping (fast tokenizers). Fall back to separate encodes.
    body_ids: list[int]
    prompt_body_len: int
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        body_ids = list(enc["input_ids"])
        offsets = enc.get("offset_mapping")
        if not offsets:
            raise TypeError("tokenizer did not return offset_mapping")
        prompt_body_len = next(
            (i for i, (start, _end) in enumerate(offsets) if start >= cut),
            len(body_ids),
        )
    except (TypeError, ValueError, AttributeError):
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            text[cut:], add_special_tokens=False
        )["input_ids"]
        body_ids = list(prompt_ids) + list(completion_ids)
        prompt_body_len = len(prompt_ids)

    full_ids = [bos] + body_ids + [eos]
    # Supervised region starts after BOS + prompt body tokens.
    prompt_len = 1 + prompt_body_len

    if len(full_ids) > max_length:
        drop = len(full_ids) - max_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len - drop)

    labels = [-100] * len(full_ids)
    for i in range(prompt_len, len(full_ids)):
        labels[i] = full_ids[i]

    if all(t == -100 for t in labels):
        raise ValueError(
            "No supervised action tokens after encoding/truncation; "
            f"hand may be malformed or longer than max_length={max_length}: "
            f"{text[:80]!r}"
        )
    return full_ids, labels

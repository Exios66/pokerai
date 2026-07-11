"""Data loading, cleaning, and dataset helpers."""

from __future__ import annotations

import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from pokerai.config import HANDS_CLEAN, HANDS_RAW, HF_DATASET, SPLIT_SEED, TEST_SIZE

# CALL 0BB is noise (no chips to put in); FOLD never carries an amount in this dataset.
_CALL_ZERO_BB = re.compile(r",CALL 0BB$")
_FOLD_WITH_AMOUNT = re.compile(r",FOLD\d+(\.\d+)?BB$")


def to_line(question: str, answer: str) -> str:
    """Flatten multiline state into one line; join with comma for prompt/completion split."""
    context = " ".join(str(question).split())
    return f"{context},{str(answer).strip()}"


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
    """Load cleaned hands and split into train/test."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/prepare_data.py"
        )
    raw = load_dataset("text", data_files={"train": str(path)})["train"]
    return raw.train_test_split(test_size=test_size, seed=seed)


def split_prompt_completion(text: str) -> tuple[str, str]:
    """Split at the last comma into (prompt including comma, completion)."""
    cut = text.rfind(",") + 1
    if cut <= 0:
        raise ValueError(f"Hand line has no comma separator: {text[:80]!r}")
    return text[:cut], text[cut:]


def encode_with_mask(
    text: str,
    tokenizer,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenize prompt and completion separately so label boundaries match BPE.

    Context tokens (and BOS) get label -100; action tokens and EOS are supervised.
    Long sequences keep the end (action) by dropping tokens from the front.
    """
    prompt_text, completion_text = split_prompt_completion(text)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    full_ids = [bos] + prompt_ids + completion_ids + [eos]
    # Supervised region starts after BOS + prompt.
    prompt_len = 1 + len(prompt_ids)

    if len(full_ids) > max_length:
        drop = len(full_ids) - max_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len - drop)

    labels = [-100] * len(full_ids)
    for i in range(prompt_len, len(full_ids)):
        labels[i] = full_ids[i]
    return full_ids, labels

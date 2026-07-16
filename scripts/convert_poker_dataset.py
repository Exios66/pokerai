"""
Convert SoelMgd/Poker_Dataset (HuggingFace) into plain-text hand files
compatible with the pokerai training pipeline.

Output (comma-separated prompt,action — same format as scripts/prepare_data.py):
    data/hands.txt         -- ALL hands (train+test), raw joined lines
    data/hands_clean.txt   -- cleaned lines (what trainers load by default)
    data/hands_train.txt   -- HF train split, cleaned
    data/hands_test.txt    -- HF test split, cleaned

Prefer ``python scripts/prepare_data.py`` for the standard path. This script
additionally preserves the dataset's official train/test split files.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import load_dataset

from pokerai.config import DATA_DIR, HANDS_CLEAN, HANDS_RAW, HF_DATASET
from pokerai.data import clean_line, to_line


def convert_split(dataset_split, path: Path) -> list[str]:
    lines = [
        clean_line(to_line(example["question"], example["answer"]))
        for example in dataset_split
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines):,} lines to {path}")
    return lines


def main() -> None:
    print(f"Downloading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET)

    print("Train split:", ds["train"])
    print("Test split:", ds["test"])

    train_lines = convert_split(ds["train"], DATA_DIR / "hands_train.txt")
    test_lines = convert_split(ds["test"], DATA_DIR / "hands_test.txt")

    # Combined raw + cleaned files for tokenizer / default trainers.
    raw_lines = [
        to_line(q, a)
        for split in ds
        for q, a in zip(ds[split]["question"], ds[split]["answer"])
    ]
    HANDS_RAW.parent.mkdir(parents=True, exist_ok=True)
    HANDS_RAW.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(raw_lines):,} lines to {HANDS_RAW}")

    cleaned = [clean_line(line) for line in raw_lines]
    HANDS_CLEAN.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    print(f"Wrote {len(cleaned):,} lines to {HANDS_CLEAN}")

    action_types = Counter(a.split()[0] for a in ds["train"]["answer"])
    print("\nAction type distribution (train):")
    for action, count in action_types.most_common():
        print(f"  {action:10s} {count:6,}  ({100 * count / len(ds['train']):.1f}%)")

    print(
        f"\nDone. Trainers load {HANDS_CLEAN} by default "
        f"({len(train_lines):,} train / {len(test_lines):,} test split files also written)."
    )


if __name__ == "__main__":
    main()

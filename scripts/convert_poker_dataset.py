"""
Convert SoelMgd/Poker_Dataset (HuggingFace) into plain-text hand files
compatible with the existing tokenizer-training / bigram-training pipeline.

Source schema:
    question: str  -- table state, ends right before the hero's decision point
    answer:   str  -- action label, e.g. "FOLD", "CALL 2BB", "RAISE 3BB", "CHECK", "BET 4BB"

Output:
    data/hands.txt              -- ALL hands (train+test), used to train the tokenizer
    data/hands_train.txt        -- train split, one hand per line, question+answer joined
    data/hands_test.txt         -- test split, same format (held out for eval)

Each line is formatted as:
    <question><SEP><answer>

where <SEP> is a literal separator token the model learns to associate with
"decision point reached." We use " => " here -- change COMPRESS() below if you
want to map this into your condensed shorthand notation instead of raw text.
"""

import os
from datasets import load_dataset

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

SEP = " => "


def compress(question: str, answer: str) -> str:
    """
    Hook for your condensed shorthand notation.

    Right now this just joins question + answer with a separator and
    collapses whitespace/newlines (the raw HF text contains literal
    newlines between betting rounds, e.g. after [STACKS] blocks).

    If/when you finalize your compressed notation (e.g. mapping
    "[TABLE_CONFIGURATION] BTN=P3 SB=P1 0.5BB BB=P2 1BB" down to a
    shorter symbolic form), do that remapping here so both hands.txt
    and the tokenizer stay in sync with a single source of truth.
    """
    q = " ".join(question.split())  # collapse newlines/extra spaces
    a = " ".join(answer.split())
    return f"{q}{SEP}{a}"


def convert_split(dataset_split, filename: str) -> int:
    path = os.path.join(OUT_DIR, filename)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            line = compress(example["question"], example["answer"])
            f.write(line + "\n")
            n += 1
    print(f"Wrote {n:,} lines to {path}")
    return n


def main():
    print("Downloading SoelMgd/Poker_Dataset ...")
    ds = load_dataset("SoelMgd/Poker_Dataset")

    print("Train split:", ds["train"])
    print("Test split:", ds["test"])

    n_train = convert_split(ds["train"], "hands_train.txt")
    n_test = convert_split(ds["test"], "hands_test.txt")

    # Combined file for tokenizer training (chunk 1 of the pipeline uses
    # data/hands.txt -- tokenizer should see both splits' vocabulary,
    # since held-out test hands can still contain notation/action tokens
    # not seen in train).
    combined_path = os.path.join(OUT_DIR, "hands.txt")
    with open(combined_path, "w", encoding="utf-8") as out:
        for fname in ("hands_train.txt", "hands_test.txt"):
            with open(os.path.join(OUT_DIR, fname), encoding="utf-8") as f:
                out.write(f.read())
    print(f"Wrote combined {n_train + n_test:,} lines to {combined_path}")

    # Quick label distribution sanity check -- useful since 'answer' has
    # only ~187 distinct string values (action + sizing combos), which
    # matters a lot for how you think about class imbalance later.
    from collections import Counter
    action_types = Counter(a.split()[0] for a in ds["train"]["answer"])
    print("\nAction type distribution (train):")
    for action, count in action_types.most_common():
        print(f"  {action:10s} {count:6,}  ({100*count/len(ds['train']):.1f}%)")


if __name__ == "__main__":
    main()

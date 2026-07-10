import re
import pandas as pd
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
df = pd.concat(
    [
        pd.read_parquet("hf://datasets/SoelMgd/Poker_Dataset/" + splits["train"]),
        pd.read_parquet("hf://datasets/SoelMgd/Poker_Dataset/" + splits["test"]),
    ],
    ignore_index=True,
)

def to_line(question: str, answer: str) -> str:
    # Flatten multiline state into one line; join with comma so training
    # scripts can split prompt/completion at the last comma.
    context = " ".join(str(question).split())
    return f"{context},{str(answer).strip()}"

lines = [to_line(q, a) for q, a in zip(df["question"], df["answer"])]

with open("data/hands.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} hands to data/hands.txt")

def clean(line: str) -> str:
    # Strip the trailing amount only off FOLD -- it's always 0, pure noise.
    return re.sub(r"FOLD\d+(\.\d+)?BB$", "FOLD", line)

cleaned = [clean(l) for l in lines]

changed = sum(1 for a, b in zip(lines, cleaned) if a != b)
print(f"Modified {changed} / {len(lines)} lines")
print("Before:", lines[0][:120], "...")
print("After: ", cleaned[0][:120], "...")

with open("data/hands_fold_update.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned) + "\n")

dataset = load_dataset("text", data_files={"train": "data/hands_fold_update.txt"})
dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)

print("\nDataset split:")
print(dataset)
print("Example hand:", dataset["train"][0]["text"][:200], "...")
print("Total train examples:", len(dataset["train"]))
print("Total eval examples:", len(dataset["test"]))

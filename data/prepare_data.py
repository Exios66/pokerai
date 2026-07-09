import re
from datasets import load_dataset

# Clean the data: remove 0BB from FOLD actions
with open("data/hands.txt", encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]

def clean(line):
    # Strip the trailing amount only off FOLD -- it's always 0, pure noise.
    return re.sub(r"FOLD\d+(\.\d+)?BB$", "FOLD", line)

cleaned = [clean(l) for l in lines]

changed = sum(1 for a, b in zip(lines, cleaned) if a != b)
print(f"Modified {changed} / {len(lines)} lines")
print("Before:", lines[0])
print("After: ", cleaned[0])

# Save cleaned data
with open("data/hands_fold_update.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned) + "\n")

# Load and split the cleaned dataset
dataset = load_dataset("text", data_files={"train": "data/hands_fold_update.txt"})
dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)

print("\nDataset split:")
print(dataset)
print("Example hand:", dataset["train"][0]["text"])
print("Total train examples:", len(dataset["train"]))
print("Total eval examples:", len(dataset["test"]))



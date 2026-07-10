import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer")
vocab_size = len(tokenizer)

dataset = load_dataset("text", data_files={"train": "data/hands_fold_update.txt"})["train"]
dataset = dataset.train_test_split(test_size=0.05, seed=42)

def encode(example):
    ids = tokenizer(example["text"])["input_ids"]
    return [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]

train_ids = [encode(ex) for ex in dataset["train"]]
val_ids = [encode(ex) for ex in dataset["test"]]

# Turn each sequence into (current_token, next_token) pairs.
def make_pairs(sequences):
    xs, ys = [], []
    for seq in sequences:
        for i in range(len(seq) - 1):
            xs.append(seq[i])
            ys.append(seq[i + 1])
    return torch.tensor(xs), torch.tensor(ys)

train_x, train_y = make_pairs(train_ids)
val_x, val_y = make_pairs(val_ids)
print(f"Train pairs: {len(train_x):,}  Val pairs: {len(val_x):,}")

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)  # row i = logits for "next token after i"

    def forward(self, idx):
        return self.table(idx)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

batch_size = 4096
n_steps = 2000
lr = 1e-2

wandb.init(
    project="pokerai",
    name="bigram-baseline",
    config={
        "model": "bigram",
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "learning_rate": lr,
        "device": device,
        "train_pairs": len(train_x),
        "val_pairs": len(val_x),
    },
)

model = Bigram(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

train_x, train_y = train_x.to(device), train_y.to(device)
val_x, val_y = val_x.to(device), val_y.to(device)

for step in range(n_steps):
    idx = torch.randint(0, len(train_x), (batch_size,))
    xb, yb = train_x[idx], train_y[idx]

    logits = model(xb)
    loss = F.cross_entropy(logits, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0 or step == n_steps - 1:
        with torch.no_grad():
            val_loss = F.cross_entropy(model(val_x), val_y)
        print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")
        wandb.log(
            {"train/loss": loss.item(), "val/loss": val_loss.item()},
            step=step,
        )

with torch.no_grad():
    final_val_loss = F.cross_entropy(model(val_x), val_y).item()

random_baseline = math.log(vocab_size)
print(f"\nRandom-guess baseline loss: {random_baseline:.4f}  (ln(vocab_size))")
print(f"Bigram model final val loss: {final_val_loss:.4f}")
wandb.summary["random_baseline_loss"] = random_baseline
wandb.summary["final_val_loss"] = final_val_loss
wandb.finish()

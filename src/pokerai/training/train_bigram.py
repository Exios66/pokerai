"""Bigram baseline trained only on action-token transitions (fairer vs GPT-2)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from pokerai.config import (
    BIGRAM_BATCH_SIZE,
    BIGRAM_LR,
    BIGRAM_STEPS,
    HANDS_CLEAN,
    MODEL_BIGRAM_DIR,
    WANDB_PROJECT,
)
from pokerai.data import encode_with_mask, load_text_split
from pokerai.models import Bigram
from pokerai.training import get_device, load_tokenizer


def _action_pairs(sequences: list[tuple[list[int], list[int]]]):
    """Build (x, y) pairs only where the next token is supervised (action region)."""
    xs, ys = [], []
    for ids, labels in sequences:
        for i in range(len(ids) - 1):
            if labels[i + 1] != -100:
                xs.append(ids[i])
                ys.append(ids[i + 1])
    return torch.tensor(xs), torch.tensor(ys)


def main(
    hands_path: Path = HANDS_CLEAN,
    output_dir: Path = MODEL_BIGRAM_DIR,
    batch_size: int = BIGRAM_BATCH_SIZE,
    n_steps: int = BIGRAM_STEPS,
    lr: float = BIGRAM_LR,
) -> None:
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    max_length = tokenizer.model_max_length or 384

    split = load_text_split(hands_path)
    train_enc = [encode_with_mask(ex["text"], tokenizer, max_length) for ex in split["train"]]
    val_enc = [encode_with_mask(ex["text"], tokenizer, max_length) for ex in split["test"]]

    train_x, train_y = _action_pairs(train_enc)
    val_x, val_y = _action_pairs(val_enc)
    print(f"Train pairs: {len(train_x):,}  Val pairs: {len(val_x):,}")

    device = get_device()
    print("Training on:", device)

    wandb.init(
        project=WANDB_PROJECT,
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
            "masked_action_only": True,
        },
    )

    model = Bigram(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    for step in range(n_steps):
        idx = torch.randint(0, len(train_x), (batch_size,))
        xb, yb = train_x[idx], train_y[idx]
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == n_steps - 1:
            with torch.no_grad():
                val_loss = F.cross_entropy(model(val_x), val_y)
            print(
                f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}"
            )
            wandb.log(
                {"train/loss": loss.item(), "val/loss": val_loss.item()},
                step=step,
            )

    with torch.no_grad():
        final_val_loss = F.cross_entropy(model(val_x), val_y).item()

    random_baseline = math.log(vocab_size)
    print(f"\nRandom-guess baseline loss: {random_baseline:.4f}")
    print(f"Bigram model final val loss: {final_val_loss:.4f}")
    wandb.summary["random_baseline_loss"] = random_baseline
    wandb.summary["final_val_loss"] = final_val_loss

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "vocab_size": vocab_size},
        output_dir / "bigram.pt",
    )
    print(f"Saved to {output_dir / 'bigram.pt'}")
    wandb.finish()


if __name__ == "__main__":
    main()

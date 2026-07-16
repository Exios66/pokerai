"""Train GPT-2 from scratch with a raw PyTorch loop (action-only loss)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pokerai.config import (
    GPT2Hyperparams,
    HANDS_CLEAN,
    MODEL_GPT2_DIR,
    WANDB_PROJECT,
)
from pokerai.data import encode_with_mask, load_text_split
from pokerai.models import build_gpt2
from pokerai.training import get_device, load_tokenizer, wandb_is_configured


class HandsDataset(Dataset):
    def __init__(self, examples: list[tuple[list[int], list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids, labels, attn_mask = [], [], []
        for ids, lbls in batch:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(lbls + [-100] * pad_len)
            attn_mask.append([1] * len(ids) + [0] * pad_len)
        return (
            torch.tensor(input_ids),
            torch.tensor(labels),
            torch.tensor(attn_mask),
        )

    return collate


def main(
    hands_path: Path = HANDS_CLEAN,
    output_dir: Path = MODEL_GPT2_DIR,
    hp: GPT2Hyperparams | None = None,
) -> None:
    hp = hp or GPT2Hyperparams()
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer is missing pad_token_id")
    model = build_gpt2(tokenizer, hp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    device = get_device()
    cast(nn.Module, model).to(device)
    print("Training on:", device)

    split = load_text_split(hands_path)
    train_enc = [
        encode_with_mask(ex["text"], tokenizer, hp.n_positions) for ex in split["train"]
    ]
    val_enc = [
        encode_with_mask(ex["text"], tokenizer, hp.n_positions) for ex in split["test"]
    ]
    train_ds = HandsDataset(train_enc)
    val_ds = HandsDataset(val_enc)
    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp.batch_size, shuffle=False, collate_fn=collate
    )

    use_wandb = wandb_is_configured()
    if use_wandb:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            name="gpt2-raw",
            config={
                "model": "gpt2",
                "n_params": n_params,
                "vocab_size": vocab_size,
                "device": device,
                "train_examples": len(train_ds),
                "eval_examples": len(val_ds),
                **hp.as_dict(),
            },
        )

    ids, labels = train_ds[0]
    print("\nTokens:", tokenizer.convert_ids_to_tokens(ids))
    print(
        "Labels:",
        [
            tokenizer.convert_ids_to_tokens([t])[0] if t != -100 else "---"
            for t in labels
        ],
    )
    print("(Only non-'---' positions contribute to the loss.)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate)

    def run_eval() -> float:
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for input_ids, batch_labels, attn_mask in val_loader:
                input_ids = input_ids.to(device)
                batch_labels = batch_labels.to(device)
                attn_mask = attn_mask.to(device)
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[
                    :, :-1, :
                ].contiguous()
                targets = batch_labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                n_tok = (targets != -100).sum().item()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok
        model.train()
        return total_loss / max(total_tokens, 1)

    step = 0
    val_loss = float("nan")
    for epoch in range(hp.num_epochs):
        for input_ids, batch_labels, attn_mask in train_loader:
            input_ids = input_ids.to(device)
            batch_labels = batch_labels.to(device)
            attn_mask = attn_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[
                :, :-1, :
            ].contiguous()
            targets = batch_labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step:5d} | train loss {loss.item():.4f}")
                if use_wandb:
                    import wandb

                    wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=step)
            step += 1

        val_loss = run_eval()
        print(f"== end of epoch {epoch}: val loss {val_loss:.4f} ==")
        if use_wandb:
            import wandb

            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=step)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved model to {output_dir}")
    if use_wandb:
        import wandb

        wandb.summary["final_val_loss"] = val_loss
        wandb.finish()


if __name__ == "__main__":
    main()

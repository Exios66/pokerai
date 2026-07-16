#!/usr/bin/env python3
"""Launch a tagged W&B training run with explicit hyperparameters.

Examples:
  python scripts/run_experiment.py bigram
  python scripts/run_experiment.py gpt2 --n-layer 4 --lr 1e-4 --tags depth-4,lr-sweep
  python scripts/run_experiment.py trl --epochs 5 --batch-size 16 --group capacity
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pokerai.config import (
    BIGRAM_BATCH_SIZE,
    BIGRAM_LR,
    BIGRAM_STEPS,
    GPT2_BATCH_SIZE,
    GPT2_EPOCHS,
    GPT2_LR,
    GPT2Hyperparams,
    HANDS_CLEAN,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    N_POSITIONS,
    WANDB_PROJECT,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "trainer",
        choices=["bigram", "gpt2", "trl"],
        help="Which training entrypoint to run",
    )
    p.add_argument("--hands", type=Path, default=HANDS_CLEAN)
    p.add_argument("--group", default=None, help="WANDB_RUN_GROUP")
    p.add_argument("--name", default=None, help="W&B run name override")
    p.add_argument("--tags", default="", help="Comma-separated WANDB_TAGS")

    # GPT-2 / TRL
    p.add_argument("--n-positions", type=int, default=N_POSITIONS)
    p.add_argument("--n-embd", type=int, default=N_EMBD)
    p.add_argument("--n-layer", type=int, default=N_LAYER)
    p.add_argument("--n-head", type=int, default=N_HEAD)
    p.add_argument("--lr", type=float, default=GPT2_LR)
    p.add_argument("--batch-size", type=int, default=GPT2_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=GPT2_EPOCHS)

    # Bigram
    p.add_argument("--bigram-steps", type=int, default=BIGRAM_STEPS)
    p.add_argument("--bigram-lr", type=float, default=BIGRAM_LR)
    p.add_argument("--bigram-batch-size", type=int, default=BIGRAM_BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    if args.group:
        os.environ["WANDB_RUN_GROUP"] = args.group
    if args.tags:
        os.environ["WANDB_TAGS"] = args.tags
    if args.name:
        os.environ["WANDB_NAME"] = args.name

    if args.trainer == "bigram":
        from pokerai.training.train_bigram import main as train

        train(
            hands_path=args.hands,
            batch_size=args.bigram_batch_size,
            n_steps=args.bigram_steps,
            lr=args.bigram_lr,
        )
        return

    if args.n_embd % args.n_head != 0:
        raise SystemExit(
            f"n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head})"
        )

    hp = GPT2Hyperparams(
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    if args.trainer == "gpt2":
        from pokerai.training.train_gpt2 import main as train

        train(hands_path=args.hands, hp=hp)
    else:
        from pokerai.training.train_trl import main as train

        train(hands_path=args.hands, hp=hp)


if __name__ == "__main__":
    main()

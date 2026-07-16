#!/usr/bin/env python3
"""Launch a tagged training run with explicit hyperparameters.

W&B logging is optional: trainers only log when W&B is configured
(API key, saved login, or WANDB_MODE=offline/online/shared). Use
``--require-wandb`` for catalog/sweep runs that must appear on the dashboard.

Examples:
  python scripts/run_experiment.py bigram
  python scripts/run_experiment.py gpt2 --n-layer 4 --lr 1e-4 --tags depth-4,lr-sweep
  python scripts/run_experiment.py trl --epochs 5 --batch-size 16 --group capacity
  python scripts/run_experiment.py gpt2 --require-wandb --group capacity --tags exp-b1
"""

from __future__ import annotations

import argparse
import os
import sys
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
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from pokerai.training import ensure_wandb_project, wandb_is_configured


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
    p.add_argument(
        "--require-wandb",
        action="store_true",
        help="Exit with an error if W&B is not configured (for catalog/sweep runs)",
    )

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


def _announce_wandb(require: bool) -> None:
    ensure_wandb_project()
    if wandb_is_configured():
        mode = os.environ.get("WANDB_MODE", "online (login/key)")
        name = os.environ.get("WANDB_NAME", "(default trainer name)")
        group = os.environ.get("WANDB_RUN_GROUP", "(none)")
        tags = os.environ.get("WANDB_TAGS", "(none)")
        entity = os.environ.get("WANDB_ENTITY", WANDB_ENTITY)
        project = os.environ.get("WANDB_PROJECT", WANDB_PROJECT)
        print(
            f"W&B logging enabled — {entity}/{project} "
            f"mode={mode} name={name} group={group} tags={tags}"
        )
        return

    msg = (
        "W&B logging disabled — training will run without experiment tracking.\n"
        "  Enable:  wandb login   OR   export WANDB_MODE=offline\n"
        "  Silence: export WANDB_MODE=disabled\n"
        "  Require: pass --require-wandb to fail if logging is unavailable\n"
        f"  Target:  https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}"
    )
    print(msg, file=sys.stderr)
    if require:
        raise SystemExit(
            "error: --require-wandb was set but W&B is not configured "
            "(no API key / login, and WANDB_MODE is not offline/online/shared)"
        )


def main() -> None:
    args = _parse_args()
    if args.group:
        os.environ["WANDB_RUN_GROUP"] = args.group
    if args.tags:
        os.environ["WANDB_TAGS"] = args.tags
    if args.name:
        os.environ["WANDB_NAME"] = args.name

    _announce_wandb(require=args.require_wandb)

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

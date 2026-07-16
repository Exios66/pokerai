#!/usr/bin/env python3
"""Launch a tagged experiment with explicit hyperparameters.

Trainers:
  bigram, gpt2, trl          — original LM / baseline stack
  majority                   — always-predict majority action type
  features                   — LogReg / RandomForest on structured features
  weighted-gpt2              — GPT-2 with inverse-frequency action weights

W&B logging is optional unless ``--require-wandb``. After generative training,
action-type charts (confusion, per-class F1) and occlusion feature importance
are logged when evaluation runs (GPT-2 / weighted-gpt2 / evaluate.py).

Examples:
  python scripts/run_experiment.py majority --group imbalance --tags exp-f2
  python scripts/run_experiment.py features --method rf --group alt-approaches
  python scripts/run_experiment.py weighted-gpt2 --group imbalance --tags exp-f3
  python scripts/run_experiment.py gpt2 --n-layer 4 --eval-max-examples 128
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


TRAINERS = ("bigram", "gpt2", "trl", "majority", "features", "weighted-gpt2")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trainer", choices=TRAINERS, help="Which training entrypoint to run")
    p.add_argument("--hands", type=Path, default=HANDS_CLEAN)
    p.add_argument("--group", default=None, help="WANDB_RUN_GROUP")
    p.add_argument("--name", default=None, help="W&B run name override")
    p.add_argument("--tags", default="", help="Comma-separated WANDB_TAGS")
    p.add_argument(
        "--require-wandb",
        action="store_true",
        help="Exit with an error if W&B is not configured (for catalog/sweep runs)",
    )

    # GPT-2 / TRL / weighted
    p.add_argument("--n-positions", type=int, default=N_POSITIONS)
    p.add_argument("--n-embd", type=int, default=N_EMBD)
    p.add_argument("--n-layer", type=int, default=N_LAYER)
    p.add_argument("--n-head", type=int, default=N_HEAD)
    p.add_argument("--lr", type=float, default=GPT2_LR)
    p.add_argument("--batch-size", type=int, default=GPT2_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=GPT2_EPOCHS)
    p.add_argument(
        "--eval-max-examples",
        type=int,
        default=256,
        help="Val hands for post-train action-type eval / charts (gpt2, weighted-gpt2)",
    )
    p.add_argument(
        "--skip-action-eval",
        action="store_true",
        help="Skip post-train generative action evaluation",
    )

    # Bigram
    p.add_argument("--bigram-steps", type=int, default=BIGRAM_STEPS)
    p.add_argument("--bigram-lr", type=float, default=BIGRAM_LR)
    p.add_argument("--bigram-batch-size", type=int, default=BIGRAM_BATCH_SIZE)

    # Features approach
    p.add_argument(
        "--method",
        choices=["rf", "logreg"],
        default="rf",
        help="features trainer: RandomForest or LogisticRegression",
    )
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization")
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


def _gpt2_hp(args: argparse.Namespace) -> GPT2Hyperparams:
    if args.n_embd % args.n_head != 0:
        raise SystemExit(
            f"n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head})"
        )
    return GPT2Hyperparams(
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
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

    if args.trainer == "majority":
        from pokerai.training.train_majority import main as train

        train(hands_path=args.hands)
        return

    if args.trainer == "features":
        from pokerai.training.train_features import main as train

        train(
            hands_path=args.hands,
            method=args.method,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            C=args.C,
        )
        return

    hp = _gpt2_hp(args)

    if args.trainer == "gpt2":
        from pokerai.training.train_gpt2 import main as train

        train(
            hands_path=args.hands,
            hp=hp,
            class_weight=False,
            run_action_eval=not args.skip_action_eval,
            eval_max_examples=args.eval_max_examples,
        )
        return

    if args.trainer == "weighted-gpt2":
        from pokerai.training.train_gpt2 import main as train
        from pokerai.training.train_weighted_gpt2 import MODEL_WEIGHTED_DIR

        train(
            hands_path=args.hands,
            output_dir=MODEL_WEIGHTED_DIR,
            hp=hp,
            class_weight=True,
            run_action_eval=not args.skip_action_eval,
            eval_max_examples=args.eval_max_examples,
        )
        return

    # trl
    from pokerai.training.train_trl import main as train

    train(hands_path=args.hands, hp=hp)


if __name__ == "__main__":
    main()

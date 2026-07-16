#!/usr/bin/env python3
"""Evaluate a saved model: action metrics, charts, feature importance.

Examples:
  python scripts/evaluate.py --model artifacts/models/gpt2 --max-examples 256
  python scripts/evaluate.py --model artifacts/models/gpt2_trl --require-wandb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pokerai.config import HANDS_CLEAN, MODEL_GPT2_DIR, MODEL_TRL_DIR
from pokerai.data import load_text_split
from pokerai.eval.report import run_lm_evaluation_report
from pokerai.inference.predict import load_model
from pokerai.training import ensure_wandb_project, wandb_is_configured


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=MODEL_TRL_DIR if MODEL_TRL_DIR.exists() else MODEL_GPT2_DIR,
        help="Directory with saved HF model + tokenizer",
    )
    p.add_argument("--hands", type=Path, default=HANDS_CLEAN)
    p.add_argument("--max-examples", type=int, default=256)
    p.add_argument("--max-importance", type=int, default=64)
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Where to write PNG/JSON (default: <model>/report)",
    )
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--require-wandb", action="store_true")
    p.add_argument("--name", default=None, help="WANDB_NAME override")
    p.add_argument("--group", default=None, help="WANDB_RUN_GROUP")
    p.add_argument("--tags", default="", help="Comma-separated WANDB_TAGS")
    args = p.parse_args()

    if args.group:
        os.environ["WANDB_RUN_GROUP"] = args.group
    if args.tags:
        os.environ["WANDB_TAGS"] = args.tags
    if args.name:
        os.environ["WANDB_NAME"] = args.name

    use_wandb = wandb_is_configured()
    if args.require_wandb and not use_wandb:
        raise SystemExit("error: --require-wandb set but W&B is not configured")
    if use_wandb:
        import wandb
        from pokerai.config import WANDB_ENTITY, WANDB_PROJECT

        ensure_wandb_project()
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=os.environ.get("WANDB_NAME", f"eval-{args.model.name}"),
            job_type="eval",
            config={"model_dir": str(args.model), "max_examples": args.max_examples},
        )

    model, tokenizer = load_model(args.model)
    split = load_text_split(args.hands)
    lines = [ex["text"] for ex in split[args.split]]
    report_dir = args.report_dir or (args.model / "report")
    summary = run_lm_evaluation_report(
        model,
        tokenizer,
        lines,
        report_dir=report_dir,
        max_examples=args.max_examples,
        max_importance=args.max_importance,
        log_wandb=use_wandb,
    )
    m = summary["metrics"]
    print(f"accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} n={m['n']}")
    for label, scores in m["per_class"].items():
        print(
            f"  {label:6s} P={scores['precision']:.3f} R={scores['recall']:.3f} "
            f"F1={scores['f1']:.3f} support={scores['support']}"
        )
    if summary.get("importance"):
        top = sorted(summary["importance"].items(), key=lambda kv: -kv[1])[:8]
        print("Occlusion importance:", ", ".join(f"{k}={v:.3f}" for k, v in top))
    print(f"Report written to {report_dir}")

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()

"""Majority-class action-type baseline (imbalance floor)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pokerai.config import ARTIFACTS_DIR, HANDS_CLEAN, WANDB_ENTITY, WANDB_PROJECT
from pokerai.data import load_text_split, split_prompt_completion
from pokerai.eval.actions import action_type, action_type_counts
from pokerai.eval.metrics import compute_action_metrics
from pokerai.eval.visualize import log_action_charts_to_wandb
from pokerai.training import ensure_wandb_project, wandb_is_configured

MODEL_MAJORITY_DIR = ARTIFACTS_DIR / "models" / "majority"


def main(
    hands_path: Path = HANDS_CLEAN,
    output_dir: Path = MODEL_MAJORITY_DIR,
) -> None:
    split = load_text_split(hands_path)
    train_actions = [
        action_type(split_prompt_completion(ex["text"])[1]) for ex in split["train"]
    ]
    val_true = [
        action_type(split_prompt_completion(ex["text"])[1]) for ex in split["test"]
    ]
    majority = Counter(train_actions).most_common(1)[0][0]
    val_pred = [majority] * len(val_true)
    metrics = compute_action_metrics(val_true, val_pred)
    dist = action_type_counts(train_actions)

    print(f"Majority class: {majority}")
    print(f"Train distribution: {dist}")
    print(
        f"Val accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f} n={metrics.n}"
    )
    for name, scores in metrics.per_class.items():
        print(
            f"  {name:6s} P={scores.precision:.3f} R={scores.recall:.3f} "
            f"F1={scores.f1:.3f} support={scores.support}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "report"
    payload = {
        "model": "majority",
        "majority_class": majority,
        "train_distribution": dist,
        "metrics": metrics.as_dict(),
    }
    (output_dir / "majority.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    use_wandb = wandb_is_configured()
    if use_wandb:
        import os

        import wandb

        ensure_wandb_project()
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=os.environ.get("WANDB_NAME", "majority-baseline"),
            config={"model": "majority", "majority_class": majority, **dist},
        )
        log_action_charts_to_wandb(
            metrics,
            y_true=val_true,
            y_pred=val_pred,
            artifact_dir=report_dir,
            prefix="eval",
        )
        wandb.summary["majority_class"] = majority
        wandb.finish()
    else:
        from pokerai.eval.visualize import save_confusion_png, save_per_class_f1_png

        save_confusion_png(metrics, report_dir / "confusion.png", title="Majority confusion")
        save_per_class_f1_png(metrics, report_dir / "per_class_f1.png")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()

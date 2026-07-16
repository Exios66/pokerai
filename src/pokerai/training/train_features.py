"""Structured-feature classifiers (LogReg / RandomForest) — non-LM approach."""

from __future__ import annotations

import json
from pathlib import Path

from pokerai.config import ARTIFACTS_DIR, HANDS_CLEAN, WANDB_ENTITY, WANDB_PROJECT
from pokerai.data import load_text_split
from pokerai.eval.actions import ACTION_TYPES
from pokerai.eval.metrics import compute_action_metrics
from pokerai.eval.visualize import log_action_charts_to_wandb
from pokerai.features import feature_names, features_and_label
from pokerai.training import ensure_wandb_project, wandb_is_configured

MODEL_FEATURES_DIR = ARTIFACTS_DIR / "models" / "features"


def main(
    hands_path: Path = HANDS_CLEAN,
    output_dir: Path = MODEL_FEATURES_DIR,
    method: str = "rf",
    n_estimators: int = 200,
    max_depth: int | None = 12,
    C: float = 1.0,
) -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for the features trainer. "
            "Install with: pip install scikit-learn"
        ) from exc

    split = load_text_split(hands_path)
    x_train, y_train = [], []
    for ex in split["train"]:
        x, y = features_and_label(ex["text"])
        x_train.append(x)
        y_train.append(y)
    x_val, y_val = [], []
    for ex in split["test"]:
        x, y = features_and_label(ex["text"])
        x_val.append(x)
        y_val.append(y)

    names = feature_names()
    if method == "logreg":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=C,
                        class_weight="balanced",
                        multi_class="auto",
                    ),
                ),
            ]
        )
        model_name = "logreg-features"
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model_name = "rf-features"
        method = "rf"

    clf.fit(x_train, y_train)
    y_pred = list(clf.predict(x_val))
    metrics = compute_action_metrics(y_val, y_pred)

    importance: dict[str, float] = {}
    if method == "rf":
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            importance = {n: float(v) for n, v in zip(names, importances)}
    else:
        # Mean absolute coefficient across classes (scaled features).
        model = clf.named_steps["model"]
        coef = getattr(model, "coef_", None)
        if coef is not None:
            import numpy as np

            mean_abs = np.mean(np.abs(coef), axis=0)
            importance = {n: float(v) for n, v in zip(names, mean_abs)}

    print(f"Model: {model_name}")
    print(
        f"Val accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f} n={metrics.n}"
    )
    for name, scores in metrics.per_class.items():
        print(
            f"  {name:6s} P={scores.precision:.3f} R={scores.recall:.3f} "
            f"F1={scores.f1:.3f} support={scores.support}"
        )
    if importance:
        top = sorted(importance.items(), key=lambda kv: -kv[1])[:8]
        print("Top features:", ", ".join(f"{k}={v:.3f}" for k, v in top))

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "report"
    try:
        import joblib

        joblib.dump(
            {"model": clf, "feature_names": names, "method": method, "labels": list(ACTION_TYPES)},
            output_dir / "features_model.joblib",
        )
    except ImportError:
        pass

    payload = {
        "model": model_name,
        "method": method,
        "metrics": metrics.as_dict(),
        "importance": importance,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    use_wandb = wandb_is_configured()
    if use_wandb:
        import os

        import wandb

        ensure_wandb_project()
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=os.environ.get("WANDB_NAME", model_name),
            config={
                "model": model_name,
                "method": method,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "C": C,
                "n_features": len(names),
            },
        )
        log_action_charts_to_wandb(
            metrics,
            y_true=y_val,
            y_pred=y_pred,
            importance=importance,
            artifact_dir=report_dir,
            prefix="eval",
        )
        wandb.finish()
    else:
        from pokerai.eval.visualize import (
            save_confusion_png,
            save_importance_png,
            save_per_class_f1_png,
        )

        save_confusion_png(metrics, report_dir / "confusion.png")
        save_per_class_f1_png(metrics, report_dir / "per_class_f1.png")
        if importance:
            save_importance_png(importance, report_dir / "feature_importance.png")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()

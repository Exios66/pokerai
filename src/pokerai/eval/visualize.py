"""Chart helpers for local artifacts and W&B."""

from __future__ import annotations

from pathlib import Path

from pokerai.eval.actions import ACTION_TYPES
from pokerai.eval.metrics import ActionMetrics


def save_confusion_png(metrics: ActionMetrics, path: Path, title: str = "Action confusion") -> Path:
    """Save a confusion-matrix heatmap PNG (requires matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array(metrics.confusion, dtype=float)
    # Row-normalize for readability when FOLD dominates.
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ACTION_TYPES)))
    ax.set_yticks(range(len(ACTION_TYPES)))
    ax.set_xticklabels(ACTION_TYPES, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_TYPES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(ACTION_TYPES)):
        for j in range(len(ACTION_TYPES)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_importance_png(
    scores: dict[str, float],
    path: Path,
    title: str = "Feature importance",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels) + 1)))
    ax.barh(labels[::-1], values[::-1], color="#2a6f97")
    ax.set_xlabel("Importance / Δ metric")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_per_class_f1_png(metrics: ActionMetrics, path: Path, title: str = "Per-action F1") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(ACTION_TYPES)
    values = [metrics.per_class[l].f1 for l in labels]
    supports = [metrics.per_class[l].support for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="#e76f51")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(title)
    for bar, support in zip(bars, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={support}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def log_action_charts_to_wandb(
    metrics: ActionMetrics,
    *,
    y_true: list[str] | None = None,
    y_pred: list[str] | None = None,
    importance: dict[str, float] | None = None,
    artifact_dir: Path | None = None,
    prefix: str = "eval",
) -> dict:
    """Log scalars + confusion/importance plots to the active W&B run (if any)."""
    payload = metrics.flat_wandb_metrics(prefix=f"{prefix}/action")
    try:
        import wandb
    except ImportError:
        return payload
    if wandb.run is None:
        return payload

    from pokerai.eval.actions import action_type

    if y_true is not None and y_pred is not None:
        payload[f"{prefix}/confusion"] = wandb.plot.confusion_matrix(
            preds=[action_type(p) for p in y_pred],
            y_true=[action_type(t) for t in y_true],
            class_names=list(ACTION_TYPES),
            title="Action-type confusion",
        )

    table = wandb.Table(columns=["action", "precision", "recall", "f1", "support"])
    for name, scores in metrics.per_class.items():
        table.add_data(name, scores.precision, scores.recall, scores.f1, scores.support)
    payload[f"{prefix}/per_class_table"] = table

    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        cm_path = save_confusion_png(metrics, artifact_dir / "confusion.png")
        f1_path = save_per_class_f1_png(metrics, artifact_dir / "per_class_f1.png")
        payload[f"{prefix}/confusion_image"] = wandb.Image(str(cm_path))
        payload[f"{prefix}/per_class_f1_image"] = wandb.Image(str(f1_path))
        if importance:
            imp_path = save_importance_png(
                importance, artifact_dir / "feature_importance.png"
            )
            payload[f"{prefix}/feature_importance_image"] = wandb.Image(str(imp_path))

    if importance:
        imp_table = wandb.Table(columns=["feature", "importance"])
        for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
            imp_table.add_data(k, v)
        payload[f"{prefix}/feature_importance_table"] = imp_table
        for k, v in importance.items():
            payload[f"{prefix}/importance/{k}"] = float(v)

    wandb.log(payload)
    for k, v in metrics.flat_wandb_metrics(prefix=f"{prefix}/action").items():
        wandb.summary[k] = v
    return payload

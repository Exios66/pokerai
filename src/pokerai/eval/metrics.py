"""Classification metrics over coarse action types."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pokerai.eval.actions import ACTION_TYPES, action_type


@dataclass(frozen=True)
class ClassScores:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ActionMetrics:
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassScores]
    confusion: list[list[int]]  # rows = true, cols = pred, order ACTION_TYPES
    n: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "n": self.n,
            "per_class": {k: asdict(v) for k, v in self.per_class.items()},
            "confusion": self.confusion,
            "labels": list(ACTION_TYPES),
        }

    def flat_wandb_metrics(self, prefix: str = "action") -> dict[str, float]:
        out: dict[str, float] = {
            f"{prefix}/accuracy": self.accuracy,
            f"{prefix}/macro_f1": self.macro_f1,
            f"{prefix}/n": float(self.n),
        }
        for name, scores in self.per_class.items():
            key = name.lower()
            out[f"{prefix}/{key}/precision"] = scores.precision
            out[f"{prefix}/{key}/recall"] = scores.recall
            out[f"{prefix}/{key}/f1"] = scores.f1
            out[f"{prefix}/{key}/support"] = float(scores.support)
        return out


def compute_action_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> ActionMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    true_t = [action_type(y) for y in y_true]
    pred_t = [action_type(y) for y in y_pred]
    n = len(true_t)
    index = {label: i for i, label in enumerate(ACTION_TYPES)}
    conf = [[0 for _ in ACTION_TYPES] for _ in ACTION_TYPES]
    correct = 0
    for t, p in zip(true_t, pred_t):
        conf[index[t]][index[p]] += 1
        if t == p:
            correct += 1

    per_class: dict[str, ClassScores] = {}
    f1s: list[float] = []
    for i, label in enumerate(ACTION_TYPES):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(len(ACTION_TYPES))) - tp
        fn = sum(conf[i]) - tp
        support = sum(conf[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[label] = ClassScores(precision, recall, f1, support)
        if support > 0:
            f1s.append(f1)

    return ActionMetrics(
        accuracy=correct / n if n else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_class=per_class,
        confusion=conf,
        n=n,
    )

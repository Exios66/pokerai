"""Evaluate generative / classifier models on action-type metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from pokerai.config import ARTIFACTS_DIR, N_POSITIONS
from pokerai.data import split_prompt_completion
from pokerai.eval.importance import occlusion_importance
from pokerai.eval.metrics import ActionMetrics, compute_action_metrics
from pokerai.eval.visualize import log_action_charts_to_wandb, save_confusion_png, save_importance_png, save_per_class_f1_png
from pokerai.inference.predict import predict_action
from pokerai.training import get_device


def evaluate_generations(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    lines: list[str],
    *,
    max_examples: int | None = None,
    max_new_tokens: int = 16,
) -> tuple[ActionMetrics, list[str], list[str], list[str]]:
    """Greedy-generate actions for each hand line; return metrics + parallel lists."""
    subset = lines[:max_examples] if max_examples is not None else lines
    prompts: list[str] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    for line in subset:
        prompt, completion = split_prompt_completion(line)
        prompts.append(prompt)
        y_true.append(completion)
        y_pred.append(
            predict_action(prompt, model, tokenizer, max_new_tokens=max_new_tokens)
        )
    metrics = compute_action_metrics(y_true, y_pred)
    return metrics, prompts, y_true, y_pred


def teacher_forced_action_type_accuracy(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    lines: list[str],
    *,
    max_examples: int | None = 256,
) -> float:
    """Accuracy of the first supervised action token under teacher forcing."""
    from pokerai.data import encode_with_mask
    from pokerai.eval.actions import action_type

    device = get_device()
    model.eval()
    model.to(device)
    subset = lines[:max_examples] if max_examples is not None else lines
    correct = total = 0
    with torch.no_grad():
        for line in subset:
            ids, labels = encode_with_mask(line, tokenizer, N_POSITIONS)
            # First supervised index
            try:
                start = next(i for i, t in enumerate(labels) if t != -100)
            except StopIteration:
                continue
            if start == 0:
                continue
            tensor = torch.tensor([ids[:start]], device=device)
            logits = model(tensor).logits[0, -1]
            pred_id = int(logits.argmax())
            true_id = ids[start]
            pred_tok = tokenizer.decode([pred_id]).strip()
            true_tok = tokenizer.decode([true_id]).strip()
            # Compare coarse type when tokens are action words; else exact token.
            if action_type(pred_tok) == action_type(true_tok) and action_type(true_tok) != "OTHER":
                correct += 1
            elif pred_id == true_id:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def run_lm_evaluation_report(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    lines: list[str],
    *,
    report_dir: Path | None = None,
    max_examples: int = 256,
    max_importance: int = 64,
    log_wandb: bool = True,
) -> dict:
    """Full eval: action metrics, charts, and occlusion importance."""
    report_dir = Path(report_dir or (ARTIFACTS_DIR / "reports" / "lm_eval"))
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics, prompts, y_true, y_pred = evaluate_generations(
        model, tokenizer, lines, max_examples=max_examples
    )

    def _acc_on_prompts(ps: list[str]) -> float:
        # Score = fraction of generations matching true action type on the paired labels.
        preds = [
            predict_action(p, model, tokenizer) for p in ps
        ]
        # Align with the same slice of y_true
        m = compute_action_metrics(y_true[: len(preds)], preds)
        return m.accuracy

    importance: dict[str, float] = {}
    if prompts:
        n_imp = min(max_importance, len(prompts))
        importance = occlusion_importance(
            prompts[:n_imp],
            lambda ps: _acc_on_prompts(ps),
        )

    save_confusion_png(metrics, report_dir / "confusion.png")
    save_per_class_f1_png(metrics, report_dir / "per_class_f1.png")
    if importance:
        save_importance_png(importance, report_dir / "feature_importance.png")

    if log_wandb:
        log_action_charts_to_wandb(
            metrics,
            y_true=y_true,
            y_pred=y_pred,
            importance=importance,
            artifact_dir=report_dir,
            prefix="eval",
        )

    summary = {
        "metrics": metrics.as_dict(),
        "importance": importance,
        "report_dir": str(report_dir),
        "max_examples": max_examples,
    }
    (report_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    return summary

"""Tests for action metrics, features, and occlusion importance."""

from __future__ import annotations

from pathlib import Path

from pokerai.eval.actions import action_type, action_type_counts
from pokerai.eval.importance import occlude_region, occlusion_importance
from pokerai.eval.metrics import compute_action_metrics
from pokerai.eval.visualize import save_confusion_png, save_per_class_f1_png
from pokerai.features import extract_features, feature_names, features_and_label


def test_action_type_parsing():
    assert action_type("FOLD") == "FOLD"
    assert action_type("CALL 2BB") == "CALL"
    assert action_type("RAISE 6.5BB") == "RAISE"
    assert action_type("CHECK") == "CHECK"
    assert action_type("BET 3BB") == "BET"
    assert action_type("ALL-IN") == "RAISE"
    assert action_type("???") == "OTHER"


def test_compute_action_metrics_perfect():
    y = ["FOLD", "CALL", "RAISE 2BB", "CHECK"]
    m = compute_action_metrics(y, y)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.per_class["FOLD"].support == 1


def test_compute_action_metrics_confusion_shape():
    m = compute_action_metrics(["FOLD", "CALL"], ["CALL", "CALL"])
    assert len(m.confusion) == 6
    assert m.accuracy == 0.5
    flat = m.flat_wandb_metrics()
    assert "action/call/f1" in flat


def test_extract_features_basic():
    prompt = (
        "[TABLE_CONFIGURATION] BTN=P3 SB=P1 0.5BB BB=P2 1BB "
        "[STACKS] P1: 44.2BB [Qh 9h] P2: 103.4BB P3: 165.2BB POT=1.5BB "
        "[PREFLOP] P3: RAISE 2BB P1:"
    )
    feats = extract_features(prompt)
    assert feats.pot_bb == 1.5
    assert feats.n_stacks == 3
    assert feats.street_preflop == 1
    assert feats.n_raises >= 1
    assert feats.has_hole_cards == 1
    assert feats.hole_suited == 1
    assert len(feats.vector()) == len(feature_names())


def test_features_and_label():
    line = "[STACKS] P1: 10BB [Ah Ad] POT=1.5BB [PREFLOP] P1:,RAISE 3BB"
    vec, label = features_and_label(line)
    assert label == "RAISE"
    assert len(vec) == len(feature_names())


def test_occlusion_changes_text():
    prompt = "[STACKS] P1: 10BB [Ah Kd] POT=2BB [PREFLOP] P1:"
    masked = occlude_region(prompt, "pot")
    assert "POT=" not in masked
    assert "POT_MASKED" in masked or "pot".upper() in masked.upper()


def test_occlusion_importance_ordering():
    prompts = ["AAAA POT=1BB", "BBBB POT=2BB", "CCCC POT=3BB"]

    def score(ps: list[str]) -> float:
        # Higher when POT= is present.
        return sum(1.0 for p in ps if "POT=" in p) / len(ps)

    scores = occlusion_importance(prompts, score, regions=["pot", "stacks"])
    assert scores["pot"] > scores["stacks"]


def test_save_charts(tmp_path: Path):
    m = compute_action_metrics(
        ["FOLD", "CALL", "RAISE", "FOLD"],
        ["FOLD", "FOLD", "RAISE", "CALL"],
    )
    cm = save_confusion_png(m, tmp_path / "cm.png")
    f1 = save_per_class_f1_png(m, tmp_path / "f1.png")
    assert cm.exists() and cm.stat().st_size > 0
    assert f1.exists() and f1.stat().st_size > 0


def test_action_type_counts():
    counts = action_type_counts(["FOLD", "FOLD", "CALL 1BB"])
    assert counts["FOLD"] == 2
    assert counts["CALL"] == 1
    assert counts["RAISE"] == 0

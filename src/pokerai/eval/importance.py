"""Occlusion-based feature importance over serialized poker states."""

from __future__ import annotations

import re
from collections.abc import Callable

# Named spans we ablate from the prompt (game state before the decision comma).
_REGION_PATTERNS: dict[str, re.Pattern[str]] = {
    "table_config": re.compile(r"\[TABLE_CONFIGURATION\][^\[\]]*"),
    "stacks": re.compile(r"\[STACKS\][^\[\]]*"),
    "pot": re.compile(r"\bPOT=\d+(\.\d+)?BB\b"),
    "preflop": re.compile(r"\[PREFLOP\][^\[\]]*"),
    "flop": re.compile(r"\[FLOP\][^\[\]]*"),
    "turn": re.compile(r"\[TURN\][^\[\]]*"),
    "river": re.compile(r"\[RIVER\][^\[\]]*"),
    "hole_cards": re.compile(r"\[[A-Za-z0-9]{2}(?:\s+[A-Za-z0-9]{2})*\]"),
}


def occlude_region(prompt: str, region: str) -> str:
    """Replace a named region with a placeholder so length stays nonzero."""
    pattern = _REGION_PATTERNS.get(region)
    if pattern is None:
        raise KeyError(f"Unknown region {region!r}; choose from {list(_REGION_PATTERNS)}")
    return pattern.sub(f"[{region.upper()}_MASKED]", prompt)


def occlusion_importance(
    prompts: list[str],
    score_fn: Callable[[list[str]], float],
    regions: list[str] | None = None,
) -> dict[str, float]:
    """Importance = baseline_score - score_after_occlusion (higher ⇒ more important).

    ``score_fn`` should return a scalar where higher is better (e.g. accuracy).
    """
    regions = regions or list(_REGION_PATTERNS)
    baseline = score_fn(prompts)
    scores: dict[str, float] = {}
    for region in regions:
        ablated = [occlude_region(p, region) for p in prompts]
        scores[region] = float(baseline - score_fn(ablated))
    return scores

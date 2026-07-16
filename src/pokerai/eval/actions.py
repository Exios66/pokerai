"""Action-type utilities for poker decision evaluation."""

from __future__ import annotations

import re
from collections import Counter

# Canonical decision classes we chart and score.
ACTION_TYPES: tuple[str, ...] = ("FOLD", "CALL", "RAISE", "CHECK", "BET", "OTHER")

_ACTION_RE = re.compile(
    r"^\s*(FOLD|CHECK|CALL|RAISE|BET|ALL[\s-]?IN)\b",
    re.IGNORECASE,
)


def action_type(text: str) -> str:
    """Map a free-form action string to a coarse class."""
    if not text or not str(text).strip():
        return "OTHER"
    m = _ACTION_RE.match(str(text).strip())
    if not m:
        return "OTHER"
    raw = m.group(1).upper().replace(" ", "").replace("-", "")
    if raw == "ALLIN":
        return "RAISE"
    if raw in ACTION_TYPES:
        return raw
    return "OTHER"


def action_type_counts(actions: list[str]) -> dict[str, int]:
    counts = Counter(action_type(a) for a in actions)
    return {k: int(counts.get(k, 0)) for k in ACTION_TYPES}

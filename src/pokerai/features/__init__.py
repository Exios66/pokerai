"""Structured feature extraction from serialized poker hands."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from pokerai.data import split_prompt_completion
from pokerai.eval.actions import action_type

_POT = re.compile(r"POT=(\d+(?:\.\d+)?)BB")
_STACK = re.compile(r"P\d+:\s*(\d+(?:\.\d+)?)BB")
_HOLE = re.compile(r"\[([A-Za-z0-9]{2})\s+([A-Za-z0-9]{2})\]")
_STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")
_RANK = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


@dataclass(frozen=True)
class HandFeatures:
    pot_bb: float
    n_stacks: int
    max_stack_bb: float
    min_stack_bb: float
    mean_stack_bb: float
    street_preflop: int
    street_flop: int
    street_turn: int
    street_river: int
    n_raises: int
    n_calls: int
    n_bets: int
    n_checks: int
    n_folds_hist: int
    facing_aggression: int
    has_hole_cards: int
    hole_high: int
    hole_low: int
    hole_suited: int
    hole_pair: int
    prompt_chars: int

    def vector(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]


def extract_features(prompt: str) -> HandFeatures:
    """Extract numeric features from a game-state prompt (no trailing action)."""
    text = prompt if not prompt.endswith(",") else prompt
    pot_m = _POT.search(text)
    pot = float(pot_m.group(1)) if pot_m else 0.0
    stacks = [float(x) for x in _STACK.findall(text)]
    street_flags = {f"street_{s.lower()}": int(f"[{s}]" in text) for s in _STREETS}
    # Prefer the last street present as "current".
    for s in reversed(_STREETS):
        if f"[{s}]" in text:
            for k in street_flags:
                street_flags[k] = 0
            street_flags[f"street_{s.lower()}"] = 1
            break

    hist = text.upper()
    n_raises = len(re.findall(r"\bRAISE\b", hist))
    n_calls = len(re.findall(r"\bCALL\b", hist))
    n_bets = len(re.findall(r"\bBET\b", hist))
    n_checks = len(re.findall(r"\bCHECK\b", hist))
    n_folds = len(re.findall(r"\bFOLD\b", hist))
    facing = int(n_raises + n_bets > 0)

    hole = _HOLE.search(text)
    has_hole = int(hole is not None)
    hole_high = hole_low = hole_suited = hole_pair = 0
    if hole:
        c1, c2 = hole.group(1), hole.group(2)
        r1, r2 = _RANK.get(c1[0].upper(), 0), _RANK.get(c2[0].upper(), 0)
        hole_high, hole_low = max(r1, r2), min(r1, r2)
        hole_suited = int(len(c1) > 1 and len(c2) > 1 and c1[1] == c2[1])
        hole_pair = int(r1 == r2 and r1 > 0)

    return HandFeatures(
        pot_bb=pot,
        n_stacks=len(stacks),
        max_stack_bb=max(stacks) if stacks else 0.0,
        min_stack_bb=min(stacks) if stacks else 0.0,
        mean_stack_bb=(sum(stacks) / len(stacks)) if stacks else 0.0,
        street_preflop=street_flags["street_preflop"],
        street_flop=street_flags["street_flop"],
        street_turn=street_flags["street_turn"],
        street_river=street_flags["street_river"],
        n_raises=n_raises,
        n_calls=n_calls,
        n_bets=n_bets,
        n_checks=n_checks,
        n_folds_hist=n_folds,
        facing_aggression=facing,
        has_hole_cards=has_hole,
        hole_high=hole_high,
        hole_low=hole_low,
        hole_suited=hole_suited,
        hole_pair=hole_pair,
        prompt_chars=len(text),
    )


def features_and_label(line: str) -> tuple[list[float], str]:
    prompt, completion = split_prompt_completion(line)
    return extract_features(prompt).vector(), action_type(completion)


# Fix HandFeatures.names() to not need a weird constructor
def feature_names() -> list[str]:
    return list(HandFeatures.__dataclass_fields__.keys())

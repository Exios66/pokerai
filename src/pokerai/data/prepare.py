"""Prepare hand histories from Hugging Face and write cleaned text files."""

from __future__ import annotations

from pokerai.config import HANDS_CLEAN, HANDS_RAW
from pokerai.data import fetch_hands, write_cleaned


def main() -> None:
    lines = fetch_hands(HANDS_RAW)
    print(f"Wrote {len(lines)} hands to {HANDS_RAW}")

    cleaned = write_cleaned(lines, HANDS_CLEAN)
    changed = sum(1 for a, b in zip(lines, cleaned) if a != b)
    print(f"Modified {changed} / {len(lines)} lines -> {HANDS_CLEAN}")
    print("Example:", cleaned[0][:160], "...")


if __name__ == "__main__":
    main()

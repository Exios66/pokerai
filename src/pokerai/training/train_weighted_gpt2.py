"""GPT-2 with inverse-frequency action-type class weights (Family F3)."""

from __future__ import annotations

from pathlib import Path

from pokerai.config import ARTIFACTS_DIR, GPT2Hyperparams, HANDS_CLEAN
from pokerai.training.train_gpt2 import main as train_gpt2_main

MODEL_WEIGHTED_DIR = ARTIFACTS_DIR / "models" / "gpt2_weighted"


def main(
    hands_path: Path = HANDS_CLEAN,
    output_dir: Path = MODEL_WEIGHTED_DIR,
    hp: GPT2Hyperparams | None = None,
) -> None:
    train_gpt2_main(
        hands_path=hands_path,
        output_dir=output_dir,
        hp=hp,
        class_weight=True,
        run_action_eval=True,
    )


if __name__ == "__main__":
    main()

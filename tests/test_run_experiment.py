"""Tests for the experiment launcher W&B gating."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_experiment.py"


def test_require_wandb_fails_when_disabled():
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env.pop("WANDB_API_KEY", None)
    proc = subprocess.run(
        [sys.executable, str(LAUNCHER), "bigram", "--require-wandb"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "not configured" in combined or "--require-wandb" in combined


def test_launcher_help_mentions_optional_wandb():
    proc = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--require-wandb" in proc.stdout
    assert "optional" in proc.stdout.lower()
    for trainer in ("majority", "features", "weighted-gpt2", "gpt2", "bigram", "trl"):
        assert trainer in proc.stdout

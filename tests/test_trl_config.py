"""Regression tests for TRL training-arg safety on CPU."""

from __future__ import annotations

import torch
from trl import SFTConfig

from pokerai.training import wandb_report_to


def test_sft_config_cpu_safe(monkeypatch):
    """TRL sets bf16=not fp16 when bf16 is None; we must set bf16=False on CPU."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    use_cuda = torch.cuda.is_available()
    cfg = SFTConfig(
        output_dir="/tmp/pokerai-trl-test",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=use_cuda,
        bf16=False,
        use_cpu=not use_cuda,
        report_to=wandb_report_to(),
    )
    assert cfg.bf16 is False
    if not use_cuda:
        assert cfg.fp16 is False
    assert cfg.dataset_kwargs["skip_prepare_dataset"] is True

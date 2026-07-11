"""Train GPT-2 via Hugging Face TRL SFTTrainer (completion-only loss)."""

from __future__ import annotations

from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer

from pokerai.config import (
    GPT2Hyperparams,
    HANDS_CLEAN,
    MODEL_TRL_DIR,
    WANDB_PROJECT,
)
from pokerai.data import load_text_split, split_prompt_completion
from pokerai.models import build_gpt2
from pokerai.training import get_device, load_tokenizer


def _map_prompt_completion(example: dict) -> dict:
    prompt, completion = split_prompt_completion(example["text"])
    return {"prompt": prompt, "completion": completion}


def main(
    hands_path: Path = HANDS_CLEAN,
    output_dir: Path = MODEL_TRL_DIR,
    hp: GPT2Hyperparams | None = None,
) -> None:
    hp = hp or GPT2Hyperparams()
    tokenizer = load_tokenizer()
    model = build_gpt2(tokenizer, hp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    print("Device preference:", get_device())

    split = load_text_split(hands_path)
    train_dataset = split["train"].map(_map_prompt_completion, remove_columns=["text"])
    eval_dataset = split["test"].map(_map_prompt_completion, remove_columns=["text"])
    print(train_dataset[0])

    # Let HF Trainer own W&B init via report_to — avoid a second wandb.init().
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=hp.num_epochs,
        per_device_train_batch_size=hp.batch_size,
        per_device_eval_batch_size=hp.batch_size,
        eval_strategy="epoch",
        logging_steps=50,
        learning_rate=hp.learning_rate,
        max_length=hp.n_positions,  # must match model n_positions
        completion_only_loss=True,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        run_name="gpt2-trl",
    )

    # Seed W&B project name before Trainer creates the run.
    import os

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()

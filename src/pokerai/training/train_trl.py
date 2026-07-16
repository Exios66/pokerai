"""Train GPT-2 via Hugging Face TRL SFTTrainer (completion-only loss)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer

from pokerai.config import (
    GPT2Hyperparams,
    HANDS_CLEAN,
    MODEL_TRL_DIR,
)
from pokerai.data import encode_with_mask, load_text_split
from pokerai.models import build_gpt2
from pokerai.training import (
    ensure_wandb_project,
    get_device,
    load_tokenizer,
    wandb_report_to,
)


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
    device = get_device()
    print("Device preference:", device)

    # Pre-tokenize with the same encode_with_mask path as the raw GPT-2 loop:
    # BOS/EOS, offset-based action mask, keep-end truncation. TRL's built-in
    # truncation_mode="keep_end" is deprecated and defaults to keep_start,
    # which can drop the action tokens entirely on long hands.
    def _tokenize(example: dict) -> dict:
        ids, labels = encode_with_mask(example["text"], tokenizer, hp.n_positions)
        return {"input_ids": ids, "labels": labels}

    split = load_text_split(hands_path)
    train_dataset = split["train"].map(_tokenize, remove_columns=["text"])
    eval_dataset = split["test"].map(_tokenize, remove_columns=["text"])
    print("Example 0 input_ids length:", len(train_dataset[0]["input_ids"]))
    print(
        "Example 0 supervised tokens:",
        sum(1 for t in train_dataset[0]["labels"] if t != -100),
    )

    use_cuda = torch.cuda.is_available()
    # TRL defaults bf16 = not fp16 when bf16 is None, which crashes on CPU.
    # Let HF Trainer own W&B init via report_to — avoid a second wandb.init().
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=hp.num_epochs,
        per_device_train_batch_size=hp.batch_size,
        per_device_eval_batch_size=hp.batch_size,
        eval_strategy="epoch",
        logging_steps=50,
        learning_rate=hp.learning_rate,
        # Dataset already truncated + masked; skip TRL's prepare/tokenize.
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=use_cuda,
        bf16=False,
        use_cpu=not use_cuda,
        report_to=wandb_report_to(),
        run_name=os.environ.get("WANDB_NAME", "gpt2-trl"),
    )

    ensure_wandb_project()

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

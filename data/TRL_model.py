import torch
import wandb
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
from trl import SFTConfig, SFTTrainer

tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")

# Same random-init model as the raw loop -- from_config, not from_pretrained.
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=384,  # HF Poker_Dataset hands can exceed the old 192 limit
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")

# Reshape into prompt/completion pairs at the last comma -- TRL masks
# the prompt and computes loss on the completion by default.
def split_prompt_completion(example):
    text = example["text"]
    cut = text.rfind(",") + 1
    return {"prompt": text[:cut], "completion": text[cut:]}

raw = load_dataset("text", data_files={"train": "data/hands_fold_update.txt"})["train"]
raw = raw.train_test_split(test_size=0.05, seed=42)
train_dataset = raw["train"].map(split_prompt_completion, remove_columns=["text"])
eval_dataset = raw["test"].map(split_prompt_completion, remove_columns=["text"])
print(train_dataset[0])

training_args = SFTConfig(
    output_dir="model_out_trl",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    logging_steps=50,
    learning_rate=3e-4,
    max_length=192,
    completion_only_loss=True,   # default for prompt/completion data, explicit here for clarity
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    run_name="gpt2-trl",
)

wandb.init(
    project="pokerai",
    name="gpt2-trl",
    config={
        "model": "gpt2",
        "n_params": n_params,
        "n_positions": config.n_positions,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "vocab_size": config.vocab_size,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "max_length": training_args.max_length,
        "completion_only_loss": training_args.completion_only_loss,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
    },
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("model_out_trl")
tokenizer.save_pretrained("model_out_trl")
wandb.finish()
print("Saved to model_out_trl/")

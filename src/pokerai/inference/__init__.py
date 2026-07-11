"""Generate an action given a serialized poker game state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from pokerai.config import MODEL_GPT2_DIR, MODEL_TRL_DIR, N_POSITIONS
from pokerai.training import get_device


def load_model(
    model_dir: Path,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No model at {model_dir}. Train first with scripts/train_gpt2.py "
            "or scripts/train_trl.py"
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    return model, tokenizer


def predict_action(
    state: str,
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    max_new_tokens: int = 16,
) -> str:
    """Predict the hero action for a game-state string (no trailing action)."""
    prompt = state if state.endswith(",") else state + ","
    device = get_device()
    model.to(device)
    model.eval()

    bos = tokenizer.bos_token_id
    input_ids = [bos] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(input_ids) > N_POSITIONS - max_new_tokens:
        input_ids = input_ids[-(N_POSITIONS - max_new_tokens) :]

    tensor = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    generated = out[0, len(input_ids) :].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a poker action from game state")
    parser.add_argument(
        "state",
        nargs="?",
        help="Serialized game state (prompt). If omitted, reads stdin.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_TRL_DIR if MODEL_TRL_DIR.exists() else MODEL_GPT2_DIR,
        help="Directory with saved model + tokenizer",
    )
    args = parser.parse_args()
    state = args.state or input("Game state: ").strip()
    model, tokenizer = load_model(args.model)
    action = predict_action(state, model, tokenizer)
    print(action)


if __name__ == "__main__":
    main()

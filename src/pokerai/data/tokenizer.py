"""Train a domain BPE tokenizer on cleaned poker hand histories."""

from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from pokerai.config import (
    BOS_TOKEN,
    EOS_TOKEN,
    HANDS_CLEAN,
    HANDS_RAW,
    N_POSITIONS,
    PAD_TOKEN,
    TOKENIZER_DIR,
    VOCAB_SIZE,
)


def train_tokenizer(
    hands_path: Path | None = None,
    output_dir: Path = TOKENIZER_DIR,
    vocab_size: int = VOCAB_SIZE,
) -> PreTrainedTokenizerFast:
    # Prefer cleaned hands so merges match what models actually train on.
    if hands_path is None:
        if HANDS_CLEAN.exists():
            hands_path = HANDS_CLEAN
        elif HANDS_RAW.exists():
            hands_path = HANDS_RAW
        else:
            raise FileNotFoundError(
                f"Missing {HANDS_CLEAN} (and {HANDS_RAW}). "
                "Run: python scripts/prepare_data.py"
            )

    if not hands_path.exists():
        raise FileNotFoundError(
            f"Missing {hands_path}. Run: python scripts/prepare_data.py"
        )

    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[str(hands_path)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = output_dir / "tokenizer.json"
    bpe.save(str(tokenizer_json))

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
    hf_tokenizer.add_special_tokens(
        {
            "pad_token": PAD_TOKEN,
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
        }
    )
    hf_tokenizer.model_max_length = N_POSITIONS
    hf_tokenizer.save_pretrained(str(output_dir))
    return hf_tokenizer


def main() -> None:
    hands_path = HANDS_CLEAN if HANDS_CLEAN.exists() else HANDS_RAW
    tokenizer = train_tokenizer(hands_path=hands_path)
    lines = hands_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{hands_path} has no lines to inspect")
    sample = lines[0]
    ids = tokenizer(sample)["input_ids"]
    print(f"Tokenizer corpus: {hands_path}")
    print("Sample hand:", sample[:120], "...")
    print("Token count (sample):", len(ids))

    lengths = [len(tokenizer(line)["input_ids"]) for line in lines]
    print("Max tokens in a hand:", max(lengths))
    print("Avg tokens per hand:", sum(lengths) / len(lengths))
    print(f"Saved tokenizer to {TOKENIZER_DIR}")
    print(f"Vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()

# Poker AI

Train a small GPT-2 decoder to predict poker actions (FOLD, CALL, RAISE, …) from a serialized game state. Loss is applied only to the action tokens (completion-only / masked training).

## Setup

```bash
git clone https://github.com/Exios66/pokerai.git
cd pokerai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt && pip install -e .
wandb login                        # optional; or WANDB_MODE=disabled
```

## Pipeline

Run from the repo root, in order:

| Step | Command | Output |
|------|---------|--------|
| 1. Fetch & clean data | `python scripts/prepare_data.py` | `data/hands.txt`, `data/hands_clean.txt` |
| 2. Train tokenizer | `python scripts/train_tokenizer.py` | `artifacts/tokenizer/` |
| 3. Bigram baseline | `python scripts/train_bigram.py` | `artifacts/models/bigram/` |
| 4a. GPT-2 (raw loop) | `python scripts/train_gpt2.py` | `artifacts/models/gpt2/` |
| 4b. GPT-2 (TRL) | `python scripts/train_trl.py` | `artifacts/models/gpt2_trl/` |
| 5. Predict | `python scripts/predict.py "<state>"` | printed action |

After `pip install -e .` you can also use `pokerai-prepare`, `pokerai-gpt2`, `pokerai-predict`, etc.

## Data

Hands are downloaded from Hugging Face [`SoelMgd/Poker_Dataset`](https://huggingface.co/datasets/SoelMgd/Poker_Dataset). Each line is `context,action` — split at the **last comma**.

Example:

```
[TABLE_CONFIGURATION] BTN=P3 SB=P1 0.5BB BB=P2 1BB [STACKS] P1: 44.2BB [Qh 9h] P2: 103.4BB P3: 165.2BB POT=1.5BB [PREFLOP] P3: RAISE 2BB P1:,FOLD
```

Cleaning normalizes redundant suffixes (`CALL 0BB` → `CALL`, legacy `FOLD0BB` → `FOLD`).

## Model

| Setting | Value |
|---------|-------|
| Architecture | GPT-2 (random init) |
| Layers / hidden / heads | 6 / 256 / 8 |
| Context length | **384** tokens |
| Vocab | Domain BPE (target size 4000; actual size depends on data) |
| Special tokens | `<\|startoftext\|>` (BOS), `<\|endoftext\|>` (EOS), `<\|pad\|>` |
| Optimizer | AdamW, lr `3e-4`, 3 epochs, batch 32 |

Training masks the game-state prefix so the model only learns to predict the hero action.

## Layout

```
pokerai/
├── src/pokerai/           # installable package
│   ├── config.py          # paths + hyperparameters
│   ├── data/              # fetch, clean, tokenize, encode/mask
│   ├── models/            # Bigram + GPT-2 factory
│   ├── training/          # bigram / gpt2 / trl trainers
│   └── inference/         # action prediction
├── scripts/               # thin CLI wrappers
├── tests/
├── data/                  # hand text only (gitignored)
└── artifacts/             # tokenizer + models (gitignored)
```

## Tests

```bash
pytest
```

## Experiment tracking

Training logs to the W&B project `pokerai` (`bigram-baseline`, `gpt2-raw`, `gpt2-trl`). Set `WANDB_MODE=disabled` to skip.

## Notes

- GPU (CUDA) is recommended; scripts also use MPS on Apple Silicon when available.
- Regenerate data, tokenizer, and models with the scripts above — they are not committed.
- Prefer `train_trl.py` for the HF Trainer path; `train_gpt2.py` is the equivalent raw loop and now saves checkpoints.

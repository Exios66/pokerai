# Poker AI

Train a small GPT-2 decoder to predict poker actions (FOLD, CALL, RAISE, …) from a serialized game state. Loss is applied only to the action tokens (completion-only / masked training).

## Setup

```bash
git clone https://github.com/Exios66/pokerai.git
cd pokerai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt && pip install -e .
```

Weights & Biases is **optional**. Training runs without it by default. To enable logging:

```bash
wandb login
# or offline: export WANDB_MODE=offline
```

To force-disable: `export WANDB_MODE=disabled`.

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| 1. Fetch & clean data | `python scripts/prepare_data.py` | `data/hands.txt`, `data/hands_clean.txt` |
| 2. Train tokenizer | `python scripts/train_tokenizer.py` | `artifacts/tokenizer/` |
| 3. Bigram baseline | `python scripts/train_bigram.py` | `artifacts/models/bigram/` |
| 4a. GPT-2 (raw loop) | `python scripts/train_gpt2.py` | `artifacts/models/gpt2/` |
| 4b. GPT-2 (TRL) | `python scripts/train_trl.py` | `artifacts/models/gpt2_trl/` |
| 5. Predict | `python scripts/predict.py "<state>"` | printed action |

After `pip install -e .` you can also use `pokerai-prepare`, `pokerai-gpt2`, `pokerai-predict`, etc.

**Optional:** `python scripts/convert_poker_dataset.py` also downloads the HF dataset, writes the same `hands.txt` / `hands_clean.txt`, and additionally preserves the official HF train/test split as `data/hands_train.txt` and `data/hands_test.txt`. Trainers still load `hands_clean.txt` by default (random 95/5 split).

## Data

Hands are downloaded from Hugging Face [`SoelMgd/Poker_Dataset`](https://huggingface.co/datasets/SoelMgd/Poker_Dataset). Each line is `context,action` — split at the **last comma**.

Example:

```
[TABLE_CONFIGURATION] BTN=P3 SB=P1 0.5BB BB=P2 1BB [STACKS] P1: 44.2BB [Qh 9h] P2: 103.4BB P3: 165.2BB POT=1.5BB [PREFLOP] P3: RAISE 2BB P1:,FOLD
```

Cleaning normalizes redundant suffixes (`CALL 0BB` → `CALL`, legacy `FOLD0BB` → `FOLD`).

The tokenizer is trained on **cleaned** hands (`data/hands_clean.txt`) so vocabulary matches what models see at train time.

## Model

| Setting | Value |
|---------|-------|
| Architecture | GPT-2 (random init) |
| Layers / hidden / heads | 6 / 256 / 8 |
| Context length | **384** tokens |
| Vocab | Domain BPE (target size 4000; actual size depends on data) |
| Special tokens | `<\|startoftext\|>` (BOS), `<\|endoftext\|>` (EOS), `<\|pad\|>` |
| Optimizer | AdamW, lr `3e-4`, 3 epochs, batch 32 |

The model uses masked language modeling:

- **Input**: Full hand history (game state + action), with BOS/EOS
- **Masking**: All tokens up to the decision separator (the game state) are masked (`-100`)
- **Prediction**: Model learns to predict only the action tokens (+ EOS)
- Long sequences **keep the end** (action), dropping tokens from the front

**Implementation detail:** the prompt/answer boundary is located by character offset in the full tokenized sequence (via `return_offsets_mapping`), not by tokenizing the prompt separately and re-joining. BPE merge behavior can differ depending on what follows a substring, so tokenizing the prompt in isolation can silently misalign the mask.

`<|startoftext|>` and `<|endoftext|>` must be distinct tokens. If BOS and EOS are the same token, the model can't tell "hand starting" from "hand ending."

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

## Tracking runs with W&B

When configured, all training scripts log to
[`mooslin-university-of-wisconsin-madison/poker-ai`](https://wandb.ai/mooslin-university-of-wisconsin-madison/poker-ai)
so runs are comparable. TRL uses `TrainingArguments(report_to=...)`; the raw-loop scripts call `wandb.init` / `wandb.log` only when W&B is available.

On offline compute nodes:

```bash
export WANDB_MODE=offline
python scripts/train_gpt2.py
wandb sync wandb/offline-run-*
```

## Experiment tracking

- **Bigram baseline**: Validation loss ~1.53
- **GPT-2 model**: Expected to significantly outperform baseline
- Compare final validation loss against baseline to assess improvement
- Check the action-type distribution before over-interpreting loss — poker decision data is typically FOLD-heavy, so raw loss alone can hide poor performance on rarer actions like RAISE

**Full experiment catalog** (configs, what each run showcases, how to run it, and what to look for on W&B): see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Launch tagged runs with `python scripts/run_experiment.py …` (add `--require-wandb` for catalog/sweep runs that must log). Capacity / model-compare sweeps: [`docs/wandb_sweep_gpt2_capacity.yaml`](docs/wandb_sweep_gpt2_capacity.yaml), [`docs/wandb_sweep_model_compare.yaml`](docs/wandb_sweep_model_compare.yaml).

## Notes

- Large files (data, models, tokenizer) are excluded from git via `.gitignore` and can be regenerated from the scripts above
- Model training is much faster with a CUDA GPU; TRL and the raw loop also run on CPU
- Adjust batch size and learning rate based on your hardware
- The bigram baseline is a sanity check — if GPT-2 does not clearly beat it, something upstream (tokenizer, data format, masking) is wrong

## Known limitations

- Class imbalance across action types is not yet handled (e.g. no weighted loss)
- No checkpoint resumption during training
- Condensed shorthand notation for the POC dataset is not yet implemented — converters pass the (flattened) raw text through

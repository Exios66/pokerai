# Poker AI

Train a small GPT-2 decoder to predict poker actions (FOLD, CALL, RAISE, …) from a serialized game state. Loss is applied only to the action tokens (completion-only / masked training).

## Live site (Posit Connect Cloud)

**Public URL:** https://019f9a68-2304-5291-83c1-e2b9574e723d.share.connect.posit.cloud/

The Quarto documentation site covers the full pipeline (setup, data, tokenizer, models, training, inference, architecture, API, W&B tracking, and limitations). To re-render and republish:

```bash
quarto render
python scripts/publish_posit_pokerai.py --skip-render --content-id 019f9a68-2304-5291-83c1-e2b9574e723d
```

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pokerAI.git
   cd pokerAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   wandb login   # optional but recommended, see Tracking Runs below
   ```

4. **Prepare your data**

   Two options:
   - **Use your own hand histories**: place them in `data/hands.txt`, one hand per line, in the format `POSITION,STACK,CARDS,PLAYER_INFO,ACTION` (see Data Format below).
   - **Use the public proof-of-concept dataset**: run the converter below to pull and format `SoelMgd/Poker_Dataset` from HuggingFace automatically.
   ```bash
   python data/convert_poker_dataset.py
   ```
   This writes `data/hands.txt` (combined), `data/hands_train.txt`, and `data/hands_test.txt`, using the dataset's existing train/test split rather than creating a new one. Prints the action-type distribution (FOLD/CALL/RAISE/etc.) so you can check class imbalance up front.

## Training Pipeline

### 1. Data Preparation (`data/prepare_data.py`)
Cleans the raw poker data and creates train/test splits:
- Removes redundant "0BB" from FOLD actions
- Splits data into 95% training / 5% validation
- Saves cleaned data to `data/hands_fold_update.txt`

Skip this step if you used `convert_poker_dataset.py` above — that script already produces cleaned, pre-split files (`hands_train.txt` / `hands_test.txt`).

```bash
git clone https://github.com/Exios66/pokerai.git
cd pokerai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt && pip install -e .
wandb login                        # optional; or WANDB_MODE=disabled
```

### 2. Tokenizer Training (`data/train_tokenizer.py`)
Trains a custom BPE tokenizer on the poker domain:
- Vocabulary size: 4000 tokens (+special tokens)
- Special tokens: `<|startoftext|>`, `<|endoftext|>`, `<|pad|>`
- Saves tokenizer to `tokenizer/` directory
- Prints token-length distribution — use the max to set `n_positions` in step 4

```bash
python data/train_tokenizer.py
```

**Note:** `<|startoftext|>` and `<|endoftext|>` must be distinct tokens. If BOS and EOS are the same token, the model can't tell "hand starting" from "hand ending," which corrupts sequence-boundary learning. If you ever add or rename special tokens, retrain the tokenizer and rerun every downstream step — token IDs shift.

### 3. Baseline Model (`data/train_bigram.py`)
Trains a simple bigram model for comparison:
- Provides a baseline for model performance
- Expected validation loss: ~1.53
- **Not the real model** — if later steps don't clearly beat this, something upstream (tokenizer, data format, masking) is broken

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

The model uses masked language modeling:
- **Input**: Full hand history (game state + action)
- **Masking**: All tokens up to the decision separator (the game state) are masked
- **Prediction**: Model learns to predict only the action tokens
- This ensures the model focuses on decision-making, not memorizing game states

**Implementation detail:** the prompt/answer boundary is located by character offset in the full tokenized sequence (via `return_offsets_mapping`), not by tokenizing the prompt separately and re-joining. BPE merge behavior can differ depending on what follows a substring, so tokenizing the prompt in isolation can silently misalign the mask. Always locate the boundary from a single full-text tokenization pass.

## Data Format

If using your own hand histories, each line should follow this format:
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

Example:
```
BTN,1.5BB,9d 10c,P1:101.0BB/0.0BB,P2:100.23BB/0.0BB,FOLD
```

If using `convert_poker_dataset.py` (the `SoelMgd/Poker_Dataset` proof-of-concept data), lines instead follow:
```
<table state question> => <action answer>
```
This uses a different separator (`" => "` instead of comma-delimited fields) — if you switch between the two data sources, make sure the separator used in the masking logic (`train_gpt2.py` / `build_model.py`) matches whichever format produced your `hands_train.txt`.

## File Structure

```
pokerAI/
├── data/
│   ├── convert_poker_dataset.py  # downloads + converts SoelMgd/Poker_Dataset (POC data source)
│   ├── prepare_data.py           # Data cleaning and splitting (custom hand histories)
│   ├── train_tokenizer.py        # Custom tokenizer training
│   ├── train_bigram.py           # Baseline model training
│   ├── build_model.py            # GPT-2 training (raw loop)
│   ├── TRL_model.py              # GPT-2 training (TRL)
│   ├── hands.txt                 # Raw/combined input data (not in repo)
│   └── hands_fold_update.txt     # Cleaned data (not in repo)
├── tokenizer/                     # Trained tokenizer (not in repo)
├── model_out_trl/                # Trained model (not in repo)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Tracking Runs with W&B

All training scripts (bigram, raw-loop GPT-2, TRL) log to the same `pokerai` W&B project so runs are directly comparable. TRL picks this up automatically via `TrainingArguments(report_to="wandb")`; the raw-loop scripts call `wandb.init`/`wandb.log` directly.

On CHTC or other offline compute nodes:
```bash
export WANDB_MODE=offline
python data/build_model.py
wandb sync wandb/offline-run-*
```

## Experiment tracking

- **Bigram baseline**: Validation loss ~1.53
- **GPT-2 model**: Expected to significantly outperform baseline
- Compare final validation loss against baseline to assess improvement
- Check the action-type distribution (printed by `convert_poker_dataset.py`) before over-interpreting loss — poker decision data is typically FOLD-heavy, so raw loss alone can hide poor performance on rarer actions like RAISE

## Notes

- Large files (data, models, tokenizer) are excluded from git via .gitignore
- These can be regenerated by running the training scripts
- Model training requires CUDA GPU for reasonable speed
- Adjust batch size and learning rate based on your hardware

## Known Limitations / Open Items

- Class imbalance across action types is not yet handled (e.g. no weighted loss)
- No checkpoint resumption during training
- Condensed shorthand notation for the POC dataset (mapping verbose HF text into a token-efficient format) is not yet implemented — `convert_poker_dataset.py` currently passes the raw text through

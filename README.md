# Poker AI Training Pipeline

A machine learning pipeline for training a GPT-2 model to predict poker actions from game states. The project uses transformer-based language modeling to learn poker decision-making from hand history data.

## Project Overview

This project trains a neural network to predict poker actions (FOLD, CALL, RAISE) given the current game state. The model learns from real poker hand histories using a masked language modeling approach where the context (game state) is provided and the model must predict the action.

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
python data/prepare_data.py
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

```bash
python data/train_bigram.py
```

### 4. GPT-2 Model Training

**Option A: Raw PyTorch Loop (`data/build_model.py`)**
- GPT-2 architecture (6 layers, 256 hidden dimensions, 8 heads)
- Context length: 192 tokens
- Masked training: only predicts the action, not the context
- 3 epochs with AdamW optimizer (lr=3e-4)

```bash
python data/build_model.py
```

**Option B: TRL Trainer (`data/TRL_model.py`)**
- Uses Hugging Face TRL library for supervised fine-tuning
- Same model architecture as raw loop
- Prompt/completion format for masked training
- Saves model to `model_out_trl/`

```bash
python data/TRL_model.py
```

## Model Architecture

- **Model Type**: GPT-2 (decoder-only transformer)
- **Parameters**: ~2.6M
- **Layers**: 6
- **Hidden Size**: 256
- **Attention Heads**: 8
- **Context Window**: 192 tokens
- **Vocabulary Size**: 4000 (domain-specific)

## Training Strategy

The model uses masked language modeling:
- **Input**: Full hand history (game state + action)
- **Masking**: All tokens up to the decision separator (the game state) are masked
- **Prediction**: Model learns to predict only the action tokens
- This ensures the model focuses on decision-making, not memorizing game states

**Implementation detail:** the prompt/answer boundary is located by character offset in the full tokenized sequence (via `return_offsets_mapping`), not by tokenizing the prompt separately and re-joining. BPE merge behavior can differ depending on what follows a substring, so tokenizing the prompt in isolation can silently misalign the mask. Always locate the boundary from a single full-text tokenization pass.

## Data Format

If using your own hand histories, each line should follow this format:
```
POSITION,STACK,CARDS,PLAYER1_INFO,PLAYER2_INFO,...,ACTION
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

## Performance

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

# W&B Experiment Catalog — Poker AI

This document is a complete catalog of experiments you can run against this repository’s models and configurations. When Weights & Biases is configured, runs log to the shared project **[`mooslin-university-of-wisconsin-madison/poker-ai`](https://wandb.ai/mooslin-university-of-wisconsin-madison/poker-ai)**. Use it to compare baselines, ablations, and training backends with a consistent evaluation story.

**W&B is optional.** Trainers only call `wandb.init` / set `report_to="wandb"` when an API key is present, `WANDB_MODE` is `offline`/`online`/`shared`, or you are already logged in. Otherwise training proceeds with **no** W&B logging (`report_to="none"`). Catalog and sweep workflows that need comparable panels should enable W&B explicitly (see below) or pass `--require-wandb` to the launcher.

| Script | Run name (default) | Key metrics (when W&B is on) |
|--------|--------------------|------------------------------|
| `scripts/train_bigram.py` | `bigram-baseline` (or `WANDB_NAME`) | `train/loss`, `val/loss`, `final_val_loss`, `random_baseline_loss` |
| `scripts/train_gpt2.py` | `gpt2-raw` (or `WANDB_NAME`) | `train/loss`, `val/loss`, `final_val_loss` |
| `scripts/train_trl.py` | `gpt2-trl` (or `WANDB_NAME`) | HF Trainer / TRL metrics (`loss`, `eval_loss`, …) via `report_to="wandb"` when configured |

---

## Shared setup (run once before any experiment)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/prepare_data.py       # -> data/hands.txt, data/hands_clean.txt
python scripts/train_tokenizer.py    # -> artifacts/tokenizer/
```

**Enable W&B for catalog / dashboard runs** (pick one):

```bash
wandb login                           # online logging
# or:
export WANDB_MODE=offline             # log locally, sync later
# force off (no logging, no prompts):
export WANDB_MODE=disabled
```

**Offline compute (e.g. CHTC):**

```bash
export WANDB_MODE=offline
# … run training …
wandb sync wandb/offline-run-*
```

**W&B hygiene (recommended whenever you want comparable panels):**

```bash
export WANDB_ENTITY=mooslin-university-of-wisconsin-madison
export WANDB_PROJECT=poker-ai
export WANDB_RUN_GROUP="<experiment-family>"   # e.g. model-comparison
export WANDB_TAGS="exp01,gpt2,baseline"
# optionally: export WANDB_NAME="gpt2-raw-lr3e4"
```

The launcher prints whether W&B is active. For sweeps or must-log experiments, fail loudly if it is not:

```bash
python scripts/run_experiment.py gpt2 --require-wandb --group capacity --tags exp-b1
```

---

## What “good” looks like (shared evaluation rubric)

Primary metric across runs: **`final_val_loss`** (or TRL’s epoch `eval_loss`) on the **action-only** (completion-masked) objective. Lower is better.

| Reference | Expected ballpark | Notes |
|-----------|-------------------|-------|
| Uniform random over vocab | `ln(V) ≈ ln(4000) ≈ 8.3` | Logged by bigram as `random_baseline_loss` |
| Bigram (default) | **~1.53** val loss | Repo baseline; if GPT-2 does not beat this, upstream is broken |
| Default GPT-2 (6L/256/8H, 3 epochs) | **Clearly below bigram**, typically well under ~1.3 if training is healthy | Exact number depends on data size / split |
| GPT-2 raw vs TRL | Should be **roughly comparable** at matched hparams | Large gaps imply masking or length mismatch |

Always also check:

1. **Train vs val gap** — large gap ⇒ overfitting or noisy small eval set.
2. **Action-type distribution** — data is typically FOLD-heavy; raw CE loss can look good while RAISE/BET sizing is weak.
3. **Smoke predictions** — `python scripts/predict.py "<state>," --model artifacts/models/...` should emit plausible actions (`FOLD`, `CALL`, `RAISE …BB`, …), not garbage tokens.
4. **Truncation** — after tokenizer training, if max hand length ≫ `n_positions` (384), long contexts are left-truncated; architecture/context experiments matter more in that regime.

---

## How to override hyperparameters

Use the experiment launcher (preferred):

```bash
python scripts/run_experiment.py bigram --group model-comparison --tags exp-a1,baseline
python scripts/run_experiment.py gpt2 --n-layer 4 --lr 1e-4 --name gpt2-depth4 --group depth-sweep --tags exp-b1
python scripts/run_experiment.py trl --epochs 5 --batch-size 16 --group optim --tags exp-c3
python scripts/run_experiment.py majority --group imbalance --tags exp-f2
python scripts/run_experiment.py features --method rf --group alt-approaches --tags exp-i2
python scripts/run_experiment.py weighted-gpt2 --group imbalance --tags exp-f3
python scripts/evaluate.py --model artifacts/models/gpt2 --max-examples 256
```

Or construct `GPT2Hyperparams` in Python / edit `src/pokerai/config.py`. For tokenizer vocab ablations, retrain with:

```python
from pokerai.data.tokenizer import train_tokenizer
train_tokenizer(vocab_size=2000)  # then re-run model training
```

Tag every W&B run with the **exact** config so panels stay comparable. `--name` / `--group` / `--tags` set `WANDB_NAME`, `WANDB_RUN_GROUP`, and `WANDB_TAGS`. Without W&B configured, those flags are still accepted but nothing is logged unless you pass `--require-wandb` (which exits with an error).

---

# Experiment families

## Family A — Model comparison (must-run showcase)

These runs are the headline W&B dashboard: same data, same tokenizer, same eval definition.

### EXP-A1 — Bigram action-only baseline

| | |
|--|--|
| **Config** | `BIGRAM_LR=1e-2`, `BIGRAM_BATCH_SIZE=4096`, `BIGRAM_STEPS=2000`; action-token pairs only (`masked_action_only=True`) |
| **Command** | `python scripts/run_experiment.py bigram --group model-comparison --name bigram-baseline --tags exp-a1` |
| **Showcases** | Strong statistical baseline that only models P(next action token \| previous token). Fair comparison vs GPT-2 because both use action-region supervision. |
| **Look for** | `final_val_loss ≈ 1.53`; clearly below `random_baseline_loss` (~8.3). Flat late-curve ⇒ converged. |

### EXP-A2 — GPT-2 raw loop (default architecture)

| | |
|--|--|
| **Config** | `n_positions=384`, `n_embd=256`, `n_layer=6`, `n_head=8`, `lr=3e-4`, `batch=32`, `epochs=3`, AdamW; action-only CE via `encode_with_mask` |
| **Command** | `python scripts/run_experiment.py gpt2 --group model-comparison --name gpt2-raw --tags exp-a2` |
| **Showcases** | Whether a small randomly-initialized decoder beats bigram when it can condition on full game state. |
| **Look for** | `final_val_loss` **≪** bigram (~1.53). Smooth `train/loss` decline; end-of-epoch `val/loss` improving then stabilizing. If val ≥ bigram, inspect tokenizer, comma split, or masking. |

### EXP-A3 — GPT-2 via TRL SFT (`completion_only_loss=True`)

| | |
|--|--|
| **Config** | Same `GPT2Hyperparams` as A2; `SFTConfig(completion_only_loss=True, max_length=n_positions, eval_strategy="epoch", fp16=cuda)` |
| **Command** | `python scripts/run_experiment.py trl --group model-comparison --name gpt2-trl --tags exp-a3` |
| **Showcases** | Equivalence (or gap) between the hand-rolled loop and HF/TRL’s completion-only path — critical for trusting production training. |
| **Look for** | `eval_loss` within a small band of A2’s `final_val_loss` (same seed/data). Materially worse TRL ⇒ prompt/completion split or `max_length` mismatch. Materially better + fp16 ⇒ note precision/speed tradeoff. |

### EXP-A4 — Side-by-side W&B panel (A1–A3)

| | |
|--|--|
| **Config** | Identical data + tokenizer; group `WANDB_RUN_GROUP=model-comparison` |
| **How to test** | Run A1→A3; in W&B create a parallel coordinates / line plot of `val/loss` (or `eval_loss`) and a bar of `final_val_loss`. |
| **Showcases** | The repo’s core claim: **GPT-2 > bigram**, and **raw ≈ TRL**. |
| **Look for** | Ordering: random ≫ bigram > GPT-2 (raw ≈ TRL). No GPT-2 run should lose to bigram after 3 epochs on the default stack. |

---

## Family B — Architecture capacity

Hold training recipe fixed (`lr=3e-4`, `batch=32`, `epochs=3`, `n_positions=384`) unless noted. Prefer `train_gpt2.py` or `train_trl.py` consistently within a sweep.

### EXP-B1 — Depth sweep (`n_layer`)

| | |
|--|--|
| **Configs** | `n_layer ∈ {2, 4, 6 (default), 8, 12}` with `n_embd=256`, `n_head=8` |
| **How to test** | `python scripts/run_experiment.py gpt2 --n-layer L --name gpt2-depth-L --group depth-sweep --tags exp-b1,depth-L` for each L. |
| **Showcases** | Returns to depth for long-horizon table state → action mapping. |
| **Look for** | Val loss decreasing then plateauing; diminishing returns past 6–8 layers. Rising val + falling train ⇒ overfit / too much capacity for data size. |

### EXP-B2 — Width sweep (`n_embd`)

| | |
|--|--|
| **Configs** | `n_embd ∈ {128, 256 (default), 384, 512}`; keep `n_head` dividing `n_embd` (e.g. 4/8/8/8 or 8/8/12/16) |
| **How to test** | Pair each width with a valid `n_head`; log `n_params` (already in raw GPT-2 W&B config). |
| **Showcases** | Embedding/capacity vs compute; useful for choosing a deployable size. |
| **Look for** | Pareto curve: val loss vs parameter count. 128d should underfit relative to 256d; 512d should help only if data is rich enough. |

### EXP-B3 — Attention head count (`n_head`)

| | |
|--|--|
| **Configs** | Fixed `n_embd=256`, `n_head ∈ {4, 8 (default), 16}` (`n_embd % n_head == 0`) |
| **How to test** | Three runs; same seed/data. |
| **Showcases** | Whether multi-head diversity matters for poker state features (position, stacks, board texture proxies in text). |
| **Look for** | Usually mild effect vs depth/width. Large degradation at extreme head sizes ⇒ unstable training or too-thin heads. |

### EXP-B4 — Context length (`n_positions`)

| | |
|--|--|
| **Configs** | `n_positions ∈ {128, 256, 384 (default), 512}`; must match tokenizer `model_max_length` and TRL `max_length` |
| **How to test** | After choosing length L, set `GPT2Hyperparams(n_positions=L)` and ensure encode/TRL max length = L. Compare fraction of truncated hands (from tokenizer length stats). |
| **Showcases** | Cost of left-truncation on long multi-street histories vs memory/speed. |
| **Look for** | If many hands exceed 128–256 tokens, raising context should **materially** cut val loss. If almost all hands fit in 256, 384→512 yields little gain. |

### EXP-B5 — Tiny vs default vs “large-small” GPT-2

| | |
|--|--|
| **Configs** | **Tiny:** 2L / 128d / 4H; **Default:** 6L / 256d / 8H; **Large-small:** 8L / 512d / 8H |
| **How to test** | Three named runs; optionally match wall-clock by adjusting epochs. |
| **Showcases** | Compact “poster” comparison of capacity classes for demos. |
| **Look for** | Clear ordering Tiny > Default ≥ Large-small on val loss (loss numbers: Tiny highest). Large-small only wins if val improves without diverging. |

---

## Family C — Optimization & schedule

Default architecture unless noted.

### EXP-C1 — Learning-rate sweep

| | |
|--|--|
| **Configs** | `lr ∈ {1e-4, 3e-4 (default), 1e-3, 3e-3}` |
| **How to test** | Four GPT-2 runs (raw or TRL); identical other hparams. |
| **Showcases** | Stability of AdamW on randomly-initialized GPT-2 for this domain. |
| **Look for** | `3e-4` near-optimal. `1e-4` slower but stable. `≥1e-3` may spike train loss / worse val. Prefer lowest stable `final_val_loss`. |

### EXP-C2 — Batch size sweep

| | |
|--|--|
| **Configs** | `batch_size ∈ {8, 16, 32 (default), 64}` (fit GPU memory; TRL uses `per_device_*_batch_size`) |
| **How to test** | Keep `lr` fixed first; optional follow-up with linear LR scaling. |
| **Showcases** | Gradient noise vs throughput; sensitivity of small poker LM to batch statistics. |
| **Look for** | Very small batches: noisier curves, possibly better generalization. Very large: faster steps but possible val degradation without LR retune. |

### EXP-C3 — Epoch / compute budget

| | |
|--|--|
| **Configs** | `num_epochs ∈ {1, 3 (default), 5, 10}` |
| **How to test** | Log val every epoch; compare best-epoch vs final. |
| **Showcases** | Underfitting vs overfitting horizon on the 95/5 split. |
| **Look for** | Val improving through epoch 3; later epochs: either small gains or rising val (overfit). Report **best** val, not only last. |

### EXP-C4 — Bigram optimization sanity

| | |
|--|--|
| **Configs** | `BIGRAM_STEPS ∈ {500, 2000 (default), 8000}`; `BIGRAM_LR ∈ {1e-3, 1e-2, 5e-2}` |
| **How to test** | `train_bigram.main(n_steps=..., lr=...)` |
| **Showcases** | That ~1.53 is a converged baseline, not an undertrained artifact. |
| **Look for** | Default 2k steps ≈ converged (extra steps ≈ flat). Much worse final val at weird LRs ⇒ don’t trust GPT-2 comparisons until bigram is retuned. |

---

## Family D — Tokenizer & representation

Retrain tokenizer **and** all models when vocab/specials change (token IDs shift).

### EXP-D1 — Vocabulary size

| | |
|--|--|
| **Configs** | `VOCAB_SIZE ∈ {1000, 2000, 4000 (default), 8000}` |
| **How to test** | For each V: `train_tokenizer(vocab_size=V)` → bigram + GPT-2 default. Compare val loss **and** avg tokens/hand. |
| **Showcases** | BPE granularity tradeoff: smaller vocab ⇒ longer sequences / more truncation; larger ⇒ rarer tokens, harder action modeling. |
| **Look for** | Sweet spot near 4k. Collapse in rare RAISE sizing tokens at tiny V; little gain + slower train at 8k. |

### EXP-D2 — Train tokenizer on raw vs cleaned text

| | |
|--|--|
| **Configs** | Tokenizer on `hands.txt` (default) vs `hands_clean.txt` |
| **How to test** | Point `train_tokenizer(hands_path=...)` at each; train identical GPT-2. |
| **Showcases** | Impact of normalizing `CALL 0BB` / `FOLD0BB` on the subword inventory. |
| **Look for** | Cleaned tokenizer should not hurt and may slightly help action-token consistency. Large gap ⇒ noisy special-case strings in vocab. |

### EXP-D3 — Distinct BOS/EOS (regression guard)

| | |
|--|--|
| **Configs** | Correct: distinct `<\|startoftext\|>` / `<\|endoftext\|>` (repo default). Ablation only if you intentionally break this. |
| **How to test** | Unit test `tests/test_config.py::test_special_tokens_are_distinct`; optionally train a broken tokenizer where BOS=EOS and compare generation stopping. |
| **Showcases** | Why the README insists BOS ≠ EOS for boundary learning / `generate` stopping. |
| **Look for** | Broken specials ⇒ worse val and/or runaway or empty generations in `predict.py`. |

---

## Family E — Data & supervision

### EXP-A reminder — always use the same split seed

Default `TEST_SIZE=0.05`, `SPLIT_SEED=42` in `load_text_split`. Changing seed without retagging makes W&B comparisons invalid.

### EXP-E1 — Train-set size scaling

| | |
|--|--|
| **Configs** | Subsample train to `{10%, 25%, 50%, 100%}` of `split["train"]` (fixed test set). |
| **How to test** | Filter the HF `Dataset` before encoding / TRL map; keep eval identical. |
| **Showcases** | Data efficiency of GPT-2 vs bigram; whether more hands still buy loss. |
| **Look for** | Monotone val improvement with more data. GPT-2 gap vs bigram widening with scale. Plateau ⇒ architecture/label noise limited, not data-limited. |

### EXP-E2 — Cleaning on vs off

| | |
|--|--|
| **Configs** | Train on `hands_clean.txt` (default) vs raw `hands.txt` |
| **How to test** | Pass `hands_path=` into `train_*` mains; same tokenizer (note D2 interaction). |
| **Showcases** | Value of the repo’s `CALL 0BB` / `FOLD…BB` normalization. |
| **Look for** | Cleaned data ≤ raw val loss; fewer pathological predicted actions like `CALL 0BB`. |

### EXP-E3 — Official HF split vs random 95/5 re-split

| | |
|--|--|
| **Configs** | **A:** `prepare_data.py` + `load_text_split` (random 5%). **B:** `scripts/convert_poker_dataset.py` train/test files (dataset’s own split). |
| **How to test** | Both paths write **comma-separated** `context,action` lines (same as `to_line` / `split_prompt_completion`). For **B**, point trainers at `data/hands_train.txt` / `data/hands_test.txt` (or load those files instead of `load_text_split` on `hands_clean.txt`). Convert also writes `hands_clean.txt` for the default pipeline. |
| **Showcases** | Generalization under the dataset author’s held-out split vs an i.i.d. reshuffle (possible leakage if correlated hands). |
| **Look for** | Val loss higher on the official test split is common and more honest. Large train/test distribution shift ⇒ report both. |

### EXP-E4 — Completion-only vs full-sequence loss (ablation)

| | |
|--|--|
| **Configs** | TRL: `completion_only_loss=True` (default) vs `False`. Raw loop: mask action-only vs supervise all tokens. |
| **How to test** | Toggle TRL flag; for raw loop temporarily label all non-pad tokens. **Compare action-region CE only** at eval time for fairness. |
| **Showcases** | Core design claim: masking state tokens focuses capacity on decisions. |
| **Look for** | Full-sequence training may lower *token* loss (easy state copying) but **action-only eval** should favor completion-only. If not, masking implementation is suspect. |

### EXP-E5 — Left-truncation stress test

| | |
|--|--|
| **Configs** | Artificially low `n_positions` (e.g. 64 or 128) vs 384 on the same data. |
| **How to test** | Pair with B4; measure % of examples with `len(ids)==max_length` after encode. |
| **Showcases** | How much poker context the model actually needs before the decision point. |
| **Look for** | Sharp val degradation when truncation eats stacks/board/prior actions; confirms front-drop policy keeps the action (labels still present) but drops useful state. |

---

## Family F — Class imbalance & decision quality

Raw CE is insufficient alone. These experiments make W&B charts **action-aware** (FOLD / CALL / RAISE / CHECK / BET).

Post-train GPT-2 eval and `scripts/evaluate.py` write confusion matrices, per-class F1 bars, and occlusion feature-importance charts under `<model>/report/` and log them to W&B when configured.

### EXP-F1 — Per-action-type eval metrics

| | |
|--|--|
| **Configs** | Default GPT-2 (or any saved HF checkpoint) |
| **How to test** | Automatic after `run_experiment.py gpt2` / `weighted-gpt2`, or: `python scripts/evaluate.py --model artifacts/models/gpt2 --max-examples 256`. Metrics: `eval/action/{fold,call,raise,...}/{precision,recall,f1}` + confusion plot. |
| **Showcases** | That low overall loss can hide failure on minority aggressive actions. |
| **Look for** | High FOLD accuracy, weaker RAISE/BET. Prefer models that improve minority classes without collapsing to always-FOLD. |

### EXP-F2 — Majority-class baseline

| | |
|--|--|
| **Configs** | Constant predictor = most frequent train action type (usually FOLD). |
| **How to test** | `python scripts/run_experiment.py majority --group imbalance --tags exp-f2` |
| **Showcases** | Floor for classification-style metrics (complementary to bigram’s token CE). |
| **Look for** | Neural / feature models should beat majority accuracy on non-FOLD slices even if overall accuracy looks “high” from FOLD dominance. |

### EXP-F3 — Weighted GPT-2 loss on action types

| | |
|--|--|
| **Configs** | Inverse-frequency weights on examples by action type (`class_weight=True`). |
| **How to test** | `python scripts/run_experiment.py weighted-gpt2 --group imbalance --tags exp-f3` |
| **Showcases** | Whether rebalancing improves RAISE/BET quality at a tolerable cost to FOLD CE. |
| **Look for** | Better minority `eval/action/raise/f1` (and BET); mild regression in overall `val/loss` is OK if decision quality improves. |

---

## Family G — Training backend & systems

### EXP-G1 — Precision: fp16 vs fp32 (TRL)

| | |
|--|--|
| **Configs** | TRL `fp16=True` (default on CUDA) vs `fp16=False` |
| **How to test** | Two `train_trl` runs; same hparams. |
| **Showcases** | Numerical sensitivity of this small GPT-2. |
| **Look for** | Nearly matching `eval_loss`; fp16 faster / less memory. Large divergence ⇒ stick to fp32 for reporting. |

### EXP-G2 — Device parity (CUDA vs CPU/MPS)

| | |
|--|--|
| **Configs** | Default hparams; force device via environment / availability. |
| **How to test** | Short 1-epoch smoke on CPU vs full GPU run; compare loss trajectories at matched step counts. |
| **Showcases** | Reproducibility across hardware for contributors. |
| **Look for** | Same qualitative curves; absolute equality not required. |

### EXP-G3 — Seed reproducibility

| | |
|--|--|
| **Configs** | 3 seeds for default GPT-2 (set `torch.manual_seed`, `numpy`, `random`, and DataLoader generators). |
| **How to test** | Report mean ± std of `final_val_loss`. |
| **Showcases** | Run-to-run variance for claiming “GPT-2 beats bigram.” |
| **Look for** | Tight cluster; GPT-2 mean − bigram gap ≫ seed std. |

### EXP-G4 — Offline W&B sync path

| | |
|--|--|
| **Configs** | `WANDB_MODE=offline` then `wandb sync` |
| **How to test** | One bigram or 1-epoch GPT-2 on a node without egress; sync later. |
| **Showcases** | Cluster-friendly tracking workflow documented in README. |
| **Look for** | Identical charts after sync; no missing summary fields (`final_val_loss`). |

---

## Family H — Inference / qualitative showcase

### EXP-H1 — Greedy prediction gallery

| | |
|--|--|
| **Configs** | Best GPT-2 raw vs TRL checkpoints; `predict_action(..., max_new_tokens=16)`, `do_sample=False` |
| **How to test** | Curate ~20 held-out states spanning FOLD/CALL/RAISE/BET/CHECK; log a W&B Table of `{state, gold, pred_raw, pred_trl}`. |
| **Showcases** | Human-readable differences that loss alone cannot show (sizing format, illegal actions). |
| **Look for** | Valid action syntax; agreement between raw/TRL; failures concentrated on rare sizes / multi-token raises. |

### EXP-H2 — Sampling vs greedy

| | |
|--|--|
| **Configs** | Temporarily enable `do_sample=True` with low temperature vs greedy (code tweak in `predict.py`). |
| **How to test** | Same prompts; compare diversity and validity rates. |
| **Showcases** | Whether the LM assigns a peaked action distribution or a diffuse one. |
| **Look for** | Greedy preferred for decision agents; sampling should not invent malformed suffixes. |

---

## Family I — Alternative modeling approaches

Different paradigms from the GPT-2 LM — useful for feature importance, imbalance floors, and comparing text LMs to tabular classifiers.

### EXP-I1 — Majority action-type baseline (see also F2)

| | |
|--|--|
| **Command** | `python scripts/run_experiment.py majority --group alt-approaches --tags exp-i1` |
| **Showcases** | Non-neural floor; charts include confusion + per-class F1. |
| **Look for** | Near-100% FOLD recall, ~0 on RAISE/BET — the imbalance story in one plot. |

### EXP-I2 — RandomForest on structured hand features

| | |
|--|--|
| **Command** | `python scripts/run_experiment.py features --method rf --group alt-approaches --tags exp-i2` |
| **Showcases** | Tabular ML with native `feature_importances_` (pot, stacks, street, aggression counts, hole cards, …). Logs importance bar chart + action confusion. |
| **Look for** | Which engineered features drive CALL vs FOLD; whether RF beats majority on RAISE/BET F1. |

### EXP-I3 — Balanced LogisticRegression on the same features

| | |
|--|--|
| **Command** | `python scripts/run_experiment.py features --method logreg --group alt-approaches --tags exp-i3` |
| **Showcases** | Linear, class-balanced alternative; importance = mean \|coefficient\|. |
| **Look for** | Comparable story to RF with a simpler decision boundary; good sanity check on feature scale. |

### EXP-I4 — Class-weighted GPT-2 (see also F3)

| | |
|--|--|
| **Command** | `python scripts/run_experiment.py weighted-gpt2 --group alt-approaches --tags exp-i4` |
| **Showcases** | Same architecture as A2, different training objective (inverse-frequency example weights) + full action eval charts. |
| **Look for** | Lift on minority action F1 vs unweighted `gpt2` at matched hparams. |

### EXP-I5 — Occlusion feature importance on a trained LM

| | |
|--|--|
| **Command** | `python scripts/evaluate.py --model artifacts/models/gpt2 --max-examples 128 --max-importance 64` |
| **Showcases** | Ablate `[TABLE_CONFIGURATION]`, stacks, pot, streets, hole cards; importance = Δ action accuracy. |
| **Look for** | Large drops when masking stacks / aggression history; confirms the model uses poker state, not just priors. |

---

## Family J — Suggested W&B sweeps (automation)

Minimal sweep for a compelling public project page:

```yaml
# docs/wandb_sweep_model_compare.yaml
program: scripts/run_experiment.py
entity: mooslin-university-of-wisconsin-madison
project: poker-ai
method: grid
parameters:
  # express via a runner that reads WANDB/sweep env into GPT2Hyperparams
  model_kind:
    values: ["bigram", "gpt2-raw", "gpt2-trl"]
```

Capacity sweep (raw GPT-2):

| Parameter | Values |
|-----------|--------|
| `n_layer` | 2, 4, 6, 8 |
| `n_embd` | 128, 256, 512 |
| `learning_rate` | 1e-4, 3e-4, 1e-3 |
| `num_epochs` | 3 |

**Constraint:** `n_embd % n_head == 0` (fix `n_head=8` when possible).

---

## Recommended execution order (showcase path)

Run in this order for a clean W&B narrative:

1. **A1** Bigram baseline  
2. **A2** GPT-2 raw default  
3. **A3** GPT-2 TRL default  
4. **A4** Dashboard: model comparison  
5. **C1** LR sweep on the winning backend  
6. **B1 + B2** Depth/width on the winning LR  
7. **B4** Context length (guided by tokenizer length histogram)  
8. **E1** Data scaling  
9. **F1 + I2 + I5** Per-action metrics, RF feature importance, occlusion charts (demo-ready)  
10. **F2 + F3 / I4** Majority floor vs weighted GPT-2

---

## Scoring card (paste into W&B notes)

For each completed experiment, record:

| Field | Value |
|-------|-------|
| Run name / group / tags | |
| Trainer (`bigram` / `gpt2-raw` / `gpt2-trl` / `majority` / `features` / `weighted-gpt2`) | |
| Architecture (`n_layer/n_embd/n_head/n_positions`) | |
| Optim (`lr`, `batch`, `epochs` or `steps`) | |
| Data (`hands_clean` path, split seed, subsample %) | |
| Tokenizer vocab / max tokens observed | |
| `final_val_loss` or best `eval_loss` | |
| Action metrics (`accuracy`, `macro_f1`, per-class F1) | |
| vs bigram Δ / vs majority Δ | |
| Notes (truncation %, qualitative fails, top features) | |

**Pass criteria for the default stack:** GPT-2 (raw and TRL) both beat bigram `final_val_loss` by a clear margin; raw≈TRL; predictions are syntactically valid poker actions on a smoke gallery; action-type charts show non-zero RAISE/BET recall (not pure majority-FOLD).

---

## Mapping to repo knobs

| Knob | Location |
|------|----------|
| GPT-2 architecture & optim | `src/pokerai/config.py` → `GPT2Hyperparams` |
| Bigram optim | `BIGRAM_LR`, `BIGRAM_BATCH_SIZE`, `BIGRAM_STEPS` |
| W&B project | `WANDB_ENTITY` / `WANDB_PROJECT` → `mooslin-university-of-wisconsin-madison/poker-ai` (logging only when W&B is configured; see setup) |
| Masked encode | `pokerai.data.encode_with_mask` (offset-based boundary; keep-end truncation) |
| TRL completion-only | Pre-tokenized via `encode_with_mask` + `skip_prepare_dataset=True` (action labels already `-100`-masked) |
| Paths | `HANDS_CLEAN`, `TOKENIZER_DIR`, `MODEL_*_DIR` |
| Hand format | Always `context,action` (last comma). Converter and prepare share `to_line`. |
| Action eval / charts | `pokerai.eval` + `scripts/evaluate.py` |
| Structured features | `pokerai.features` → RF / LogReg trainers |
| Class-weighted GPT-2 | `scripts/run_experiment.py weighted-gpt2` |

Experiments marked as needing further instrumentation (**H2** sampling toggle) remain optional polish. **F1–F3** and **I1–I5** are implemented via the CLI trainers and `evaluate.py`.

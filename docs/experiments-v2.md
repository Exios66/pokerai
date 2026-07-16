## Python Scripts

```python
python scripts/run_experiment.py majority          # imbalance floor
python scripts/run_experiment.py features --method rf    # RF + feature importances
python scripts/run_experiment.py features --method logreg
python scripts/run_experiment.py weighted-gpt2     # class-weighted GPT-2
python scripts/evaluate.py --model artifacts/models/gpt2 --max-examples 256
```
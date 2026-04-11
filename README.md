# Polyhedral Interventions for Sparse Autoencoders

## Setting up

```bash
pip install uv
uv sync
uv run wandb login
```

## Project Structure

```
├── configs/
│   ├── spiked_model.yaml
│   └── spiked_model_regularization_ablation.yaml
├── scripts/
│   └── run_spiked.py
├── adaptive_elastic_sae/
│   ├── data/
│   ├── saes/
│   └── training/
├── notebooks/
├── pyproject.toml
└── README.md
```

## Run Experiment 1.1

Main experiment:

```
uv run scripts/run_spiked.py --config configs/spiked_model.yaml --use-wandb
```

Regularization ablation:

```
uv run scripts/run_spiked.py --config configs/spiked_model_regularization_ablation.yaml --use-wandb
```

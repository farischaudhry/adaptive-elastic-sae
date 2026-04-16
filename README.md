# Polyhedral Interventions for Sparse Autoencoders

This repository contains two aligned experiment tracks:

- Synthetic experiments for controlled mechanistic analysis.
- LLM activation experiments which use a model and dataset from Hugging Face directly.

## Setting up

```bash
pip install uv
uv sync
uv run wandb login
```

## Project Structure

```
├── configs/
│   ├── spiked/
│   ├── pythia70m/
│   │   └── llm_pythia70m_test.yaml
│   └── llama8b/
│       └── llm_llama8b_test.yaml
├── scripts/
│   ├── run_spiked.py
│   └── run_llm.py
├── adaptive_elastic_sae/
│   ├── data/                # synthetic and LLM stream data loaders
│   │   ├── synthetic.py     # spiked-data generation and sampling utilities
│   │   └── llm_streamer.py  # Hugging Face / TransformerLens activation streaming
│   ├── saes/                # SAE model implementations and shared base classes
│   │   ├── base.py          # common SAE interface and utilities
│   │   ├── vanilla.py       # L1 / ghost-variant sparse autoencoders
│   │   ├── polyhedral.py    # adaptive lasso and adaptive elastic net variants
│   │   └── top_k.py         # top-k SAE baseline
│   └── training/            # trainers, metrics, batching, and validation logic
│       ├── trainer.py       # synthetic SAE training loop
│       ├── llm_trainer.py   # LLM streaming training loop
│       ├── metrics.py       # synthetic and shared diagnostic metrics
│       ├── llm_metrics.py   # downstream patching / CE / KL validation metrics
│       ├── llm_batch_provider.py # batch adapter for streamed LLM activations
│       ├── trainer_utils.py  # trainer configs and batch-provider helpers
│       └── gpu_metrics.py   # FLOPs and throughput instrumentation
├── notebooks/
├── pyproject.toml
└── README.md
```

## Run Synthetic Experiment

Main experiment:

```
uv run scripts/run_spiked.py --config configs/spiked_model.yaml --use-wandb
```

Regularization ablation:

```
uv run scripts/run_spiked.py --config configs/spiked_model_regularization_ablation.yaml --use-wandb
```

## Run LLM Experiments

Pythia-70M test pattern:

```bash
uv run scripts/run_llm.py --config configs/pythia70m/llm_pythia70m_test.yaml --use-wandb
```

Llama 3.1 8B test pattern:

```bash
uv run scripts/run_llm.py --config configs/llama8b/llm_llama8b_test.yaml --use-wandb
```

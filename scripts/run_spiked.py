#!/usr/bin/env python3
"""Entry point: Run spiked-model rho sweep with all SAE variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Add repository root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_elastic_sae.data.synthetic import SpikedDataGenerator
from adaptive_elastic_sae.saes.vanilla import GhostVanillaSAE, VanillaSAE
from adaptive_elastic_sae.saes.polyhedral import (
    AdaptiveElasticNetSAE,
    AdaptiveLassoSAE,
    ElasticNetSAE,
)
from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.training.trainer import SAETrainer, TrainerConfig


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def instantiate_model(
    model_name: str,
    model_config: dict[str, Any],
    n_dim: int,
    d_dict: int,
    device: str,
    dtype_str: str,
) -> Any:
    """Create a model instance from config."""
    dtype = torch.float32 if dtype_str == "float32" else torch.float64
    class_name = model_config["class_name"]

    shared_args = {
        "n_dim": n_dim,
        "d_dict": d_dict,
        "device": device,
        "dtype": dtype,
    }

    if class_name == "VanillaSAE":
        return VanillaSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            **shared_args,
        )
    elif class_name == "GhostVanillaSAE":
        return GhostVanillaSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            ghost_scale=model_config.get("ghost_scale", 1e-4),
            dead_threshold=model_config.get("dead_threshold", 1e-12),
            ghost_activation=model_config.get("ghost_activation", "softplus"),
            ghost_exp_clip=model_config.get("ghost_exp_clip", 6.0),
            firing_ema_beta=model_config.get("firing_ema_beta", 0.999),
            persistent_dead_threshold=model_config.get(
                "persistent_dead_threshold", 1e-5
            ),
            min_steps_before_ghost=model_config.get("min_steps_before_ghost", 500),
            **shared_args,
        )
    elif class_name == "ElasticNetSAE":
        return ElasticNetSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            lambda_2=model_config.get("lambda_2", 0.01),
            **shared_args,
        )
    elif class_name == "AdaptiveLassoSAE":
        return AdaptiveLassoSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            gamma=model_config.get("gamma", 1.0),
            ema_beta=model_config.get("ema_beta", 0.999),
            weight_min=model_config.get("weight_min", 0.1),
            weight_max=model_config.get("weight_max", 10.0),
            **shared_args,
        )
    elif class_name == "AdaptiveElasticNetSAE":
        return AdaptiveElasticNetSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            lambda_2=model_config.get("lambda_2", 0.01),
            gamma=model_config.get("gamma", 1.0),
            ema_beta=model_config.get("ema_beta", 0.999),
            weight_min=model_config.get("weight_min", 0.1),
            weight_max=model_config.get("weight_max", 10.0),
            **shared_args,
        )
    elif class_name == "TopKSAE":
        return TopKSAE(
            k=model_config.get("k", 32),
            **shared_args,
        )
    else:
        raise ValueError(f"Unknown model class: {class_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spiked-model rho sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spiked_model.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B logging",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Extract sections
    data_cfg = config["data"]
    train_cfg = config["training"]
    models_cfg = config["models"]
    wandb_cfg = config.get("wandb", {})

    device = train_cfg.get("device", "cpu")
    dtype_str = train_cfg.get("dtype", "float32")
    use_wandb = args.use_wandb and wandb_cfg.get("enabled", False)

    print(f"\n{'=' * 80}")
    print("SAE Spiked-Model Mechanistic Validation")
    print(f"{'=' * 80}\n")

    # Get seeds from experiment config
    experiment_cfg = config.get("experiment", {})
    seeds = experiment_cfg.get("seeds", [404])

    # Seed sweep
    for seed in seeds:
        print(f"\n{'─' * 80}")
        print(f"SEED = {seed}")
        print(f"{'─' * 80}\n")

        # Rho sweep
        rho_values = data_cfg.get("rho_values", [0.0, 0.5, 0.9])
        for rho in rho_values:
            print(f"\n{'=' * 80}")
            print(f"RHO = {rho:.2f}")
            print(f"{'=' * 80}\n")

            # Create data generator with current seed
            gen = SpikedDataGenerator(
                n_dim=data_cfg["n_dim"],
                d_dict=data_cfg["d_dict"],
                k_sparse=data_cfg["k_sparse"],
                rho=rho,
                noise_std=data_cfg.get("noise_std", 0.0),
                allow_negative_codes=data_cfg.get("allow_negative_codes", False),
                seed=seed,
                device=device,
                dtype=torch.float32 if dtype_str == "float32" else torch.float64,
            )

            # Log dictionary statistics
            dict_stats = gen.dictionary_stats()
            print(f"Dictionary stats (rho={rho:.2f}):")
            for k, v in dict_stats.items():
                print(f"  {k}: {v:.4f}")
            print()

            # Train each model variant
            for model_name, model_config in models_cfg.items():
                print(f"{'─' * 80}")
                print(f"Training {model_name.upper()} on rho={rho:.2f}")
                print(f"{'─' * 80}\n")

                # Instantiate model
                model = instantiate_model(
                    model_name,
                    model_config,
                    data_cfg["n_dim"],
                    data_cfg["d_dict"],
                    device,
                    dtype_str,
                )

                # Create trainer config
                trainer_config = TrainerConfig(
                    num_steps=train_cfg["num_steps"],
                    batch_size=train_cfg["batch_size"],
                    learning_rate=train_cfg["learning_rate"],
                    warmup_steps=train_cfg.get("warmup_steps", 10_000_000),
                    max_activations_window=train_cfg.get(
                        "max_activations_window", 1_000_000
                    ),
                    log_interval=train_cfg.get("log_interval", 100),
                    seed=seed,
                    model_type=model_name,
                    device=device,
                    dtype=torch.float32 if dtype_str == "float32" else torch.float64,
                )

                run_name = f"spiked-rho{rho:.2f}-{model_name}-seed{seed}"
                run_tag_templates = wandb_cfg.get("run_tag_templates", [])
                run_tags = [
                    tpl.format(rho=rho, seed=seed, model_name=model_name)
                    for tpl in run_tag_templates
                ]

                # Train
                trainer = SAETrainer.from_synthetic(model, trainer_config, gen)
                trainer.train(
                    use_wandb=use_wandb,
                    run_name=run_name,
                    wandb_config=wandb_cfg,
                    run_metadata={
                        "seed": seed,
                        "rho": rho,
                        "model_type": model_name,
                    },
                    run_tags=run_tags,
                )

                print(f"\nCompleted {model_name} on rho={rho:.2f}\n")

    print(f"\n{'=' * 80}")
    print("Spiked-model sweep complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

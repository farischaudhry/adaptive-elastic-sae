from __future__ import annotations

import argparse
import itertools
from logging import getLogger
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Add repository root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer
from adaptive_elastic_sae.saes.polyhedral import (
    AdaptiveElasticNetSAE,
    AdaptiveLassoSAE,
    ElasticNetSAE,
)
from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.vanilla import GhostVanillaSAE, VanillaSAE
from adaptive_elastic_sae.training.llm_batch_provider import LLMActivationBatchProvider
from adaptive_elastic_sae.training.llm_trainer import LLMTrainerConfig, LLMSAETrainer

logger = getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "float64":
        return torch.float64
    return torch.float32


def expand_sweep_dict(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand a dict by Cartesian product over list-valued entries.

    Example:
      {"lambda_1": [1e-3, 2e-3], "lambda_2": [0.0, 1e-4], "class_name": "ElasticNetSAE"}
    becomes 4 concrete dicts.
    """
    sweep_keys = [k for k, v in config.items() if isinstance(v, list)]
    if not sweep_keys:
        return [dict(config)]

    fixed = {k: v for k, v in config.items() if k not in sweep_keys}
    values_product = itertools.product(*(config[k] for k in sweep_keys))

    expanded: list[dict[str, Any]] = []
    for values in values_product:
        variant = dict(fixed)
        for key, value in zip(sweep_keys, values):
            variant[key] = value
        expanded.append(variant)
    return expanded


def build_model_variants(models_cfg: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Expand model sweep configs into concrete run variants."""
    variants: list[tuple[str, dict[str, Any]]] = []

    for model_name, model_cfg in models_cfg.items():
        expanded = expand_sweep_dict(model_cfg)
        if len(expanded) == 1:
            variants.append((model_name, expanded[0]))
            continue

        for idx, cfg in enumerate(expanded):
            variants.append((f"{model_name}_sweep{idx:03d}", cfg))

    return variants


def instantiate_model(
    model_config: dict[str, Any],
    n_dim: int,
    d_dict: int,
    device: str,
    dtype: torch.dtype,
):
    class_name = model_config["class_name"]

    shared_args = {
        "n_dim": n_dim,
        "d_dict": d_dict,
        "device": device,
        "dtype": dtype,
    }

    if class_name == "VanillaSAE":
        return VanillaSAE(lambda_1=model_config.get("lambda_1", 0.1), **shared_args)
    if class_name == "GhostVanillaSAE":
        return GhostVanillaSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            ghost_scale=model_config.get("ghost_scale", 1e-4),
            dead_threshold=model_config.get("dead_threshold", 1e-12),
            ghost_activation=model_config.get("ghost_activation", "softplus"),
            ghost_exp_clip=model_config.get("ghost_exp_clip", 6.0),
            firing_ema_beta=model_config.get("firing_ema_beta", 0.999),
            persistent_dead_threshold=model_config.get("persistent_dead_threshold", 1e-5),
            min_steps_before_ghost=model_config.get("min_steps_before_ghost", 500),
            **shared_args,
        )
    if class_name == "ElasticNetSAE":
        return ElasticNetSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            lambda_2=model_config.get("lambda_2", 0.01),
            **shared_args,
        )
    if class_name == "AdaptiveLassoSAE":
        return AdaptiveLassoSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            gamma=model_config.get("gamma", 1.0),
            ema_beta=model_config.get("ema_beta", 0.999),
            weight_min=model_config.get("weight_min", 0.1),
            weight_max=model_config.get("weight_max", 10.0),
            ref_k_top_pct=model_config.get("ref_k_top_pct", 0.01),
            **shared_args,
        )
    if class_name == "AdaptiveElasticNetSAE":
        return AdaptiveElasticNetSAE(
            lambda_1=model_config.get("lambda_1", 0.1),
            lambda_2=model_config.get("lambda_2", 0.01),
            gamma=model_config.get("gamma", 1.0),
            ema_beta=model_config.get("ema_beta", 0.999),
            weight_min=model_config.get("weight_min", 0.1),
            weight_max=model_config.get("weight_max", 10.0),
            ref_k_top_pct=model_config.get("ref_k_top_pct", 0.01),
            **shared_args,
        )
    if class_name == "TopKSAE":
        return TopKSAE(k=model_config.get("k", 32), **shared_args)

    raise ValueError(f"Unknown model class: {class_name}")


def make_streamer(
    base_cfg: dict[str, Any],
    override_cfg: dict[str, Any] | None,
    device: str,
    model_dtype: str,
) -> PythiaActivationStreamer:
    cfg = dict(base_cfg)
    if override_cfg:
        cfg.update(override_cfg)
    cfg["device"] = device
    cfg["model_dtype"] = model_dtype
    return PythiaActivationStreamer(LLMStreamConfig(**cfg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM SAE training with sweep support")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_pythia.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    experiment_cfg = config.get("experiment", {})
    training_cfg = config["training"]
    data_cfg = config["data"]
    models_cfg = config["models"]
    wandb_cfg = config.get("wandb", {})
    checkpoint_cfg = config.get("checkpoint", {})

    device = training_cfg.get("device", "cuda")
    model_dtype = str(
        training_cfg.get("model_dtype", training_cfg.get("dtype", "float32"))
    )
    dtype = torch_dtype_from_str(training_cfg.get("dtype", "float32"))
    use_wandb = args.use_wandb and wandb_cfg.get("enabled", False)
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", True))
    checkpoint_dir = Path(checkpoint_cfg.get("dir", "checkpoints/llm_pythia"))
    if checkpoint_enabled:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = experiment_cfg.get("seeds", [0])
    model_variants = build_model_variants(models_cfg)

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Expanded model variants: {len(model_variants)}")

    for seed in seeds:
        torch.manual_seed(seed)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"SEED = {seed}")
        logger.info(f"{'=' * 80}\n")

        # Streamers and providers
        train_streamer = make_streamer(
            base_cfg=data_cfg["train"],
            override_cfg=None,
            device=device,
            model_dtype=model_dtype,
        )
        train_provider = LLMActivationBatchProvider(train_streamer)

        enable_validation = bool(training_cfg.get("enable_validation", False))
        validation_provider_online = None
        validation_provider_final = None

        if enable_validation:
            online_streamer = make_streamer(
                base_cfg=data_cfg["train"],
                override_cfg=data_cfg.get("validation_online", {}),
                device=device,
                model_dtype=model_dtype,
            )
            final_streamer = make_streamer(
                base_cfg=data_cfg["train"],
                override_cfg=data_cfg.get("validation_final", {}),
                device=device,
                model_dtype=model_dtype,
            )
            validation_provider_online = LLMActivationBatchProvider(online_streamer)
            validation_provider_final = LLMActivationBatchProvider(final_streamer)

        # Infer activation width from model if not pinned in config
        n_dim = int(data_cfg.get("n_dim", train_streamer.model.cfg.d_model))
        d_dict = int(data_cfg["d_dict"])

        trainer_config = LLMTrainerConfig(
            num_steps=int(training_cfg["num_steps"]),
            batch_size=int(training_cfg["batch_size"]),
            learning_rate=float(training_cfg["learning_rate"]),
            warmup_steps=int(training_cfg.get("warmup_steps", 10_000)),
            validation_log_interval=int(training_cfg.get("validation_log_interval", 1_000)),
            geometry_log_interval=int(training_cfg.get("geometry_log_interval", 3_000)),
            validation_num_batches=int(training_cfg.get("validation_num_batches", 100)),
            online_validation_enabled=bool(
                training_cfg.get("online_validation_enabled", True)
            ),
            final_validation_num_batches=(
                int(training_cfg["final_validation_num_batches"])
                if training_cfg.get("final_validation_num_batches") is not None
                else None
            ),
            final_validation_enabled=bool(
                training_cfg.get("final_validation_enabled", True)
            ),
            enable_validation=enable_validation,
            log_interval=int(training_cfg.get("log_interval", 100)),
            device=device,
            dtype=dtype,
        )

        for variant_name, model_cfg in model_variants:
            logger.info(f"{'-' * 80}")
            logger.info(f"Training {variant_name}")
            logger.info(f"{'-' * 80}\n")

            model = instantiate_model(
                model_config=model_cfg,
                n_dim=n_dim,
                d_dict=d_dict,
                device=device,
                dtype=dtype,
            )

            trainer = LLMSAETrainer(
                model=model,
                config=trainer_config,
                batch_provider=train_provider,
                validation_provider_online=validation_provider_online,
                validation_provider_final=validation_provider_final,
                llm=train_streamer.model,
                hook_name=train_streamer.hook_name,
            )

            run_name = f"llm-{variant_name}-seed{seed}"
            run_metadata = {
                "seed": seed,
                "variant_name": variant_name,
                "model_class": model_cfg.get("class_name"),
                "d_dict": d_dict,
                "n_dim": n_dim,
                "hook_name": train_streamer.hook_name,
                "tl_model_name": data_cfg["train"].get("tl_model_name", "pythia-70m-deduped"),
            }
            for k, v in model_cfg.items():
                if k != "class_name":
                    run_metadata[f"hp_{k}"] = v

            tags = list(wandb_cfg.get("tags", []))
            tags.append(f"seed={seed}")
            tags.append(f"variant={variant_name}")

            trainer.train(
                checkpoint_path=(
                    str(checkpoint_dir / f"{run_name}.pt") if checkpoint_enabled else None
                ),
                use_wandb=use_wandb,
                run_name=run_name,
                wandb_config={
                    **wandb_cfg,
                    "tags": tags,
                },
                run_metadata=run_metadata,
            )

            logger.info(f"Completed {variant_name} (seed={seed})\n")


if __name__ == "__main__":
    main()

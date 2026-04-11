"""Unified training loop for SAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.optim import Adam

from ..saes.base import BaseSAE
from .metrics import (
    active_gram_spectrum,
    dead_neurons_pct,
    explained_variance,
    feature_shrinkage_ratio,
    interaction_leakage_frobenius_approx,
    l0_active_features,
    l0_vs_l1_ratio,
    mean_max_cosine_similarity,
)


class BatchProvider(Protocol):
    """Interface for data providers used by SAETrainer."""

    def next_batch(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Return a batch dict containing at least key 'x'."""


class SyntheticBatchProvider:
    """Adapter that wraps SpikedDataGenerator into the generic BatchProvider interface."""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def next_batch(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        x, h_true = self.generator.generate_batch(batch_size)
        return {
            "x": x.to(device=device, dtype=dtype),
            "h_true": h_true.to(device=device, dtype=dtype),
        }


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    num_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    warmup_steps: int = 1_000_000
    max_activations_window: int = 1_000_000  # For dead neuron tracking
    log_interval: int = 100
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


class SAETrainer:
    """Trainer for synthetic SAE experiments."""

    def __init__(
        self,
        model: BaseSAE,
        config: TrainerConfig,
        batch_provider: BatchProvider,
    ) -> None:
        self.model = model
        self.config = config
        self.batch_provider = batch_provider
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # Track max activations for dead neuron metrics
        self.max_activations = torch.zeros(
            model.d_dict,
            device=self.device,
            dtype=config.dtype,
        )
        self.max_activations_history = []

    @classmethod
    def from_synthetic(
        cls,
        model: BaseSAE,
        config: TrainerConfig,
        generator: Any,
    ) -> "SAETrainer":
        """Construct trainer using synthetic generator adapter."""
        return cls(model=model, config=config, batch_provider=SyntheticBatchProvider(generator))

    def train(
        self,
        use_wandb: bool = False,
        run_name: str = "sae-train",
        wandb_config: dict | None = None,
    ) -> dict[str, Any]:
        """Run full training, return aggregate metrics."""
        if wandb_config is None:
            wandb_config = {}

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_config.get("project", "adaptive-elastic-sae"),
                    entity=wandb_config.get("entity"),
                    name=run_name,
                    tags=wandb_config.get("tags", []),
                    reinit=True,
                )
            except ImportError:
                use_wandb = False

        self.model.train()
        metrics_history = []

        for step in range(self.config.num_steps):
            # Generate batch
            batch = self.batch_provider.next_batch(
                batch_size=self.config.batch_size,
                device=self.device,
                dtype=self.config.dtype,
            )
            x = batch["x"]

            # Forward
            x_hat, h = self.model.forward(x)

            # Loss (support models that return either Tensor or (Tensor, dict)).
            loss_components: dict[str, float] = {}
            try:
                loss_out = self.model.compute_loss(
                    x,
                    x_hat,
                    h,
                    warmup_steps=self.config.warmup_steps,
                )
            except TypeError:
                loss_out = self.model.compute_loss(x, x_hat, h)

            if isinstance(loss_out, tuple):
                loss, loss_components = loss_out
            else:
                loss = loss_out
                loss_components = {"loss_total": loss.item()}

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Normalize decoder
            self.model.normalize_decoder()

            # Track max activations for dead neuron calc
            self.max_activations = torch.max(
                self.max_activations,
                h.abs().max(dim=0).values,
            )

            # Periodic logging
            if (step + 1) % self.config.log_interval == 0:
                metrics = self._compute_metrics(x, x_hat, h)
                metrics["step"] = step + 1
                metrics["loss"] = loss.item()
                metrics.update(loss_components)
                metrics_history.append(metrics)

                if use_wandb:
                    wandb.log(metrics)

                print(
                    f"Step {step + 1}/{self.config.num_steps} | "
                    f"Loss: {loss.item():.6f} | "
                    f"L0: {metrics['l0_active_features']:.2f} | "
                    f"Dead: {metrics['dead_neurons_pct']:.1f}%"
                )

            # Reset max activations periodically
            if (step + 1) % self.config.max_activations_window == 0:
                self.max_activations_history.append(self.max_activations.clone())
                self.max_activations.zero_()

        if use_wandb:
            wandb.finish()

        return {
            "metrics_history": metrics_history,
            "model_state": self.model.state_dict(),
        }

    @torch.no_grad()
    def _compute_metrics(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
    ) -> dict[str, float]:
        """Compute all diagnostic metrics for the current batch."""
        metrics = {}

        # Dead neurons
        metrics["dead_neurons_pct"] = dead_neurons_pct(
            self.max_activations,
            eps=1e-12,
        )

        # Sparsity
        metrics["l0_active_features"] = l0_active_features(h, eps=1e-12)
        metrics["l0_vs_l1_ratio"] = l0_vs_l1_ratio(h, eps=1e-12)

        # Reconstruction
        metrics["explained_variance"] = explained_variance(x, x_hat, eps=1e-12)

        # Shrinkage
        metrics["feature_shrinkage_ratio"] = feature_shrinkage_ratio(
            x,
            x_hat,
            eps=1e-12,
        )

        # Geometric: decoder coherence and conditioning (local active set, not batch union)
        local_active_mask = h[0] > 1e-12
        if local_active_mask.any():
            metrics["interaction_leakage_frobenius"] = (
                interaction_leakage_frobenius_approx(
                    self.model.decoder.weight.data,
                    local_active_mask,
                )
            )

            gram_metrics = active_gram_spectrum(
                self.model.decoder.weight.data,
                local_active_mask,
            )
            metrics.update(gram_metrics)

            metrics["mean_max_cosine_similarity"] = mean_max_cosine_similarity(
                self.model.decoder.weight.data,
                eps=1e-12,
            )

        return metrics

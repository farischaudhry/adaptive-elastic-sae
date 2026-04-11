from __future__ import annotations

from typing import Any, Protocol
from dataclasses import dataclass

import torch


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    num_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    warmup_steps: int = 1_000_000
    max_activations_window: int = 1_000_000
    log_interval: int = 100
    seed: int = -1
    model_type: str = "unknown"
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


class BatchProvider(Protocol):
    """Interface for data providers used by SAETrainer."""

    def next_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Return a batch dict containing at least key 'x'."""
        ...


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

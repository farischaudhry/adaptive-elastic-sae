from __future__ import annotations

import torch

from adaptive_elastic_sae.saes.base import BaseSAE


class ElasticNetSAE(BaseSAE):
    """SAE with L1 and L2 penalties (constant weights)."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        lambda_1: float = 0.1,
        lambda_2: float = 0.01,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(n_dim, d_dict, dtype=dtype, device=device)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Total loss: reconstruction + L1 + L2."""
        recon_loss = self.compute_reconstruction_loss(x, x_hat)
        l1_loss = self.lambda_1 * h.abs().sum(dim=1).mean()
        l2_loss = self.lambda_2 * (h**2).sum(dim=1).mean()
        total_loss = recon_loss + l1_loss + l2_loss
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": l1_loss.item(),
            "loss_l2": l2_loss.item(),
        }


class AdaptiveLassoSAE(BaseSAE):
    """SAE with EMA-based adaptive L1 weights."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        lambda_1: float = 0.1,
        gamma: float = 1.0,
        ema_beta: float = 0.999,
        weight_min: float = 0.1,
        weight_max: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(n_dim, d_dict, dtype=dtype, device=device)
        self.lambda_1 = lambda_1
        self.gamma = gamma
        self.ema_beta = ema_beta
        self.weight_min = weight_min
        self.weight_max = weight_max

        # EMA of absolute activations per feature
        self.register_buffer(
            "ema_abs_activations",
            torch.ones(d_dict, dtype=dtype, device=device),
        )
        self._step_count = 0

    def update_ema(self, h: torch.Tensor) -> None:
        """Update EMA of absolute activations."""
        batch_mean_abs = h.detach().abs().mean(dim=0)
        with torch.no_grad():
            self.ema_abs_activations.mul_(self.ema_beta).add_(
                batch_mean_abs,
                alpha=1.0 - self.ema_beta,
            )

    def get_adaptive_weights(self) -> torch.Tensor:
        """Compute adaptive penalty weights: 1 / (bar_a_i^gamma + eps)."""
        eps = torch.tensor(1e-5, dtype=self.dtype, device=self.device)
        weights = 1.0 / (torch.clamp(self.ema_abs_activations, min=eps) ** self.gamma)
        weights = torch.clamp(weights, min=self.weight_min, max=self.weight_max)
        return weights

    def warmup_factor(self, warmup_steps: int = 0) -> float:
        """Linearly ramp adaptive weights from 0 to 1 over warmup steps."""
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, self._step_count / max(1, warmup_steps))

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        warmup_steps: int = 10_000,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss with adaptive L1 weights + warmup schedule."""
        self.update_ema(h)
        self._step_count += 1

        recon_loss = self.compute_reconstruction_loss(x, x_hat)

        # Compute L1 with adaptive weights, linearly ramped during warmup
        adaptive_weights = self.get_adaptive_weights()
        warmup_factor = self.warmup_factor(warmup_steps)
        effective_weights = 1.0 + (adaptive_weights - 1.0) * warmup_factor

        l1_loss = self.lambda_1 * (effective_weights * h.abs()).sum(dim=1).mean()
        total_loss = recon_loss + l1_loss
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": l1_loss.item(),
            "warmup_factor": warmup_factor,
        }


class AdaptiveElasticNetSAE(BaseSAE):
    """SAE with adaptive L1 weights and fixed L2 penalty."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        lambda_1: float = 0.1,
        lambda_2: float = 0.01,
        gamma: float = 1.0,
        ema_beta: float = 0.999,
        weight_min: float = 0.1,
        weight_max: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(n_dim, d_dict, dtype=dtype, device=device)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gamma = gamma
        self.ema_beta = ema_beta
        self.weight_min = weight_min
        self.weight_max = weight_max

        # EMA of absolute activations per feature
        self.register_buffer(
            "ema_abs_activations",
            torch.ones(d_dict, dtype=dtype, device=device),
        )
        self._step_count = 0

    def update_ema(self, h: torch.Tensor) -> None:
        """Update EMA of absolute activations."""
        batch_mean_abs = h.detach().abs().mean(dim=0)
        with torch.no_grad():
            self.ema_abs_activations.mul_(self.ema_beta).add_(
                batch_mean_abs,
                alpha=1.0 - self.ema_beta,
            )

    def get_adaptive_weights(self) -> torch.Tensor:
        """Compute adaptive penalty weights: 1 / (bar_a_i^gamma + eps)."""
        eps = torch.tensor(1e-5, dtype=self.dtype, device=self.device)
        weights = 1.0 / (torch.clamp(self.ema_abs_activations, min=eps) ** self.gamma)
        weights = torch.clamp(weights, min=self.weight_min, max=self.weight_max)
        return weights

    def warmup_factor(self, warmup_steps: int = 0) -> float:
        """Linearly ramp adaptive weights from 0 to 1 over warmup steps."""
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, self._step_count / max(1, warmup_steps))

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        warmup_steps: int = 10_000,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss with adaptive L1 weights + fixed L2 + warmup."""
        self.update_ema(h)
        self._step_count += 1

        recon_loss = self.compute_reconstruction_loss(x, x_hat)

        # Adaptive L1 with warmup
        adaptive_weights = self.get_adaptive_weights()
        warmup_factor = self.warmup_factor(warmup_steps)
        effective_weights = 1.0 + (adaptive_weights - 1.0) * warmup_factor

        l1_loss = self.lambda_1 * (effective_weights * h.abs()).sum(dim=1).mean()
        l2_loss = self.lambda_2 * (h**2).sum(dim=1).mean()
        total_loss = recon_loss + l1_loss + l2_loss
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": l1_loss.item(),
            "loss_l2": l2_loss.item(),
            "warmup_factor": warmup_factor,
        }

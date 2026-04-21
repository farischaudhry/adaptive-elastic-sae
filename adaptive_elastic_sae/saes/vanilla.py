from __future__ import annotations

from typing import Literal

import torch

from adaptive_elastic_sae.saes.base import BaseSAE


class VanillaSAE(BaseSAE):
    """Standard SAE with L1 sparsity penalty."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        lambda_1: float = 0.1,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(n_dim, d_dict, dtype=dtype, device=device)
        self.lambda_1 = lambda_1

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Total loss: reconstruction + L1 penalty."""
        recon_loss = self.compute_reconstruction_loss(x, x_hat)
        l1_loss = self.lambda_1 * h.abs().sum(dim=1).mean()
        total_loss = recon_loss + l1_loss
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": l1_loss.item(),
        }


class GhostVanillaSAE(VanillaSAE):
    """Vanilla SAE with optional ghost-gradient proxy loss for dead features."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        lambda_1: float = 0.1,
        ghost_scale: float = 0.0,
        dead_threshold: float = 1e-12,
        ghost_activation: Literal["softplus", "exp", "relu"] = "softplus",
        ghost_exp_clip: float = 6.0,
        firing_ema_beta: float = 0.999,
        persistent_dead_threshold: float = 1e-5,
        min_steps_before_ghost: int = 500,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(n_dim, d_dict, lambda_1=lambda_1, dtype=dtype, device=device)
        self.ghost_scale = ghost_scale
        self.dead_threshold = dead_threshold
        self.ghost_activation = ghost_activation
        self.ghost_exp_clip = ghost_exp_clip
        self.firing_ema_beta = firing_ema_beta
        self.persistent_dead_threshold = persistent_dead_threshold
        self.min_steps_before_ghost = min_steps_before_ghost
        self._step_count = 0

        self.register_buffer(
            "firing_ema",
            torch.ones(d_dict, dtype=dtype, device=device),
        )

    def _update_firing_ema(self, h: torch.Tensor) -> None:
        """Track persistent inactivity, not just per-batch silence."""
        with torch.no_grad():
            fired = (h.detach().abs() > self.dead_threshold).float().mean(dim=0)
            self.firing_ema.mul_(self.firing_ema_beta).add_(
                fired,
                alpha=1.0 - self.firing_ema_beta,
            )

    def _ghost_proxy_activations(self, proxy_logits: torch.Tensor) -> torch.Tensor:
        """Compute ghost proxy activations with a stable nonlinearity."""
        if self.ghost_activation == "exp":
            return torch.exp(torch.clamp(proxy_logits, max=self.ghost_exp_clip))
        if self.ghost_activation == "relu":
            return torch.relu(proxy_logits)
        return torch.nn.functional.softplus(proxy_logits)

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Vanilla loss plus optional ghost proxy loss for persistently dead features."""
        self._step_count += 1
        self._update_firing_ema(h)

        recon_loss = self.compute_reconstruction_loss(x, x_hat)
        l1_loss = self.lambda_1 * h.abs().sum(dim=1).mean()

        ghost_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        dead_features_pct_batch = (
            h <= self.dead_threshold
        ).float().mean().item() * 100.0

        persistent_dead_mask_vec = self.firing_ema <= self.persistent_dead_threshold
        persistent_dead_pct = persistent_dead_mask_vec.float().mean().item() * 100.0

        if self.ghost_scale > 0.0 and self._step_count >= self.min_steps_before_ghost:
            # Use detached residual to avoid coupling ghost target dynamics to decoder updates.
            residual = (x - x_hat).detach()

            dead_mask = persistent_dead_mask_vec.float().unsqueeze(0).expand_as(h)
            # Residual is already in reconstruction-error coordinates; avoid re-centering with b_dec.
            proxy_logits = self.encoder(residual)
            proxy_acts = self._ghost_proxy_activations(proxy_logits) * dead_mask

            # Reconstruct residual using raw decoder map (without adding bias term).
            ghost_recon = self.decoder(proxy_acts)
            ghost_loss = self.ghost_scale * ((residual - ghost_recon) ** 2).mean()

        total_loss = recon_loss + l1_loss + ghost_loss
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": l1_loss.item(),
            "loss_ghost": ghost_loss.item(),
            "dead_features_pct_batch": dead_features_pct_batch,
            "dead_features_pct_persistent": persistent_dead_pct,
            "persistent_dead_threshold": self.persistent_dead_threshold,
            "ghost_scale": self.ghost_scale,
        }

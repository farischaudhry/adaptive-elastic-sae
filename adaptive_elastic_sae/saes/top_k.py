from __future__ import annotations

import torch

from adaptive_elastic_sae.saes.base import BaseSAE


class TopKSAE(BaseSAE):
    """SAE using hard Top-K activation instead of an L1 penalty."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        k: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(n_dim, d_dict, dtype=dtype, device=device)
        if not (1 <= k <= d_dict):
            raise ValueError(f"k must be in [1, d_dict], got k={k}, d_dict={d_dict}")
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard Top-K masking to pre-activations."""
        x_centered = x - self.b_dec
        pre_acts = self.encoder(x_centered)

        top_k_vals, top_k_idx = torch.topk(pre_acts, k=self.k, dim=-1)

        h = torch.zeros_like(pre_acts)
        h.scatter_(dim=-1, index=top_k_idx, src=torch.relu(top_k_vals))
        return h

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Total loss: pure reconstruction (MSE) with no L1/L2 penalties."""
        recon_loss = self.compute_reconstruction_loss(x, x_hat)
        return recon_loss, {
            "loss_total": recon_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": 0.0,
            "loss_l2": 0.0,
            "warmup_factor": 1.0,
        }

from __future__ import annotations

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
    ) -> torch.Tensor:
        """Total loss: reconstruction + L1 penalty."""
        recon_loss = self.compute_reconstruction_loss(x, x_hat)
        l1_loss = self.lambda_1 * h.abs().sum(dim=1).mean()
        return recon_loss + l1_loss

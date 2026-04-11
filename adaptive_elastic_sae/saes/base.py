from __future__ import annotations

import torch
import torch.nn as nn


class BaseSAE(nn.Module):
    """Shared SAE encoder-decoder architecture."""

    def __init__(
        self,
        n_dim: int,
        d_dict: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.d_dict = d_dict
        self.dtype = dtype
        self.device = torch.device(device)

        # Traditional AEs have no bias parameter in the decoder
        # SAEs include it to avoid centering bias (to avoid wasting capacity on the mean)
        self.b_dec = nn.Parameter(torch.zeros(n_dim, dtype=dtype, device=device))

        # Encoder: n_dim -> d_dict
        self.encoder = nn.Linear(n_dim, d_dict, dtype=dtype, device=device)

        # Decoder: d_dict -> n_dim with unit-norm columns
        self.decoder = nn.Linear(d_dict, n_dim, dtype=dtype, device=device, bias=False)

        # Initialize decoder to unit norm
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        """Normalize decoder columns to unit length."""
        with torch.no_grad():
            self.decoder.weight.data = self.decoder.weight.data / torch.clamp(
                self.decoder.weight.data.norm(dim=0, keepdim=True),
                min=1e-12,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute latent sparse activations after centering the data."""
        x_centered = x - self.b_dec
        return torch.relu(self.encoder(x_centered))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent and add the center back."""
        return self.decoder(h) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, decode, return (reconstruction, latent)."""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """MSE reconstruction loss."""
        return ((x - x_hat) ** 2).mean()

    def normalize_decoder(self) -> None:
        """Externally callable normalization; should be called after each optimization step."""
        self._normalize_decoder()

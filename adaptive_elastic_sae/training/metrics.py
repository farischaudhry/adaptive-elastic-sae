from __future__ import annotations

from typing import Any

import torch


def dead_neurons_pct(max_activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Percent of features that never activated above eps in a window."""
    dead_mask = max_activations <= eps
    return 100.0 * dead_mask.float().mean().item()


def dead_neuron_recovery_rate(
    prev_max_activations: torch.Tensor,
    curr_max_activations: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Fraction of previously dead features that became alive in the next window."""
    prev_dead = prev_max_activations <= eps
    denom = prev_dead.sum().item()
    if denom == 0:
        return 0.0

    curr_alive = curr_max_activations > eps
    recovered = (prev_dead & curr_alive).sum().item()
    return recovered / denom


@torch.no_grad()
def interaction_leakage_frobenius_approx(
    decoder: torch.Tensor, active_mask: torch.Tensor
) -> float:
    """Approximate leakage as ||D_Ac^T D_A||_F (avoids unstable inverse term)."""
    if active_mask.dtype != torch.bool:
        active_mask = active_mask.bool()

    active_count = active_mask.sum().item()
    inactive_count = (~active_mask).sum().item()
    if active_count == 0 or inactive_count == 0:
        return 0.0

    d_a = decoder[:, active_mask]
    d_ac = decoder[:, ~active_mask]
    return torch.norm(d_ac.T @ d_a, p="fro").item()


@torch.no_grad()
def active_gram_spectrum(
    decoder: torch.Tensor,
    active_mask: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return min/max eigenvalues and condition number of active Gram block."""
    if active_mask.dtype != torch.bool:
        active_mask = active_mask.bool()

    active_count = active_mask.sum().item()
    if active_count == 0:
        return {
            "active_min_eigenvalue": 0.0,
            "active_max_eigenvalue": 0.0,
            "active_condition_number": 0.0,
        }

    d_a = decoder[:, active_mask]
    gram = d_a.T @ d_a
    eigvals = torch.linalg.eigvalsh(gram)
    min_eig = torch.clamp(eigvals.min(), min=eps)
    max_eig = eigvals.max()

    return {
        "active_min_eigenvalue": min_eig.item(),
        "active_max_eigenvalue": max_eig.item(),
        "active_condition_number": (max_eig / min_eig).item(),
    }


@torch.no_grad()
def compute_cross_leverage(
    decoder: torch.Tensor, active_mask: torch.Tensor, k_top: int = 5, eps: float = 1e-12
) -> dict[str, Any]:
    """
    Computes h_j(A) = (1/n) ||P_A a_j||^2.
    Identifies features 'shadowed' by the active subspace.
    """
    if not active_mask.any():
        return {
            "mean_h_j": 0.0,
            "total_h_j": 0.0,
            "shadowed_ids": [],
            "shadowed_scores": [],
        }

    n_dim = decoder.shape[0]
    inactive_indices = torch.where(~active_mask)[0]

    d_a = decoder[:, active_mask]  # Active vectors
    d_ac = decoder[:, ~active_mask]  # Inactive vectors

    # Local Gram matrices
    g_aa = (d_a.T @ d_a) / n_dim
    g_aca = (d_ac.T @ d_a) / n_dim

    try:
        # Regression of inactive onto active
        sol = torch.linalg.lstsq(
            g_aa + eps * torch.eye(g_aa.shape[0], device=g_aa.device), g_aca.T
        )
        inv_g_aa_g_at_ac = sol.solution
    except (RuntimeError, torch.linalg.LinAlgError):
        # Fallback for catastrophic failure:
        # Return -1.0 to indicate a singularity in wandb logs.
        return {
            "mean_h_j": -1.0,
            "total_h_j": -1.0,
            "shadowed_ids": [],
            "shadowed_scores": [],
        }

    # h_j alignment scores
    h_j = n_dim * torch.sum(g_aca * inv_g_aa_g_at_ac.T, dim=1)

    actual_k = min(k_top, h_j.size(0))
    top_scores, top_local_idx = torch.topk(h_j, k=actual_k)

    return {
        "mean_h_j": h_j.mean().item(),
        "total_h_j": h_j.sum().item(),
        "shadowed_ids": inactive_indices[top_local_idx].tolist(),
        "shadowed_scores": top_scores.tolist(),
    }


def feature_shrinkage_ratio(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Measures reconstruction shrinkage pathology as ||x_hat||_1 / ||x||_1.
    A ratio < 1.0 indicates magnitude suppression.
    """
    norm_x_hat = x_hat.abs().sum(dim=1).mean()
    norm_x = torch.clamp(x.abs().sum(dim=1).mean(), min=eps)
    return (norm_x_hat / norm_x).item()


def l0_active_features(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Average number of active features per sample."""
    return (activations.abs() > eps).sum(dim=1).float().mean().item()


def l0_vs_l1_ratio(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute ||a||_1 / ||a||_0 averaged across samples."""
    l1 = activations.abs().sum(dim=1)
    l0 = torch.clamp((activations.abs() > eps).sum(dim=1).float(), min=1.0)
    return (l1 / l0).mean().item()


def explained_variance(
    x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-12
) -> float:
    """1 - Var(x - x_hat) / Var(x) calculated per-feature and summed."""
    residual_var = (x - x_hat).var(dim=0, unbiased=False).sum()
    total_var = torch.clamp(x.var(dim=0, unbiased=False).sum(), min=eps)
    return (1.0 - residual_var / total_var).item()


@torch.no_grad()
def mean_max_cosine_similarity(decoder: torch.Tensor, eps: float = 1e-12) -> float:
    """For each decoder atom, average cosine similarity to its nearest neighbor."""
    d = decoder / torch.clamp(decoder.norm(dim=0, keepdim=True), min=eps)
    sim = d.T @ d
    sim.fill_diagonal_(-1.0)
    max_sim, _ = sim.max(dim=1)
    return max_sim.mean().item()


def activation_effective_sample_size(
    mean_abs_activations: torch.Tensor, eps: float = 1e-12
) -> float:
    """ESS proxy: (sum a_i)^2 / sum a_i^2 for nonnegative feature activity."""
    a = torch.clamp(mean_abs_activations, min=0.0)
    num = a.sum() ** 2
    den = torch.clamp((a**2).sum(), min=eps)
    return (num / den).item()


def weight_bimodality_ratio(
    weights: torch.Tensor,
    delta_sig: float,
    delta_noise: float,
    eps: float = 1e-12,
) -> float:
    """Ratio of low-weight (signal) to high-weight (noise) feature counts."""
    low = (weights < delta_sig).sum().float()
    high = torch.clamp((weights > delta_noise).sum().float(), min=eps)
    return (low / high).item()

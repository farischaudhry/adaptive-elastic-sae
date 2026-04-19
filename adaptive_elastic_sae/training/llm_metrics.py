from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from adaptive_elastic_sae.training.metrics import summary_stats


def _safe_next_token_ce(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute next-token CE with raw semantics and finite-value fallback."""
    shifted_logits = logits[:, :-1, :].contiguous().float()
    shifted_targets = tokens[:, 1:].contiguous()

    raw_ce = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_targets.reshape(-1),
        reduction="mean",
    )
    if torch.isfinite(raw_ce):
        return raw_ce

    # Fallback only when numerics are broken under extreme ablations.
    shifted_logits = torch.nan_to_num(
        shifted_logits,
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    )
    shifted_logits = torch.clamp(shifted_logits, min=-80.0, max=80.0)
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_targets.reshape(-1),
        reduction="mean",
    )


@torch.no_grad()
def evaluate_downstream_degradation(
    llm,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    ablation_mode: str = "batch_mean",
    verbose_nan_debug: bool = False,
) -> dict[str, float]:
    """
    Evaluate downstream degradation under SAE patching.

    Returns CE degradation, CE loss recovered, and KL divergence metrics.
    
    Args:
        verbose_nan_debug: If True, log which values are NaN/Inf (for debugging)
    """
    clean_logits = llm(tokens)

    def sae_patch_hook(
        activations: torch.Tensor,
        hook=None,
    ) -> torch.Tensor:
        original_shape = activations.shape
        flat_acts = activations.reshape(-1, activations.shape[-1])
        
        # Debug: check for NaN/Inf in activations before SAE
        if verbose_nan_debug:
            if torch.isnan(flat_acts).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[DEBUG] Activations contain NaN before SAE")
            if torch.isinf(flat_acts).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[DEBUG] Activations contain Inf before SAE")
        
        reconstructed_acts, _ = sae(flat_acts)
        
        # Debug: check for NaN/Inf in reconstructed activations
        if verbose_nan_debug:
            if torch.isnan(reconstructed_acts).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[DEBUG] Reconstructed activations contain NaN after SAE")
            if torch.isinf(reconstructed_acts).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[DEBUG] Reconstructed activations contain Inf after SAE")
        
        return reconstructed_acts.reshape(original_shape)

    patched_logits = llm.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, sae_patch_hook)],
    )

    # Baseline ablation for normalized CE recovery metric.
    def baseline_patch_hook(
        activations: torch.Tensor,
        hook=None,
    ) -> torch.Tensor:
        if ablation_mode == "zero":
            return torch.zeros_like(activations)
        if ablation_mode == "batch_mean":
            # Mean over batch and sequence dimensions, preserving feature dimension.
            mean_acts = activations.mean(dim=(0, 1), keepdim=True)
            return mean_acts.expand_as(activations)
        raise ValueError(
            f"Unsupported ablation_mode '{ablation_mode}'. "
            "Expected one of: 'zero', 'batch_mean'."
        )

    zero_logits = llm.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, baseline_patch_hook)],
    )

    clean_loss = _safe_next_token_ce(clean_logits, tokens)
    patched_loss = _safe_next_token_ce(patched_logits, tokens)
    zero_loss = _safe_next_token_ce(zero_logits, tokens)

    # Compute KL in float32 with finite-value sanitization.
    clean_logits_f32 = clean_logits.float()
    patched_logits_f32 = patched_logits.float()
    clean_logits_f32 = torch.nan_to_num(
        clean_logits_f32,
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    )
    patched_logits_f32 = torch.nan_to_num(
        patched_logits_f32,
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    )
    log_probs_clean = torch.log_softmax(clean_logits_f32, dim=-1)
    log_probs_patched = torch.log_softmax(patched_logits_f32, dim=-1)
    log_probs_clean = torch.clamp(log_probs_clean, min=-100.0, max=0.0)
    log_probs_patched = torch.clamp(log_probs_patched, min=-100.0, max=0.0)
    kl_div = (
        (torch.exp(log_probs_clean) * (log_probs_clean - log_probs_patched))
        .sum(dim=-1)
        .mean()
    )

    clean_loss_val = clean_loss.item()
    patched_loss_val = patched_loss.item()
    zero_loss_val = zero_loss.item()
    kl_div_val = kl_div.item()

    denom = zero_loss_val - clean_loss_val
    if (not math.isfinite(denom)) or abs(denom) < 1e-12:
        ce_loss_recovered = 0.0
    else:
        ce_loss_recovered = 1.0 - ((patched_loss_val - clean_loss_val) / denom)

    # Check which values are invalid for debugging
    value_dict = {
        "clean_loss": clean_loss_val,
        "patched_loss": patched_loss_val,
        "zero_loss": zero_loss_val,
        "kl_divergence": kl_div_val,
        "ce_loss_recovered": ce_loss_recovered,
    }
    
    invalid_keys = [k for k, v in value_dict.items() if not math.isfinite(v)]
    is_finite = len(invalid_keys) == 0
    
    if verbose_nan_debug and invalid_keys:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[DEBUG] evaluate_downstream_degradation found invalid values: {invalid_keys}")
        for k, v in value_dict.items():
            if not math.isfinite(v):
                logger.debug(f"  {k} = {v}")

    return {
        "ce_loss_degradation": patched_loss_val - clean_loss_val,
        "ce_loss_recovered": ce_loss_recovered,
        "ce_loss_recovered_pct": 100.0 * ce_loss_recovered,
        "kl_divergence": kl_div_val,
        "clean_loss": clean_loss_val,
        "patched_loss": patched_loss_val,
        "zero_loss": zero_loss_val,
        "is_finite": float(is_finite),
    }


def aggregate_downstream_degradation(
    results: list[dict[str, float]],
    label: str,
    verbose_nan_debug: bool = False,
) -> dict[str, Any]:
    """Aggregate batch-level downstream degradation metrics with uncertainty summaries."""
    if not results:
        return {}

    # Core metrics required for a batch to be considered valid.
    required_keys = (
        "ce_loss_degradation",
        "ce_loss_recovered",
        "ce_loss_recovered_pct",
        "kl_divergence",
        "clean_loss",
        "patched_loss",
    )
    # Diagnostic-only metric: zero-ablation can be unstable for some hooks/models.
    optional_keys = ("zero_loss",)
    tracked_keys = required_keys + optional_keys
    invalid_counts = {k: 0 for k in tracked_keys}
    valid_results = []
    
    if verbose_nan_debug and len(results) > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[DEBUG] aggregate_downstream_degradation: Processing {len(results)} results")
    
    for batch_idx, r in enumerate(results):
        row_is_valid = True
        batch_invalid_keys = []
        for k in tracked_keys:
            if not math.isfinite(float(r[k])):
                invalid_counts[k] += 1
                batch_invalid_keys.append(k)
                if k in required_keys:
                    row_is_valid = False
        
        if verbose_nan_debug and batch_invalid_keys:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[DEBUG]   Batch {batch_idx} invalid for keys: {batch_invalid_keys}")
            for k in batch_invalid_keys:
                logger.debug(f"[DEBUG]     {k} = {r.get(k, 'N/A')}")
        
        if row_is_valid:
            valid_results.append(r)

    skipped = len(results) - len(valid_results)
    if not valid_results:
        if verbose_nan_debug:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[DEBUG] {label}: All {skipped} batches filtered due to NaN/Inf. "
                f"Invalid counts per metric: {invalid_counts}"
            )
        metrics = {
            f"{label}_valid_batches": 0.0,
            f"{label}_skipped_batches": float(skipped),
        }
        for k, c in invalid_counts.items():
            metrics[f"{label}_invalid_{k}"] = float(c)
        return metrics

    ce_degradations = [r["ce_loss_degradation"] for r in valid_results]
    ce_recoveries = [r["ce_loss_recovered"] for r in valid_results]
    ce_recoveries_pct = [r["ce_loss_recovered_pct"] for r in valid_results]
    kl_divs = [r["kl_divergence"] for r in valid_results]
    clean_losses = [r["clean_loss"] for r in valid_results]
    patched_losses = [r["patched_loss"] for r in valid_results]
    finite_zero_losses = [
        float(r["zero_loss"])
        for r in valid_results
        if math.isfinite(float(r["zero_loss"]))
    ]

    metrics: dict[str, float] = {
        f"{label}_ce_loss_degradation": sum(ce_degradations) / len(ce_degradations),
        f"{label}_ce_loss_recovered": sum(ce_recoveries) / len(ce_recoveries),
        f"{label}_ce_loss_recovered_pct": sum(ce_recoveries_pct)
        / len(ce_recoveries_pct),
        f"{label}_kl_div": sum(kl_divs) / len(kl_divs),
        f"{label}_clean_loss": sum(clean_losses) / len(clean_losses),
        f"{label}_patched_loss": sum(patched_losses) / len(patched_losses),
        f"{label}_valid_batches": float(len(valid_results)),
        f"{label}_skipped_batches": float(skipped),
        f"{label}_zero_loss_valid_batches": float(len(finite_zero_losses)),
    }

    if finite_zero_losses:
        metrics[f"{label}_zero_loss"] = sum(finite_zero_losses) / len(finite_zero_losses)

    for k, c in invalid_counts.items():
        metrics[f"{label}_invalid_{k}"] = float(c)

    metrics.update(
        summary_stats(
            ce_recoveries,
            f"{label}_ce_recovered",
            include_histogram=True,
        )
    )
    metrics.update(
        summary_stats(
            ce_recoveries_pct,
            f"{label}_ce_recovered_pct",
            include_histogram=True,
        )
    )
    metrics.update(
        summary_stats(
            kl_divs,
            f"{label}_kl_div",
            include_histogram=True,
        )
    )
    return metrics


def aggregate_reconstruction_validation(
    recon_losses: list[float],
    kl_divs: list[float],
    label: str,
) -> dict[str, float]:
    """Aggregate non-patching validation metrics for reconstruction/KL branches."""
    metrics: dict[str, float] = {}
    if recon_losses:
        metrics[f"{label}_recon_loss"] = sum(recon_losses) / len(recon_losses)
    if kl_divs:
        metrics[f"{label}_kl_div"] = sum(kl_divs) / len(kl_divs)
    return metrics

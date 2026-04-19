from __future__ import annotations

import math
from typing import Any

import torch

from adaptive_elastic_sae.training.metrics import summary_stats


@torch.no_grad()
def evaluate_downstream_degradation(
    llm,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    verbose_nan_debug: bool = False,
) -> dict[str, float]:
    """
    Evaluate downstream degradation under SAE patching.

    Returns CE degradation, CE loss recovered, and KL divergence metrics.
    
    Args:
        verbose_nan_debug: If True, log which values are NaN/Inf (for debugging)
    """
    clean_logits = llm(tokens)
    clean_loss = llm(tokens, return_type="loss")

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
    patched_loss = llm.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, sae_patch_hook)],
    )

    # Zero-ablation baseline for normalized CE recovery metric.
    def zero_patch_hook(
        activations: torch.Tensor,
        hook=None,
    ) -> torch.Tensor:
        return torch.zeros_like(activations)

    zero_loss = llm.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_patch_hook)],
    )

    # Compute KL in float32 for numerical stability under fp16 model execution.
    clean_logits_f32 = clean_logits.float()
    patched_logits_f32 = patched_logits.float()
    log_probs_clean = torch.log_softmax(clean_logits_f32, dim=-1)
    log_probs_patched = torch.log_softmax(patched_logits_f32, dim=-1)
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

    required_keys = (
        "ce_loss_degradation",
        "ce_loss_recovered",
        "ce_loss_recovered_pct",
        "kl_divergence",
        "clean_loss",
        "patched_loss",
        "zero_loss",
    )
    invalid_counts = {k: 0 for k in required_keys}
    valid_results = []
    
    if verbose_nan_debug and len(results) > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[DEBUG] aggregate_downstream_degradation: Processing {len(results)} results")
    
    for batch_idx, r in enumerate(results):
        row_is_valid = True
        batch_invalid_keys = []
        for k in required_keys:
            if not math.isfinite(float(r[k])):
                invalid_counts[k] += 1
                row_is_valid = False
                batch_invalid_keys.append(k)
        
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
    zero_losses = [r["zero_loss"] for r in valid_results]

    metrics: dict[str, float] = {
        f"{label}_ce_loss_degradation": sum(ce_degradations) / len(ce_degradations),
        f"{label}_ce_loss_recovered": sum(ce_recoveries) / len(ce_recoveries),
        f"{label}_ce_loss_recovered_pct": sum(ce_recoveries_pct)
        / len(ce_recoveries_pct),
        f"{label}_kl_div": sum(kl_divs) / len(kl_divs),
        f"{label}_clean_loss": sum(clean_losses) / len(clean_losses),
        f"{label}_patched_loss": sum(patched_losses) / len(patched_losses),
        f"{label}_zero_loss": sum(zero_losses) / len(zero_losses),
        f"{label}_valid_batches": float(len(valid_results)),
        f"{label}_skipped_batches": float(skipped),
    }

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

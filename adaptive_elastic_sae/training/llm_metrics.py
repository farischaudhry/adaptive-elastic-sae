from __future__ import annotations

import math

import torch

from adaptive_elastic_sae.training.metrics import summary_stats


@torch.no_grad()
def evaluate_downstream_degradation(
    llm,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
) -> dict[str, float]:
    """
    Evaluate downstream degradation under SAE patching.

    Returns CE degradation, CE loss recovered, and KL divergence metrics.
    """
    clean_logits = llm(tokens)
    clean_loss = llm(tokens, return_type="loss")

    def sae_patch_hook(
        activations: torch.Tensor,
        hook=None,
    ) -> torch.Tensor:
        original_shape = activations.shape
        flat_acts = activations.reshape(-1, activations.shape[-1])
        reconstructed_acts, _ = sae(flat_acts)
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

    is_finite = all(
        math.isfinite(v)
        for v in (
            clean_loss_val,
            patched_loss_val,
            zero_loss_val,
            kl_div_val,
            ce_loss_recovered,
        )
    )

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
) -> dict[str, float]:
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
    for r in results:
        row_is_valid = True
        for k in required_keys:
            if not math.isfinite(float(r[k])):
                invalid_counts[k] += 1
                row_is_valid = False
        if row_is_valid:
            valid_results.append(r)

    skipped = len(results) - len(valid_results)
    if not valid_results:
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

    metrics.update(summary_stats(ce_recoveries, f"{label}_ce_recovered"))
    metrics.update(summary_stats(ce_recoveries_pct, f"{label}_ce_recovered_pct"))
    metrics.update(summary_stats(kl_divs, f"{label}_kl_div"))
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

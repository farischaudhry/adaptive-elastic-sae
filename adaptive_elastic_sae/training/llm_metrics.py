from __future__ import annotations

import torch


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

    def sae_patch_hook(activations: torch.Tensor, _hook) -> torch.Tensor:
        original_shape = activations.shape
        flat_acts = activations.reshape(-1, activations.shape[-1])
        reconstructed_acts, _ = sae(flat_acts)
        return reconstructed_acts.reshape(original_shape)

    patched_logits = llm.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, sae_patch_hook)],
    )
    patched_loss = llm(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, sae_patch_hook)],
    )

    # Zero-ablation baseline for normalized CE recovery metric.
    def zero_patch_hook(activations: torch.Tensor, _hook) -> torch.Tensor:
        return torch.zeros_like(activations)

    zero_loss = llm(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_patch_hook)],
    )

    log_probs_clean = torch.log_softmax(clean_logits, dim=-1)
    log_probs_patched = torch.log_softmax(patched_logits, dim=-1)
    kl_div = (
        (torch.exp(log_probs_clean) * (log_probs_clean - log_probs_patched))
        .sum(dim=-1)
        .mean()
    )

    denom = (zero_loss - clean_loss).item()
    if abs(denom) < 1e-12:
        ce_loss_recovered = 0.0
    else:
        ce_loss_recovered = 1.0 - ((patched_loss - clean_loss).item() / denom)

    return {
        "ce_loss_degradation": (patched_loss - clean_loss).item(),
        "ce_loss_recovered": ce_loss_recovered,
        "kl_divergence": kl_div.item(),
        "clean_loss": clean_loss.item(),
        "patched_loss": patched_loss.item(),
        "zero_loss": zero_loss.item(),
    }

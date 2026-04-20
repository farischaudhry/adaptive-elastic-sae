from __future__ import annotations

import math
from logging import getLogger
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam

from adaptive_elastic_sae.saes.base import BaseSAE
from adaptive_elastic_sae.training.gpu_metrics import (
    flops_progress_metrics,
    measure_training_step_flops,
)
from adaptive_elastic_sae.training.llm_metrics import (
    aggregate_downstream_degradation,
    aggregate_reconstruction_validation,
    evaluate_downstream_degradation,
)
from adaptive_elastic_sae.training.metrics import (
    activation_effective_sample_size,
    adaptive_weight_summary,
    active_gram_spectrum,
    compute_cross_leverage,
    dead_neuron_recovery_rate,
    dead_neurons_pct,
    explained_variance,
    dictionary_coherence_summary,
    feature_utilization_summary,
    feature_shrinkage_ratio,
    interaction_leakage_frobenius_approx,
    l0_active_features,
    l0_vs_l1_ratio,
    weight_bimodality_ratio,
)
from adaptive_elastic_sae.training.trainer_utils import BatchProvider

logger = getLogger(__name__)


@dataclass
class LLMTrainerConfig:
    """LLM trainer hyperparameters."""

    num_steps: int = 50_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    warmup_steps: int = 10_000
    max_activations_window: int = 10_000
    log_interval: int = 100
    validation_log_interval: int = 1_000
    geometry_log_interval: int = 2_000
    validation_num_batches: int = 100
    online_validation_enabled: bool = True
    final_validation_enabled: bool = True
    final_validation_num_batches: int | None = None
    validation_ablation_mode: str = "batch_mean"
    enable_validation: bool = False
    device: str | torch.device = "cuda"
    dtype: torch.dtype = torch.float32


class LLMSAETrainer:
    """Trainer for LLM-based SAE experiments with validation sets."""

    def __init__(
        self,
        model: BaseSAE,
        config: LLMTrainerConfig,
        batch_provider: BatchProvider,
        validation_provider_online: BatchProvider | None = None,
        validation_provider_final: BatchProvider | None = None,
        llm: Any | None = None,
        hook_name: str | None = None,
        validation_token_streamer: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.batch_provider = batch_provider
        self.validation_provider_online = validation_provider_online
        self.validation_provider_final = validation_provider_final

        # Optional direct LLM handles for downstream CE/KL patch metrics.
        inferred_streamer = getattr(batch_provider, "streamer", None)
        self.llm = llm if llm is not None else getattr(inferred_streamer, "model", None)
        inferred_hook = getattr(inferred_streamer, "hook_name", None)
        self.hook_name = hook_name if hook_name is not None else inferred_hook
        self.validation_token_streamer = (
            validation_token_streamer
            if validation_token_streamer is not None
            else getattr(validation_provider_online, "streamer", None)
        )

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # Track max feature activations over windows for persistent dead-neuron metrics.
        self.max_activations = torch.zeros(
            model.d_dict,
            device=self.device,
            dtype=config.dtype,
        )
        self.max_activations_history: list[torch.Tensor] = []

    def train(
        self,
        use_wandb: bool = False,
        run_name: str = "llm-sae-train",
        wandb_config: dict | None = None,
        run_metadata: dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Run full training with online and final validation."""
        if wandb_config is None:
            wandb_config = {}
        if run_metadata is None:
            run_metadata = {}

        run_config = asdict(self.config)
        run_config["device"] = str(run_config["device"])
        run_config["dtype"] = str(run_config["dtype"])
        run_config.update(run_metadata)

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_config.get("project", "adaptive-elastic-sae"),
                    entity=wandb_config.get("entity"),
                    name=run_name,
                    tags=wandb_config.get("tags", []),
                    config=run_config,
                    reinit="finish_previous",
                )
                # Use training step as the canonical x-axis for all metrics.
                wandb.define_metric("step")
                wandb.define_metric("*", step_metric="step")
            except ImportError:
                use_wandb = False

        self.model.train()
        metrics_history = []
        measured_flops_per_step: float | None = None
        last_log_time = time.time()
        seq_len = self._infer_seq_len(default=128)

        for step in range(self.config.num_steps):
            # Generate batch
            batch = self.batch_provider.next_batch(
                batch_size=self.config.batch_size,
                device=self.device,
                dtype=self.config.dtype,
            )
            x = batch["x"]

            if step == 0:
                flops_metrics = measure_training_step_flops(
                    model=self.model,
                    optimizer=self.optimizer,
                    x=x,
                    warmup_steps=self.config.warmup_steps,
                )
                measured_flops_per_step = flops_metrics.get("measured_flops_per_step")
                run_config.update(flops_metrics)
                if use_wandb:
                    wandb.config.update(flops_metrics, allow_val_change=True)

            # Forward
            x_hat, h = self.model.forward(x)

            # Track max activations for dead-neuron window metrics without
            # retaining autograd history across training steps.
            with torch.no_grad():
                self.max_activations = torch.maximum(
                    self.max_activations,
                    h.detach().abs().amax(dim=0),
                )

            # Loss (support both Tensor and (Tensor, dict) returns)
            loss_components: dict[str, float] = {}
            try:
                loss_out = self.model.compute_loss(
                    x,
                    x_hat,
                    h,
                    warmup_steps=self.config.warmup_steps,
                )
            except TypeError:
                loss_out = self.model.compute_loss(x, x_hat, h)

            if isinstance(loss_out, tuple):
                loss, loss_components = loss_out
            else:
                loss = loss_out
                loss_components = {"loss_total": loss.item()}

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Normalize decoder
            self.model.normalize_decoder()

            # Periodic logging
            if (step + 1) % self.config.log_interval == 0:
                now = time.time()
                elapsed = max(now - last_log_time, 1e-12)
                steps_per_sec = self.config.log_interval / elapsed
                tokens_per_sec = steps_per_sec * self.config.batch_size * seq_len
                last_log_time = now

                metrics = {
                    "step": step + 1,
                    "loss": loss.item(),
                    "throughput/steps_per_sec": steps_per_sec,
                    "throughput/tokens_per_sec": tokens_per_sec,
                    "dead_neurons_pct_window": dead_neurons_pct(
                        self.max_activations,
                        eps=1e-12,
                    ),
                }
                metrics.update(self._compute_grad_health())
                metrics.update(self._compute_lightweight_metrics(x, x_hat, h))
                metrics.update(
                    flops_progress_metrics(
                        measured_flops_per_step=measured_flops_per_step,
                        step=step + 1,
                    )
                )
                tokens_seen = getattr(self.batch_provider, "tokens_seen", None)
                if tokens_seen is not None:
                    metrics["tokens_seen"] = float(tokens_seen)
                metrics.update(loss_components)
                metrics_history.append(metrics)

                if use_wandb:
                    wandb.log(metrics, step=step + 1)

                logger.info(
                    f"Step {step + 1}/{self.config.num_steps} | "
                    f"Loss: {loss.item():.6f}"
                )

            # Windowed dead-neuron recovery logging.
            if (step + 1) % self.config.max_activations_window == 0:
                self.max_activations_history.append(self.max_activations.clone())

                window_metrics = {
                    "step": step + 1,
                    "dead_neurons_pct_window": dead_neurons_pct(
                        self.max_activations,
                        eps=1e-12,
                    ),
                }
                if len(self.max_activations_history) >= 2:
                    recovery_rate = dead_neuron_recovery_rate(
                        self.max_activations_history[-2],
                        self.max_activations_history[-1],
                        eps=1e-12,
                    )
                    window_metrics["dead_neuron_recovery_rate"] = recovery_rate
                    window_metrics["dead_neuron_recovery_pct"] = 100.0 * recovery_rate

                if use_wandb:
                    wandb.log(window_metrics, step=step + 1)

                self.max_activations.zero_()

            # LLM evals
            if (
                self.config.enable_validation
                and self.config.online_validation_enabled
                and self.validation_provider_online
                and (step + 1) % self.config.validation_log_interval == 0
            ):
                val_metrics = self._evaluate_on_validation(
                    self.validation_provider_online,
                    n_batches=self.config.validation_num_batches,
                    label="val_online",
                )
                if val_metrics.get("val_online_valid_batches", 1.0) == 0.0:
                    logger.warning(
                        "Online validation produced 0 valid batches at step %d; "
                        "check val_online_invalid_* metrics in W&B for root cause.",
                        step + 1,
                    )
                if use_wandb and val_metrics:
                    wandb.log({**val_metrics, "step": step + 1}, step=step + 1)

            # Geometry evals
            if (
                self.config.geometry_log_interval > 0
                and (step + 1) % self.config.geometry_log_interval == 0
            ):
                geometry_metrics = self._evaluate_geometry(h)
                cond = geometry_metrics.get("active_condition_number", 0.0)
                if cond > 0:
                    geometry_metrics["log_active_condition_number"] = math.log10(cond)
                if use_wandb and geometry_metrics:
                    wandb.log({**geometry_metrics, "step": step + 1}, step=step + 1)

        # Final validation on exhaustive set
        if (
            self.config.enable_validation
            and self.config.final_validation_enabled
            and self.validation_provider_final
        ):
            final_metrics = self._evaluate_on_validation(
                self.validation_provider_final,
                n_batches=self.config.final_validation_num_batches,
                label="val_final",
            )
            if use_wandb:
                wandb.log(
                    {**final_metrics, "step": self.config.num_steps},
                    step=self.config.num_steps,
                )

            metrics_history.append({"step": "final", **final_metrics})

        saved_checkpoint_path: str | None = None
        if checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "trainer_config": asdict(self.config),
                    "run_name": run_name,
                    "run_metadata": run_metadata,
                    "metrics_history": metrics_history,
                },
                checkpoint_file,
            )
            saved_checkpoint_path = str(checkpoint_file)
            logger.info(f"Saved final checkpoint: {saved_checkpoint_path}")

        if use_wandb:
            wandb.finish()

        return {
            "metrics_history": metrics_history,
            "model_state": self.model.state_dict(),
            "checkpoint_path": saved_checkpoint_path,
        }

    @torch.no_grad()
    def _compute_lightweight_metrics(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
    ) -> dict[str, float]:
        """Cheap per-batch metrics that do not require extra LLM forward passes or linear algebra."""
        eps = 1e-12
        dead_batch_pct = 100.0 * (h.abs().max(dim=0).values <= eps).float().mean().item()

        metrics = {
            "recon_mse": ((x - x_hat) ** 2).mean().item(),
            "explained_variance": explained_variance(x, x_hat, eps=eps),
            "feature_shrinkage_ratio": feature_shrinkage_ratio(x, x_hat, eps=eps),
            "l0_active_features": l0_active_features(h, eps=eps),
            "l0_vs_l1_ratio": l0_vs_l1_ratio(h, eps=eps),
            "dead_features_pct_batch": dead_batch_pct,
        }

        # Prefer persistent firing-EMA utilization when available; fallback to batch frequencies.
        if hasattr(self.model, "firing_ema"):
            firing_rates = self.model.firing_ema
            util_prefix = "feature_utilization_ema"
        else:
            firing_rates = (h.abs() > eps).float().mean(dim=0)
            util_prefix = "feature_utilization_batch"
        metrics.update(feature_utilization_summary(firing_rates, prefix=util_prefix, eps=eps))

        if hasattr(self.model, "ema_abs_activations"):
            metrics["activation_effective_sample_size"] = activation_effective_sample_size(
                self.model.ema_abs_activations,
                eps=eps,
            )

        if hasattr(self.model, "get_adaptive_weights"):
            try:
                weights = self.model.get_adaptive_weights()
                metrics["weight_bimodality_ratio"] = weight_bimodality_ratio(
                    weights,
                    delta_sig=0.5,
                    delta_noise=5.0,
                    eps=eps,
                )
            except Exception:
                pass

        return metrics

    @torch.no_grad()
    def _compute_grad_health(self) -> dict[str, float]:
        """O(N) gradient diagnostics and update-to-weight ratios."""
        metrics: dict[str, float] = {}
        total_grad_sq = 0.0
        enc_grad_sq = 0.0
        dec_grad_sq = 0.0
        enc_weight_sq = 0.0
        dec_weight_sq = 0.0

        for name, param in self.model.named_parameters():
            weight_sq = param.detach().pow(2).sum().item()
            is_encoder = "encoder" in name
            is_decoder = "decoder" in name or "W_dec" in name

            if is_encoder:
                enc_weight_sq += weight_sq
            elif is_decoder:
                dec_weight_sq += weight_sq

            if param.grad is None:
                continue

            grad_sq = param.grad.detach().pow(2).sum().item()
            total_grad_sq += grad_sq
            if is_encoder:
                enc_grad_sq += grad_sq
            elif is_decoder:
                dec_grad_sq += grad_sq

        enc_grad = enc_grad_sq**0.5
        dec_grad = dec_grad_sq**0.5
        enc_weight = enc_weight_sq**0.5
        dec_weight = dec_weight_sq**0.5

        metrics["grad_norm/global"] = total_grad_sq**0.5
        metrics["grad_norm/encoder"] = enc_grad
        metrics["grad_norm/decoder"] = dec_grad
        metrics["update_ratio/encoder"] = (
            self.config.learning_rate * enc_grad / max(enc_weight, 1e-8)
        )
        metrics["update_ratio/decoder"] = (
            self.config.learning_rate * dec_grad / max(dec_weight, 1e-8)
        )
        return metrics

    def _infer_seq_len(self, default: int = 128) -> int:
        """Infer token sequence length from streamer config when available."""
        streamer = getattr(self.batch_provider, "streamer", None)
        seq_len = getattr(getattr(streamer, "cfg", None), "seq_len", default)
        try:
            return max(int(seq_len), 1)
        except (TypeError, ValueError):
            return default

    @torch.no_grad()
    def _evaluate_on_validation(
        self,
        provider: BatchProvider,
        n_batches: int | None = None,
        label: str = "validation",
    ) -> dict[str, float]:
        """
        Evaluate reconstruction loss and KL divergence on validation set.

        Args:
            provider: BatchProvider yielding validation batches with key "x" and optionally "logits_original"
            n_batches: Number of batches to eval. If None, drain provider.
            label: Prefix for metric names in output dict

        Returns:
            dict with keys like "{label}_recon_loss", "{label}_kl_div", etc.
        """
        self.model.eval()

        # Prefer true downstream degradation via model patching when a token streamer is available.
        # Use the provider streamer first so each split (online/final) evaluates on its own dataset window.
        token_streamer = getattr(provider, "streamer", None)
        if token_streamer is None:
            token_streamer = self.validation_token_streamer

        # Non-loop validation streams (e.g., final validation) are finite and can be
        # exhausted by prior runs/variants. Reset so each evaluation gets a fresh pass.
        loop_dataset = getattr(getattr(token_streamer, "cfg", None), "loop_dataset", True)
        if token_streamer is not None and not loop_dataset and hasattr(token_streamer, "reset_stream"):
            try:
                token_streamer.reset_stream()
            except Exception as e:
                logger.warning(
                    "%s: failed to reset non-loop validation stream before eval (%s)",
                    label,
                    e,
                )

        if (
            self.llm is not None
            and self.hook_name is not None
            and token_streamer is not None
        ):
            downstream_results: list[dict[str, float]] = []

            # Check SAE health before validation
            has_nan = False
            has_inf = False
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    has_nan = True
                    logger.debug(f"SAE parameter {name} contains NaN")
                if torch.isinf(param).any():
                    has_inf = True
                    logger.debug(f"SAE parameter {name} contains Inf")
            if has_nan or has_inf:
                logger.warning(f"SAE health check FAILED: nan={has_nan}, inf={has_inf}, returning empty metrics")
                self.model.train()
                return {}

            # If split loops forever and n_batches is unspecified, cap for safety.
            eval_batches: int | None
            if n_batches is not None:
                eval_batches = n_batches
            elif loop_dataset:
                eval_batches = 100
            else:
                eval_batches = None

            batch_count = 0
            while eval_batches is None or batch_count < eval_batches:
                try:
                    tokens = token_streamer.next_token_batch()
                except StopIteration:
                    break

                result = evaluate_downstream_degradation(
                    self.llm,
                    self.model,
                    tokens,
                    self.hook_name,
                    ablation_mode=self.config.validation_ablation_mode,
                    verbose_nan_debug=False,
                )
                downstream_results.append(result)
                batch_count += 1

            if batch_count == 0:
                # One defensive retry for finite streams: transient iterator state can
                # occasionally produce an empty read despite available documents.
                if token_streamer is not None and not loop_dataset and hasattr(token_streamer, "reset_stream"):
                    try:
                        token_streamer.reset_stream()
                        tokens = token_streamer.next_token_batch()
                        result = evaluate_downstream_degradation(
                            self.llm,
                            self.model,
                            tokens,
                            self.hook_name,
                            ablation_mode=self.config.validation_ablation_mode,
                            verbose_nan_debug=False,
                        )
                        downstream_results.append(result)
                        batch_count = 1
                    except StopIteration:
                        pass
                    except Exception as e:
                        logger.warning(
                            "%s: validation retry after reset failed (%s)",
                            label,
                            e,
                        )

            if batch_count == 0:
                cfg = getattr(token_streamer, "cfg", None)
                logger.warning(
                    "%s: validation produced zero batches (loop_dataset=%s, skip_docs=%s, take_docs=%s)",
                    label,
                    loop_dataset,
                    getattr(cfg, "skip_docs", None),
                    getattr(cfg, "take_docs", None),
                )
                self.model.train()
                return {
                    f"{label}_valid_batches": 0.0,
                    f"{label}_skipped_batches": 0.0,
                    f"{label}_empty_stream": 1.0,
                }

            self.model.train()
            metrics: dict[str, float] = aggregate_downstream_degradation(
                downstream_results,
                label,
                verbose_nan_debug=False,
            )

            if hasattr(self.model, "get_adaptive_weights"):
                try:
                    weights = self.model.get_adaptive_weights().detach().float()
                    metrics.update(
                        adaptive_weight_summary(
                            weights,
                            prefix=f"{label}_adaptive_weight",
                            weight_min=getattr(self.model, "weight_min", None),
                            weight_max=getattr(self.model, "weight_max", None),
                        )
                    )
                except Exception:
                    pass

            return metrics

        recon_losses = []
        kl_divs = []

        batch_count = 0
        while True:
            try:
                batch = provider.next_batch(
                    batch_size=self.config.batch_size,
                    device=self.device,
                    dtype=self.config.dtype,
                )
            except StopIteration:
                break

            x = batch["x"]
            x_hat, h = self.model.forward(x)

            # Reconstruction loss (MSE on residual stream)
            recon_loss = ((x - x_hat) ** 2).mean().item()
            recon_losses.append(recon_loss)

            # KL divergence (if LM logits provided)
            if "logits_original" in batch and "logits_sae" in batch:
                logits_orig = batch["logits_original"]  # Shape: (batch, vocab_size)
                logits_sae = batch["logits_sae"]  # SAE-reconstructed logits

                log_probs_orig = torch.log_softmax(logits_orig, dim=-1)
                log_probs_sae = torch.log_softmax(logits_sae, dim=-1)

                kl = (
                    torch.exp(log_probs_orig) * (log_probs_orig - log_probs_sae)
                ).sum(dim=-1).mean()
                kl_divs.append(kl.item())

            batch_count += 1
            if n_batches is not None and batch_count >= n_batches:
                break

        self.model.train()

        return aggregate_reconstruction_validation(recon_losses, kl_divs, label)

    @torch.no_grad()
    def _evaluate_geometry(
        self,
        h_batch: torch.Tensor,
        eps: float = 1e-12,
    ) -> dict[str, float]:
        """Heavy geometric evaluations, logged sparingly alongside validation."""
        # Safely extract decoder dictionary D in shape (n_dim, d_dict)
        if hasattr(self.model, "W_dec"):
            d = self.model.W_dec
            if d.shape[0] == self.model.d_dict:
                d = d.T
        elif hasattr(self.model, "decoder") and hasattr(self.model.decoder, "weight"):
            d = self.model.decoder.weight
        else:
            return {}

        # Prefer EMA activity for global active set if available, fallback to batch activity
        if hasattr(self.model, "ema_abs_activations"):
            active_mask = self.model.ema_abs_activations > eps
        else:
            active_mask = h_batch.abs().max(dim=0).values > eps

        if not active_mask.any():
            return {
                "active_min_eigenvalue": 0.0,
                "active_max_eigenvalue": 0.0,
                "active_condition_number": 0.0,
                "interaction_leakage_frobenius": 0.0,
                "mean_max_cosine_similarity": 0.0,
                "mean_h_j": 0.0,
                "total_h_j": 0.0,
            }

        metrics: dict[str, float] = {}
        metrics.update(active_gram_spectrum(d, active_mask, eps=eps))
        metrics["interaction_leakage_frobenius"] = interaction_leakage_frobenius_approx(
            d, active_mask
        )
        metrics.update(dictionary_coherence_summary(d, eps=eps))

        leverage_stats = compute_cross_leverage(d, active_mask, eps=eps)
        metrics["mean_h_j"] = float(leverage_stats["mean_h_j"])
        metrics["total_h_j"] = float(leverage_stats["total_h_j"])

        return metrics

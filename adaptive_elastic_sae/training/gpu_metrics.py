from __future__ import annotations

import copy
from typing import Any

import torch


def measure_training_step_flops(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    warmup_steps: int,
) -> dict[str, float]:
    """Measure real FLOPs for one full training step with torch.profiler."""
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return {}

    activities = [ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    model_snapshot = copy.deepcopy(model.state_dict())

    optimizer.zero_grad(set_to_none=True)
    try:
        with profile(activities=activities, with_flops=True) as prof:
            x_hat, h = model.forward(x)
            try:
                loss_out = model.compute_loss(
                    x,
                    x_hat,
                    h,
                    warmup_steps=warmup_steps,
                )
            except TypeError:
                loss_out = model.compute_loss(x, x_hat, h)

            if isinstance(loss_out, tuple):
                loss, _ = loss_out
            else:
                loss = loss_out

            loss.backward()

        flops = float(prof.key_averages().total_average().flops)
        if flops <= 0.0:
            return {}
        return {
            "measured_flops_per_step": flops,
            "measured_gflops_per_step": flops / 1e9,
        }
    except Exception:
        return {}
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.load_state_dict(model_snapshot)


def flops_progress_metrics(
    *,
    measured_flops_per_step: float | None,
    step: int,
) -> dict[str, Any]:
    """Compute cumulative FLOPs metrics for the current step index."""
    if measured_flops_per_step is None:
        return {}

    cumulative = measured_flops_per_step * float(step)
    return {
        "measured_flops_per_step": measured_flops_per_step,
        "measured_gflops_per_step": measured_flops_per_step / 1e9,
        "measured_cumulative_flops": cumulative,
        "measured_cumulative_tflops": cumulative / 1e12,
    }

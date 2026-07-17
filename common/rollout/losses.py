from collections.abc import Callable

import torch

from common.rollout.metrics import mse, per_step_mse
from torch import Tensor
from common.rollout.types import ScalarTensor


def rollout_mse(predicted: Tensor, target: Tensor) -> ScalarTensor:
    return mse(predicted, target)


def weighted_rollout_mse(
    predicted: Tensor,
    target: Tensor,
    weights: Tensor | None = None,
) -> ScalarTensor:
    losses = per_step_mse(predicted, target)

    if weights is None:
        return losses.mean()

    weights = weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def final_step_mse(predicted: Tensor, target: Tensor) -> ScalarTensor:
    return mse(predicted[-1], target[-1])


def _broadcast_dynamics_scale(scale: Tensor, reference: Tensor) -> Tensor:
    """Reshape a scalar/per-coordinate scale for ``[time, bodies, dim]`` data."""

    flat_scale = scale.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if flat_scale.numel() == 1:
        return flat_scale[0]
    if flat_scale.numel() == reference.shape[-1]:
        return flat_scale.reshape(1, 1, reference.shape[-1])
    if flat_scale.numel() == reference.shape[-2] * reference.shape[-1]:
        return flat_scale.reshape(1, reference.shape[-2], reference.shape[-1])
    raise ValueError(
        "Dynamics normalization scale must be scalar, one value per spatial "
        "coordinate, or one value per body coordinate. Got "
        f"{flat_scale.numel()} values for errors with shape {tuple(reference.shape)}."
    )


def teacher_forced_dynamics_mse(
    positions: Tensor,
    velocities: Tensor,
    target_accelerations: Tensor,
    masses: Tensor,
    predict_acceleration: Callable[[Tensor, Tensor], Tensor],
    normalization_scale: Tensor,
    *,
    target_kind: str,
) -> ScalarTensor:
    """Measure the learned vector field on true states in a rollout window.

    ``positions``, ``velocities``, and ``target_accelerations`` contain the
    states that generate transitions, so they normally have length equal to
    the rollout horizon (the final target state is excluded). The prediction
    callback receives one position/velocity state at a time. Force-supervised
    models are compared after multiplying acceleration error by body mass.
    """

    if positions.shape != velocities.shape or positions.shape != target_accelerations.shape:
        raise ValueError(
            "Dynamics-loss positions, velocities, and accelerations must have "
            f"matching shapes. Got {positions.shape}, {velocities.shape}, and "
            f"{target_accelerations.shape}."
        )
    if target_kind not in {"acceleration", "force"}:
        raise ValueError(
            "target_kind must be 'acceleration' or 'force', got "
            f"{target_kind!r}."
        )

    predicted_accelerations = torch.stack(
        [predict_acceleration(position, velocity) for position, velocity in zip(positions, velocities)],
        dim=0,
    )
    dynamics_error = predicted_accelerations - target_accelerations
    if target_kind == "force":
        dynamics_error = dynamics_error * masses.unsqueeze(0)

    scale = _broadcast_dynamics_scale(normalization_scale, dynamics_error)
    return (dynamics_error / scale).square().mean()


def dynamics_to_position_scale(
    baseline_position_mse: Tensor | float,
    baseline_dynamics_mse: Tensor | float,
) -> float:
    """Scale normalized dynamics MSE into raw rollout-position-MSE units."""

    position_value = float(torch.as_tensor(baseline_position_mse).detach().cpu())
    dynamics_value = float(torch.as_tensor(baseline_dynamics_mse).detach().cpu())
    if not position_value > 0.0 or not dynamics_value > 0.0:
        raise ValueError(
            "Baseline position and dynamics MSE values must both be positive. "
            f"Got position={position_value} and dynamics={dynamics_value}."
        )
    return position_value / dynamics_value


def composite_rollout_loss(
    position_mse: Tensor,
    *,
    velocity_mse: Tensor | None = None,
    velocity_loss_weight: float = 0.0,
    dynamics_mse: Tensor | None = None,
    dynamics_loss_weight: float = 0.0,
    dynamics_position_scale: float = 1.0,
) -> ScalarTensor:
    """Combine trajectory and vector-field losses without changing old scales.

    Velocity weighting retains its historical raw-MSE meaning. The normalized
    dynamics loss is multiplied by a fixed parent-checkpoint calibration ratio
    so a dynamics weight of one initially contributes about as much as position
    MSE rather than overwhelming it by several orders of magnitude.
    """

    loss = position_mse
    if velocity_loss_weight > 0.0:
        if velocity_mse is None:
            raise ValueError("velocity_mse is required when velocity loss is enabled.")
        loss = loss + velocity_loss_weight * velocity_mse
    if dynamics_loss_weight > 0.0:
        if dynamics_mse is None:
            raise ValueError("dynamics_mse is required when dynamics loss is enabled.")
        loss = loss + dynamics_loss_weight * dynamics_position_scale * dynamics_mse
    return loss


def sample_rollout_windows(
    trajectory_indices: list[int],
    *,
    num_steps: int,
    rollout_horizon: int,
    num_samples: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Choose reproducible global trajectory/time windows without data leakage."""

    if not trajectory_indices:
        raise ValueError("At least one trajectory index is required.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if rollout_horizon <= 0 or rollout_horizon >= num_steps:
        raise ValueError(
            "rollout_horizon must be positive and smaller than num_steps."
        )

    generator = torch.Generator().manual_seed(seed)
    local_trajectories = torch.randint(
        0,
        len(trajectory_indices),
        (num_samples,),
        generator=generator,
    )
    times = torch.randint(
        0,
        num_steps - rollout_horizon,
        (num_samples,),
        generator=generator,
    )
    return [
        (trajectory_indices[local_index], time_index)
        for local_index, time_index in zip(
            local_trajectories.tolist(),
            times.tolist(),
        )
    ]

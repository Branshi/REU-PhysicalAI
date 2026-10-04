import torch
from torch import Tensor

from common.rollout.integrators import rk4_step
from common.rollout.types import (
    BodyDimensionTensor,
    BodyTensor,
    StateTensor,
    StepFn,
)


def pack_mlp_state(
    positions: BodyDimensionTensor,
    velocities: BodyDimensionTensor,
) -> StateTensor:
    if positions.shape != velocities.shape:
        raise ValueError(
            "positions and velocities must have the same shape. "
            f"Got {positions.shape} and {velocities.shape}."
        )

    spatial_dim = positions.shape[-2] * positions.shape[-1]
    q = positions.reshape(*positions.shape[:-2], spatial_dim)
    v = velocities.reshape(*velocities.shape[:-2], spatial_dim)

    if q.ndim == 1:
        q = q.unsqueeze(0)
        v = v.unsqueeze(0)

    return torch.cat([q, v], dim=-1)


def unpack_mlp_state(
    state: StateTensor,
    num_bodies: int,
    dim: int,
) -> tuple[Tensor, Tensor]:
    if num_bodies <= 0:
        raise ValueError("num_bodies must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    spatial_dim = num_bodies * dim
    if state.shape[-1] != 2 * spatial_dim:
        raise ValueError(
            "Packed MLP state must have size 2 * num_bodies * dim along the "
            f"last axis. Got last dimension {state.shape[-1]} for "
            f"num_bodies={num_bodies}, dim={dim}."
        )

    positions = state[..., :spatial_dim].reshape(
        *state.shape[:-1], num_bodies, dim
    )
    velocities = state[..., spatial_dim:].reshape(
        *state.shape[:-1], num_bodies, dim
    )
    return positions, velocities


def make_mlp_acceleration_fn(
    model,
    masses: BodyTensor,
    acc_mean: Tensor,
    acc_std: Tensor,
    num_bodies: int,
    dim: int,
):
    if num_bodies <= 0:
        raise ValueError("num_bodies must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    mass_features = masses.reshape(num_bodies)

    def expand_mass_features(q):
        masses_t = mass_features.to(device=q.device, dtype=q.dtype)
        view_shape = (1,) * (q.ndim - 1) + (num_bodies,)
        return masses_t.reshape(view_shape).expand(*q.shape[:-1], num_bodies)

    def expand_acceleration_stat(stat, positions, name):
        values = stat.reshape(-1).to(
            device=positions.device,
            dtype=positions.dtype,
        )
        if values.numel() == 1:
            return values.reshape((1,) * positions.ndim)

        spatial_dim = num_bodies * dim
        if values.numel() != spatial_dim:
            raise ValueError(
                f"{name} must be scalar or have one value per acceleration "
                f"coordinate. Got {values.numel()} values for "
                f"num_bodies={num_bodies}, dim={dim}."
            )

        view_shape = (1,) * (positions.ndim - 2) + (num_bodies, dim)
        return values.reshape(view_shape)

    def predict_acceleration(positions, velocities):
        if positions.shape != velocities.shape:
            raise ValueError(
                "positions and velocities must have the same shape. "
                f"Got {positions.shape} and {velocities.shape}."
            )

        spatial_dim = num_bodies * dim
        q = positions.reshape(*positions.shape[:-2], spatial_dim)
        v = velocities.reshape(*velocities.shape[:-2], spatial_dim)
        model_input = torch.cat([q, v, expand_mass_features(q)], dim=-1)

        predicted_normalized = model(model_input).reshape(
            *positions.shape[:-2],
            num_bodies,
            dim,
        )
        mean = expand_acceleration_stat(acc_mean, positions, "acc_mean")
        std = expand_acceleration_stat(acc_std, positions, "acc_std")
        return predicted_normalized * std + mean

    return predict_acceleration


def make_mlp_step_fn(
    model,
    masses: BodyTensor,
    acc_mean: Tensor,
    acc_std: Tensor,
    dt: float,
    num_bodies: int,
    dim: int,
) -> StepFn:
    predict_acceleration = make_mlp_acceleration_fn(
        model=model,
        masses=masses,
        acc_mean=acc_mean,
        acc_std=acc_std,
        num_bodies=num_bodies,
        dim=dim,
    )

    def dynamics(state: StateTensor) -> StateTensor:
        positions, velocities = unpack_mlp_state(
            state,
            num_bodies=num_bodies,
            dim=dim,
        )
        accelerations = predict_acceleration(positions, velocities)
        return pack_mlp_state(velocities, accelerations)

    def step_fn(state: StateTensor) -> StateTensor:
        return rk4_step(state, dynamics, dt)

    return step_fn

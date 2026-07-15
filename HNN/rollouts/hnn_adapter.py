import torch
from torch import Tensor

from common.rollout.integrators import velocity_verlet_step
from common.rollout.types import (
    BodyDimensionTensor,
    BodyTensor,
    StateTensor,
    StepFn,
)


def pack_hnn_state(
    positions: BodyDimensionTensor, velocities: BodyDimensionTensor
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


def unpack_hnn_state(
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
            "Packed HNN state must have size 2 * num_bodies * dim along the "
            f"last axis. Got last dimension {state.shape[-1]} for "
            f"num_bodies={num_bodies}, dim={dim}."
        )

    q = state[..., :spatial_dim]
    v = state[..., spatial_dim:]

    positions = q.reshape(*state.shape[:-1], num_bodies, dim)
    velocities = v.reshape(*state.shape[:-1], num_bodies, dim)

    return positions, velocities


def make_hnn_step_fn(
    model,
    masses: BodyTensor,
    force_std: Tensor,
    dt: float,
    num_bodies: int,
    dim: int,
) -> StepFn:
    if num_bodies <= 0:
        raise ValueError("num_bodies must be positive.")

    if dim <= 0:
        raise ValueError("dim must be positive.")

    mass_features = masses.reshape(num_bodies)
    mass_columns = masses.reshape(num_bodies, 1)
    force_scale = force_std.reshape(-1)

    def expand_mass_features(q):
        masses_t = mass_features.to(device=q.device, dtype=q.dtype)
        view_shape = (1,) * (q.ndim - 1) + (num_bodies,)
        return masses_t.reshape(view_shape).expand(*q.shape[:-1], num_bodies)

    def expand_mass_columns(positions):
        masses_t = mass_columns.to(device=positions.device, dtype=positions.dtype)
        view_shape = (1,) * (positions.ndim - 2) + (num_bodies, 1)
        return masses_t.reshape(view_shape).expand(
            *positions.shape[:-2],
            num_bodies,
            1,
        )

    def expand_force_scale(positions):
        scale = force_scale.to(device=positions.device, dtype=positions.dtype)
        if scale.numel() == 1:
            view_shape = (1,) * positions.ndim
            return scale.reshape(view_shape)

        if scale.numel() != num_bodies * dim:
            raise ValueError(
                "force_std must be scalar or have one value per position coordinate. "
                f"Got {scale.numel()} values for num_bodies={num_bodies}, dim={dim}."
            )

        view_shape = (1,) * (positions.ndim - 2) + (num_bodies, dim)
        return scale.reshape(view_shape)

    def predict_acceleration(positions):
        spatial_dim = num_bodies * dim
        q = positions.reshape(*positions.shape[:-2], spatial_dim)
        model_input = torch.cat([q, expand_mass_features(q)], dim=-1)

        predicted_force_normalized = model.force(model_input)
        predicted_force = predicted_force_normalized.reshape(
            *positions.shape[:-2],
            num_bodies,
            dim,
        )
        predicted_force = predicted_force * expand_force_scale(positions)

        return predicted_force / expand_mass_columns(positions)

    def step_fn(state: StateTensor) -> StateTensor:
        positions, velocities = unpack_hnn_state(
            state,
            num_bodies=num_bodies,
            dim=dim,
        )
        next_positions, next_velocities, _ = velocity_verlet_step(
            positions,
            velocities,
            predict_acceleration,
            dt,
        )
        return pack_hnn_state(next_positions, next_velocities)

    return step_fn

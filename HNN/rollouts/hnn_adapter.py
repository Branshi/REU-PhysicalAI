import torch
from torch import Tensor

from common.rollout.integrators import rk4_step
from common.rollout.types import (
    BodyDimensionTensor,
    BodyTensor,
    IntegratorFn,
    StateTensor,
    StepFn,
)


def pack_hnn_state(
    positions: BodyDimensionTensor, momenta: BodyDimensionTensor
) -> StateTensor:
    if positions.shape != momenta.shape:
        raise ValueError(
            "positions and momenta must have the same shape. "
            f"Got {positions.shape} and {momenta.shape}."
        )

    q_dim = positions.shape[-2] * positions.shape[-1]
    q = positions.reshape(*positions.shape[:-2], q_dim)
    p = momenta.reshape(*momenta.shape[:-2], q_dim)

    if q.ndim == 1:
        q = q.unsqueeze(0)
        p = p.unsqueeze(0)

    return torch.cat([q, p], dim=-1)


def unpack_hnn_state(
    state: StateTensor,
    num_bodies: int,
    dim: int,
) -> tuple[Tensor, Tensor]:
    if num_bodies <= 0:
        raise ValueError("num_bodies must be positive.")

    if dim <= 0:
        raise ValueError("dim must be positive.")

    q_dim = num_bodies * dim

    if state.shape[-1] != 2 * q_dim:
        raise ValueError(
            "Packed HNN state must have size 2 * num_bodies * dim along the "
            f"last axis. Got last dimension {state.shape[-1]} for "
            f"num_bodies={num_bodies}, dim={dim}."
        )

    q = state[..., :q_dim]
    p = state[..., q_dim:]

    positions = q.reshape(*state.shape[:-1], num_bodies, dim)
    momenta = p.reshape(*state.shape[:-1], num_bodies, dim)

    return positions, momenta


def make_hnn_step_fn(
    model,
    masses: BodyTensor,
    dt: float,
    num_bodies: int,
    integrator: IntegratorFn = rk4_step,
) -> StepFn:
    if num_bodies <= 0:
        raise ValueError("num_bodies must be positive.")

    m = masses.reshape(1, num_bodies)

    def dynamics(state):
        if m.device != state.device or m.dtype != state.dtype:
            masses_t = m.to(device=state.device, dtype=state.dtype)
        else:
            masses_t = m

        while masses_t.ndim < state.ndim:
            masses_t = masses_t.unsqueeze(0)

        masses_t = masses_t.expand(*state.shape[:-1], num_bodies)
        model_input = torch.cat([state, masses_t], dim=-1)
        return model.time_derivative(model_input)

    def step_fn(state: StateTensor) -> StateTensor:
        if state.shape[-1] % 2 != 0:
            raise ValueError(
                "Packed HNN state must contain equal q and p halves. "
                f"Got odd last dimension {state.shape[-1]}."
            )

        q_dim = state.shape[-1] // 2
        if q_dim % num_bodies != 0:
            raise ValueError(
                "Cannot split packed HNN q state evenly across bodies. "
                f"Got q_dim={q_dim} and num_bodies={num_bodies}."
            )

        return integrator(state, dynamics, dt)

    return step_fn

import torch
from torch import Tensor

from common.rollout.types import BodyTensor, StateTensor, StepFn


def pack_gns_state(positions: Tensor, velocities: Tensor) -> StateTensor:
    if positions.shape != velocities.shape:
        raise ValueError(
            "positions and velocities must have the same shape. "
            f"Got {positions.shape} and {velocities.shape}."
        )

    return torch.cat([positions, velocities], dim=-1)


def unpack_gns_state(state: StateTensor, dim: int) -> tuple[Tensor, Tensor]:
    if dim <= 0:
        raise ValueError("dim must be positive.")

    if state.shape[-1] != 2 * dim:
        raise ValueError(
            "Packed GNS state must have size 2 * dim along the last axis. "
            f"Got last dimension {state.shape[-1]} for dim={dim}."
        )

    positions = state[..., :dim]
    velocities = state[..., dim:]

    return positions, velocities


def make_gns_step_fn(
    simulator,
    masses: BodyTensor,
    dim: int | None = None,
) -> StepFn:
    def step_fn(state: StateTensor) -> StateTensor:
        state_dim = dim
        if state_dim is None:
            if state.shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot infer position dimension from a packed GNS state "
                    f"with odd last dimension {state.shape[-1]}."
                )
            state_dim = state.shape[-1] // 2

        positions, velocities = unpack_gns_state(state, state_dim)

        next_positions, next_velocities, _ = simulator(
            positions=positions,
            velocities=velocities,
            masses=masses,
        )

        return pack_gns_state(next_positions, next_velocities)

    return step_fn

import torch

from common.rollout.types import StepFn, StateTensor, TrajectoryTensor


def rollout_steps(
    initial_state: StateTensor,
    step_fn: StepFn,
    num_steps: int,
    detach_between_steps: bool = False,
) -> TrajectoryTensor:
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    if detach_between_steps:
        initial_state = initial_state.detach()

    current_state = initial_state
    trajectory = [current_state]

    for _ in range(num_steps):
        next_state = step_fn(current_state)

        if detach_between_steps:
            next_state = next_state.detach()

        trajectory.append(next_state)
        current_state = next_state
    # torch.stack joins tensors along a newly created dimension in this case the time dimension
    return torch.stack(trajectory, dim=0)

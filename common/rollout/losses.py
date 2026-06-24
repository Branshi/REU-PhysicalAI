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

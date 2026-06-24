from collections.abc import Callable
from typing import Any

from torch import Tensor
from jaxtyping import Float

# Generic packed state used by integrators and model adapters.
# Examples: flat HNN [q, p], packed GNS [positions, velocities], or batched forms.
StateTensor = Float[Tensor, "..."]

ScalarTensor = Float[Tensor, ""]

# A stacked sequence of packed states returned by a rollout loop.
TrajectoryTensor = Float[Tensor, "time ..."]

# DynamicsFn returns a time derivative for the current state.
DynamicsFn = Callable[[StateTensor], StateTensor]

# StepFn advances the current state by one discrete rollout step.
StepFn = Callable[[StateTensor], StateTensor]

# AccelerationFn returns acceleration from positions plus optional physics args.
AccelerationFn = Callable[..., StateTensor]

# IntegratorFn advances a state using a continuous-time dynamics function.
IntegratorFn = Callable[[StateTensor, DynamicsFn, float], StateTensor]

# Physical N-body trajectory tensors used by metrics.
BodyStateTensor = Float[Tensor, "*batch time bodies dim"]
BodyTensor = Float[Tensor, "*batch bodies"] | Float[Tensor, "*batch bodies 1"]
BodyDimensionTensor = Float[Tensor, "*batch bodies dim"]
BodyValueTensor = Float[Tensor, "*batch time bodies"]
TimeTensor = Float[Tensor, "*batch time"]
VectorSeriesTensor = Float[Tensor, "*batch time dim"]

# Extra args/kwargs accepted by adapter factories or physics helper functions.
ExtraArgs = tuple[Any, ...]
ExtraKwargs = dict[str, Any]

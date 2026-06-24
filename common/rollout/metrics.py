import torch
from torch import Tensor
from jaxtyping import Float

from common.rollout.types import (
    BodyStateTensor,
    BodyTensor,
    BodyValueTensor,
    TimeTensor,
    VectorSeriesTensor,
    ScalarTensor,
)


def mse(predicted: Tensor, true: Tensor) -> ScalarTensor:
    # or equivalently squared_error.mean() == squared_error.mean(dim=(0,1,2))
    return ((predicted - true) ** 2).mean()


def rmse(predicted: Tensor, true: Tensor) -> ScalarTensor:
    return torch.sqrt(mse(predicted, true))


def per_step_mse(
    predicted: BodyStateTensor,
    true: BodyStateTensor,
) -> TimeTensor:
    squared_error = (predicted - true) ** 2
    # dim=(1,2) averages over bodies and coordinate dimensions, but not over time.
    # squared_error.shape = [300, 3, 2]
    # for example, squared_error[0] has shape [3,2] with 3 bodies, 2 coordinates [[0.1, 0.2], [0.3,0.1], [0.4, 0.3]].
    # squared_error.mean(dim=(1,2)) then averages all these values, 0.1 + 0.2 + 0.3 + 0.1 + 0.4 + 0.3 / 6
    # per_step_mse = [300]
    return squared_error.mean(dim=(-2, -1))


def per_step_mean_error(
    predicted: BodyStateTensor,
    true: BodyStateTensor,
) -> TimeTensor:
    # position_error.shape == [300, 3, 2]
    # distance_error.shape == [300, 3]
    error = predicted - true
    distance_error = torch.linalg.vector_norm(error, dim=-1)
    # distance_error = [num_steps]
    return distance_error.mean(dim=-1)


def mean_final_error(predicted: BodyStateTensor, true: BodyStateTensor) -> ScalarTensor:
    return per_step_mean_error(predicted, true)[..., -1]


def max_step_error(predicted: BodyStateTensor, true: BodyStateTensor) -> ScalarTensor:
    return per_step_mean_error(predicted, true).max(dim=-1).values


# this utility function is because masses typically has the shape [num_bodies, 1], however its more useful to work with [num_bodies].
# torch.squeeze only removes dimensions of size 1, unlike torch.reshape which change the shape of your tensor more generally.
# for example, [3,1].squeeze(-1).shape == [3]
# also, [300,3,2].squeeze().shape == [300,3,2] since there are no dimensions of size 1.
def _squeeze_masses(masses: BodyTensor) -> Float[Tensor, "*batch bodies"]:
    # note that squeezing a [1] vector will turn it into a scalar, but in the code below we only squeeze the last dimension [1,1] so we would get atmost [1].
    # we check to make sure that our tensor shape has more than 1 dimension.
    # ndim is the same as len(masses.shape), which gives you the number of dimension of your tensor.
    if masses.ndim > 1 and masses.shape[-1] == 1:
        return masses.squeeze(-1)
    return masses


# this utility function is used to make sure that the masses match tensors.
# torch.unsqueeze() inserts a new dimension of size 1 at the specificed index, in our case right before the num_bodies.
# So we have masses.unsqueeze(-2) == [1,num_bodies] since we squeezed it into [num_bodies] before unsqueezing it.
# unsqueeze serves the same purpose as None.
# this is done repeatedly until mass.ndim matches body_values.ndim
def _broadcast_masses_to_body_values(
    masses: BodyTensor,
    body_values: BodyValueTensor,
) -> BodyValueTensor:
    masses = _squeeze_masses(masses)
    while masses.ndim < body_values.ndim:
        masses = masses.unsqueeze(-2)
    return masses


def kinetic_energy_from_velocities(
    velocities: BodyStateTensor,
    masses: BodyTensor,
) -> TimeTensor:
    speed_squared = (velocities**2).sum(dim=-1)  # shape: [num_steps, num_bodies]
    masses = _broadcast_masses_to_body_values(masses, speed_squared)  # [1, num_bodies]
    body_kinetic_energy = 0.5 * masses * speed_squared  # broadcasting
    return body_kinetic_energy.sum(dim=-1)


def kinetic_energy_from_momenta(
    momenta: BodyStateTensor,
    masses: BodyTensor,
) -> TimeTensor:
    momentum_squared = (momenta**2).sum(dim=-1)
    masses = _broadcast_masses_to_body_values(masses, momentum_squared)
    body_kinetic_energy = momentum_squared / (2.0 * masses)
    return body_kinetic_energy.sum(dim=-1)


def potential_energy(
    positions: BodyStateTensor,
    masses: BodyTensor,
    G: float = 1.0,
    epsilon: float = 0.15,
) -> TimeTensor:
    masses = _squeeze_masses(masses)
    # None creates a dimension of size 1 for broadcasting purposes
    # the ... syntax says to keep all earlier indices that are not explicitly given. So for example,
    # suppose tensor has shape [20,5,3,2,2]. Then tensor[..., None, None, :, :] has shape [20, 5, 3, 1, 1, 2, 2]
    # in our case ... below only grabs num_steps
    displacement = (
        positions[..., :, None, :] - positions[..., None, :, :]
    )  # computing pairwise displacement of each object.
    distance_squared = (displacement**2).sum(dim=-1)
    softened_distance = torch.sqrt(distance_squared + epsilon**2)

    num_bodies = positions.shape[-2]
    # torch.triu is used to turn a matrix into upper traingular, either by zeroing out entries or setting them to False.
    # the diagonal argument tells the function along which diagonal should values on and above remain true
    # for example diagonal = 0 keeps all values on the main diagonal and above, while diagonal = 1 keeps only everything above the
    # main diagonal which in our case is what we want to avoid self interactions.
    upper_triangle_mask = torch.triu(
        torch.ones(num_bodies, num_bodies, dtype=torch.bool, device=positions.device),
        diagonal=1,
    )

    pair_mass_product = masses[..., :, None] * masses[..., None, :]
    while pair_mass_product.ndim < softened_distance.ndim:
        pair_mass_product = pair_mass_product.unsqueeze(-3)

    pair_potential = -G * pair_mass_product / softened_distance
    return pair_potential[..., upper_triangle_mask].sum(dim=-1)


def total_energy_from_velocities(
    positions: BodyStateTensor,
    velocities: BodyStateTensor,
    masses: BodyTensor,
    G: float = 1.0,
    epsilon: float = 0.15,
) -> TimeTensor:
    return kinetic_energy_from_velocities(velocities, masses) + potential_energy(
        positions, masses, G=G, epsilon=epsilon
    )


def total_energy_from_momenta(
    positions: BodyStateTensor,
    momenta: BodyStateTensor,
    masses: BodyTensor,
    G: float = 1.0,
    epsilon: float = 0.15,
) -> TimeTensor:
    return kinetic_energy_from_momenta(momenta, masses) + potential_energy(
        positions, masses, G=G, epsilon=epsilon
    )


def energy_drift(energy: TimeTensor) -> TimeTensor:
    initial_energy = energy[..., 0]
    return energy - initial_energy[..., None]


def relative_energy_drift(energy: TimeTensor, eps: float = 1e-8) -> TimeTensor:
    initial_energy = energy[..., 0]
    # initial_energy.abs().clamp_min(eps) is a scalar, [..., None] adds a dimension for batched trajectories, for example [2,300] where 300 is the energy drift at each time step
    return energy_drift(energy) / initial_energy.abs().clamp_min(eps)[..., None]


def linear_momentum_from_velocities(
    velocities: BodyStateTensor,
    masses: BodyTensor,
) -> VectorSeriesTensor:
    masses = _broadcast_masses_to_body_values(masses, velocities[..., 0])
    return (masses[..., None] * velocities).sum(dim=-2)


def linear_momentum_from_momenta(momenta: BodyStateTensor) -> VectorSeriesTensor:
    return momenta.sum(dim=-2)


def linear_momentum_drift(linear_momentum: VectorSeriesTensor) -> TimeTensor:
    initial_momentum = linear_momentum[..., 0, :]
    momentum_change = linear_momentum - initial_momentum[..., None, :]
    return torch.linalg.vector_norm(momentum_change, dim=-1)


def center_of_mass(
    positions: BodyStateTensor,
    masses: BodyTensor,
) -> VectorSeriesTensor:
    masses = _broadcast_masses_to_body_values(masses, positions[..., 0])
    total_mass = masses.sum(dim=-1)
    weighted_positions = (masses[..., None] * positions).sum(dim=-2)
    return weighted_positions / total_mass[..., None]


def center_of_mass_drift(
    positions: BodyStateTensor,
    masses: BodyTensor,
) -> TimeTensor:
    com = center_of_mass(positions, masses)
    initial_com = com[..., 0, :]
    com_change = com - initial_com[..., None, :]
    return torch.linalg.vector_norm(com_change, dim=-1)

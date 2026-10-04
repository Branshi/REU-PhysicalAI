import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.rollout.integrators import velocity_verlet_step


INITIALIZATION_MODES = ("random", "solar-like")
SIMULATION_DTYPES = ("auto", "float32", "float64")
STORAGE_DTYPES = ("float32", "float64")

DEFAULT_SOLAR_LIKE_CONFIG = {
    "central_mass_min": 0.8,
    "central_mass_max": 1.2,
    "planet_mass_min": 0.001,
    "planet_mass_max": 0.03,
    "semi_major_axis_min": 0.4,
    "semi_major_axis_max": 3.0,
    "min_orbit_ratio": 1.4,
    "packed_min_orbit_ratio": 1.25,
    "typical_eccentricity_max": 0.15,
    "eccentric_eccentricity_max": 0.4,
    "typical_inclination_max_degrees": 5.0,
    "inclined_inclination_max_degrees": 20.0,
    "eccentric_system_fraction": 0.15,
    "packed_system_fraction": 0.05,
    "min_central_distance": 0.15,
    "max_orbital_radius": 8.0,
    "shuffle_body_order": True,
}


def resolve_torch_dtype(dtype_name, initialization):
    if dtype_name == "auto":
        return torch.float64 if initialization == "solar-like" else torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(
        f"Unknown simulation dtype {dtype_name!r}. Expected one of {SIMULATION_DTYPES}."
    )


def resolve_storage_dtype(dtype_name):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(
        f"Unknown storage dtype {dtype_name!r}. Expected one of {STORAGE_DTYPES}."
    )


def compute_acceleration(positions, masses, G=1.0, epsilon=0.0):
    """
    Computes gravitational acceleration for each body.

    Parameters:
        positions : torch.Tensor
            Shape: [num_bodies, dim]

        masses: torch.Tensor
            Shape: [num_bodies, 1]

        G: float
            Gravitational constant

        epsilon : float
            Softening constant to prevent division by zero.

        Returns:
            acceleration : torch.Tensor
                Shape : [num_bodies, dim]

    """

    # This line computes the displacement for every pair (i, j), i.e., x_i - x_j
    # Note that even though position has 2 indicies, we use None to add a new dimension.
    # PyTorch uses broadcasting in order to compute the displacement for each pair
    r = positions[None, :, :] - positions[:, None, :]

    num_bodies = positions.shape[0]

    # A body should not gravitationally pull itself. Set self-distances to
    # infinity before the inverse power so epsilon=0 does not create inf * 0.
    interaction_mask = ~torch.eye(
        num_bodies, dtype=torch.bool, device=positions.device
    ).view(num_bodies, num_bodies, 1)

    # dim = -1 means sum very last dimension of the array
    dist_sq = (r**2).sum(dim=-1, keepdim=True)

    # The role of softened_dist_sq is to prevent the gravitational force from becoming infinitely large when two bodies get extremely close.
    # The denomintor in the acceleration equation is ||x_i - x_j||^3 which if approaches 0 the acceleration blows up to infinty, to prevent this we add a small
    # number to the dist_sq.
    softened_dist_sq = dist_sq + epsilon**2
    softened_dist_sq = softened_dist_sq.masked_fill(~interaction_mask, float("inf"))

    # Computes the denominator of acceleration = G * mass_j * direction_vector / distance^3.
    inv_dist_cubed = softened_dist_sq ** (-1.5)

    acceleration = G * (masses[None, :, :] * r * inv_dist_cubed).sum(dim=1)

    return acceleration


def min_pairwise_distance(positions):
    num_bodies = positions.shape[0]

    r = positions[None, :, :] - positions[:, None, :]
    distances = torch.linalg.vector_norm(r, ord=2, dim=-1)

    mask = ~torch.eye(num_bodies, dtype=torch.bool, device=positions.device)

    return distances[mask].min()


def trajectory_min_pairwise_distance(positions):
    num_bodies = positions.shape[1]

    r = positions[:, None, :, :] - positions[:, :, None, :]
    distances = torch.linalg.vector_norm(r, ord=2, dim=-1)

    mask = ~torch.eye(num_bodies, dtype=torch.bool, device=positions.device)

    return distances[:, mask].min()


def compute_total_energy(positions, velocities, masses, G=1.0, epsilon=0.0):
    """Compute kinetic plus pairwise potential energy for one or more states."""

    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape")
    if positions.shape[-2] != masses.shape[0]:
        raise ValueError("masses must have one entry for every body")

    body_masses = masses.squeeze(-1)
    kinetic = 0.5 * (
        body_masses * velocities.square().sum(dim=-1)
    ).sum(dim=-1)

    displacements = positions[..., :, None, :] - positions[..., None, :, :]
    distance_squared = displacements.square().sum(dim=-1) + epsilon**2
    pair_mask = torch.triu(
        torch.ones(
            masses.shape[0],
            masses.shape[0],
            dtype=torch.bool,
            device=positions.device,
        ),
        diagonal=1,
    )
    pair_distances = torch.sqrt(distance_squared[..., pair_mask])
    pair_masses = (
        body_masses[:, None] * body_masses[None, :]
    )[pair_mask]
    potential = -G * (pair_masses / pair_distances).sum(dim=-1)

    return kinetic + potential


def relative_energy_drift(
    positions,
    velocities,
    masses,
    G=1.0,
    epsilon=0.0,
):
    energies = compute_total_energy(
        positions=positions,
        velocities=velocities,
        masses=masses,
        G=G,
        epsilon=epsilon,
    )
    initial_energy_scale = energies[0].abs().clamp_min(1e-12)
    return ((energies - energies[0]).abs() / initial_energy_scale).max()


def trajectory_central_distances(positions, masses):
    """Return every orbiter's distance from the most massive body over time."""

    central_index = masses.squeeze(-1).argmax()
    relative_positions = positions - positions[:, central_index : central_index + 1]
    distances = torch.linalg.vector_norm(relative_positions, dim=-1)
    body_mask = torch.arange(masses.shape[0], device=positions.device) != central_index
    return distances[:, body_mask]


def trajectory_is_valid(
    positions,
    accelerations,
    velocities=None,
    masses=None,
    G=1.0,
    epsilon=0.0,
    min_trajectory_distance=None,
    max_acceleration=None,
    min_central_distance=None,
    max_orbital_radius=None,
    max_relative_energy_drift=None,
):
    if not torch.isfinite(positions).all():
        return False

    if not torch.isfinite(accelerations).all():
        return False

    if velocities is not None and not torch.isfinite(velocities).all():
        return False

    if masses is not None and not torch.isfinite(masses).all():
        return False

    if min_trajectory_distance is not None:
        if trajectory_min_pairwise_distance(positions) < min_trajectory_distance:
            return False

    if max_acceleration is not None:
        if accelerations.abs().max() > max_acceleration:
            return False

    if min_central_distance is not None or max_orbital_radius is not None:
        if masses is None:
            raise ValueError("masses are required for central-distance validation")
        central_distances = trajectory_central_distances(positions, masses)
        if min_central_distance is not None:
            if central_distances.min() < min_central_distance:
                return False
        if max_orbital_radius is not None:
            if central_distances.max() > max_orbital_radius:
                return False

    if max_relative_energy_drift is not None:
        if velocities is None or masses is None:
            raise ValueError(
                "velocities and masses are required for energy-drift validation"
            )
        energy_drift = relative_energy_drift(
            positions=positions,
            velocities=velocities,
            masses=masses,
            G=G,
            epsilon=epsilon,
        )
        if not torch.isfinite(energy_drift):
            return False
        if energy_drift > max_relative_energy_drift:
            return False

    return True


def sample_initial_conditions(
    num_bodies=2,
    dim=2,
    position_scale=0.8,
    velocity_scale=0.8,
    mass_min=0.5,
    mass_max=2.0,
    min_distance=0.4,
    max_attempts=1000,
    device="cpu",
    dtype=torch.float32,
):
    for _ in range(max_attempts):
        positions = position_scale * torch.randn(
            num_bodies,
            dim,
            device=device,
            dtype=dtype,
        )

        total_mass_dummy = torch.ones(
            num_bodies,
            1,
            device=device,
            dtype=dtype,
        )
        center_of_mass_position = (total_mass_dummy * positions).sum(
            dim=0, keepdim=True
        ) / total_mass_dummy.sum()
        positions = positions - center_of_mass_position

        if min_pairwise_distance(positions) >= min_distance:
            break
    else:
        raise RuntimeError("Could not sample valid initial positions.")

    velocities = velocity_scale * torch.randn(
        num_bodies,
        dim,
        device=device,
        dtype=dtype,
    )

    masses = mass_min + (mass_max - mass_min) * torch.rand(
        num_bodies,
        1,
        device=device,
        dtype=dtype,
    )

    total_mass = masses.sum()

    center_of_mass_velocity = (masses * velocities).sum(
        dim=0, keepdim=True
    ) / total_mass
    velocities = velocities - center_of_mass_velocity

    center_of_mass_position = (masses * positions).sum(dim=0, keepdim=True) / total_mass
    positions = positions - center_of_mass_position

    return positions, velocities, masses


def validate_solar_like_config(config, num_bodies, dim, G):
    if num_bodies < 2:
        raise ValueError("solar-like initialization requires at least two bodies")
    if dim not in (2, 3):
        raise ValueError("solar-like initialization supports only dim=2 or dim=3")
    if G <= 0.0:
        raise ValueError("solar-like initialization requires G > 0")

    positive_ranges = (
        ("central mass", config["central_mass_min"], config["central_mass_max"]),
        ("planet mass", config["planet_mass_min"], config["planet_mass_max"]),
        (
            "semi-major axis",
            config["semi_major_axis_min"],
            config["semi_major_axis_max"],
        ),
    )
    for name, minimum, maximum in positive_ranges:
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(
                f"{name} bounds must satisfy 0 < minimum <= maximum"
            )

    if config["central_mass_min"] <= config["planet_mass_max"]:
        raise ValueError("the central mass must always exceed every planet mass")

    for name in ("min_orbit_ratio", "packed_min_orbit_ratio"):
        if config[name] <= 1.0:
            raise ValueError(f"{name} must be greater than 1")

    num_planets = num_bodies - 1
    available_ratio = (
        config["semi_major_axis_max"] / config["semi_major_axis_min"]
    )
    for name in ("min_orbit_ratio", "packed_min_orbit_ratio"):
        required_ratio = config[name] ** max(num_planets - 1, 0)
        if required_ratio > available_ratio:
            raise ValueError(
                f"{name}={config[name]} cannot fit {num_planets} planets inside "
                "the configured semi-major-axis range"
            )

    typical_eccentricity = config["typical_eccentricity_max"]
    eccentric_eccentricity = config["eccentric_eccentricity_max"]
    if not 0.0 <= typical_eccentricity <= eccentric_eccentricity < 1.0:
        raise ValueError(
            "eccentricities must satisfy 0 <= typical maximum <= "
            "eccentric maximum < 1"
        )

    for name in (
        "typical_inclination_max_degrees",
        "inclined_inclination_max_degrees",
    ):
        if not 0.0 <= config[name] <= 180.0:
            raise ValueError(f"{name} must be between 0 and 180 degrees")
    if (
        config["typical_inclination_max_degrees"]
        > config["inclined_inclination_max_degrees"]
    ):
        raise ValueError(
            "typical inclination maximum cannot exceed inclined maximum"
        )

    eccentric_fraction = config["eccentric_system_fraction"]
    packed_fraction = config["packed_system_fraction"]
    if eccentric_fraction < 0.0 or packed_fraction < 0.0:
        raise ValueError("solar-like profile fractions must be nonnegative")
    if eccentric_fraction + packed_fraction > 1.0:
        raise ValueError("solar-like profile fractions must sum to at most 1")

    if config["min_central_distance"] <= 0.0:
        raise ValueError("min_central_distance must be positive")
    if config["max_orbital_radius"] <= config["min_central_distance"]:
        raise ValueError(
            "max_orbital_radius must be greater than min_central_distance"
        )


def resolve_solar_like_config(solar_like_config, num_bodies, dim, G):
    config = dict(DEFAULT_SOLAR_LIKE_CONFIG)
    if solar_like_config is not None:
        unknown_keys = set(solar_like_config) - set(config)
        if unknown_keys:
            raise ValueError(
                "Unknown solar-like configuration keys: "
                + ", ".join(sorted(unknown_keys))
            )
        config.update(solar_like_config)
    validate_solar_like_config(config, num_bodies=num_bodies, dim=dim, G=G)
    return config


def sample_separated_semi_major_axes(
    num_planets,
    minimum,
    maximum,
    minimum_ratio,
    device,
    dtype,
):
    """Sample sorted orbital scales with a guaranteed adjacent ratio."""

    if num_planets == 1:
        log_axis = torch.empty((), device=device, dtype=dtype).uniform_(
            math.log(minimum),
            math.log(maximum),
        )
        return log_axis.exp().reshape(1)

    minimum_log_spacing = math.log(minimum_ratio)
    free_log_span = (
        math.log(maximum / minimum)
        - (num_planets - 1) * minimum_log_spacing
    )
    free_log_span = max(0.0, free_log_span)
    offsets = torch.sort(
        torch.rand(num_planets, device=device, dtype=dtype) * free_log_span
    ).values
    base = torch.arange(num_planets, device=device, dtype=dtype)
    log_axes = math.log(minimum) + base * minimum_log_spacing + offsets
    return log_axes.exp()


def solve_kepler_equation(mean_anomaly, eccentricity, num_iterations=12):
    """Solve M = E - e sin(E) with Newton iterations."""

    eccentric_anomaly = mean_anomaly.clone()
    for _ in range(num_iterations):
        residual = (
            eccentric_anomaly
            - eccentricity * torch.sin(eccentric_anomaly)
            - mean_anomaly
        )
        derivative = 1.0 - eccentricity * torch.cos(eccentric_anomaly)
        eccentric_anomaly = eccentric_anomaly - residual / derivative
    return eccentric_anomaly


def orbital_plane_basis(ascending_node, inclination, argument_of_periapsis):
    """Return orthonormal periapsis and transverse basis vectors in 3D."""

    cos_node = torch.cos(ascending_node)
    sin_node = torch.sin(ascending_node)
    cos_inclination = torch.cos(inclination)
    sin_inclination = torch.sin(inclination)
    cos_periapsis = torch.cos(argument_of_periapsis)
    sin_periapsis = torch.sin(argument_of_periapsis)

    periapsis_basis = torch.stack(
        [
            cos_node * cos_periapsis
            - sin_node * sin_periapsis * cos_inclination,
            sin_node * cos_periapsis
            + cos_node * sin_periapsis * cos_inclination,
            sin_periapsis * sin_inclination,
        ]
    )
    transverse_basis = torch.stack(
        [
            -cos_node * sin_periapsis
            - sin_node * cos_periapsis * cos_inclination,
            -sin_node * sin_periapsis
            + cos_node * cos_periapsis * cos_inclination,
            cos_periapsis * sin_inclination,
        ]
    )
    return periapsis_basis, transverse_basis


def sample_solar_like_initial_conditions(
    num_bodies,
    dim=3,
    G=1.0,
    solar_like_config=None,
    device="cpu",
    dtype=torch.float64,
):
    """Sample one dominant body and Keplerian orbiters in random 3D planes."""

    config = resolve_solar_like_config(
        solar_like_config,
        num_bodies=num_bodies,
        dim=dim,
        G=G,
    )
    num_planets = num_bodies - 1

    central_mass = torch.empty((), device=device, dtype=dtype).uniform_(
        config["central_mass_min"],
        config["central_mass_max"],
    )
    log_planet_masses = torch.empty(
        num_planets,
        device=device,
        dtype=dtype,
    ).uniform_(
        math.log10(config["planet_mass_min"]),
        math.log10(config["planet_mass_max"]),
    )
    planet_masses = torch.pow(10.0, log_planet_masses)

    profile_draw = torch.rand((), device=device, dtype=dtype).item()
    packed_fraction = config["packed_system_fraction"]
    eccentric_fraction = config["eccentric_system_fraction"]
    if profile_draw < packed_fraction:
        minimum_ratio = config["packed_min_orbit_ratio"]
        eccentricity_min = 0.0
        eccentricity_max = min(config["typical_eccentricity_max"], 0.08)
        inclination_max_degrees = config["typical_inclination_max_degrees"]
    elif profile_draw < packed_fraction + eccentric_fraction:
        minimum_ratio = config["min_orbit_ratio"]
        eccentricity_min = config["typical_eccentricity_max"]
        eccentricity_max = config["eccentric_eccentricity_max"]
        inclination_max_degrees = config["inclined_inclination_max_degrees"]
    else:
        minimum_ratio = config["min_orbit_ratio"]
        eccentricity_min = 0.0
        eccentricity_max = config["typical_eccentricity_max"]
        inclination_max_degrees = config["typical_inclination_max_degrees"]

    semi_major_axes = sample_separated_semi_major_axes(
        num_planets=num_planets,
        minimum=config["semi_major_axis_min"],
        maximum=config["semi_major_axis_max"],
        minimum_ratio=minimum_ratio,
        device=device,
        dtype=dtype,
    )
    eccentricities = eccentricity_min + (
        eccentricity_max - eccentricity_min
    ) * torch.rand(num_planets, device=device, dtype=dtype)

    # Keep every sampled osculating periapsis outside the central exclusion zone.
    eccentricity_limits = 1.0 - (
        config["min_central_distance"] / semi_major_axes
    )
    eccentricities = torch.minimum(
        eccentricities,
        eccentricity_limits.clamp_min(0.0),
    )

    full_turn = 2.0 * math.pi
    mean_anomalies = full_turn * torch.rand(
        num_planets,
        device=device,
        dtype=dtype,
    )
    eccentric_anomalies = solve_kepler_equation(
        mean_anomaly=mean_anomalies,
        eccentricity=eccentricities,
    )
    arguments_of_periapsis = full_turn * torch.rand(
        num_planets,
        device=device,
        dtype=dtype,
    )

    if dim == 3:
        ascending_nodes = full_turn * torch.rand(
            num_planets,
            device=device,
            dtype=dtype,
        )
        max_inclination = math.radians(inclination_max_degrees)
        # Sample cos(i) uniformly so orbital normals are not biased by angle.
        cos_inclinations = 1.0 - torch.rand(
            num_planets,
            device=device,
            dtype=dtype,
        ) * (1.0 - math.cos(max_inclination))
        inclinations = torch.acos(cos_inclinations)
    else:
        ascending_nodes = torch.zeros(num_planets, device=device, dtype=dtype)
        inclinations = torch.zeros(num_planets, device=device, dtype=dtype)

    positions_3d = torch.zeros(num_bodies, 3, device=device, dtype=dtype)
    velocities_3d = torch.zeros_like(positions_3d)

    for planet_index in range(num_planets):
        semi_major_axis = semi_major_axes[planet_index]
        eccentricity = eccentricities[planet_index]
        eccentric_anomaly = eccentric_anomalies[planet_index]
        one_minus_e_cos = 1.0 - eccentricity * torch.cos(eccentric_anomaly)
        sqrt_one_minus_e_squared = torch.sqrt(1.0 - eccentricity.square())

        plane_x = semi_major_axis * (
            torch.cos(eccentric_anomaly) - eccentricity
        )
        plane_y = (
            semi_major_axis
            * sqrt_one_minus_e_squared
            * torch.sin(eccentric_anomaly)
        )

        gravitational_parameter = G * (
            central_mass + planet_masses[planet_index]
        )
        mean_motion = torch.sqrt(
            gravitational_parameter / semi_major_axis.pow(3)
        )
        plane_vx = (
            -semi_major_axis
            * mean_motion
            * torch.sin(eccentric_anomaly)
            / one_minus_e_cos
        )
        plane_vy = (
            semi_major_axis
            * mean_motion
            * sqrt_one_minus_e_squared
            * torch.cos(eccentric_anomaly)
            / one_minus_e_cos
        )

        periapsis_basis, transverse_basis = orbital_plane_basis(
            ascending_node=ascending_nodes[planet_index],
            inclination=inclinations[planet_index],
            argument_of_periapsis=arguments_of_periapsis[planet_index],
        )
        body_index = planet_index + 1
        positions_3d[body_index] = (
            plane_x * periapsis_basis + plane_y * transverse_basis
        )
        velocities_3d[body_index] = (
            plane_vx * periapsis_basis + plane_vy * transverse_basis
        )

    masses = torch.cat(
        [central_mass.reshape(1), planet_masses],
        dim=0,
    ).unsqueeze(-1)
    positions = positions_3d[:, :dim]
    velocities = velocities_3d[:, :dim]

    total_mass = masses.sum()
    positions = positions - (masses * positions).sum(
        dim=0,
        keepdim=True,
    ) / total_mass
    velocities = velocities - (masses * velocities).sum(
        dim=0,
        keepdim=True,
    ) / total_mass

    if config["shuffle_body_order"]:
        permutation = torch.randperm(num_bodies, device=device)
        positions = positions[permutation]
        velocities = velocities[permutation]
        masses = masses[permutation]

    return positions, velocities, masses


def simulate_trajectory(
    num_bodies=2,
    num_steps=100,
    dt=0.01,
    dim=2,
    G=1.0,
    epsilon=0.0,
    position_scale=0.8,
    velocity_scale=0.8,
    min_distance=0.4,
    initialization="random",
    solar_like_config=None,
    simulation_dtype=torch.float32,
    device="cpu",
):
    """
    Simulates one N body trajectory

    Returns:
         positions : torch.Tensor
             Shape : [num_bodies, dim]

         velocities : torch.Tensor
             Shape : [num_bodies, dim]

         acceleration : torch.Tensor
             Shape : [num_bodies, dim]

         masses : torch.Tensor
             Shape : [num_bodies, 1]
    """

    if initialization == "random":
        positions, velocities, masses = sample_initial_conditions(
            num_bodies=num_bodies,
            dim=dim,
            position_scale=position_scale,
            velocity_scale=velocity_scale,
            min_distance=min_distance,
            device=device,
            dtype=simulation_dtype,
        )
    elif initialization == "solar-like":
        positions, velocities, masses = sample_solar_like_initial_conditions(
            num_bodies=num_bodies,
            dim=dim,
            G=G,
            solar_like_config=solar_like_config,
            device=device,
            dtype=simulation_dtype,
        )
    else:
        raise ValueError(
            f"Unknown initialization {initialization!r}. "
            f"Expected one of {INITIALIZATION_MODES}."
        )

    position_history = []
    velocity_history = []
    acceleration_history = []

    acceleration = compute_acceleration(
        positions=positions, masses=masses, G=G, epsilon=epsilon
    )

    for _ in range(num_steps):
        position_history.append(positions)
        velocity_history.append(velocities)
        acceleration_history.append(acceleration)

        positions, velocities, acceleration = velocity_verlet_step(
            positions,
            velocities,
            compute_acceleration,
            dt,
            masses,
            acceleration=acceleration,
            G=G,
            epsilon=epsilon,
        )

    positions = torch.stack(position_history, dim=0)
    velocities = torch.stack(velocity_history, dim=0)
    accelerations = torch.stack(acceleration_history, dim=0)

    return positions, velocities, accelerations, masses


def simulate_trajectory_from_initial_conditions(
    positions,
    velocities,
    masses,
    num_steps,
    dt=0.01,
    G=1.0,
    epsilon=0.0,
):
    """
    Simulate a trajectory from user-provided initial conditions.

    Unlike simulate_trajectory, this includes the initial state and then
    advances num_steps times, so returned tensors have length num_steps + 1.
    """

    position_history = []
    velocity_history = []
    acceleration_history = []

    acceleration = compute_acceleration(
        positions=positions,
        masses=masses,
        G=G,
        epsilon=epsilon,
    )

    for _ in range(num_steps):
        position_history.append(positions)
        velocity_history.append(velocities)
        acceleration_history.append(acceleration)

        positions, velocities, acceleration = velocity_verlet_step(
            positions,
            velocities,
            compute_acceleration,
            dt,
            masses,
            acceleration=acceleration,
            G=G,
            epsilon=epsilon,
        )

    position_history.append(positions)
    velocity_history.append(velocities)
    acceleration_history.append(acceleration)

    positions = torch.stack(position_history, dim=0)
    velocities = torch.stack(velocity_history, dim=0)
    accelerations = torch.stack(acceleration_history, dim=0)

    return positions, velocities, accelerations


def generate_dataset(
    num_trajectories,
    num_bodies,
    num_steps,
    dt=0.01,
    dim=2,
    G=1.0,
    epsilon=0.0,
    position_scale=0.8,
    velocity_scale=0.8,
    min_distance=0.4,
    min_trajectory_distance=None,
    max_acceleration=None,
    max_resample_attempts=100,
    initialization="random",
    solar_like_config=None,
    max_relative_energy_drift=None,
    simulation_dtype="auto",
    storage_dtype="float32",
    device="cpu",
):
    """
    Generates many N body trajectories

    Returns:
        dataset : dict
            Dictionary containing positions, velocities, accelerations, and masses.
    """

    if initialization not in INITIALIZATION_MODES:
        raise ValueError(
            f"Unknown initialization {initialization!r}. "
            f"Expected one of {INITIALIZATION_MODES}."
        )
    if num_trajectories <= 0 or num_bodies <= 0 or num_steps <= 0:
        raise ValueError(
            "num_trajectories, num_bodies, and num_steps must all be positive"
        )
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if max_resample_attempts <= 0:
        raise ValueError("max_resample_attempts must be positive")

    if min_trajectory_distance is None:
        min_trajectory_distance = 0.05 if initialization == "solar-like" else 0.3
    if max_acceleration is None:
        max_acceleration = 200.0 if initialization == "solar-like" else 100.0
    if min_trajectory_distance < 0.0:
        raise ValueError("min_trajectory_distance must be nonnegative")
    if max_acceleration <= 0.0:
        raise ValueError("max_acceleration must be positive")
    if max_relative_energy_drift is None and initialization == "solar-like":
        max_relative_energy_drift = 1e-3
    if max_relative_energy_drift is not None and max_relative_energy_drift < 0.0:
        raise ValueError("max_relative_energy_drift must be nonnegative")

    resolved_solar_like_config = None
    if initialization == "solar-like":
        resolved_solar_like_config = resolve_solar_like_config(
            solar_like_config,
            num_bodies=num_bodies,
            dim=dim,
            G=G,
        )

    simulation_torch_dtype = resolve_torch_dtype(
        simulation_dtype,
        initialization=initialization,
    )
    storage_torch_dtype = resolve_storage_dtype(storage_dtype)

    all_positions = []
    all_velocities = []
    all_accelerations = []
    all_masses = []
    accepted_energy_drifts = []

    rejected_trajectories = 0

    for _ in range(num_trajectories):
        for attempt in range(max_resample_attempts):
            positions, velocities, accelerations, masses = simulate_trajectory(
                num_bodies=num_bodies,
                num_steps=num_steps,
                dt=dt,
                dim=dim,
                G=G,
                epsilon=epsilon,
                position_scale=position_scale,
                velocity_scale=velocity_scale,
                min_distance=min_distance,
                initialization=initialization,
                solar_like_config=resolved_solar_like_config,
                simulation_dtype=simulation_torch_dtype,
                device=device,
            )

            if trajectory_is_valid(
                positions=positions,
                accelerations=accelerations,
                velocities=velocities,
                masses=masses,
                G=G,
                epsilon=epsilon,
                min_trajectory_distance=min_trajectory_distance,
                max_acceleration=max_acceleration,
                min_central_distance=(
                    resolved_solar_like_config["min_central_distance"]
                    if resolved_solar_like_config is not None
                    else None
                ),
                max_orbital_radius=(
                    resolved_solar_like_config["max_orbital_radius"]
                    if resolved_solar_like_config is not None
                    else None
                ),
                max_relative_energy_drift=max_relative_energy_drift,
            ):
                break

            rejected_trajectories += 1
        else:
            raise RuntimeError(
                "Could not sample a valid trajectory. Try increasing "
                "max_resample_attempts, increasing epsilon, or relaxing "
                "the distance, acceleration, orbital-radius, or energy-drift "
                "filters."
            )

        if max_relative_energy_drift is not None:
            accepted_energy_drifts.append(
                relative_energy_drift(
                    positions=positions,
                    velocities=velocities,
                    masses=masses,
                    G=G,
                    epsilon=epsilon,
                ).item()
            )

        # Integrate and validate in the requested precision, then cast only the
        # accepted trajectory to its compact on-disk representation.
        all_positions.append(positions.to(dtype=storage_torch_dtype))
        all_velocities.append(velocities.to(dtype=storage_torch_dtype))
        all_accelerations.append(accelerations.to(dtype=storage_torch_dtype))
        all_masses.append(masses.to(dtype=storage_torch_dtype))

    positions = torch.stack(all_positions, dim=0)
    velocities = torch.stack(all_velocities, dim=0)
    accelerations = torch.stack(all_accelerations, dim=0)
    masses = torch.stack(all_masses, dim=0)

    # masses shape:       [num_trajectories, num_bodies, 1]
    # velocities shape:   [num_trajectories, num_steps, num_bodies, dim]
    # Add a time dimension to masses so it broadcasts correctly.
    momenta = masses[:, None, :, :] * velocities

    # Optional but very useful for HNN training:
    # p_dot = force = m * acceleration
    forces = masses[:, None, :, :] * accelerations

    dataset = {
        "positions": positions,
        "velocities": velocities,
        "momenta": momenta,
        "accelerations": accelerations,
        "forces": forces,
        "masses": masses,
        "metadata": {
            "dt": dt,
            "G": G,
            "epsilon": epsilon,
            "position_scale": position_scale,
            "velocity_scale": velocity_scale,
            "min_distance": min_distance,
            "min_trajectory_distance": min_trajectory_distance,
            "max_acceleration": max_acceleration,
            "initialization": initialization,
            "simulation_dtype": str(simulation_torch_dtype).removeprefix("torch."),
            "storage_dtype": str(storage_torch_dtype).removeprefix("torch."),
            "max_relative_energy_drift": max_relative_energy_drift,
            "rejected_trajectories": rejected_trajectories,
        },
    }
    if resolved_solar_like_config is not None:
        dataset["metadata"]["solar_like_config"] = resolved_solar_like_config
    if accepted_energy_drifts:
        dataset["metadata"]["accepted_relative_energy_drift_mean"] = sum(
            accepted_energy_drifts
        ) / len(accepted_energy_drifts)
        dataset["metadata"]["accepted_relative_energy_drift_max"] = max(
            accepted_energy_drifts
        )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an N-body dataset.")
    parser.add_argument("--dataset-name", default="nbody_3body_eps0_close")
    parser.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT / "common" / "datasets" / "nbody_3body_eps0_close.pt"
        ),
    )
    parser.add_argument(
        "--split-output-path",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "splits"
            / "nbody_3body_eps0_close.json"
        ),
    )
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--num-bodies", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_MODES,
        default="random",
        help="Initial-condition family. solar-like creates a star plus orbiters.",
    )
    parser.add_argument("--position-scale", type=float, default=0.8)
    parser.add_argument("--velocity-scale", type=float, default=0.8)
    parser.add_argument("--min-distance", type=float, default=0.4)
    parser.add_argument(
        "--min-trajectory-distance",
        type=float,
        default=None,
        help="Defaults to 0.3 for random data and 0.05 for solar-like data.",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=None,
        help="Defaults to 100 for random data and 200 for solar-like data.",
    )
    parser.add_argument(
        "--max-relative-energy-drift",
        type=float,
        default=None,
        help="Defaults to 1e-3 for solar-like data and disabled for random data.",
    )
    parser.add_argument(
        "--simulation-dtype",
        choices=SIMULATION_DTYPES,
        default="auto",
        help="auto uses float64 for solar-like data and float32 otherwise.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=STORAGE_DTYPES,
        default="float32",
        help="Tensor precision used in the saved dataset.",
    )
    parser.add_argument("--max-resample-attempts", type=int, default=100)

    solar_group = parser.add_argument_group("solar-like initialization")
    solar_group.add_argument("--central-mass-min", type=float, default=0.8)
    solar_group.add_argument("--central-mass-max", type=float, default=1.2)
    solar_group.add_argument("--planet-mass-min", type=float, default=0.001)
    solar_group.add_argument("--planet-mass-max", type=float, default=0.03)
    solar_group.add_argument("--semi-major-axis-min", type=float, default=0.4)
    solar_group.add_argument("--semi-major-axis-max", type=float, default=3.0)
    solar_group.add_argument("--min-orbit-ratio", type=float, default=1.4)
    solar_group.add_argument(
        "--packed-min-orbit-ratio",
        type=float,
        default=1.25,
    )
    solar_group.add_argument(
        "--typical-eccentricity-max",
        type=float,
        default=0.15,
    )
    solar_group.add_argument(
        "--eccentric-eccentricity-max",
        type=float,
        default=0.4,
    )
    solar_group.add_argument(
        "--typical-inclination-max-degrees",
        type=float,
        default=5.0,
    )
    solar_group.add_argument(
        "--inclined-inclination-max-degrees",
        type=float,
        default=20.0,
    )
    solar_group.add_argument(
        "--eccentric-system-fraction",
        type=float,
        default=0.15,
    )
    solar_group.add_argument(
        "--packed-system-fraction",
        type=float,
        default=0.05,
    )
    solar_group.add_argument("--min-central-distance", type=float, default=0.15)
    solar_group.add_argument("--max-orbital-radius", type=float, default=8.0)
    solar_group.add_argument(
        "--shuffle-body-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize which tensor row contains the central body.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate split settings before running the more expensive trajectory
    # simulation, and require each partition to receive at least one trajectory.
    split_fractions = (
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
    )
    if any(fraction <= 0.0 for fraction in split_fractions):
        raise ValueError("train/val/test fractions must all be positive")

    total_fraction = sum(split_fractions)
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")

    num_train = int(args.train_fraction * args.num_trajectories)
    num_val = int(args.val_fraction * args.num_trajectories)
    num_test = args.num_trajectories - num_train - num_val
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            "train/val/test fractions and num-trajectories must give each split "
            "at least one trajectory"
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    solar_like_config = {
        "central_mass_min": args.central_mass_min,
        "central_mass_max": args.central_mass_max,
        "planet_mass_min": args.planet_mass_min,
        "planet_mass_max": args.planet_mass_max,
        "semi_major_axis_min": args.semi_major_axis_min,
        "semi_major_axis_max": args.semi_major_axis_max,
        "min_orbit_ratio": args.min_orbit_ratio,
        "packed_min_orbit_ratio": args.packed_min_orbit_ratio,
        "typical_eccentricity_max": args.typical_eccentricity_max,
        "eccentric_eccentricity_max": args.eccentric_eccentricity_max,
        "typical_inclination_max_degrees": (
            args.typical_inclination_max_degrees
        ),
        "inclined_inclination_max_degrees": (
            args.inclined_inclination_max_degrees
        ),
        "eccentric_system_fraction": args.eccentric_system_fraction,
        "packed_system_fraction": args.packed_system_fraction,
        "min_central_distance": args.min_central_distance,
        "max_orbital_radius": args.max_orbital_radius,
        "shuffle_body_order": args.shuffle_body_order,
    }

    dataset = generate_dataset(
        num_trajectories=args.num_trajectories,
        num_bodies=args.num_bodies,
        num_steps=args.num_steps,
        dt=args.dt,
        dim=args.dim,
        G=args.G,
        epsilon=args.epsilon,
        position_scale=args.position_scale,
        velocity_scale=args.velocity_scale,
        min_distance=args.min_distance,
        min_trajectory_distance=args.min_trajectory_distance,
        max_acceleration=args.max_acceleration,
        max_resample_attempts=args.max_resample_attempts,
        initialization=args.initialization,
        solar_like_config=solar_like_config,
        max_relative_energy_drift=args.max_relative_energy_drift,
        simulation_dtype=args.simulation_dtype,
        storage_dtype=args.storage_dtype,
    )
    dataset["metadata"]["seed"] = args.seed

    # Store an absolute dataset path so every trainer can verify that the split
    # manifest belongs to this exact generated file.
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    # Use a dedicated generator so changing the split seed changes only the
    # trajectory assignment, not the simulated trajectories themselves.
    split_seed = args.split_seed if args.split_seed is not None else 0
    generator = torch.Generator().manual_seed(split_seed)

    indices = torch.randperm(args.num_trajectories, generator=generator).tolist()

    # Slice one seeded permutation so the three partitions are reproducible,
    # non-overlapping, and collectively contain every trajectory.
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]

    # Keep both the exact IDs and the configuration needed to audit an
    # experiment later from its checkpoint or MLflow metadata.
    split_manifest = {
        "name": args.dataset_name,
        "dataset_path": str(output_path),
        "num_trajectories": args.num_trajectories,
        "num_steps": args.num_steps,
        "num_bodies": args.num_bodies,
        "dim": args.dim,
        "split": {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        },
        "split_config": {
            "split_mode": "random_seeded",
            "split_seed": split_seed,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "num_test": len(test_indices),
        },
        "dataset_metadata": dataset["metadata"],
    }

    split_output_path = Path(args.split_output_path).expanduser().resolve()
    split_output_path.parent.mkdir(parents=True, exist_ok=True)

    with split_output_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"Saved dataset to {output_path}.")
    print(f"Saved trajectory split to {split_output_path}.")
    print("acc mean:", dataset["accelerations"].mean().item())
    print("acc std:", dataset["accelerations"].std().item())
    print("acc max abs:", dataset["accelerations"].abs().max().item())
    print("rejected trajectories:", dataset["metadata"]["rejected_trajectories"])
    if "accepted_relative_energy_drift_max" in dataset["metadata"]:
        print(
            "max accepted relative energy drift:",
            dataset["metadata"]["accepted_relative_energy_drift_max"],
        )


if __name__ == "__main__":
    main()

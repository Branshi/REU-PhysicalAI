import argparse

import torch

try:
    from common.rollout.integrators import velocity_verlet_step
except ModuleNotFoundError:
    from rollout.integrators import velocity_verlet_step


def compute_acceleration(positions, masses, G=1.0, epsilon=0.05):
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


def trajectory_is_valid(
    positions,
    accelerations,
    min_trajectory_distance=None,
    max_acceleration=None,
):
    if not torch.isfinite(positions).all():
        return False

    if not torch.isfinite(accelerations).all():
        return False

    if min_trajectory_distance is not None:
        if trajectory_min_pairwise_distance(positions) < min_trajectory_distance:
            return False

    if max_acceleration is not None:
        if accelerations.abs().max() > max_acceleration:
            return False

    return True


def sample_initial_conditions(
    num_bodies=2,
    dim=2,
    position_scale=0.8,
    velocity_scale=0.15,
    mass_min=0.5,
    mass_max=2.0,
    min_distance=0.35,
    max_attempts=1000,
    device="cpu",
):
    for _ in range(max_attempts):
        positions = position_scale * torch.randn(num_bodies, dim, device=device)

        total_mass_dummy = torch.ones(num_bodies, 1, device=device)
        center_of_mass_position = (total_mass_dummy * positions).sum(
            dim=0, keepdim=True
        ) / total_mass_dummy.sum()
        positions = positions - center_of_mass_position

        if min_pairwise_distance(positions) >= min_distance:
            break
    else:
        raise RuntimeError("Could not sample valid initial positions.")

    velocities = velocity_scale * torch.randn(num_bodies, dim, device=device)

    masses = mass_min + (mass_max - mass_min) * torch.rand(num_bodies, 1, device=device)

    total_mass = masses.sum()

    center_of_mass_velocity = (masses * velocities).sum(
        dim=0, keepdim=True
    ) / total_mass
    velocities = velocities - center_of_mass_velocity

    center_of_mass_position = (masses * positions).sum(dim=0, keepdim=True) / total_mass
    positions = positions - center_of_mass_position

    return positions, velocities, masses


def simulate_trajectory(
    num_bodies=2,
    num_steps=100,
    dt=0.01,
    dim=2,
    G=1.0,
    epsilon=0.15,
    position_scale=0.8,
    velocity_scale=0.15,
    min_distance=0.35,
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

    positions, velocities, masses = sample_initial_conditions(
        num_bodies=num_bodies,
        dim=dim,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        min_distance=min_distance,
        device=device,
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
    epsilon=0.15,
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
    epsilon=0.15,
    position_scale=0.8,
    velocity_scale=0.15,
    min_distance=0.35,
    min_trajectory_distance=None,
    max_acceleration=80.0,
    max_resample_attempts=100,
    device="cpu",
):
    """
    Generates many N body trajectories

    Returns:
        dataset : dict
            Dictionary containing positions, velocities, accelerations, and masses.
    """

    all_positions = []
    all_velocities = []
    all_accelerations = []
    all_masses = []

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
                device=device,
            )

            if trajectory_is_valid(
                positions=positions,
                accelerations=accelerations,
                min_trajectory_distance=min_trajectory_distance,
                max_acceleration=max_acceleration,
            ):
                break

            rejected_trajectories += 1
        else:
            raise RuntimeError(
                "Could not sample a valid trajectory. Try increasing "
                "max_resample_attempts, increasing epsilon, or relaxing "
                "min_trajectory_distance/max_acceleration."
            )

        all_positions.append(positions)
        all_velocities.append(velocities)
        all_accelerations.append(accelerations)
        all_masses.append(masses)

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
            "rejected_trajectories": rejected_trajectories,
        },
    }
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an N-body dataset.")
    parser.add_argument("--output-path", default="nbody_3body_dataset.pt")
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--num-bodies", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--position-scale", type=float, default=0.8)
    parser.add_argument("--velocity-scale", type=float, default=0.15)
    parser.add_argument("--min-distance", type=float, default=0.35)
    parser.add_argument("--min-trajectory-distance", type=float, default=None)
    parser.add_argument("--max-acceleration", type=float, default=80.0)
    parser.add_argument("--max-resample-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

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
    )
    torch.save(dataset, args.output_path)

    print(f"Saved dataset to {args.output_path}.")
    print("positions:", dataset["positions"].shape)
    print("velocities:", dataset["velocities"].shape)
    print("momenta:", dataset["momenta"].shape)
    print("accelerations:", dataset["accelerations"].shape)
    print("forces:", dataset["forces"].shape)
    print("masses:", dataset["masses"].shape)
    print("acc mean:", dataset["accelerations"].mean().item())
    print("acc std:", dataset["accelerations"].std().item())
    print("acc max abs:", dataset["accelerations"].abs().max().item())
    print(
        "acc 95th percentile:",
        torch.quantile(dataset["accelerations"].abs().flatten(), 0.95).item(),
    )
    print(
        "acc 99th percentile:",
        torch.quantile(dataset["accelerations"].abs().flatten(), 0.99).item(),
    )
    print("rejected trajectories:", dataset["metadata"]["rejected_trajectories"])


if __name__ == "__main__":
    main()

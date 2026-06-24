import argparse
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GNS_ROOT) not in sys.path:
    sys.path.insert(0, str(GNS_ROOT))

from common.initial_conditions import (
    get_condition,
    load_initial_conditions,
    normalize_masses,
    parse_json_tensor,
    resolve_rollout_dt,
    validate_body_tensor,
)
from common.nbody_data import simulate_trajectory_from_initial_conditions
from common.rollout.differentiable import rollout_steps
from common.visualize import animate_trajectories, plot_trajectories

from models.graph_network import EncodeProcessDecode
from models.learned_simulator import LearnedSimulator
from rollouts.gns_adapter import make_gns_step_fn, pack_gns_state, unpack_gns_state

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
if "--no-show" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def rollout(simulator, initial_positions, initial_velocities, masses, num_steps):
    """
    Autoregressively roll out the learned simulator.

    Each predicted state becomes the next input state.
    """

    simulator.eval()

    dim = initial_positions.shape[-1]
    initial_state = pack_gns_state(initial_positions, initial_velocities)
    step_fn = make_gns_step_fn(simulator, masses, dim=dim)

    with torch.no_grad():
        predicted_states = rollout_steps(
            initial_state=initial_state,
            step_fn=step_fn,
            num_steps=num_steps,
            # setting this to true is redudant because we are in a no_grad context but we leave it anyways
            detach_between_steps=True,
        )

    predicted_positions, predicted_velocities = unpack_gns_state(
        predicted_states,
        dim=dim,
    )

    return predicted_positions, predicted_velocities


def load_checkpoint(checkpoint_path, dataset, device):
    """
    Load either a full checkpoint dictionary or an older raw state_dict.

    Full checkpoint format should contain:
        model_state_dict
        acc_mean
        acc_std
        model_config
        dt

    If an older raw state_dict is found, this function uses default config
    and recomputes acc_mean/acc_std from the dataset.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        acc_mean = checkpoint["acc_mean"].to(device)
        acc_std = checkpoint["acc_std"].to(device)
        config = checkpoint["model_config"]
        dt = checkpoint["dt"]
    else:
        print("Loaded older raw state_dict checkpoint.")
        print(
            "Using default model config and recomputing acc_mean/acc_std from dataset."
        )

        model_state_dict = checkpoint

        accelerations = dataset["accelerations"].to(device)

        acc_mean = accelerations.mean(dim=(0, 1, 2), keepdim=True)
        acc_std = accelerations.std(dim=(0, 1, 2), keepdim=True) + 1e-8
        processor_indices = {
            int(key.split(".")[1])
            for key in model_state_dict
            if key.startswith("processors.")
        }

        config = {
            "node_input_dim": model_state_dict["node_encoder.net.0.weight"].shape[1],
            "edge_input_dim": model_state_dict["edge_encoder.net.0.weight"].shape[1],
            "output_dim": model_state_dict["node_decoder.net.4.weight"].shape[0],
            "latent_dim": model_state_dict["node_encoder.net.4.weight"].shape[0],
            "hidden_dim": model_state_dict["node_encoder.net.0.weight"].shape[0],
            "num_message_passing_steps": len(processor_indices),
        }

        dt = 0.01

    if "node_mean" not in model_state_dict:
        model_state_dict["node_mean"] = torch.zeros(
            config["node_input_dim"], device=device
        )
    if "node_std" not in model_state_dict:
        model_state_dict["node_std"] = torch.ones(
            config["node_input_dim"], device=device
        )
    if "edge_mean" not in model_state_dict:
        model_state_dict["edge_mean"] = torch.zeros(
            config["edge_input_dim"], device=device
        )
    if "edge_std" not in model_state_dict:
        model_state_dict["edge_std"] = torch.ones(
            config["edge_input_dim"], device=device
        )

    return model_state_dict, acc_mean, acc_std, config, dt


def choose_dynamic_trajectory(positions):
    displacements = torch.linalg.vector_norm(
        positions[:, -1] - positions[:, 0],
        dim=-1,
    )
    mean_displacements = displacements.mean(dim=-1)
    return torch.argmax(mean_displacements).item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roll out and animate the learned N-body graph simulator."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "nbody_3body_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "gns" / "rollout.pt"
        ),
        help="Checkpoint to evaluate. Defaults to experiments/checkpoints/gns/rollout.pt.",
    )
    parser.add_argument("--traj-idx", type=int, default=0)
    parser.add_argument(
        "--initial-conditions",
        default=None,
        help=(
            "Optional JSON file with positions, velocities, and masses for a "
            "custom rollout initial state."
        ),
    )
    parser.add_argument(
        "--initial-positions",
        default=None,
        help="Inline JSON array with shape [num_bodies, dim].",
    )
    parser.add_argument(
        "--initial-velocities",
        default=None,
        help="Inline JSON array with shape [num_bodies, dim].",
    )
    parser.add_argument(
        "--masses",
        default=None,
        help="Inline JSON array with one mass per body.",
    )
    parser.add_argument(
        "--traj-mode",
        choices=["indexed", "dynamic"],
        default="indexed",
        help="Use traj-idx directly, or choose the trajectory with the most motion.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Number of predicted steps. Defaults to the full saved trajectory.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional rollout timestep override for custom initial conditions.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help=(
            "Optional orbit period for custom initial conditions. If provided "
            "without --dt, uses dt = period / rollout_steps."
        ),
    )
    parser.add_argument(
        "--true-epsilon",
        type=float,
        default=None,
        help="Optional epsilon override for the generated true trajectory.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Delay between animation frames in milliseconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS used when saving. Defaults to 1000 / interval.",
    )
    parser.add_argument("--save-path", default=None)
    parser.add_argument(
        "--style",
        choices=["dark", "clean"],
        default="dark",
        help="Animation visual style.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=None,
        help="Number of recent frames to show in trails. Defaults to full trails.",
    )
    parser.add_argument("--skip-static-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    dataset = torch.load(args.dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    masses = dataset["masses"].to(device)
    metadata = dataset.get("metadata", {})

    print("Using checkpoint:", args.checkpoint_path)
    model_state_dict, acc_mean, acc_std, config, dt = load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        dataset=dataset,
        device=device,
    )

    graph_network = EncodeProcessDecode(
        node_input_dim=config["node_input_dim"],
        edge_input_dim=config["edge_input_dim"],
        output_dim=config["output_dim"],
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        num_message_passing_steps=config["num_message_passing_steps"],
        node_mean=model_state_dict.get("node_mean"),
        node_std=model_state_dict.get("node_std"),
        edge_mean=model_state_dict.get("edge_mean"),
        edge_std=model_state_dict.get("edge_std"),
    ).to(device)

    graph_network.load_state_dict(model_state_dict)

    simulator = LearnedSimulator(
        graph_network=graph_network,
        acc_mean=acc_mean,
        acc_std=acc_std,
        dt=dt,
        edge_feature_dim=config["edge_input_dim"],
    ).to(device)

    conditions = load_initial_conditions(args.initial_conditions)
    custom_initial_positions = parse_json_tensor(
        get_condition(
            conditions,
            args.initial_positions,
            "positions",
            "initial_positions",
        ),
        device=device,
        name="initial_positions",
    )
    custom_initial_velocities = parse_json_tensor(
        get_condition(
            conditions,
            args.initial_velocities,
            "velocities",
            "initial_velocities",
        ),
        device=device,
        name="initial_velocities",
    )
    custom_masses = parse_json_tensor(
        get_condition(conditions, args.masses, "masses"),
        device=device,
        name="masses",
    )
    use_custom_initial_conditions = any(
        value is not None
        for value in [custom_initial_positions, custom_initial_velocities, custom_masses]
    )

    if use_custom_initial_conditions:
        if (
            custom_initial_positions is None
            or custom_initial_velocities is None
            or custom_masses is None
        ):
            raise ValueError(
                "Custom GNS rollouts require positions, velocities, and masses."
            )

        initial_positions = validate_body_tensor(
            custom_initial_positions,
            name="initial_positions",
            dim=config["output_dim"],
        )
        initial_velocities = validate_body_tensor(
            custom_initial_velocities,
            name="initial_velocities",
            num_bodies=initial_positions.shape[0],
            dim=initial_positions.shape[1],
        )
        masses_t = normalize_masses(custom_masses, initial_positions.shape[0])
        rollout_steps = (
            args.rollout_steps
            if args.rollout_steps is not None
            else positions.shape[1] - 1
        )
        rollout_dt = resolve_rollout_dt(
            dt,
            rollout_steps,
            dt_override=args.dt,
            period=args.period,
        )
        true_epsilon = (
            args.true_epsilon
            if args.true_epsilon is not None
            else metadata.get("epsilon", 0.15)
        )
        simulator.dt = rollout_dt
        true_positions, _true_velocities, _ = simulate_trajectory_from_initial_conditions(
            positions=initial_positions,
            velocities=initial_velocities,
            masses=masses_t,
            num_steps=rollout_steps,
            dt=rollout_dt,
            G=metadata.get("G", 1.0),
            epsilon=true_epsilon,
        )
        print("Using custom initial conditions with generated true trajectory.")
        print(f"rollout dt: {rollout_dt}")
        print(f"true epsilon: {true_epsilon}")
    else:
        rollout_dt = dt
        if args.traj_mode == "dynamic":
            traj_idx = choose_dynamic_trajectory(positions)
            print(f"Selected dynamic trajectory: {traj_idx}")
        else:
            traj_idx = args.traj_idx

        max_rollout_steps = positions.shape[1] - 1

        if args.rollout_steps is None:
            rollout_steps = max_rollout_steps
        else:
            rollout_steps = min(args.rollout_steps, max_rollout_steps)

        if traj_idx < 0 or traj_idx >= positions.shape[0]:
            raise ValueError(f"traj_idx must be between 0 and {positions.shape[0] - 1}.")

        if args.rollout_steps is not None and args.rollout_steps > max_rollout_steps:
            print(
                f"Requested {args.rollout_steps} rollout steps, but this dataset only "
                f"supports {max_rollout_steps}. Using {rollout_steps}."
            )

        initial_positions = positions[traj_idx, 0]
        initial_velocities = velocities[traj_idx, 0]
        masses_t = masses[traj_idx]
        true_positions = positions[traj_idx, : rollout_steps + 1]

    predicted_positions, predicted_velocities = rollout(
        simulator=simulator,
        initial_positions=initial_positions,
        initial_velocities=initial_velocities,
        masses=masses_t,
        num_steps=rollout_steps,
    )

    initial_distances = torch.cdist(predicted_positions[0], predicted_positions[0])
    pair_mask = ~torch.eye(
        initial_distances.shape[0],
        dtype=torch.bool,
        device=initial_distances.device,
    )

    print("predicted_positions shape:", predicted_positions.shape)
    print("initial pair distances:", initial_distances[pair_mask].tolist())
    if true_positions is not None:
        position_mse = torch.mean((predicted_positions - true_positions) ** 2)
        true_displacement = torch.linalg.vector_norm(
            true_positions[-1] - true_positions[0],
            dim=-1,
        )
        print("true_positions shape:", true_positions.shape)
        print("position rollout MSE:", position_mse.item())
        print("true final displacement per body:", true_displacement.tolist())
    print(
        "animation length:",
        f"{predicted_positions.shape[0] * args.interval / 1000:.2f} seconds",
    )

    if not args.skip_static_plot:
        if true_positions is None:
            plot_trajectories(
                true_positions=None,
                predicted_positions=predicted_positions,
                title="Predicted GNS rollout",
                show=not args.no_show,
            )
        else:
            plot_trajectories(
                true_positions=true_positions,
                predicted_positions=predicted_positions,
                title="True vs learned GNS rollout",
                show=not args.no_show,
            )

    if args.no_show and args.save_path is None:
        print("Skipping animation display because --no-show was provided.")
    else:
        animate_trajectories(
            true_positions=true_positions,
            predicted_positions=predicted_positions,
            masses=masses_t,
            dt=rollout_dt,
            interval=args.interval,
            save_path=args.save_path,
            fps=args.fps,
            show=not args.no_show,
            style=args.style,
            trail_length=args.trail_length,
        )


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HNN_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HNN_ROOT) not in sys.path:
    sys.path.insert(0, str(HNN_ROOT))

import torch
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
from common.rollout.integrators import rk4_step
from common.visualize import animate_trajectories, plot_trajectories
from models.hamiltonian_network import HNN
from rollouts.hnn_adapter import make_hnn_step_fn, pack_hnn_state, unpack_hnn_state

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

if "--no-show" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ROLLOUT


def rollout(model, initial_positions, initial_momenta, masses, num_steps, dt):

    model.eval()

    num_bodies, dim = initial_positions.shape

    initial_state = pack_hnn_state(initial_positions, initial_momenta)
    # We don't use with torch,no_grad(): here because we need gradients still for differentiation.
    step_fn = make_hnn_step_fn(model, masses, dt, num_bodies, rk4_step)
    predicted_states = rollout_steps(
        initial_state, step_fn, num_steps, detach_between_steps=True
    )

    positions, momenta = unpack_hnn_state(predicted_states, num_bodies, dim)

    if positions.shape[1] == 1:
        positions = positions.squeeze(1)
        momenta = momenta.squeeze(1)

    return positions, momenta


def load_checkpoint(checkpoint_path, dataset, device, default_hidden_dim=128):
    #   checkpoint file
    #   ↓
    #   read model_config
    #   ↓
    #   recreate HNN with same input_dim and hidden_dim
    #   ↓
    #   load model_state_dict into that HNN
    #   ↓
    #   use model for rollout
    checkpoint = torch.load(checkpoint_path, map_location=device)

    dataset_metadata = dataset.get("metadata", {})

    positions = dataset["positions"]
    _, _, num_bodies, dim = positions.shape

    q_dim = num_bodies * dim
    mass_dim = num_bodies
    input_dim = 2 * q_dim + mass_dim

    default_model_config = {
        "input_dim": input_dim,
        "hidden_dim": default_hidden_dim,
        "num_bodies": num_bodies,
        "dim": dim,
        "q_dim": q_dim,
    }

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]  # Learned weights
        model_config = checkpoint.get("model_config", default_model_config)
        training_config = checkpoint.get("training_config", {})
        checkpoint_metadata = checkpoint.get("dataset_metadata", {})
        dt = checkpoint_metadata.get("dt", dataset_metadata.get("dt", 0.01))

    # This case is for if someone saved the model weights directly
    # torch.save(model.state_dict(), "experiments/checkpoints/hnn/one_step.pt")
    elif isinstance(checkpoint, dict):
        print("Using default model config")

        model_state_dict = checkpoint
        model_config = default_model_config
        training_config = {}
        dt = dataset_metadata.get("dt", 0.01)

    else:
        raise ValueError(
            "Checkpoint format not recognized. Expected a full checkpoint with model_state_dict or a raw model.state_dict()"
        )

    model = HNN(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        q_dim=model_config.get("q_dim"),
    ).to(device)
    # loading the weights into the model
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, model_config, training_config, dt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "nbody_3body_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "hnn" / "rollout.pt"
        ),
        help="Checkpoint to evaluate. Defaults to experiments/checkpoints/hnn/rollout.pt.",
    )
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--skip-static-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--traj-idx", type=int, default=0)
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
        "--initial-conditions",
        default=None,
        help=(
            "Optional JSON file with positions, masses, and either momenta or "
            "velocities for a custom rollout initial state."
        ),
    )
    parser.add_argument(
        "--initial-positions",
        default=None,
        help="Inline JSON array with shape [num_bodies, dim].",
    )
    parser.add_argument(
        "--initial-momenta",
        default=None,
        help="Inline JSON array with shape [num_bodies, dim].",
    )
    parser.add_argument(
        "--initial-velocities",
        default=None,
        help="Inline JSON array with shape [num_bodies, dim]. Converted to momenta.",
    )
    parser.add_argument(
        "--masses",
        default=None,
        help="Inline JSON array with one mass per body.",
    )
    parser.add_argument("--fps", type=int, default=None)  # Save FPS.
    parser.add_argument(
        "--style", choices=["dark", "clean"], default="dark"
    )  # Visual style.
    parser.add_argument("--trail-length", type=int, default=None)

    return parser.parse_args()


def main():

    args = parse_args()
    device = get_device()

    print("Using device: ", device)

    dataset = torch.load(args.dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    momenta = dataset["momenta"].to(device)
    masses = dataset["masses"].to(device)
    metadata = dataset.get("metadata", {})

    print("Using checkpoint:", args.checkpoint_path)
    model, model_config, _, dt = load_checkpoint(args.checkpoint_path, dataset, device)

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
    custom_initial_momenta = parse_json_tensor(
        get_condition(
            conditions,
            args.initial_momenta,
            "momenta",
            "initial_momenta",
        ),
        device=device,
        name="initial_momenta",
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
        for value in [
            custom_initial_positions,
            custom_initial_momenta,
            custom_initial_velocities,
            custom_masses,
        ]
    )

    if use_custom_initial_conditions:
        if custom_initial_positions is None or custom_masses is None:
            raise ValueError("Custom HNN rollouts require positions and masses.")
        if custom_initial_momenta is None and custom_initial_velocities is None:
            raise ValueError("Custom HNN rollouts require momenta or velocities.")
        if custom_initial_momenta is not None and custom_initial_velocities is not None:
            raise ValueError("Provide either momenta or velocities, not both.")

        init_pos = validate_body_tensor(
            custom_initial_positions,
            name="initial_positions",
            num_bodies=model_config.get("num_bodies"),
            dim=model_config.get("dim"),
        )
        masses_t = normalize_masses(custom_masses, init_pos.shape[0])

        if custom_initial_momenta is not None:
            init_mom = validate_body_tensor(
                custom_initial_momenta,
                name="initial_momenta",
                num_bodies=init_pos.shape[0],
                dim=init_pos.shape[1],
            )
            initial_velocities = init_mom / masses_t
        else:
            initial_velocities = validate_body_tensor(
                custom_initial_velocities,
                name="initial_velocities",
                num_bodies=init_pos.shape[0],
                dim=init_pos.shape[1],
            )
            init_mom = masses_t * initial_velocities

        rollout_steps_count = (
            args.rollout_steps
            if args.rollout_steps is not None
            else positions.shape[1] - 1
        )
        rollout_dt = resolve_rollout_dt(
            dt,
            rollout_steps_count,
            dt_override=args.dt,
            period=args.period,
        )
        true_epsilon = (
            args.true_epsilon
            if args.true_epsilon is not None
            else metadata.get("epsilon", 0.15)
        )
        true_positions, _true_velocities, _ = simulate_trajectory_from_initial_conditions(
            positions=init_pos,
            velocities=initial_velocities,
            masses=masses_t,
            num_steps=rollout_steps_count,
            dt=rollout_dt,
            G=metadata.get("G", 1.0),
            epsilon=true_epsilon,
        )
        print("Using custom initial conditions with generated true trajectory.")
        print(f"rollout dt: {rollout_dt}")
        print(f"true epsilon: {true_epsilon}")
    else:
        rollout_dt = dt
        if args.traj_idx < 0 or args.traj_idx >= positions.shape[0]:
            raise ValueError(f"traj_idx must be between 0 and {positions.shape[0] - 1}.")

        max_rollout_steps = positions.shape[1] - 1
        rollout_steps_count = (
            max_rollout_steps
            if args.rollout_steps is None
            else min(args.rollout_steps, max_rollout_steps)
        )
        if args.rollout_steps is not None and args.rollout_steps > max_rollout_steps:
            print(
                f"Requested {args.rollout_steps} rollout steps, but this dataset only "
                f"supports {max_rollout_steps}. Using {rollout_steps_count}."
            )

        init_pos = positions[args.traj_idx, 0]
        init_mom = momenta[args.traj_idx, 0]
        masses_t = masses[args.traj_idx]
        true_positions = positions[args.traj_idx, : rollout_steps_count + 1]

    pred_positions, _ = rollout(
        model,
        init_pos,
        init_mom,
        masses_t,
        num_steps=rollout_steps_count,
        dt=rollout_dt,
    )

    print("predicted_positions shape:", pred_positions.shape)
    if true_positions is not None:
        position_mse = torch.mean((pred_positions - true_positions) ** 2)
        print("true_positions shape:", true_positions.shape)
        print("position rollout MSE:", position_mse.item())  # Debug error.

    if not args.skip_static_plot:
        if true_positions is None:
            plot_trajectories(
                true_positions=None,
                predicted_positions=pred_positions,
                title="Predicted HNN rollout",
                show=not args.no_show,
            )
        else:
            plot_trajectories(
                true_positions=true_positions,
                predicted_positions=pred_positions,
                show=not args.no_show,
            )

    if args.no_show and args.save_path is None:  # Nothing to display or save.
        print("Skipping animation display because --no-show was provided.")
    else:  # Display and/or save animation.
        animate_trajectories(  # Build the animation.
            true_positions=true_positions,
            predicted_positions=pred_positions,
            masses=masses_t,  # Marker size scaling.
            dt=rollout_dt,  # Time label.
            interval=args.interval,  # Playback interval.
            save_path=args.save_path,  # Optional export path.
            fps=args.fps,  # Optional save FPS.
            show=not args.no_show,  # Show unless disabled.
            style=args.style,  # Dark or clean.
            trail_length=args.trail_length,  # Full or recent trails.
        )


if __name__ == "__main__":
    main()

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
from common.rollout.test_suite import (
    add_test_suite_arguments,
    evaluate_fixed_test_suite,
    should_run_test_suite,
)
from common.splits import (
    load_split_manifest,
    resolve_project_path,
    resolve_split_trajectory_index,
    validate_checkpoint_split,
)
from common.visualize import animate_trajectories, plot_trajectories
from models.hamiltonian_network import HNN
from rollouts.hnn_adapter import (
    make_hnn_acceleration_fn,
    make_hnn_step_fn,
    pack_hnn_state,
    unpack_hnn_state,
)

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


def rollout(
    model,
    initial_positions,
    initial_velocities,
    masses,
    force_std,
    num_steps,
    dt,
):
    model.eval()

    num_bodies, dim = initial_positions.shape
    initial_state = pack_hnn_state(initial_positions, initial_velocities)
    step_fn = make_hnn_step_fn(
        model=model,
        masses=masses,
        force_std=force_std,
        dt=dt,
        num_bodies=num_bodies,
        dim=dim,
    )

    with torch.no_grad():
        predicted_states = rollout_steps(
            initial_state,
            step_fn,
            num_steps,
            detach_between_steps=True,
        )

    positions, velocities = unpack_hnn_state(predicted_states, num_bodies, dim)

    if positions.shape[1] == 1:
        positions = positions.squeeze(1)
        velocities = velocities.squeeze(1)

    return positions, velocities


def load_checkpoint(
    checkpoint_path,
    dataset,
    device,
    train_indices=None,
    default_hidden_dim=128,
    split_manifest=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if split_manifest is not None:
        validate_checkpoint_split(checkpoint, split_manifest)

    dataset_metadata = dataset.get("metadata", {})
    positions = dataset["positions"]
    _, _, num_bodies, dim = positions.shape
    spatial_dim = num_bodies * dim
    mass_dim = num_bodies
    input_dim = spatial_dim + mass_dim

    default_model_config = {
        "input_dim": input_dim,
        "hidden_dim": default_hidden_dim,
        "num_bodies": num_bodies,
        "dim": dim,
        "spatial_dim": spatial_dim,
        "mass_dim": mass_dim,
        "output_dim": dim,
    }

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "force_std" not in checkpoint:
            raise ValueError(
                "This checkpoint predates potential-only HNN force learning. "
                "Retrain HNN/train.py before evaluating with velocity Verlet."
            )

        model_state_dict = checkpoint["model_state_dict"]
        force_std = checkpoint["force_std"].to(device)
        model_config = checkpoint.get("model_config", default_model_config)
        training_config = checkpoint.get("training_config", {})
        checkpoint_metadata = checkpoint.get("dataset_metadata", {})
        dt = checkpoint.get(
            "dt",
            checkpoint_metadata.get("dt", dataset_metadata.get("dt", 0.01)),
        )
    elif isinstance(checkpoint, dict):
        print("Using raw state_dict checkpoint.")
        print("Recomputing force_std from manifest training trajectories.")

        model_state_dict = checkpoint
        forces = dataset["forces"].to(device)
        if train_indices is not None:
            forces = forces[train_indices]
        force_std = forces.std().clamp_min(1e-8).reshape(1)
        model_config = {
            **default_model_config,
            "input_dim": model_state_dict["net.0.weight"].shape[1],
            "hidden_dim": model_state_dict["net.0.weight"].shape[0],
        }
        training_config = {}
        dt = dataset_metadata.get("dt", 0.01)
    else:
        raise ValueError(
            "Checkpoint format not recognized. Expected a full checkpoint with "
            "model_state_dict or a raw model.state_dict()."
        )

    checkpoint_spatial_dim = model_config.get(
        "spatial_dim", model_config.get("q_dim", spatial_dim)
    )
    model_config.setdefault("spatial_dim", checkpoint_spatial_dim)
    model_config.setdefault("mass_dim", mass_dim)
    model_config.setdefault("output_dim", dim)

    if checkpoint_spatial_dim != spatial_dim:
        raise ValueError(
            "Checkpoint spatial_dim does not match the dataset. "
            f"Got {checkpoint_spatial_dim}, expected {spatial_dim}."
        )
    if model_config["input_dim"] != input_dim:
        raise ValueError(
            "Potential-only HNN checkpoints must use input_dim = num_bodies * dim "
            "+ num_bodies. "
            f"Got checkpoint input_dim={model_config['input_dim']}, expected "
            f"{input_dim}. Retrain HNN/train.py."
        )

    model = HNN(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        spatial_dim=checkpoint_spatial_dim,
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, model_config, training_config, force_std, dt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "datasets" / "nbody_dataset.pt"),
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
    parser.add_argument(
        "--static-save-path",
        "--figure-save-path",
        dest="static_save_path",
        default=None,
        help="Optional PNG, PDF, or SVG path for the static trajectory plot.",
    )
    parser.add_argument("--skip-static-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--split-path",
        default=str(PROJECT_ROOT / "experiments" / "splits" / "nbody_dataset.json"),
        help="Trajectory-level train/validation/test split manifest.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "val", "test"],
        default="test",
        help="Manifest split from which dataset trajectories may be evaluated.",
    )
    add_test_suite_arguments(parser)
    parser.add_argument(
        "--traj-idx",
        type=int,
        default=None,
        help="Global dataset trajectory ID. Defaults to the first ID in eval-split.",
    )
    parser.add_argument(
        "--traj-number",
        type=int,
        default=None,
        help="One-based trajectory number within eval-split (for example, 1-500).",
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
        help="Inline JSON array with shape [num_bodies, dim]. Converted to velocities.",
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
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--style", choices=["dark", "clean"], default="dark")
    parser.add_argument("--trail-length", type=int, default=None)
    parser.add_argument(
        "--loop-animation",
        action="store_true",
        help="Loop an interactive 3D animation until its window is closed.",
    )
    parser.add_argument(
        "--predicted-only",
        action="store_true",
        help="Hide reference trajectories and visualize only model predictions.",
    )
    parser.add_argument(
        "--bloom",
        action="store_true",
        help="Apply post-processed bloom when saving a 3D animation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("Using device: ", device)

    dataset_path = resolve_project_path(args.dataset_path)
    split_path = resolve_project_path(args.split_path)
    checkpoint_path = resolve_project_path(args.checkpoint_path)
    static_save_path = (
        resolve_project_path(args.static_save_path)
        if args.static_save_path is not None
        else None
    )
    dataset = torch.load(dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    accelerations = dataset["accelerations"].to(device)
    masses = dataset["masses"].to(device)
    metadata = dataset.get("metadata", {})

    manifest, split_indices = load_split_manifest(
        split_path=split_path,
        dataset_path=dataset_path,
        dataset_shape=positions.shape,
    )
    allowed_indices = split_indices[args.eval_split]

    print("Using checkpoint:", checkpoint_path)
    model, model_config, _, force_std, dt = load_checkpoint(
        checkpoint_path,
        dataset,
        device,
        train_indices=split_indices["train"],
        split_manifest=manifest,
    )

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

    if should_run_test_suite(args, use_custom_initial_conditions):
        test_suite_output = (
            resolve_project_path(args.test_suite_output)
            if args.test_suite_output is not None
            else None
        )
        evaluate_fixed_test_suite(
            model_name="HNN",
            rollout_fn=lambda initial_positions, initial_velocities, masses_t, steps: rollout(
                model=model,
                initial_positions=initial_positions,
                initial_velocities=initial_velocities,
                masses=masses_t,
                force_std=force_std,
                num_steps=steps,
                dt=dt,
            ),
            acceleration_prediction_fn=lambda positions_t, _velocities_t, masses_t: make_hnn_acceleration_fn(
                model=model,
                masses=masses_t,
                force_std=force_std,
                num_bodies=positions_t.shape[-2],
                dim=positions_t.shape[-1],
            )(positions_t),
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            masses=masses,
            allowed_indices=allowed_indices,
            split_name=args.eval_split,
            num_trajectories=args.num_test_trajectories,
            requested_rollout_steps=args.rollout_steps,
            failure_threshold=args.failure_threshold,
            gravitational_constant=float(metadata.get("G", 1.0)),
            softening_epsilon=float(metadata.get("epsilon", 0.15)),
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            split_path=split_path,
            output_path=test_suite_output,
        )
        return

    if use_custom_initial_conditions:
        if custom_initial_positions is None or custom_masses is None:
            raise ValueError("Custom HNN rollouts require positions and masses.")
        if custom_initial_momenta is None and custom_initial_velocities is None:
            raise ValueError("Custom HNN rollouts require momenta or velocities.")
        if (
            custom_initial_momenta is not None
            and custom_initial_velocities is not None
        ):
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
            init_vel = init_mom / masses_t
        else:
            init_vel = validate_body_tensor(
                custom_initial_velocities,
                name="initial_velocities",
                num_bodies=init_pos.shape[0],
                dim=init_pos.shape[1],
            )

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
        true_positions, _true_velocities, _ = (
            simulate_trajectory_from_initial_conditions(
                positions=init_pos,
                velocities=init_vel,
                masses=masses_t,
                num_steps=rollout_steps_count,
                dt=rollout_dt,
                G=metadata.get("G", 1.0),
                epsilon=true_epsilon,
            )
        )
        print("Using custom initial conditions with generated true trajectory.")
        print(f"rollout dt: {rollout_dt}")
        print(f"true epsilon: {true_epsilon}")
    else:
        rollout_dt = dt
        traj_idx = resolve_split_trajectory_index(
            allowed_indices,
            args.eval_split,
            trajectory_index=args.traj_idx,
            trajectory_number=args.traj_number,
        )
        if args.traj_number is not None:
            print(
                f"Selected {args.eval_split} trajectory number "
                f"{args.traj_number}: global ID {traj_idx}"
            )
        print(f"Evaluating {args.eval_split} trajectory: {traj_idx}")

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

        init_pos = positions[traj_idx, 0]
        init_vel = velocities[traj_idx, 0]
        masses_t = masses[traj_idx]
        true_positions = positions[traj_idx, : rollout_steps_count + 1]

    pred_positions, _ = rollout(
        model=model,
        initial_positions=init_pos,
        initial_velocities=init_vel,
        masses=masses_t,
        force_std=force_std,
        num_steps=rollout_steps_count,
        dt=rollout_dt,
    )

    print("predicted_positions shape:", pred_positions.shape)
    if true_positions is not None:
        rollout_position_rmse = torch.sqrt(
            torch.mean((pred_positions - true_positions) ** 2)
        )
        print("true_positions shape:", true_positions.shape)
        print("rollout position RMSE:", rollout_position_rmse.item())

    display_true_positions = None if args.predicted_only else true_positions

    if not args.skip_static_plot:
        if display_true_positions is None:
            plot_trajectories(
                true_positions=None,
                predicted_positions=pred_positions,
                title="Predicted HNN rollout",
                show=not args.no_show,
                save_path=static_save_path,
            )
        else:
            plot_trajectories(
                true_positions=display_true_positions,
                predicted_positions=pred_positions,
                show=not args.no_show,
                save_path=static_save_path,
            )

    if args.no_show and args.save_path is None:
        print("Skipping animation display because --no-show was provided.")
    else:
        animate_trajectories(
            true_positions=display_true_positions,
            predicted_positions=pred_positions,
            masses=masses_t,
            dt=rollout_dt,
            interval=args.interval,
            save_path=args.save_path,
            fps=args.fps,
            show=not args.no_show,
            style=args.style,
            trail_length=args.trail_length,
            loop=args.loop_animation,
            bloom=args.bloom,
        )


if __name__ == "__main__":
    main()

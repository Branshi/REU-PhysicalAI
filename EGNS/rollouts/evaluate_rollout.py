import argparse
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EGNS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EGNS_ROOT) not in sys.path:
    sys.path.insert(0, str(EGNS_ROOT))

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


def load_checkpoint(
    checkpoint_path,
    dataset,
    device,
    train_indices=None,
    split_manifest=None,
):
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
    # New checkpoints record their split. Verify it before reporting a result as
    # validation/test performance under the supplied manifest.
    if split_manifest is not None:
        validate_checkpoint_split(checkpoint, split_manifest)

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
        # Rebuild missing legacy normalization from training data only.
        if train_indices is not None:
            accelerations = accelerations[train_indices]

        positions = dataset["positions"]
        _, _, _, dim = positions.shape

        acc_mean = torch.zeros(1, 1, 1, 1, device=device, dtype=accelerations.dtype)
        acc_std = accelerations.std().clamp_min(1e-8).view(1, 1, 1, 1)
        processor_indices = {
            int(key.split(".")[1])
            for key in model_state_dict
            if key.startswith("processors.")
        }

        config = {
            "node_input_dim": model_state_dict["node_encoder.net.0.weight"].shape[1],
            "edge_input_dim": model_state_dict["edge_encoder.net.0.weight"].shape[1],
            "output_dim": dim,
            "latent_dim": model_state_dict["node_encoder.net.4.weight"].shape[0],
            "hidden_dim": model_state_dict["node_encoder.net.0.weight"].shape[0],
            "num_message_passing_steps": len(processor_indices),
        }

        dt = dataset.get("metadata", {}).get("dt", 0.01)

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
        description="Roll out and animate the learned N-body equivariant graph simulator."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "datasets" / "nbody_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "egns" / "rollout.pt"
        ),
        help="Checkpoint to evaluate. Defaults to experiments/checkpoints/egns/rollout.pt.",
    )
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
        "--static-save-path",
        "--figure-save-path",
        dest="static_save_path",
        default=None,
        help="Optional PNG, PDF, or SVG path for the static trajectory plot.",
    )
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
    parser.add_argument("--skip-static-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

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

    # Evaluation defaults to held-out test trajectories. --eval-split can select
    # train or validation explicitly for diagnostics without changing the file.
    manifest, split_indices = load_split_manifest(
        split_path=split_path,
        dataset_path=dataset_path,
        dataset_shape=positions.shape,
    )
    allowed_indices = split_indices[args.eval_split]

    print("Using checkpoint:", checkpoint_path)
    model_state_dict, acc_mean, acc_std, config, dt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        device=device,
        train_indices=split_indices["train"],
        split_manifest=manifest,
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
        for value in [
            custom_initial_positions,
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
            model_name="EGNS",
            rollout_fn=lambda initial_positions, initial_velocities, masses_t, steps: rollout(
                simulator=simulator,
                initial_positions=initial_positions,
                initial_velocities=initial_velocities,
                masses=masses_t,
                num_steps=steps,
            ),
            acceleration_prediction_fn=lambda positions_t, velocities_t, masses_t: simulator.predict_acceleration(
                positions_t,
                velocities_t,
                masses_t,
            ),
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
        if (
            custom_initial_positions is None
            or custom_initial_velocities is None
            or custom_masses is None
        ):
            raise ValueError(
                "Custom EGNS rollouts require positions, velocities, and masses."
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
        true_positions, _true_velocities, _ = (
            simulate_trajectory_from_initial_conditions(
                positions=initial_positions,
                velocities=initial_velocities,
                masses=masses_t,
                num_steps=rollout_steps,
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
        if args.traj_mode == "dynamic":
            # Find the most dynamic trajectory within the allowed split, then map
            # the local result back to its global dataset trajectory ID.
            local_idx = choose_dynamic_trajectory(positions[allowed_indices])
            traj_idx = allowed_indices[local_idx]
            print(f"Selected dynamic trajectory: {traj_idx}")
        else:
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

        if args.rollout_steps is None:
            rollout_steps = max_rollout_steps
        else:
            rollout_steps = min(args.rollout_steps, max_rollout_steps)

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
        rollout_position_rmse = torch.sqrt(
            torch.mean((predicted_positions - true_positions) ** 2)
        )
        true_displacement = torch.linalg.vector_norm(
            true_positions[-1] - true_positions[0],
            dim=-1,
        )
        print("true_positions shape:", true_positions.shape)
        print("rollout position RMSE:", rollout_position_rmse.item())
        print("true final displacement per body:", true_displacement.tolist())
    print(
        "animation length:",
        f"{predicted_positions.shape[0] * args.interval / 1000:.2f} seconds",
    )

    display_true_positions = None if args.predicted_only else true_positions

    if not args.skip_static_plot:
        if display_true_positions is None:
            plot_trajectories(
                true_positions=None,
                predicted_positions=predicted_positions,
                title="Predicted EGNS rollout",
                show=not args.no_show,
                save_path=static_save_path,
            )
        else:
            plot_trajectories(
                true_positions=display_true_positions,
                predicted_positions=predicted_positions,
                title="True vs learned EGNS rollout",
                show=not args.no_show,
                save_path=static_save_path,
            )

    if args.no_show and args.save_path is None:
        print("Skipping animation display because --no-show was provided.")
    else:
        animate_trajectories(
            true_positions=display_true_positions,
            predicted_positions=predicted_positions,
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

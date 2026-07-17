import argparse
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HNN_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HNN_ROOT) not in sys.path:
    sys.path.insert(0, str(HNN_ROOT))

from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from common.rollout.differentiable import rollout_steps
from common.rollout.losses import (
    composite_rollout_loss,
    dynamics_to_position_scale,
    rollout_mse,
    sample_rollout_windows,
    teacher_forced_dynamics_mse,
)
from common.splits import (
    load_split_manifest,
    resolve_project_path,
    validate_checkpoint_split,
)
from models.hamiltonian_network import HNN
from rollouts.hnn_adapter import (
    make_hnn_acceleration_fn,
    make_hnn_step_fn,
    pack_hnn_state,
    unpack_hnn_state,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained potential HNN with rollout loss."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "datasets" / "nbody_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "hnn" / "one_step.pt"
        ),
        help="Pretrained one-step HNN checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "hnn" / "rollout.pt"
        ),
        help="Checkpoint path for the best rollout fine-tuned model.",
    )
    parser.add_argument(
        "--split-path",
        default=str(PROJECT_ROOT / "experiments" / "splits" / "nbody_dataset.json"),
        help="Trajectory-level train/validation/test split manifest.",
    )
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=100,
        help="Number of fixed validation rollout windows evaluated after each epoch.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=1234,
        help="Seed used once to choose the fixed validation rollout windows.",
    )
    parser.add_argument("--num-rollout-steps", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--velocity-loss-weight", type=float, default=None)
    parser.add_argument(
        "--momentum-loss-weight",
        type=float,
        default=None,
        help="Deprecated alias for --velocity-loss-weight.",
    )
    parser.add_argument(
        "--dynamics-loss-weight",
        type=float,
        default=1.0,
        help="Weight for calibrated normalized force MSE on true states.",
    )
    parser.add_argument("--num-loss-calibration-samples", type=int, default=50)
    parser.add_argument("--loss-calibration-seed", type=int, default=4321)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Stop training after this many wall-clock hours.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional MLflow run description shown in the web UI.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Run without creating an MLflow run. Useful for smoke tests.",
    )
    return parser.parse_args()


def resolve_velocity_loss_weight(args):
    if args.velocity_loss_weight is not None:
        return args.velocity_loss_weight
    if args.momentum_loss_weight is not None:
        return args.momentum_loss_weight
    return 0.0


def load_checkpoint(
    checkpoint_path,
    dataset,
    device,
    train_indices=None,
    split_manifest=None,
    default_hidden_dim=128,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if split_manifest is not None:
        validate_checkpoint_split(checkpoint, split_manifest)

    positions = dataset["positions"]
    _, _, num_bodies, dim = positions.shape
    spatial_dim = num_bodies * dim
    input_dim = spatial_dim + num_bodies

    default_model_config = {
        "input_dim": input_dim,
        "hidden_dim": default_hidden_dim,
        "num_bodies": num_bodies,
        "dim": dim,
        "spatial_dim": spatial_dim,
        "mass_dim": num_bodies,
        "output_dim": dim,
    }

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "force_std" not in checkpoint:
            raise ValueError(
                "This checkpoint predates potential-only HNN force learning. "
                "Retrain HNN/train.py before rollout fine-tuning."
            )
        model_state_dict = checkpoint["model_state_dict"]
        force_std = checkpoint["force_std"].to(device)
        model_config = checkpoint.get("model_config", default_model_config)
        dataset_metadata = checkpoint.get("dataset_metadata", {})
        dt = checkpoint.get(
            "dt",
            dataset_metadata.get("dt", dataset.get("metadata", {}).get("dt", 0.01)),
        )
    elif isinstance(checkpoint, dict):
        print("Loaded raw state_dict checkpoint.")
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
        dt = dataset.get("metadata", {}).get("dt", 0.01)
    else:
        raise ValueError(
            "Checkpoint format not recognized. Expected a full checkpoint with "
            "model_state_dict or a raw model.state_dict()."
        )

    checkpoint_spatial_dim = model_config.get(
        "spatial_dim", model_config.get("q_dim", spatial_dim)
    )
    model_config.setdefault("spatial_dim", checkpoint_spatial_dim)
    model_config.setdefault("mass_dim", num_bodies)
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
        spatial_dim=spatial_dim,
    ).to(device)
    model.load_state_dict(model_state_dict)

    return model, model_config, force_std, dt


def evaluate_validation_rollout(
    model,
    positions,
    velocities,
    accelerations,
    masses,
    force_std,
    dt,
    num_bodies,
    dim,
    num_rollout_steps,
    velocity_loss_weight,
    dynamics_loss_weight,
    dynamics_position_scale,
    validation_windows,
):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_position_mse = 0.0
    total_dynamics_mse = 0.0

    # HNN.force locally enables coordinate gradients to compute force, while
    # eval mode disables the higher-order graph used for parameter updates.
    with torch.no_grad():
        for traj_idx, time_idx in validation_windows:
            positions_t = positions[traj_idx, time_idx]
            velocities_t = velocities[traj_idx, time_idx]
            masses_t = masses[traj_idx]

            initial_state = pack_hnn_state(positions_t, velocities_t)
            step_fn = make_hnn_step_fn(
                model=model,
                masses=masses_t,
                force_std=force_std,
                dt=dt,
                num_bodies=num_bodies,
                dim=dim,
            )
            predicted_states = rollout_steps(
                initial_state=initial_state,
                step_fn=step_fn,
                num_steps=num_rollout_steps,
                detach_between_steps=False,
            )
            predicted_positions, predicted_velocities = unpack_hnn_state(
                predicted_states,
                num_bodies=num_bodies,
                dim=dim,
            )

            if predicted_positions.shape[1] == 1:
                predicted_positions = predicted_positions.squeeze(1)
                predicted_velocities = predicted_velocities.squeeze(1)

            target_start = time_idx
            target_end = time_idx + num_rollout_steps + 1
            target_positions = positions[traj_idx, target_start:target_end]
            target_velocities = velocities[traj_idx, target_start:target_end]
            target_accelerations = accelerations[
                traj_idx, target_start : target_end - 1
            ]

            position_mse = rollout_mse(
                predicted_positions[1:], target_positions[1:]
            )
            velocity_mse = rollout_mse(
                predicted_velocities[1:], target_velocities[1:]
            )
            predict_acceleration = make_hnn_acceleration_fn(
                model=model,
                masses=masses_t,
                force_std=force_std,
                num_bodies=num_bodies,
                dim=dim,
            )
            dynamics_mse = teacher_forced_dynamics_mse(
                positions=target_positions[:-1],
                velocities=target_velocities[:-1],
                target_accelerations=target_accelerations,
                masses=masses_t,
                predict_acceleration=lambda position, _velocity: predict_acceleration(
                    position
                ),
                normalization_scale=force_std,
                target_kind="force",
            )
            loss = composite_rollout_loss(
                position_mse,
                velocity_mse=velocity_mse,
                velocity_loss_weight=velocity_loss_weight,
                dynamics_mse=dynamics_mse,
                dynamics_loss_weight=dynamics_loss_weight,
                dynamics_position_scale=dynamics_position_scale,
            )

            total_loss += loss.item()
            total_position_mse += position_mse.item()
            total_dynamics_mse += dynamics_mse.item()

    if was_training:
        model.train()

    num_windows = len(validation_windows)
    mean_position_mse = total_position_mse / num_windows
    mean_dynamics_mse = total_dynamics_mse / num_windows
    return {
        "objective": total_loss / num_windows,
        "position_mse": mean_position_mse,
        "position_rmse": mean_position_mse**0.5,
        "dynamics_mse": mean_dynamics_mse,
        "dynamics_rmse": mean_dynamics_mse**0.5,
    }


def main():
    args = parse_args()
    velocity_loss_weight = resolve_velocity_loss_weight(args)
    device = get_device()
    print("Using device:", device)

    output_path, run_dir, metrics_path = prepare_run_outputs(
        project_root=PROJECT_ROOT,
        run_name="hnn_rollout",
        args=args,
        output_path=args.output_path,
    )
    if run_dir is not None:
        print("Run directory:", run_dir)

    mlflow_logger = MLflowLogger(
        project_root=PROJECT_ROOT,
        experiment_name="hnn_rollout",
        run_name=run_dir.name if run_dir is not None else output_path.stem,
        tags={"model": "HNN", "training_stage": "rollout", "device": device},
        enabled=not args.disable_mlflow,
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_params({"velocity_loss_weight_resolved": velocity_loss_weight})
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    dataset_path = resolve_project_path(args.dataset_path)
    split_path = resolve_project_path(args.split_path)
    checkpoint_path = resolve_project_path(args.checkpoint_path)
    dataset = torch.load(dataset_path, map_location=device)
    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    accelerations = dataset["accelerations"].to(device)
    masses = dataset["masses"].to(device)

    num_trajectories, num_steps, num_bodies, dim = positions.shape
    manifest, split_indices = load_split_manifest(
        split_path=split_path,
        dataset_path=dataset_path,
        dataset_shape=positions.shape,
    )
    train_indices = split_indices["train"]
    validation_trajectory_indices = split_indices["val"]
    validation_positions = positions[validation_trajectory_indices]
    validation_velocities = velocities[validation_trajectory_indices]
    validation_accelerations = accelerations[validation_trajectory_indices]
    validation_masses = masses[validation_trajectory_indices]
    num_train_trajectories = len(train_indices)
    num_validation_trajectories = len(validation_trajectory_indices)

    if args.num_rollout_steps <= 0:
        raise ValueError("num-rollout-steps must be positive.")
    if args.num_rollout_steps >= num_steps:
        raise ValueError(
            "num-rollout-steps must be smaller than the number of saved "
            f"trajectory steps. Got {args.num_rollout_steps} for {num_steps} steps."
        )
    if args.dynamics_loss_weight < 0.0:
        raise ValueError("dynamics-loss-weight must be nonnegative.")
    if args.num_loss_calibration_samples <= 0:
        raise ValueError("num-loss-calibration-samples must be positive.")

    model, model_config, force_std, dt = load_checkpoint(
        checkpoint_path,
        dataset,
        device,
        train_indices=train_indices,
        split_manifest=manifest,
    )
    mlflow_logger.log_params({
        "data": {
            "num_trajectories": num_trajectories,
            "num_steps": num_steps,
            "num_bodies": num_bodies,
            "dim": dim,
            "num_train_trajectories": num_train_trajectories,
            "num_val_trajectories": len(split_indices["val"]),
            "num_test_trajectories": len(split_indices["test"]),
            "num_validation_samples": args.num_validation_samples,
            "dt": dt,
        },
        "split": {
            "manifest_path": str(split_path),
            **manifest.get("split_config", {}),
        },
        "pretrained_model": model_config,
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    if args.num_validation_samples <= 0:
        raise ValueError("num-validation-samples must be positive.")

    validation_generator = torch.Generator().manual_seed(args.validation_seed)
    validation_trajectories = torch.randint(
        0,
        num_validation_trajectories,
        (args.num_validation_samples,),
        generator=validation_generator,
    )
    validation_times = torch.randint(
        0,
        num_steps - args.num_rollout_steps,
        (args.num_validation_samples,),
        generator=validation_generator,
    )
    validation_windows = list(
        zip(
            validation_trajectories.tolist(),
            validation_times.tolist(),
        )
    )
    validation_global_windows = [
        (validation_trajectory_indices[traj_idx], time_idx)
        for traj_idx, time_idx in validation_windows
    ]

    calibration_windows = sample_rollout_windows(
        train_indices,
        num_steps=num_steps,
        rollout_horizon=args.num_rollout_steps,
        num_samples=args.num_loss_calibration_samples,
        seed=args.loss_calibration_seed,
    )
    calibration_metrics = evaluate_validation_rollout(
        model=model,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        masses=masses,
        force_std=force_std,
        dt=dt,
        num_bodies=num_bodies,
        dim=dim,
        num_rollout_steps=args.num_rollout_steps,
        velocity_loss_weight=0.0,
        dynamics_loss_weight=0.0,
        dynamics_position_scale=1.0,
        validation_windows=calibration_windows,
    )
    dynamics_position_scale = dynamics_to_position_scale(
        calibration_metrics["position_mse"],
        calibration_metrics["dynamics_mse"],
    )
    mlflow_logger.log_params({
        "loss_calibration": {
            "num_samples": args.num_loss_calibration_samples,
            "seed": args.loss_calibration_seed,
            "baseline_position_mse": calibration_metrics["position_mse"],
            "baseline_dynamics_mse": calibration_metrics["dynamics_mse"],
            "dynamics_position_scale": dynamics_position_scale,
            "dynamics_target": "force",
        }
    })

    best_validation_loss = float("inf")
    start_time = time.monotonic()
    max_seconds = None if args.max_hours is None else args.max_hours * 60 * 60
    stop_training = False

    print("fine-tuning from:", checkpoint_path)
    print("training trajectories:", num_train_trajectories)
    print("rollout steps:", args.num_rollout_steps)
    print("dynamics loss target: normalized force")
    print("dynamics-to-position scale:", dynamics_position_scale)
    if max_seconds is not None:
        print(f"max training time: {args.max_hours:.2f} hours")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_position_mse = 0.0
        total_dynamics_mse = 0.0
        steps_completed = 0

        for _ in range(args.steps_per_epoch):
            if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                stop_training = True
                break

            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            batch_position_mse = 0.0
            batch_dynamics_mse = 0.0

            for _ in range(args.batch_size):
                train_index = torch.randint(0, num_train_trajectories, (1,)).item()
                traj_idx = train_indices[train_index]
                time_idx = torch.randint(
                    0,
                    num_steps - args.num_rollout_steps,
                    (1,),
                ).item()

                positions_t = positions[traj_idx, time_idx]
                velocities_t = velocities[traj_idx, time_idx]
                masses_t = masses[traj_idx]

                initial_state = pack_hnn_state(positions_t, velocities_t)
                step_fn = make_hnn_step_fn(
                    model=model,
                    masses=masses_t,
                    force_std=force_std,
                    dt=dt,
                    num_bodies=num_bodies,
                    dim=dim,
                )
                predicted_states = rollout_steps(
                    initial_state=initial_state,
                    step_fn=step_fn,
                    num_steps=args.num_rollout_steps,
                    detach_between_steps=False,
                )
                predicted_positions, predicted_velocities = unpack_hnn_state(
                    predicted_states,
                    num_bodies=num_bodies,
                    dim=dim,
                )

                if predicted_positions.shape[1] == 1:
                    predicted_positions = predicted_positions.squeeze(1)
                    predicted_velocities = predicted_velocities.squeeze(1)

                target_start = time_idx
                target_end = time_idx + args.num_rollout_steps + 1
                target_positions = positions[traj_idx, target_start:target_end]
                target_velocities = velocities[traj_idx, target_start:target_end]
                target_accelerations = accelerations[
                    traj_idx, target_start : target_end - 1
                ]

                position_mse = rollout_mse(
                    predicted_positions[1:], target_positions[1:]
                )
                velocity_mse = rollout_mse(
                    predicted_velocities[1:], target_velocities[1:]
                )
                predict_acceleration = make_hnn_acceleration_fn(
                    model=model,
                    masses=masses_t,
                    force_std=force_std,
                    num_bodies=num_bodies,
                    dim=dim,
                )
                dynamics_mse = teacher_forced_dynamics_mse(
                    positions=target_positions[:-1],
                    velocities=target_velocities[:-1],
                    target_accelerations=target_accelerations,
                    masses=masses_t,
                    predict_acceleration=lambda position, _velocity: predict_acceleration(
                        position
                    ),
                    normalization_scale=force_std,
                    target_kind="force",
                )
                loss = composite_rollout_loss(
                    position_mse,
                    velocity_mse=velocity_mse,
                    velocity_loss_weight=velocity_loss_weight,
                    dynamics_mse=dynamics_mse,
                    dynamics_loss_weight=args.dynamics_loss_weight,
                    dynamics_position_scale=dynamics_position_scale,
                )

                batch_loss = batch_loss + loss
                batch_position_mse += position_mse.item()
                batch_dynamics_mse += dynamics_mse.item()

            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()

            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip_norm,
                )

            optimizer.step()

            total_loss += batch_loss.item()
            total_position_mse += batch_position_mse / args.batch_size
            total_dynamics_mse += batch_dynamics_mse / args.batch_size
            steps_completed += 1

        if steps_completed == 0:
            break

        avg_train_loss = total_loss / steps_completed
        train_rollout_position_rmse = math.sqrt(
            total_position_mse / steps_completed
        )
        train_dynamics_rmse = math.sqrt(total_dynamics_mse / steps_completed)
        validation_metrics = evaluate_validation_rollout(
            model=model,
            positions=validation_positions,
            velocities=validation_velocities,
            accelerations=validation_accelerations,
            masses=validation_masses,
            force_std=force_std,
            dt=dt,
            num_bodies=num_bodies,
            dim=dim,
            num_rollout_steps=args.num_rollout_steps,
            velocity_loss_weight=velocity_loss_weight,
            dynamics_loss_weight=args.dynamics_loss_weight,
            dynamics_position_scale=dynamics_position_scale,
            validation_windows=validation_windows,
        )
        validation_rollout_loss = validation_metrics["objective"]
        validation_rollout_position_rmse = validation_metrics["position_rmse"]
        validation_dynamics_rmse = validation_metrics["dynamics_rmse"]

        if validation_rollout_loss < best_validation_loss:
            best_validation_loss = validation_rollout_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "force_std": force_std,
                "model_config": model_config,
                "dt": dt,
                "training_config": {
                    "fine_tuned_from": str(checkpoint_path),
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "num_rollout_steps": args.num_rollout_steps,
                    "velocity_loss_weight": velocity_loss_weight,
                    "dynamics_loss_weight": args.dynamics_loss_weight,
                    "dynamics_target": "force",
                    "dynamics_position_scale": dynamics_position_scale,
                    "baseline_position_mse": calibration_metrics["position_mse"],
                    "baseline_dynamics_mse": calibration_metrics["dynamics_mse"],
                    "num_loss_calibration_samples": args.num_loss_calibration_samples,
                    "loss_calibration_seed": args.loss_calibration_seed,
                    "grad_clip_norm": args.grad_clip_norm,
                    "num_train_trajectories": num_train_trajectories,
                    "num_val_trajectories": len(split_indices["val"]),
                    "num_test_trajectories": len(split_indices["test"]),
                    "num_validation_samples": args.num_validation_samples,
                    "validation_seed": args.validation_seed,
                    "validation_windows": validation_global_windows,
                    "validation_local_windows": validation_windows,
                    "split_path": str(split_path),
                    "split_seed": manifest.get("split_config", {}).get("split_seed"),
                },
                "split_manifest": manifest,
                "dataset_metadata": dataset.get("metadata", {}),
                "train_rollout_loss_at_best": avg_train_loss,
                "best_validation_rollout_objective": best_validation_loss,
                "validation_rollout_position_rmse_at_best": validation_rollout_position_rmse,
                "validation_dynamics_rmse_at_best": validation_dynamics_rmse,
            }
            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)
            print("Saved best validation rollout fine-tuned model.")

        elapsed_hours = (time.monotonic() - start_time) / 3600
        mlflow_logger.log_metrics(
            {
                "train_rollout_objective": avg_train_loss,
                "train_rollout_position_rmse": train_rollout_position_rmse,
                "train_dynamics_rmse": train_dynamics_rmse,
                "validation_rollout_objective": validation_rollout_loss,
                "validation_rollout_position_rmse": validation_rollout_position_rmse,
                "validation_dynamics_rmse": validation_dynamics_rmse,
                "best_validation_rollout_objective": best_validation_loss,
                "steps_completed": steps_completed,
                "elapsed_hours": elapsed_hours,
            },
            step=epoch,
        )
        print(
            f"Epoch {epoch}/{args.epochs}, "
            f"steps = {steps_completed}, "
            f"train rollout objective = {avg_train_loss:.6e}, "
            f"train rollout position RMSE = {train_rollout_position_rmse:.6e}, "
            f"train normalized force RMSE = {train_dynamics_rmse:.6e}, "
            f"validation rollout objective = {validation_rollout_loss:.6e}, "
            f"validation rollout position RMSE = {validation_rollout_position_rmse:.6e}, "
            f"validation normalized force RMSE = {validation_dynamics_rmse:.6e}, "
            f"elapsed = {elapsed_hours:.2f}h"
        )

        if stop_training:
            print("Reached max training time.")
            break

    metrics = {
        "best_validation_rollout_objective": best_validation_loss,
        "epochs_completed": epoch,
        "checkpoint_path": str(output_path),
    }
    if metrics_path is not None:
        save_json(metrics_path, metrics)
        mlflow_logger.log_artifact(metrics_path, artifact_path="run_metadata")

    mlflow_logger.log_metrics(metrics)
    mlflow_logger.end()


if __name__ == "__main__":
    main()

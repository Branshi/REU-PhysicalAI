import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_builder import build_graph
from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from common.splits import load_split_manifest, resolve_project_path
from models.graph_network import EncodeProcessDecode


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the N-body equivariant graph simulator with one-step acceleration loss."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "datasets" / "nbody_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "egns" / "one_step.pt"
        ),
        help="Checkpoint path for the best one-step model.",
    )
    parser.add_argument(
        "--split-path",
        default=str(PROJECT_ROOT / "experiments" / "splits" / "nbody_dataset.json"),
        help="Trajectory-level train/validation/test split manifest.",
    )
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=500,
        help="Number of fixed trajectory/time pairs evaluated after each epoch.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=1234,
        help="Seed used once to choose the fixed validation samples.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-messages", type=int, default=6)
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


def compute_graph_feature_stats(
    positions,
    velocities,
    masses,
    num_train_trajectories,
    num_steps,
):
    node_feature_list = []
    edge_feature_list = []
    num_stat_samples = min(10000, num_train_trajectories * num_steps)

    for _ in range(num_stat_samples):
        traj_idx = torch.randint(0, num_train_trajectories, (1,)).item()
        time_idx = torch.randint(0, num_steps, (1,)).item()

        graph = build_graph(
            positions=positions[traj_idx, time_idx],
            velocities=velocities[traj_idx, time_idx],
            masses=masses[traj_idx],
        )
        node_feature_list.append(graph["node_features"])
        edge_feature_list.append(graph["edge_features"])

    all_node_features = torch.cat(node_feature_list, dim=0)
    all_edge_features = torch.cat(edge_feature_list, dim=0)

    node_mean = all_node_features.mean(dim=0)
    node_std = all_node_features.std(dim=0).clamp_min(1e-8)
    edge_mean = all_edge_features.mean(dim=0)
    edge_std = all_edge_features.std(dim=0).clamp_min(1e-8)

    return node_mean, node_std, edge_mean, edge_std


def evaluate_validation(
    model,
    positions,
    velocities,
    accelerations,
    masses,
    validation_indices,
    acc_mean,
    acc_std,
):
    # Remember the incoming mode so this helper does not unexpectedly change it.
    was_training = model.training

    # Evaluation mode disables any training-only layer behavior.
    model.eval()

    # Keep the native normalized objective separate from the physical comparison
    # metric used across model families.
    total_normalized_acceleration_mse = 0.0
    total_physical_squared_error = 0.0
    total_physical_values = 0

    # Disable graph recording around validation because EGNS predicts acceleration
    # directly and does not need validation gradients.
    with torch.no_grad():
        # Reuse the same trajectory/time pairs on every epoch.
        for traj_idx, time_idx in validation_indices:
            # Select the particle positions for this validation state.
            positions_t = positions[traj_idx, time_idx]
            velocities_t = velocities[traj_idx, time_idx]

            # Select the constant particle masses for this trajectory.
            masses_t = masses[traj_idx]

            # Load the physical acceleration target for this state.
            target_acceleration = accelerations[traj_idx, time_idx]
            target_normalized = (
                target_acceleration - acc_mean.squeeze()
            ) / acc_std.squeeze()
            # Build the graph from the current positions and masses.
            graph = build_graph(
                positions=positions_t,
                velocities=velocities_t,
                masses=masses_t,
            )

            predicted_acceleration_normalized = model(graph)

            # Match the training objective: normalized acceleration MSE.
            acceleration_loss = F.mse_loss(
                predicted_acceleration_normalized,
                target_normalized,
            )

            predicted_acceleration = predicted_acceleration_normalized * acc_std.squeeze()
            physical_squared_error = (
                predicted_acceleration - target_acceleration
            ).square()

            # Add this state's scalar acceleration loss to the running sum.
            total_normalized_acceleration_mse += acceleration_loss.item()
            total_physical_squared_error += physical_squared_error.sum().item()
            total_physical_values += physical_squared_error.numel()

    # Restore training mode when the model entered this helper in training mode.
    if was_training:
        model.train()

    # Count the fixed states so the accumulated losses can be averaged.
    num_samples = len(validation_indices)

    # The normalized objective is for this model's checkpoint selection. Physical
    # acceleration RMSE is comparable across all model families.
    return {
        "normalized_acceleration_mse": (
            total_normalized_acceleration_mse / num_samples
        ),
        "acceleration_rmse": (
            total_physical_squared_error / total_physical_values
        ) ** 0.5,
    }


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    output_path, run_dir, metrics_path = prepare_run_outputs(
        project_root=PROJECT_ROOT,
        run_name="egns_onestep",
        args=args,
        output_path=args.checkpoint_path,
    )
    if run_dir is not None:
        print("Run directory:", run_dir)

    mlflow_logger = MLflowLogger(
        project_root=PROJECT_ROOT,
        experiment_name="egns_onestep",
        run_name=run_dir.name if run_dir is not None else output_path.stem,
        tags={"model": "EGNS", "training_stage": "one_step", "device": device},
        enabled=not args.disable_mlflow,
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    # Validate the shared trajectory manifest before constructing any training
    # views, so this model uses the same IDs as every other architecture.
    dataset_path = resolve_project_path(args.dataset_path)
    split_path = resolve_project_path(args.split_path)
    dataset = torch.load(dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    accelerations = dataset["accelerations"].to(device)
    masses = dataset["masses"].to(device)
    dt = dataset.get("metadata", {}).get("dt", 0.01)

    num_trajectories, num_steps, _, dim = positions.shape
    manifest, split_indices = load_split_manifest(
        split_path=split_path,
        dataset_path=dataset_path,
        dataset_shape=positions.shape,
    )
    # Replace the full tensors with train-only views. Every later random
    # trajectory/time sample and every learned statistic is therefore train-only.
    train_indices = split_indices["train"]
    val_indices = split_indices["val"]

    train_positions = positions[train_indices]
    validation_positions = positions[val_indices]
    train_velocities = velocities[train_indices]
    validation_velocities = velocities[val_indices]
    train_accelerations = accelerations[train_indices]
    validation_accelerations = accelerations[val_indices]
    train_masses = masses[train_indices]
    validation_masses = masses[val_indices]

    num_train_trajectories = len(train_indices)
    num_val_trajectories = len(val_indices)

    sample_graph = build_graph(
        positions=train_positions[0, 0],
        velocities=train_velocities[0, 0],
        masses=train_masses[0],
    )
    node_input_dim = sample_graph["node_features"].shape[-1]
    edge_input_dim = sample_graph["edge_features"].shape[-1]

    print("acc mean:", train_accelerations.mean().item())
    print("acc std:", train_accelerations.std().item())
    print("acc max abs:", train_accelerations.abs().max().item())
    print(
        "acc 95th percentile:",
        torch.quantile(train_accelerations.abs().flatten(), 0.95).item(),
    )
    print(
        "acc 99th percentile:",
        torch.quantile(train_accelerations.abs().flatten(), 0.99).item(),
    )

    # EGNS keeps a zero mean and one rotationally invariant scalar scale. The
    # scalar is computed only from training accelerations to avoid leakage.
    # Mean is set to 0 for equivariant reasons.
    acc_mean = torch.zeros(1, 1, 1, 1, device=device, dtype=train_accelerations.dtype)
    acc_std = train_accelerations.std().clamp_min(1e-8).view(1, 1, 1, 1)

    mlflow_logger.log_params({
        "data": {
            "dataset_path": str(dataset_path),
            "num_trajectories": num_trajectories,
            "num_steps": num_steps,
            "dim": dim,
            "num_train_trajectories": num_train_trajectories,
            "num_val_trajectories": len(split_indices["val"]),
            "num_test_trajectories": len(split_indices["test"]),
            "num_validation_samples": args.num_validation_samples,
            "validation_seed": args.validation_seed,
            "dt": dt,
        },
        "split": {
            "manifest_path": str(split_path),
            **manifest.get("split_config", {}),
        },
        "model": {
            "node_input_dim": node_input_dim,
            "edge_input_dim": edge_input_dim,
            "output_dim": dim,
        },
    })

    # Graph input normalization is also estimated from training trajectories
    # only, using the already-subset tensors above.
    node_mean, node_std, edge_mean, edge_std = compute_graph_feature_stats(
        positions=train_positions,
        velocities=train_velocities,
        masses=train_masses,
        num_train_trajectories=num_train_trajectories,
        num_steps=num_steps,
    )

    model = EncodeProcessDecode(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        output_dim=dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.num_messages,
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    best_loss = float("inf")
    start_time = time.monotonic()
    max_seconds = None if args.max_hours is None else args.max_hours * 60 * 60
    stop_training = False

    print("training trajectories:", num_train_trajectories)
    if max_seconds is not None:
        print(f"max training time: {args.max_hours:.2f} hours")

    if args.num_validation_samples <= 0:
        raise ValueError("num-validation-samples must be positive.")

    # Create a CPU generator so validation sampling is reproducible on any device.
    validation_generator = torch.Generator().manual_seed(args.validation_seed)

    # Choose local trajectory IDs only from the manifest validation tensors.
    validation_trajectories = torch.randint(
        0,
        num_val_trajectories,
        (args.num_validation_samples,),
        generator=validation_generator,
    )

    # Choose one saved time index for each sampled validation trajectory.
    validation_times = torch.randint(
        0,
        num_steps,
        (args.num_validation_samples,),
        generator=validation_generator,
    )

    # Store ordinary integer pairs so exactly the same states are reused each epoch.
    validation_indices = list(
        zip(
            validation_trajectories.tolist(),
            validation_times.tolist(),
        )
    )
    # Validation runs on compact validation tensors, but checkpoints store the
    # corresponding global IDs so the sampled states are easy to audit later.
    validation_global_indices = [
        (val_indices[traj_idx], time_idx) for traj_idx, time_idx in validation_indices
    ]

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        steps_completed = 0

        for _ in range(args.steps_per_epoch):
            if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                stop_training = True
                break

            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)

            for _ in range(args.batch_size):
                traj_idx = torch.randint(0, num_train_trajectories, (1,)).item()
                time_idx = torch.randint(0, num_steps, (1,)).item()

                positions_t = train_positions[traj_idx, time_idx]
                velocities_t = train_velocities[traj_idx, time_idx]
                masses_t = train_masses[traj_idx]
                target_acceleration = train_accelerations[traj_idx, time_idx]

                target_acceleration_normalized = (
                    target_acceleration - acc_mean.squeeze()
                ) / acc_std.squeeze()

                graph = build_graph(
                    positions=positions_t,
                    velocities=velocities_t,
                    masses=masses_t,
                )
                predicted_acceleration = model(graph)
                loss = F.mse_loss(
                    predicted_acceleration,
                    target_acceleration_normalized,
                )
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            steps_completed += 1

        if steps_completed == 0:
            break

        avg_train_loss = total_loss / steps_completed

        validation_metrics = evaluate_validation(
            model,
            validation_positions,
            validation_velocities,
            validation_accelerations,
            validation_masses,
            validation_indices,
            acc_mean,
            acc_std,
        )

        validation_normalized_acceleration_mse = validation_metrics[
            "normalized_acceleration_mse"
        ]
        validation_acceleration_rmse = validation_metrics["acceleration_rmse"]

        if validation_normalized_acceleration_mse < best_loss:
            best_loss = validation_normalized_acceleration_mse
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "model_config": {
                    "node_input_dim": node_input_dim,
                    "edge_input_dim": edge_input_dim,
                    "output_dim": dim,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_message_passing_steps": args.num_messages,
                },
                "dt": dt,
                "training_config": {
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "num_train_trajectories": num_train_trajectories,
                    "num_val_trajectories": len(split_indices["val"]),
                    "num_test_trajectories": len(split_indices["test"]),
                    "num_validation_samples": args.num_validation_samples,
                    "validation_seed": args.validation_seed,
                    "validation_indices": validation_global_indices,
                    "validation_local_indices": validation_indices,
                    "split_path": str(split_path),
                    "split_seed": manifest.get("split_config", {}).get("split_seed"),
                },
                # Save the exact split with the model for reproducible comparison.
                "split_manifest": manifest,
                "dataset_metadata": dataset.get("metadata", {}),
                "best_validation_normalized_acceleration_mse": best_loss,
                "validation_acceleration_rmse_at_best": validation_acceleration_rmse,
            }
            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)
            print("Saved best one-step model.")

        elapsed_hours = (time.monotonic() - start_time) / 3600
        mlflow_logger.log_metrics(
            {
                "train_normalized_acceleration_mse": avg_train_loss,
                "validation_normalized_acceleration_mse": validation_normalized_acceleration_mse,
                "validation_acceleration_rmse": validation_acceleration_rmse,
                "best_validation_normalized_acceleration_mse": best_loss,
                "steps_completed": steps_completed,
                "elapsed_hours": elapsed_hours,
            },
            step=epoch + 1,
        )
        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, "
            f"steps = {steps_completed}, "
            f"train normalized acceleration MSE = {avg_train_loss:.6f}, "
            f"validation normalized acceleration MSE = {validation_normalized_acceleration_mse:.6f}, "
            f"validation acceleration RMSE = {validation_acceleration_rmse:.6f}, "
            f"elapsed = {elapsed_hours:.2f}h"
        )

        if stop_training:
            print("Reached max training time.")
            break

    metrics = {
        "best_validation_normalized_acceleration_mse": best_loss,
        "final_validation_acceleration_rmse": validation_acceleration_rmse,
        "epochs_completed": epoch + 1,
        "checkpoint_path": str(output_path),
    }
    if metrics_path is not None:
        save_json(metrics_path, metrics)
        mlflow_logger.log_artifact(metrics_path, artifact_path="run_metadata")

    mlflow_logger.log_metrics(metrics)
    mlflow_logger.end()


if __name__ == "__main__":
    main()

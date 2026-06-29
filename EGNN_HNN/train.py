import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from EGNN_HNN.graph_builder import build_graph
from EGNN_HNN.models.graph_network import EncodeProcessDecode


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the N-body EGNN-HNN with one-step force loss."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "nbody_3body_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "egnn_hnn" / "one_step.pt"
        ),
        help="Checkpoint path for the best one-step model.",
    )
    parser.add_argument("--num-train-trajectories", type=int, default=None)
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
    masses,
    num_train_trajectories,
):
    node_feature_list = []
    edge_feature_list = []

    # Node and edge features now depend only on masses, which are constant over
    # a trajectory, so each training trajectory needs to be included only once.
    for traj_idx in range(num_train_trajectories):
        graph = build_graph(
            positions=positions[traj_idx, 0],
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
    accelerations,
    masses,
    force_std,
    validation_indices,
):
    # Remember the incoming mode so this helper does not unexpectedly change it.
    was_training = model.training

    # Evaluation mode disables any training-only layer behavior.
    model.eval()

    # Accumulate normalized force MSE across the fixed validation samples.
    total_force_loss = 0.0

    # Accumulate physical acceleration MSE for comparison with EGNS.
    total_acceleration_loss = 0.0

    # Disable ordinary graph recording around validation. The model temporarily
    # re-enables gradients internally only to differentiate potential by position.
    with torch.no_grad():
        # Reuse the same trajectory/time pairs on every epoch.
        for traj_idx, time_idx in validation_indices:
            # Select the particle positions for this validation state.
            positions_t = positions[traj_idx, time_idx]

            # Select the constant particle masses for this trajectory.
            masses_t = masses[traj_idx]

            # Load the physical acceleration target for this state.
            target_acceleration = accelerations[traj_idx, time_idx]

            # Convert acceleration to physical force using F = m * a.
            target_force = target_acceleration * masses_t

            # Express the target in the normalized units learned by the model.
            target_force_normalized = target_force / force_std.squeeze()

            # Build the graph from the current positions and masses.
            graph = build_graph(
                positions=positions_t,
                masses=masses_t,
            )

            # Predict normalized force without constructing a higher-order graph.
            predicted_force_normalized = model(
                graph,
                create_graph=False,
            )

            # Measure normalized force error, which matches the training objective.
            force_loss = F.mse_loss(
                predicted_force_normalized,
                target_force_normalized,
            )

            # Convert the normalized prediction back to physical force units.
            predicted_force = predicted_force_normalized * force_std

            # Convert physical force to acceleration using a = F / m.
            predicted_acceleration = predicted_force / masses_t

            # Measure physical acceleration error for a fair EGNS comparison.
            acceleration_loss = F.mse_loss(
                predicted_acceleration,
                target_acceleration,
            )

            # Add this state's scalar normalized force loss to the running sum.
            total_force_loss += force_loss.item()

            # Add this state's scalar acceleration loss to the running sum.
            total_acceleration_loss += acceleration_loss.item()

    # Restore training mode when the model entered this helper in training mode.
    if was_training:
        model.train()

    # Count the fixed states so the accumulated losses can be averaged.
    num_samples = len(validation_indices)

    # Return both metrics as ordinary Python floats.
    return {
        "force_loss": total_force_loss / num_samples,
        "acceleration_mse": total_acceleration_loss / num_samples,
    }


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    output_path, run_dir, metrics_path = prepare_run_outputs(
        project_root=PROJECT_ROOT,
        run_name="egnn_hnn_onestep",
        args=args,
        output_path=args.checkpoint_path,
    )
    if run_dir is not None:
        print("Run directory:", run_dir)

    mlflow_logger = MLflowLogger(
        project_root=PROJECT_ROOT,
        experiment_name="egnn_hnn_onestep",
        run_name=run_dir.name if run_dir is not None else output_path.stem,
        tags={"model": "EGNN-HNN", "training_stage": "one_step", "device": device},
        enabled=not args.disable_mlflow,
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    dataset = torch.load(args.dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    accelerations = dataset["accelerations"].to(device)
    masses = dataset["masses"].to(device)
    dt = dataset.get("metadata", {}).get("dt", 0.01)
    epsilon = dataset.get("metadata", {}).get("epsilon", 0.15)

    sample_graph = build_graph(
        positions=positions[0, 0],
        masses=masses[0],
    )
    node_input_dim = sample_graph["node_features"].shape[-1]
    edge_input_dim = sample_graph["edge_features"].shape[-1]

    print("acc mean:", accelerations.mean().item())
    print("acc std:", accelerations.std().item())
    print("acc max abs:", accelerations.abs().max().item())
    print(
        "acc 95th percentile:",
        torch.quantile(accelerations.abs().flatten(), 0.95).item(),
    )
    print(
        "acc 99th percentile:",
        torch.quantile(accelerations.abs().flatten(), 0.99).item(),
    )

    num_trajectories, num_steps, _, dim = positions.shape

    # When no explicit split is supplied, reserve the final 20% for validation.
    if args.num_train_trajectories is None:
        num_train_trajectories = max(1, int(0.8 * num_trajectories))
    else:
        # Prevent an oversized request from indexing beyond the loaded dataset.
        num_train_trajectories = min(args.num_train_trajectories, num_trajectories)

    # Training requires at least one trajectory.
    if num_train_trajectories <= 0:
        raise ValueError("num-train-trajectories must be positive.")

    # Fixed validation requires at least one trajectory outside the training split.
    if num_train_trajectories >= num_trajectories:
        raise ValueError(
            "num-train-trajectories must leave at least one trajectory for validation."
        )

    # Validation averaging requires at least one fixed sample.
    if args.num_validation_samples <= 0:
        raise ValueError("num-validation-samples must be positive.")

    # Convert every state in the training split from acceleration to force.
    training_forces = accelerations[:num_train_trajectories] * masses[
        :num_train_trajectories
    ].unsqueeze(1)

    # Use one global scalar so force normalization preserves rotational symmetry.
    force_std = training_forces.std().clamp_min(1e-8).reshape(1)

    mlflow_logger.log_params({
        "data": {
            "num_trajectories": num_trajectories,
            "num_steps": num_steps,
            "dim": dim,
            "num_train_trajectories": num_train_trajectories,
            "num_validation_trajectories": num_trajectories
            - num_train_trajectories,
            "num_validation_samples": args.num_validation_samples,
            "dt": dt,
            "epsilon": epsilon,
        },
        "model": {
            "node_input_dim": node_input_dim,
            "edge_input_dim": edge_input_dim,
            "output_dim": dim,
        },
    })

    node_mean, node_std, edge_mean, edge_std = compute_graph_feature_stats(
        positions=positions,
        masses=masses,
        num_train_trajectories=num_train_trajectories,
    )

    model = EncodeProcessDecode(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_message_passing_steps=args.num_messages,
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
        epsilon=epsilon,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    best_validation_loss = float("inf")
    start_time = time.monotonic()
    max_seconds = None if args.max_hours is None else args.max_hours * 60 * 60
    stop_training = False

    # The validation split begins immediately after the training trajectories.
    validation_start = num_train_trajectories

    # Create a CPU generator so validation sampling is reproducible on any device.
    validation_generator = torch.Generator().manual_seed(args.validation_seed)

    # Choose validation trajectories only from the held-out portion of the dataset.
    validation_trajectories = torch.randint(
        validation_start,
        num_trajectories,
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

    print("training trajectories:", num_train_trajectories)
    if max_seconds is not None:
        print(f"max training time: {args.max_hours:.2f} hours")

    # Track completed epochs explicitly so early stopping cannot leave it undefined.
    epochs_completed = 0

    # Initialize final metrics for the run summary.
    last_train_loss = float("nan")
    last_validation_force_loss = float("nan")
    last_validation_acceleration_mse = float("nan")

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

                positions_t = positions[traj_idx, time_idx]
                masses_t = masses[traj_idx]
                target_force = accelerations[traj_idx, time_idx] * masses_t

                target_force_normalized = target_force / force_std.squeeze()

                graph = build_graph(
                    positions=positions_t,
                    masses=masses_t,
                )
                predicted_force = model(graph)
                loss = F.mse_loss(
                    predicted_force,
                    target_force_normalized,
                )
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            steps_completed += 1

        if steps_completed == 0:
            break

        # Average the randomized training batches completed during this epoch.
        avg_loss = total_loss / steps_completed

        # Evaluate the same held-out states after every epoch.
        validation_metrics = evaluate_validation(
            model=model,
            positions=positions,
            accelerations=accelerations,
            masses=masses,
            force_std=force_std,
            validation_indices=validation_indices,
        )

        # Extract normalized validation force MSE for model selection.
        validation_force_loss = validation_metrics["force_loss"]

        # Extract physical acceleration MSE for comparison with EGNS.
        validation_acceleration_mse = validation_metrics["acceleration_mse"]

        # Record the latest completed epoch and its metrics for the run summary.
        epochs_completed = epoch + 1
        last_train_loss = avg_loss
        last_validation_force_loss = validation_force_loss
        last_validation_acceleration_mse = validation_acceleration_mse

        # Save only when the fixed validation force loss reaches a new minimum.
        if validation_force_loss < best_validation_loss:
            best_validation_loss = validation_force_loss

            # Package model parameters, normalization, architecture, and metrics.
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "force_std": force_std,
                "model_config": {
                    "node_input_dim": node_input_dim,
                    "edge_input_dim": edge_input_dim,
                    "output_dim": dim,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_message_passing_steps": args.num_messages,
                    "epsilon": epsilon,
                },
                "dt": dt,
                "training_config": {
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "num_train_trajectories": num_train_trajectories,
                    "num_validation_samples": args.num_validation_samples,
                    "validation_seed": args.validation_seed,
                },
                "train_loss_at_best": avg_loss,
                "best_loss": best_validation_loss,
                "best_validation_force_loss": best_validation_loss,
                "validation_acceleration_mse_at_best": validation_acceleration_mse,
            }

            # Write the best validation checkpoint to the configured path.
            torch.save(checkpoint, output_path)

            # Copy the saved checkpoint into the active MLflow run when enabled.
            mlflow_logger.log_checkpoint(output_path)

            # Make checkpoint improvements visible in the terminal.
            print("Saved best validation one-step model.")

        # Measure wall-clock duration after training and validation for this epoch.
        elapsed_hours = (time.monotonic() - start_time) / 3600

        # Log directly comparable training and fixed-validation metrics.
        mlflow_logger.log_metrics(
            {
                "train_loss": avg_loss,
                "validation_force_loss": validation_force_loss,
                "validation_acceleration_mse": validation_acceleration_mse,
                "best_validation_force_loss": best_validation_loss,
                "steps_completed": steps_completed,
                "elapsed_hours": elapsed_hours,
            },
            step=epoch + 1,
        )
        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, "
            f"steps = {steps_completed}, "
            f"train loss = {avg_loss:.6f}, "
            f"validation force loss = {validation_force_loss:.6f}, "
            f"validation acceleration MSE = {validation_acceleration_mse:.6f}, "
            f"elapsed = {elapsed_hours:.2f}h"
        )

        if stop_training:
            print("Reached max training time.")
            break

    # Summarize the completed run for JSON and MLflow output.
    metrics = {
        "best_loss": best_validation_loss,
        "best_validation_force_loss": best_validation_loss,
        "final_train_loss": last_train_loss,
        "final_validation_force_loss": last_validation_force_loss,
        "final_validation_acceleration_mse": last_validation_acceleration_mse,
        "epochs_completed": epochs_completed,
        "checkpoint_path": str(output_path),
    }

    # Persist the run summary next to the checkpoint when run outputs are enabled.
    if metrics_path is not None:
        save_json(metrics_path, metrics)
        mlflow_logger.log_artifact(metrics_path, artifact_path="run_metadata")

    # Log final summary values and close the MLflow run.
    mlflow_logger.log_metrics(metrics)
    mlflow_logger.end()


if __name__ == "__main__":
    main()

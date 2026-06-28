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
        default=str(PROJECT_ROOT / "common" / "nbody_3body_dataset.pt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "egns" / "one_step.pt"
        ),
        help="Checkpoint path for the best one-step model.",
    )
    parser.add_argument("--num-train-trajectories", type=int, default=None)
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

    dataset = torch.load(args.dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    accelerations = dataset["accelerations"].to(device)
    masses = dataset["masses"].to(device)
    dt = dataset.get("metadata", {}).get("dt", 0.01)

    sample_graph = build_graph(
        positions=positions[0, 0],
        velocities=velocities[0, 0],
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

    acc_mean = torch.zeros(1, 1, 1, 1, device=device, dtype=accelerations.dtype)
    acc_std = accelerations.std().clamp_min(1e-8).view(1, 1, 1, 1)

    num_trajectories, num_steps, _, dim = positions.shape
    if args.num_train_trajectories is None:
        num_train_trajectories = num_trajectories
    else:
        num_train_trajectories = min(args.num_train_trajectories, num_trajectories)

    mlflow_logger.log_params({
        "data": {
            "num_trajectories": num_trajectories,
            "num_steps": num_steps,
            "dim": dim,
            "num_train_trajectories": num_train_trajectories,
            "dt": dt,
        },
        "model": {
            "node_input_dim": node_input_dim,
            "edge_input_dim": edge_input_dim,
            "output_dim": dim,
        },
    })

    node_mean, node_std, edge_mean, edge_std = compute_graph_feature_stats(
        positions=positions,
        velocities=velocities,
        masses=masses,
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
                velocities_t = velocities[traj_idx, time_idx]
                masses_t = masses[traj_idx]
                target_acceleration = accelerations[traj_idx, time_idx]

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

        avg_loss = total_loss / steps_completed
        if avg_loss < best_loss:
            best_loss = avg_loss
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
                },
                "best_loss": best_loss,
            }
            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)
            print("Saved best one-step model.")

        elapsed_hours = (time.monotonic() - start_time) / 3600
        mlflow_logger.log_metrics(
            {
                "train_loss": avg_loss,
                "best_loss": best_loss,
                "steps_completed": steps_completed,
                "elapsed_hours": elapsed_hours,
            },
            step=epoch + 1,
        )
        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, "
            f"steps = {steps_completed}, "
            f"loss = {avg_loss:.6f}, "
            f"elapsed = {elapsed_hours:.2f}h"
        )

        if stop_training:
            print("Reached max training time.")
            break

    metrics = {
        "best_loss": best_loss,
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

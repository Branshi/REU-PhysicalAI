import argparse
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EGNS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EGNS_ROOT) not in sys.path:
    sys.path.insert(0, str(EGNS_ROOT))

from common.rollout.differentiable import rollout_steps
from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from common.rollout.losses import rollout_mse
from common.splits import (
    load_split_manifest,
    resolve_project_path,
    validate_checkpoint_split,
)
from models.graph_network import EncodeProcessDecode
from models.learned_simulator import LearnedSimulator
from rollouts.gns_adapter import make_gns_step_fn, pack_gns_state, unpack_gns_state


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained EGNS with differentiable rollout loss."
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
        help="Pretrained one-step EGNS checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "egns" / "rollout.pt"
        ),
        help="Checkpoint path for the best rollout fine-tuned model.",
    )
    parser.add_argument(
        "--split-path",
        default=str(PROJECT_ROOT / "experiments" / "splits" / "nbody_dataset.json"),
        help="Trajectory-level train/validation/test split manifest.",
    )
    parser.add_argument("--num-rollout-steps", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.0)
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


def load_checkpoint(
    checkpoint_path,
    dataset,
    device,
    train_indices=None,
    split_manifest=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # A fine-tuning run must inherit the one-step checkpoint's partition; using
    # another manifest could silently turn prior validation/test data into train.
    if split_manifest is not None:
        validate_checkpoint_split(checkpoint, split_manifest)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        acc_mean = checkpoint["acc_mean"].to(device)
        acc_std = checkpoint["acc_std"].to(device)
        config = checkpoint["model_config"]
        dt = checkpoint.get("dt", dataset.get("metadata", {}).get("dt", 0.01))
    elif isinstance(checkpoint, dict):
        print("Loaded older raw state_dict checkpoint.")
        print(
            "Using default model config and recomputing acc_mean/acc_std from dataset."
        )

        model_state_dict = checkpoint
        accelerations = dataset["accelerations"].to(device)
        # Legacy raw checkpoints do not contain normalization. Reconstruct it
        # from manifest training trajectories rather than from the full dataset.
        if train_indices is not None:
            accelerations = accelerations[train_indices]
        positions = dataset["positions"]
        _, _, num_bodies, dim = positions.shape

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
    else:
        raise ValueError(
            "Checkpoint format not recognized. Expected a full checkpoint with "
            "model_state_dict or a raw model.state_dict()."
        )

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


def build_simulator(model_state_dict, acc_mean, acc_std, config, dt, device):
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

    return graph_network, simulator


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    output_path, run_dir, metrics_path = prepare_run_outputs(
        project_root=PROJECT_ROOT,
        run_name="egns_rollout",
        args=args,
        output_path=args.output_path,
    )
    if run_dir is not None:
        print("Run directory:", run_dir)

    mlflow_logger = MLflowLogger(
        project_root=PROJECT_ROOT,
        experiment_name="egns_rollout",
        run_name=run_dir.name if run_dir is not None else output_path.stem,
        tags={"model": "EGNS", "training_stage": "rollout", "device": device},
        enabled=not args.disable_mlflow,
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    # Load the same manifest used by one-step training and keep validation/test
    # IDs out of every differentiable rollout optimization window.
    dataset_path = resolve_project_path(args.dataset_path)
    split_path = resolve_project_path(args.split_path)
    checkpoint_path = resolve_project_path(args.checkpoint_path)
    dataset = torch.load(dataset_path, map_location=device)
    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    masses = dataset["masses"].to(device)

    num_trajectories, num_steps, _, dim = positions.shape
    manifest, split_indices = load_split_manifest(
        split_path=split_path,
        dataset_path=dataset_path,
        dataset_shape=positions.shape,
    )
    train_indices = split_indices["train"]
    num_train_trajectories = len(train_indices)

    if args.num_rollout_steps <= 0:
        raise ValueError("num-rollout-steps must be positive.")
    if args.num_rollout_steps >= num_steps:
        raise ValueError(
            "num-rollout-steps must be smaller than the number of saved "
            f"trajectory steps. Got {args.num_rollout_steps} for {num_steps} steps."
        )

    mlflow_logger.log_params({
        "data": {
            "dataset_path": str(dataset_path),
            "num_trajectories": num_trajectories,
            "num_steps": num_steps,
            "dim": dim,
            "num_train_trajectories": num_train_trajectories,
            "num_val_trajectories": len(split_indices["val"]),
            "num_test_trajectories": len(split_indices["test"]),
        },
        "split": {
            "manifest_path": str(split_path),
            **manifest.get("split_config", {}),
        },
    })

    model_state_dict, acc_mean, acc_std, config, dt = load_checkpoint(
        checkpoint_path,
        dataset,
        device,
        train_indices=train_indices,
        split_manifest=manifest,
    )
    mlflow_logger.log_params({"pretrained_model": config, "data": {"dt": dt}})
    graph_network, simulator = build_simulator(
        model_state_dict,
        acc_mean,
        acc_std,
        config,
        dt,
        device,
    )

    optimizer = torch.optim.Adam(graph_network.parameters(), lr=args.learning_rate)
    simulator.train()

    best_loss = float("inf")
    start_time = time.monotonic()
    max_seconds = None if args.max_hours is None else args.max_hours * 60 * 60
    stop_training = False

    print("fine-tuning from:", checkpoint_path)
    print("training trajectories:", num_train_trajectories)
    print("rollout steps:", args.num_rollout_steps)
    if max_seconds is not None:
        print(f"max training time: {args.max_hours:.2f} hours")

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        total_position_mse = 0.0
        steps_completed = 0

        for _ in range(args.steps_per_epoch):
            if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                stop_training = True
                break

            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            batch_position_mse = 0.0

            for _ in range(args.batch_size):
                # Split IDs are seeded but noncontiguous, so sample within the
                # manifest list and map that location back to a dataset ID.
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

                initial_state = pack_gns_state(positions_t, velocities_t)
                step_fn = make_gns_step_fn(simulator, masses_t, dim=dim)
                predicted_states = rollout_steps(
                    initial_state=initial_state,
                    step_fn=step_fn,
                    num_steps=args.num_rollout_steps,
                    detach_between_steps=False,
                )
                predicted_positions, predicted_velocities = unpack_gns_state(
                    predicted_states,
                    dim=dim,
                )

                target_start = time_idx
                target_end = time_idx + args.num_rollout_steps + 1
                target_positions = positions[traj_idx, target_start:target_end]
                target_velocities = velocities[traj_idx, target_start:target_end]

                position_mse = rollout_mse(
                    predicted_positions[1:], target_positions[1:]
                )
                loss = position_mse
                if args.velocity_loss_weight > 0.0:
                    loss = loss + args.velocity_loss_weight * rollout_mse(
                        predicted_velocities[1:],
                        target_velocities[1:],
                    )

                batch_loss = batch_loss + loss
                batch_position_mse += position_mse.item()

            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()

            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    graph_network.parameters(),
                    max_norm=args.grad_clip_norm,
                )

            optimizer.step()

            total_loss += batch_loss.item()
            total_position_mse += batch_position_mse / args.batch_size
            steps_completed += 1

        if steps_completed == 0:
            break

        avg_loss = total_loss / steps_completed
        train_rollout_position_rmse = math.sqrt(
            total_position_mse / steps_completed
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "model_state_dict": graph_network.state_dict(),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "model_config": config,
                "dt": dt,
                "training_config": {
                    "fine_tuned_from": str(checkpoint_path),
                    "num_rollout_steps": args.num_rollout_steps,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "velocity_loss_weight": args.velocity_loss_weight,
                    "grad_clip_norm": args.grad_clip_norm,
                    "num_train_trajectories": num_train_trajectories,
                    "num_val_trajectories": len(split_indices["val"]),
                    "num_test_trajectories": len(split_indices["test"]),
                    "split_path": str(split_path),
                    "split_seed": manifest.get("split_config", {}).get("split_seed"),
                },
                # Preserve split provenance in the rollout-fine-tuned checkpoint.
                "split_manifest": manifest,
                "dataset_metadata": dataset.get("metadata", {}),
                "best_rollout_objective": best_loss,
            }
            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)
            print("Saved best rollout fine-tuned model.")

        elapsed_hours = (time.monotonic() - start_time) / 3600
        mlflow_logger.log_metrics(
            {
                "train_rollout_objective": avg_loss,
                "train_rollout_position_rmse": train_rollout_position_rmse,
                "best_rollout_objective": best_loss,
                "steps_completed": steps_completed,
                "elapsed_hours": elapsed_hours,
            },
            step=epoch + 1,
        )
        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, "
            f"steps = {steps_completed}, "
            f"train rollout objective = {avg_loss:.6e}, "
            f"train rollout position RMSE = {train_rollout_position_rmse:.6e}, "
            f"elapsed = {elapsed_hours:.2f}h"
        )

        if stop_training:
            print("Reached max training time.")
            break

    metrics = {
        "best_rollout_objective": best_loss,
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

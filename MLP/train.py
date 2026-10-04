import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLP_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MLP_ROOT) not in sys.path:
    sys.path.insert(0, str(MLP_ROOT))

from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from common.splits import load_split_manifest, resolve_project_path
from models.mlp import MLP


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a global MLP with one-step acceleration loss."
    )

    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "datasets" / "nbody_dataset.pt"),
    )
    parser.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "mlp" / "one_step.pt"
        ),
        help="Checkpoint path for the best one-step model.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=234,
        help=(
            "Hidden width. The default gives 115,134 trainable parameters for "
            "the repository's 3-body, 2D dataset."
        ),
    )
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help=(
            "Optional wall-clock training limit in hours. The current partial "
            "epoch is validated before training exits."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for model initialization and training permutations.",
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
    parser.add_argument(
        "--split-path",
        default=str(PROJECT_ROOT / "experiments" / "splits" / "nbody_dataset.json"),
        help="Trajectory-level train/validation/test split manifest.",
    )

    return parser.parse_args()


def flatten_acceleration_inputs(
    positions,
    velocities,
    accelerations,
    masses,
    num_traj,
    num_steps,
    spatial_dim,
    num_bodies,
):
    # q, v, acceleration: [N, T, B * D]
    # masses:              [N, T, B]
    q_t = positions.reshape(num_traj, num_steps, spatial_dim)
    v_t = velocities.reshape(num_traj, num_steps, spatial_dim)
    m_t = masses.reshape(num_traj, 1, num_bodies)
    m_t = m_t.expand(-1, num_steps, -1)
    acceleration_t = accelerations.reshape(num_traj, num_steps, spatial_dim)

    inputs = torch.cat([q_t, v_t, m_t], dim=-1)
    num_samples = num_traj * num_steps
    return (
        inputs.reshape(num_samples, inputs.shape[-1]),
        acceleration_t.reshape(num_samples, acceleration_t.shape[-1]),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    print("Using device: ", device)

    output_path, run_dir, metrics_path = prepare_run_outputs(
        project_root=PROJECT_ROOT,
        run_name="mlp_onestep",
        args=args,
        output_path=args.output_path,
    )
    if run_dir is not None:
        print("Run directory:", run_dir)

    mlflow_logger = MLflowLogger(
        project_root=PROJECT_ROOT,
        experiment_name="mlp_onestep",
        run_name=run_dir.name if run_dir is not None else output_path.stem,
        tags={"model": "MLP", "training_stage": "one_step", "device": device},
        enabled=not args.disable_mlflow,
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    dataset_path = resolve_project_path(args.dataset_path)
    split_path = resolve_project_path(args.split_path)
    dataset = torch.load(dataset_path, map_location=device)

    manifest, split_indices = load_split_manifest(
        split_path=split_path,
        dataset_path=dataset_path,
        dataset_shape=dataset["positions"].shape,
    )
    train_indices = split_indices["train"]
    val_indices = split_indices["val"]

    train_pos = dataset["positions"][train_indices].to(device)
    train_velocities = dataset["velocities"][train_indices].to(device)
    train_accelerations = dataset["accelerations"][train_indices].to(device)
    train_masses = dataset["masses"][train_indices].to(device)

    val_pos = dataset["positions"][val_indices].to(device)
    val_velocities = dataset["velocities"][val_indices].to(device)
    val_accelerations = dataset["accelerations"][val_indices].to(device)
    val_masses = dataset["masses"][val_indices].to(device)

    num_train_traj, num_steps, num_bodies, dim = train_pos.shape
    num_val_traj = val_pos.shape[0]
    spatial_dim = num_bodies * dim
    mass_dim = num_bodies
    input_dim = 2 * spatial_dim + mass_dim
    dt = dataset.get("metadata", {}).get("dt", 0.01)

    train_inputs, train_targets = flatten_acceleration_inputs(
        train_pos,
        train_velocities,
        train_accelerations,
        train_masses,
        num_train_traj,
        num_steps,
        spatial_dim,
        num_bodies,
    )
    val_inputs, val_targets = flatten_acceleration_inputs(
        val_pos,
        val_velocities,
        val_accelerations,
        val_masses,
        num_val_traj,
        num_steps,
        spatial_dim,
        num_bodies,
    )

    if args.num_validation_samples <= 0:
        raise ValueError("num-validation-samples must be positive.")

    validation_generator = torch.Generator().manual_seed(args.validation_seed)
    validation_trajectories = torch.randint(
        0,
        num_val_traj,
        (args.num_validation_samples,),
        generator=validation_generator,
    )
    validation_times = torch.randint(
        0,
        num_steps,
        (args.num_validation_samples,),
        generator=validation_generator,
    )
    validation_flat_indices = validation_trajectories * num_steps + validation_times
    validation_inputs = val_inputs[validation_flat_indices.to(device)]
    validation_targets = val_targets[validation_flat_indices.to(device)]
    validation_local_indices = list(
        zip(validation_trajectories.tolist(), validation_times.tolist())
    )
    validation_global_indices = [
        (val_indices[traj_idx], time_idx)
        for traj_idx, time_idx in validation_local_indices
    ]

    train_input_mean = train_inputs.mean(dim=0)
    train_input_std = train_inputs.std(dim=0).clamp_min(1e-8)
    # Match GNS: zero mean and one scalar train-only acceleration scale.
    acc_mean = torch.zeros(1, device=device, dtype=train_targets.dtype)
    acc_std = train_targets.std().clamp_min(1e-8).reshape(1)

    print("train inputs:", train_inputs.shape)
    print("train acceleration targets:", train_targets.shape)
    print("validation inputs:", val_inputs.shape)
    print("validation acceleration targets:", val_targets.shape)

    mlflow_logger.log_params({
        "data": {
            "dataset_path": str(dataset_path),
            "num_train_trajectories": num_train_traj,
            "num_val_trajectories": num_val_traj,
            "num_test_trajectories": len(split_indices["test"]),
            "num_validation_samples": args.num_validation_samples,
            "validation_seed": args.validation_seed,
            "num_steps": num_steps,
            "num_bodies": num_bodies,
            "dim": dim,
            "dt": dt,
        },
        "split": {
            "manifest_path": str(split_path),
            **manifest.get("split_config", {}),
        },
        "model": {
            "input_dim": input_dim,
            "spatial_dim": spatial_dim,
            "mass_dim": mass_dim,
            "output_dim": spatial_dim,
            "input_features": "flattened_positions_velocities_masses",
            "target": "normalized_acceleration",
        },
    })

    # Reset immediately before construction so preprocessing cannot change the
    # initial weights selected by the configured seed.
    torch.manual_seed(args.seed)
    model = MLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=spatial_dim,
        state_mean=train_input_mean,
        state_std=train_input_std,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {parameter_count:,}")
    mlflow_logger.log_params({"model": {"trainable_parameters": parameter_count}})

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate permutations on CPU for consistent seeded behavior across CPU,
    # CUDA, and MPS, then move each permutation to the selected device.
    training_generator = torch.Generator(device="cpu").manual_seed(args.seed)

    best_val_loss = float("inf")
    best_validation_acceleration_rmse = float("inf")
    last_train_loss = float("nan")
    last_validation_acceleration_loss = float("nan")
    last_validation_acceleration_rmse = float("nan")
    epochs_completed = 0
    start_time = time.monotonic()
    max_seconds = None if args.max_hours is None else args.max_hours * 60 * 60
    stop_training = False

    if max_seconds is not None:
        print(f"max training time: {args.max_hours:.2f} hours")

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(
            train_inputs.shape[0],
            generator=training_generator,
            device="cpu",
        ).to(device)

        total_train_loss = 0.0
        num_train_batches = 0

        for start in range(0, train_inputs.shape[0], args.batch_size):
            if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
                stop_training = True
                break

            end = start + args.batch_size
            batch_indices = permutation[start:end]

            x_batch = train_inputs[batch_indices]
            acceleration_batch = train_targets[batch_indices]
            target_acceleration_normalized = (
                acceleration_batch - acc_mean
            ) / acc_std

            predicted_acceleration_normalized = model(x_batch)
            loss = F.mse_loss(
                predicted_acceleration_normalized,
                target_acceleration_normalized,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        if num_train_batches == 0:
            break

        avg_train_loss = total_train_loss / num_train_batches

        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        total_acceleration_squared_error = 0.0
        total_acceleration_values = 0

        for start in range(0, validation_inputs.shape[0], args.batch_size):
            end = start + args.batch_size

            x_batch = validation_inputs[start:end]
            acceleration_batch = validation_targets[start:end]
            target_acceleration_normalized = (
                acceleration_batch - acc_mean
            ) / acc_std

            with torch.no_grad():
                predicted_acceleration_normalized = model(x_batch)
            val_loss = F.mse_loss(
                predicted_acceleration_normalized,
                target_acceleration_normalized,
            )

            predicted_acceleration = (
                predicted_acceleration_normalized * acc_std + acc_mean
            ).reshape(
                -1, num_bodies, dim
            )
            target_acceleration = acceleration_batch.reshape(-1, num_bodies, dim)
            acceleration_squared_error = (
                predicted_acceleration - target_acceleration
            ).square()

            total_val_loss += val_loss.item()
            num_val_batches += 1
            total_acceleration_squared_error += acceleration_squared_error.sum().item()
            total_acceleration_values += acceleration_squared_error.numel()

        avg_val_loss = total_val_loss / num_val_batches
        validation_acceleration_rmse = (
            total_acceleration_squared_error / total_acceleration_values
        ) ** 0.5
        last_train_loss = avg_train_loss
        last_validation_acceleration_loss = avg_val_loss
        last_validation_acceleration_rmse = validation_acceleration_rmse

        if epoch % 50 == 0:
            print(
                f"Epoch: {epoch:04d} | "
                f"train normalized acceleration MSE: {avg_train_loss:.6f} | "
                f"validation normalized acceleration MSE: {avg_val_loss:.6f} | "
                f"validation acceleration RMSE: {validation_acceleration_rmse:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_validation_acceleration_rmse = validation_acceleration_rmse
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "model_config": {
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_hidden_layers": args.num_hidden_layers,
                    "num_bodies": num_bodies,
                    "dim": dim,
                    "spatial_dim": spatial_dim,
                    "mass_dim": mass_dim,
                    "output_dim": spatial_dim,
                    "input_features": "flattened_positions_velocities_masses",
                    "target": "normalized_acceleration",
                    "trainable_parameters": parameter_count,
                },
                "dt": dt,
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                    "split_path": str(split_path),
                    "split_seed": manifest.get("split_config", {}).get("split_seed"),
                    "num_train_trajectories": num_train_traj,
                    "num_val_trajectories": num_val_traj,
                    "num_test_trajectories": len(split_indices["test"]),
                    "num_validation_samples": args.num_validation_samples,
                    "validation_seed": args.validation_seed,
                    "validation_indices": validation_global_indices,
                    "validation_local_indices": validation_local_indices,
                },
                "split_manifest": manifest,
                "dataset_metadata": dataset.get("metadata", {}),
                "train_loss_at_best": avg_train_loss,
                "best_validation_normalized_acceleration_mse": best_val_loss,
                "validation_acceleration_rmse_at_best": validation_acceleration_rmse,
            }

            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)

        mlflow_logger.log_metrics(
            {
                "train_normalized_acceleration_mse": avg_train_loss,
                "validation_normalized_acceleration_mse": avg_val_loss,
                "validation_acceleration_rmse": validation_acceleration_rmse,
                "best_validation_normalized_acceleration_mse": best_val_loss,
            },
            step=epoch,
        )

        epochs_completed = epoch
        if stop_training:
            break

    metrics = {
        "best_validation_normalized_acceleration_mse": best_val_loss,
        "final_train_normalized_acceleration_mse": last_train_loss,
        "final_validation_normalized_acceleration_mse": last_validation_acceleration_loss,
        "validation_acceleration_rmse_at_best": best_validation_acceleration_rmse,
        "final_validation_acceleration_rmse": last_validation_acceleration_rmse,
        "epochs_completed": epochs_completed,
        "checkpoint_path": str(output_path),
    }
    if metrics_path is not None:
        save_json(metrics_path, metrics)
        mlflow_logger.log_artifact(metrics_path, artifact_path="run_metadata")

    mlflow_logger.log_metrics(metrics)
    mlflow_logger.end()

    print(f"Saved best model to {output_path}")
    print(f"Best validation normalized acceleration MSE: {best_val_loss:.6f}")
    print(
        "Validation acceleration RMSE at best checkpoint: "
        f"{best_validation_acceleration_rmse:.6f}"
    )


if __name__ == "__main__":
    main()

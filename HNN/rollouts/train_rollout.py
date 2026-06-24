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

from common.rollout.differentiable import rollout_steps
from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from common.rollout.integrators import rk4_step
from common.rollout.losses import rollout_mse
from models.hamiltonian_network import HNN
from rollouts.hnn_adapter import make_hnn_step_fn, pack_hnn_state, unpack_hnn_state


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained HNN with differentiable rollout loss."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "nbody_3body_dataset.pt"),
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
    parser.add_argument("--num-train-trajectories", type=int, default=None)
    parser.add_argument("--num-rollout-steps", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum-loss-weight", type=float, default=0.0)
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
    return parser.parse_args()


def load_checkpoint(checkpoint_path, dataset, device, default_hidden_dim=128):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    positions = dataset["positions"]
    _, _, num_bodies, dim = positions.shape
    q_dim = num_bodies * dim
    input_dim = 2 * q_dim + num_bodies

    default_model_config = {
        "input_dim": input_dim,
        "hidden_dim": default_hidden_dim,
        "num_bodies": num_bodies,
        "dim": dim,
        "q_dim": q_dim,
    }

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint.get("model_config", default_model_config)
        dataset_metadata = checkpoint.get("dataset_metadata", {})
    elif isinstance(checkpoint, dict):
        print("Loaded older raw state_dict checkpoint.")
        model_state_dict = checkpoint
        model_config = default_model_config
        dataset_metadata = {}
    else:
        raise ValueError(
            "Checkpoint format not recognized. Expected a full checkpoint with "
            "model_state_dict or a raw model.state_dict()."
        )

    dt = dataset_metadata.get("dt", dataset.get("metadata", {}).get("dt", 0.01))

    model = HNN(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        q_dim=model_config["q_dim"],
    ).to(device)
    model.load_state_dict(model_state_dict)

    return model, model_config, dt


def main():
    args = parse_args()
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
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    dataset = torch.load(args.dataset_path, map_location=device)
    positions = dataset["positions"].to(device)
    momenta = dataset["momenta"].to(device)
    masses = dataset["masses"].to(device)

    num_trajectories, num_steps, num_bodies, dim = positions.shape
    if args.num_rollout_steps <= 0:
        raise ValueError("num-rollout-steps must be positive.")
    if args.num_rollout_steps >= num_steps:
        raise ValueError(
            "num-rollout-steps must be smaller than the number of saved "
            f"trajectory steps. Got {args.num_rollout_steps} for {num_steps} steps."
        )

    if args.num_train_trajectories is None:
        num_train_trajectories = num_trajectories
    else:
        num_train_trajectories = min(args.num_train_trajectories, num_trajectories)

    model, model_config, dt = load_checkpoint(args.checkpoint_path, dataset, device)
    mlflow_logger.log_params(
        {
            "data": {
                "num_trajectories": num_trajectories,
                "num_steps": num_steps,
                "num_bodies": num_bodies,
                "dim": dim,
                "num_train_trajectories": num_train_trajectories,
                "dt": dt,
            },
            "pretrained_model": model_config,
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    best_loss = float("inf")
    start_time = time.monotonic()
    max_seconds = None if args.max_hours is None else args.max_hours * 60 * 60
    stop_training = False

    print("fine-tuning from:", args.checkpoint_path)
    print("training trajectories:", num_train_trajectories)
    print("rollout steps:", args.num_rollout_steps)
    if max_seconds is not None:
        print(f"max training time: {args.max_hours:.2f} hours")

    for epoch in range(1, args.epochs + 1):
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
                time_idx = torch.randint(
                    0,
                    num_steps - args.num_rollout_steps,
                    (1,),
                ).item()

                positions_t = positions[traj_idx, time_idx]
                momenta_t = momenta[traj_idx, time_idx]
                masses_t = masses[traj_idx]

                initial_state = pack_hnn_state(positions_t, momenta_t)
                step_fn = make_hnn_step_fn(
                    model=model,
                    masses=masses_t,
                    dt=dt,
                    num_bodies=num_bodies,
                    integrator=rk4_step,
                )
                predicted_states = rollout_steps(
                    initial_state=initial_state,
                    step_fn=step_fn,
                    num_steps=args.num_rollout_steps,
                    detach_between_steps=False,
                )
                predicted_positions, predicted_momenta = unpack_hnn_state(
                    predicted_states,
                    num_bodies=num_bodies,
                    dim=dim,
                )

                if predicted_positions.shape[1] == 1:
                    predicted_positions = predicted_positions.squeeze(1)
                    predicted_momenta = predicted_momenta.squeeze(1)

                target_start = time_idx
                target_end = time_idx + args.num_rollout_steps + 1
                target_positions = positions[traj_idx, target_start:target_end]
                target_momenta = momenta[traj_idx, target_start:target_end]

                loss = rollout_mse(predicted_positions[1:], target_positions[1:])
                if args.momentum_loss_weight > 0.0:
                    loss = loss + args.momentum_loss_weight * rollout_mse(
                        predicted_momenta[1:],
                        target_momenta[1:],
                    )

                batch_loss = batch_loss + loss

            batch_loss = batch_loss / args.batch_size
            batch_loss.backward()

            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip_norm,
                )

            optimizer.step()

            total_loss += batch_loss.item()
            steps_completed += 1

        if steps_completed == 0:
            break

        avg_loss = total_loss / steps_completed
        avg_rmse = math.sqrt(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "training_config": {
                    "fine_tuned_from": args.checkpoint_path,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "num_rollout_steps": args.num_rollout_steps,
                    "momentum_loss_weight": args.momentum_loss_weight,
                    "grad_clip_norm": args.grad_clip_norm,
                    "num_train_trajectories": num_train_trajectories,
                },
                "dataset_metadata": dataset.get("metadata", {}),
                "best_rollout_loss": best_loss,
            }
            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)
            print("Saved best rollout fine-tuned model.")

        elapsed_hours = (time.monotonic() - start_time) / 3600
        mlflow_logger.log_metrics(
            {
                "rollout_loss": avg_loss,
                "rollout_rmse": avg_rmse,
                "best_rollout_loss": best_loss,
                "steps_completed": steps_completed,
                "elapsed_hours": elapsed_hours,
            },
            step=epoch,
        )
        print(
            f"Epoch {epoch}/{args.epochs}, "
            f"steps = {steps_completed}, "
            f"loss = {avg_loss:.6e}, "
            f"rmse = {avg_rmse:.6e}, "
            f"elapsed = {elapsed_hours:.2f}h"
        )

        if stop_training:
            print("Reached max training time.")
            break

    metrics = {
        "best_rollout_loss": best_loss,
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

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HNN_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HNN_ROOT) not in sys.path:
    sys.path.insert(0, str(HNN_ROOT))

from common.experiment_runs import prepare_run_outputs, save_json
from common.mlflow_logger import MLflowLogger
from models.hamiltonian_network import HNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Hamiltonian network with one-step derivative loss."
    )

    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "common" / "nbody_3body_dataset.pt"),
    )
    parser.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT / "experiments" / "checkpoints" / "hnn" / "one_step.pt"
        ),
        help="Checkpoint path for the best one-step model.",
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
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


def main():
    args = parse_args()
    device = get_device()
    print("Using device: ", device)

    output_path, run_dir, metrics_path = prepare_run_outputs(
        project_root=PROJECT_ROOT,
        run_name="hnn_onestep",
        args=args,
        output_path=args.output_path,
    )
    if run_dir is not None:
        print("Run directory:", run_dir)

    mlflow_logger = MLflowLogger(
        project_root=PROJECT_ROOT,
        experiment_name="hnn_onestep",
        run_name=run_dir.name if run_dir is not None else output_path.stem,
        tags={"model": "HNN", "training_stage": "one_step", "device": device},
        enabled=not args.disable_mlflow,
    ).start()
    mlflow_logger.log_params(args)
    mlflow_logger.log_description(args.description)
    mlflow_logger.log_tags({"checkpoint_path": output_path, "run_dir": run_dir})

    dataset = torch.load(args.dataset_path, map_location=device)

    positions = dataset["positions"].to(device)
    velocities = dataset["velocities"].to(device)
    momenta = dataset["momenta"].to(device)
    forces = dataset["forces"].to(device)
    masses = dataset["masses"].to(device)

    # We must properly flatten data in order to pass into the HNN

    # Original shapes:
    # positions:  [num_trajectories, num_steps, num_bodies, dim]
    # momenta:    [num_trajectories, num_steps, num_bodies, dim]
    # velocities: [num_trajectories, num_steps, num_bodies, dim]
    # forces:     [num_trajectories, num_steps, num_bodies, dim]

    num_trajectories, num_steps, num_bodies, dim = positions.shape

    q_dim = num_bodies * dim
    mass_dim = num_bodies
    qp_dim = 2 * q_dim
    input_dim = qp_dim + mass_dim  # mass for each body

    mlflow_logger.log_params(
        {
            "data": {
                "num_trajectories": num_trajectories,
                "num_steps": num_steps,
                "num_bodies": num_bodies,
                "dim": dim,
            },
            "model": {
                "input_dim": input_dim,
                "q_dim": q_dim,
                "mass_dim": mass_dim,
                "qp_dim": qp_dim,
            },
        }
    )

    # Flatten body and coordinate dimensions.
    # q:     [N, T, B * D]
    # p:     [N, T, B * D]
    # q_dot: [N, T, B * D]
    # p_dot: [N, T, B * D]
    q = positions.reshape(num_trajectories, num_steps, q_dim)
    p = momenta.reshape(num_trajectories, num_steps, q_dim)
    m = masses.reshape(num_trajectories, 1, num_bodies)
    m = m.expand(-1, num_steps, -1)

    q_dot = velocities.reshape(num_trajectories, num_steps, q_dim)
    p_dot = forces.reshape(num_trajectories, num_steps, q_dim)

    # Build state and derivative target.
    # state:  [N, T, 2 * B * D] = [q, p]
    # target: [N, T, 2 * B * D] = [q_dot, p_dot]
    states = torch.cat([q, p, m], dim=-1)
    targets = torch.cat([q_dot, p_dot], dim=-1)

    # Combine trajectory dimension and time dimension.
    # states:  [N * T, input_dim]
    # targets: [N * T, input_dim]
    states = states.reshape(num_trajectories * num_steps, input_dim)
    targets = targets.reshape(num_trajectories * num_steps, qp_dim)

    print("states:", states.shape)
    print("targets:", targets.shape)

    # Data Split

    num_samples = states.shape[0]
    indices = torch.randperm(num_samples, device=device)

    train_size = int(args.train_fraction * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_states = states[train_indices]
    train_targets = targets[train_indices]

    train_state_mean = train_states.mean(dim=0)
    train_state_std = train_states.std(dim=0).clamp_min(1e-8)
    train_target_std = train_targets.std(dim=0).clamp_min(1e-8)

    val_states = states[val_indices]
    val_targets = targets[val_indices]

    print("train states:", train_states.shape)
    print("val states:", val_states.shape)

    # Model Construction

    model = HNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        q_dim=q_dim,
        state_mean=train_state_mean,
        state_std=train_state_std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # put the model in train mode
        model.train()

        # creates a random ordering of training sample indicies
        permutation = torch.randperm(train_states.shape[0], device=device)

        total_train_loss = 0.0
        num_train_batches = 0

        for start in range(0, train_states.shape[0], args.batch_size):
            end = start + args.batch_size
            batch_indices = permutation[start:end]

            x_batch = train_states[batch_indices]
            dxdt_batch = train_targets[batch_indices]

            pred_dxdt = model.time_derivative(x_batch)

            loss = F.mse_loss(
                pred_dxdt / train_target_std, dxdt_batch / train_target_std
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # Validation
        # Important: do NOT use torch.no_grad() here as HNN validation still needs autograd with respect to inputs

        model.eval()

        total_val_loss = 0.0
        num_val_batches = 0

        for start in range(0, val_states.shape[0], args.batch_size):
            end = start + args.batch_size

            x_batch = val_states[start:end]
            dxdt_batch = val_targets[start:end]

            pred_dxdt = model.time_derivative(x_batch)

            val_loss = F.mse_loss(
                pred_dxdt / train_target_std, dxdt_batch / train_target_std
            )

            total_val_loss += val_loss.item()
            num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        if epoch % 50 == 0:
            print(
                f"Epoch: {epoch:04d} | "
                f"train loss: {avg_train_loss:.6f} | "
                f"val loss: {avg_val_loss}"
            )

        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": {
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_bodies": num_bodies,
                    "dim": dim,
                    "q_dim": q_dim,
                },
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "train_fraction": args.train_fraction,
                },
                "dataset_metadata": dataset.get("metadata", {}),
                "best_val_loss": best_val_loss,
            }

            torch.save(checkpoint, output_path)
            mlflow_logger.log_checkpoint(output_path)

        mlflow_logger.log_metrics(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            },
            step=epoch,
        )

    metrics = {
        "best_val_loss": best_val_loss,
        "epochs_completed": args.epochs,
        "checkpoint_path": str(output_path),
    }
    if metrics_path is not None:
        save_json(metrics_path, metrics)
        mlflow_logger.log_artifact(metrics_path, artifact_path="run_metadata")

    mlflow_logger.log_metrics(metrics)
    mlflow_logger.end()

    print(f"Saved best model to {output_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

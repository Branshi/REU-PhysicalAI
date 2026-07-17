"""Deterministic multi-trajectory rollout evaluation shared by every model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from common.rollout.metrics import (
    center_of_mass_drift,
    energy_drift,
    relative_energy_drift,
    total_energy_from_velocities,
)


DEFAULT_NUM_TEST_TRAJECTORIES = 50
DEFAULT_FAILURE_THRESHOLD = 0.1
CONSERVATION_METRIC_NAMES = (
    "final_absolute_energy_drift",
    "max_absolute_energy_drift",
    "final_absolute_relative_energy_drift",
    "max_absolute_relative_energy_drift",
    "final_center_of_mass_drift",
    "max_center_of_mass_drift",
)


def add_test_suite_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common fixed-suite CLI without removing single-rollout support."""

    parser.add_argument(
        "--test-suite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate a deterministic prefix of the selected manifest split. "
            "Enabled by default for dataset-based evaluation; use "
            "--no-test-suite for the legacy single-trajectory visualization."
        ),
    )
    parser.add_argument(
        "--num-test-trajectories",
        "--num-traj",
        dest="num_test_trajectories",
        type=int,
        default=DEFAULT_NUM_TEST_TRAJECTORIES,
        help=(
            "Number of trajectories from the fixed manifest-order suite. "
            f"Defaults to {DEFAULT_NUM_TEST_TRAJECTORIES}."
        ),
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=DEFAULT_FAILURE_THRESHOLD,
        help=(
            "A trajectory fails if any predicted step has position RMSE above "
            f"this value. Non-finite predictions always fail. Defaults to "
            f"{DEFAULT_FAILURE_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--test-suite-output",
        default=None,
        help="Optional JSON path for the complete suite report.",
    )


def should_run_test_suite(args, use_custom_initial_conditions: bool) -> bool:
    """Return whether this invocation requests aggregate dataset evaluation."""

    if use_custom_initial_conditions or not args.test_suite:
        return False
    if args.traj_idx is not None:
        return False
    return getattr(args, "traj_mode", "indexed") == "indexed"


def _resolve_rollout_steps(requested_steps: int | None, num_saved_steps: int) -> int:
    max_rollout_steps = num_saved_steps - 1
    if max_rollout_steps <= 0:
        raise ValueError("The dataset must contain at least two saved states.")
    if requested_steps is None:
        return max_rollout_steps
    if requested_steps <= 0:
        raise ValueError("rollout-steps must be positive for test-suite evaluation.")
    if requested_steps > max_rollout_steps:
        print(
            f"Requested {requested_steps} rollout steps, but the dataset supports "
            f"{max_rollout_steps}. Using {max_rollout_steps}."
        )
        return max_rollout_steps
    return requested_steps


def _summarize(values: list[float], trajectory_indices: list[int]) -> dict:
    if not values:
        return {
            "finite_count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "worst": None,
            "worst_trajectory_index": None,
        }

    tensor = torch.tensor(values, dtype=torch.float64)
    worst_local_index = int(torch.argmax(tensor).item())
    return {
        "finite_count": len(values),
        "mean": float(tensor.mean().item()),
        # Population standard deviation describes the evaluated suite itself.
        "std": float(tensor.std(unbiased=False).item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "worst": float(tensor[worst_local_index].item()),
        "worst_trajectory_index": trajectory_indices[worst_local_index],
    }


def _print_metric_summary(label: str, summary: dict) -> None:
    if summary["finite_count"] == 0:
        print(f"{label}: no finite trajectories")
        return
    print(
        f"{label}: "
        f"mean={summary['mean']:.6e}, "
        f"std={summary['std']:.6e}, "
        f"median={summary['median']:.6e}, "
        f"worst={summary['worst']:.6e} "
        f"(trajectory {summary['worst_trajectory_index']})"
    )


def _compute_conservation_metrics(
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    gravitational_constant: float,
    softening_epsilon: float,
) -> dict[str, float | None]:
    empty = {name: None for name in CONSERVATION_METRIC_NAMES}
    if not bool(
        torch.isfinite(positions).all().item()
        and torch.isfinite(velocities).all().item()
        and torch.isfinite(masses).all().item()
    ):
        return empty

    energy = total_energy_from_velocities(
        positions,
        velocities,
        masses,
        G=gravitational_constant,
        epsilon=softening_epsilon,
    )
    absolute_energy_drift = energy_drift(energy).abs()
    absolute_relative_energy_drift = relative_energy_drift(energy).abs()
    com_drift = center_of_mass_drift(positions, masses)

    if not bool(
        torch.isfinite(absolute_energy_drift).all().item()
        and torch.isfinite(absolute_relative_energy_drift).all().item()
        and torch.isfinite(com_drift).all().item()
    ):
        return empty

    return {
        "final_absolute_energy_drift": float(absolute_energy_drift[-1].item()),
        "max_absolute_energy_drift": float(absolute_energy_drift.max().item()),
        "final_absolute_relative_energy_drift": float(
            absolute_relative_energy_drift[-1].item()
        ),
        "max_absolute_relative_energy_drift": float(
            absolute_relative_energy_drift.max().item()
        ),
        "final_center_of_mass_drift": float(com_drift[-1].item()),
        "max_center_of_mass_drift": float(com_drift.max().item()),
    }


def _empty_conservation_accumulator() -> dict:
    return {
        series_name: {
            metric_name: {"values": [], "trajectory_indices": []}
            for metric_name in CONSERVATION_METRIC_NAMES
        }
        for series_name in ("predicted", "true_numerical", "predicted_minus_true")
    }


def _record_conservation_metrics(
    accumulator: dict,
    series_name: str,
    metrics: dict[str, float | None],
    trajectory_index: int,
) -> None:
    for metric_name, value in metrics.items():
        if value is None:
            continue
        accumulator[series_name][metric_name]["values"].append(value)
        accumulator[series_name][metric_name]["trajectory_indices"].append(
            trajectory_index
        )


def _summarize_conservation(accumulator: dict) -> dict:
    return {
        series_name: {
            metric_name: _summarize(
                recorded["values"],
                recorded["trajectory_indices"],
            )
            for metric_name, recorded in series.items()
        }
        for series_name, series in accumulator.items()
    }


def evaluate_fixed_test_suite(
    *,
    model_name: str,
    rollout_fn: Callable[[Tensor, Tensor, Tensor, int], tuple[Tensor, Tensor]],
    acceleration_prediction_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    positions: Tensor,
    velocities: Tensor,
    accelerations: Tensor,
    masses: Tensor,
    allowed_indices: list[int],
    split_name: str,
    num_trajectories: int,
    requested_rollout_steps: int | None,
    failure_threshold: float,
    gravitational_constant: float,
    softening_epsilon: float,
    checkpoint_path: Path,
    dataset_path: Path,
    split_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Evaluate a fixed manifest-order trajectory suite and aggregate errors."""

    if num_trajectories <= 0:
        raise ValueError("num-test-trajectories must be positive.")
    if num_trajectories > len(allowed_indices):
        raise ValueError(
            f"Requested {num_trajectories} trajectories, but the {split_name} "
            f"split contains only {len(allowed_indices)}."
        )
    if failure_threshold < 0.0:
        raise ValueError("failure-threshold must be non-negative.")

    rollout_steps = _resolve_rollout_steps(
        requested_rollout_steps,
        positions.shape[1],
    )
    # Manifest order is already fixed by the saved split. Taking its prefix is
    # deterministic and gives every model exactly the same global trajectory IDs.
    trajectory_indices = [int(index) for index in allowed_indices[:num_trajectories]]

    position_mse_values: list[float] = []
    position_mse_value_indices: list[int] = []
    rollout_values: list[float] = []
    rollout_value_indices: list[int] = []
    final_values: list[float] = []
    final_value_indices: list[int] = []
    max_step_values: list[float] = []
    max_step_value_indices: list[int] = []
    acceleration_mse_values: list[float] = []
    acceleration_mse_value_indices: list[int] = []
    acceleration_rmse_values: list[float] = []
    acceleration_rmse_value_indices: list[int] = []
    conservation_accumulator = _empty_conservation_accumulator()
    per_trajectory = []
    failure_indices = []
    non_finite_indices = []
    non_finite_acceleration_indices = []
    start_time = time.monotonic()

    print(f"Running fixed {split_name} suite for {model_name}.")
    print(f"trajectories: {num_trajectories}")
    print(f"rollout steps per trajectory: {rollout_steps}")
    print(f"failure threshold (max-step position RMSE): {failure_threshold}")
    print(
        "conservation physics: "
        f"G={gravitational_constant}, epsilon={softening_epsilon}"
    )

    for suite_index, trajectory_index in enumerate(trajectory_indices, start=1):
        predicted_positions, predicted_velocities = rollout_fn(
            positions[trajectory_index, 0],
            velocities[trajectory_index, 0],
            masses[trajectory_index],
            rollout_steps,
        )
        true_positions = positions[trajectory_index, : rollout_steps + 1]
        true_velocities = velocities[trajectory_index, : rollout_steps + 1]
        true_accelerations = accelerations[
            trajectory_index, : rollout_steps + 1
        ]
        trajectory_masses = masses[trajectory_index]

        if predicted_positions.shape != true_positions.shape:
            raise ValueError(
                f"Trajectory {trajectory_index} produced positions with shape "
                f"{tuple(predicted_positions.shape)}, expected "
                f"{tuple(true_positions.shape)}."
            )
        if predicted_velocities.shape != true_velocities.shape:
            raise ValueError(
                f"Trajectory {trajectory_index} produced velocities with shape "
                f"{tuple(predicted_velocities.shape)}, expected "
                f"{tuple(true_velocities.shape)}."
            )

        predictions_are_finite = bool(
            torch.isfinite(predicted_positions).all().item()
            and torch.isfinite(predicted_velocities).all().item()
        )

        if predictions_are_finite:
            # State zero is supplied rather than predicted, so exclude it from
            # every learned-rollout error statistic.
            squared_error = (predicted_positions[1:] - true_positions[1:]) ** 2
            per_step_rmse = torch.sqrt(squared_error.mean(dim=(-2, -1)))
            position_mse = float(squared_error.mean().item())
            rollout_rmse = position_mse**0.5
            final_rmse = float(per_step_rmse[-1].item())
            max_step_rmse = float(per_step_rmse.max().item())

            position_mse_values.append(position_mse)
            position_mse_value_indices.append(trajectory_index)
            rollout_values.append(rollout_rmse)
            rollout_value_indices.append(trajectory_index)
            final_values.append(final_rmse)
            final_value_indices.append(trajectory_index)
            max_step_values.append(max_step_rmse)
            max_step_value_indices.append(trajectory_index)
            failed = max_step_rmse > failure_threshold
        else:
            position_mse = None
            rollout_rmse = None
            final_rmse = None
            max_step_rmse = None
            failed = True
            non_finite_indices.append(trajectory_index)

        with torch.no_grad():
            predicted_accelerations = torch.stack(
                [
                    acceleration_prediction_fn(
                        true_positions[time_index],
                        true_velocities[time_index],
                        trajectory_masses,
                    )
                    for time_index in range(rollout_steps + 1)
                ],
                dim=0,
            )
        if predicted_accelerations.shape != true_accelerations.shape:
            raise ValueError(
                f"Trajectory {trajectory_index} produced accelerations with shape "
                f"{tuple(predicted_accelerations.shape)}, expected "
                f"{tuple(true_accelerations.shape)}."
            )

        accelerations_are_finite = bool(
            torch.isfinite(predicted_accelerations).all().item()
        )
        if accelerations_are_finite:
            acceleration_mse = float(
                ((predicted_accelerations - true_accelerations) ** 2).mean().item()
            )
            acceleration_rmse = acceleration_mse**0.5
            acceleration_mse_values.append(acceleration_mse)
            acceleration_mse_value_indices.append(trajectory_index)
            acceleration_rmse_values.append(acceleration_rmse)
            acceleration_rmse_value_indices.append(trajectory_index)
        else:
            acceleration_mse = None
            acceleration_rmse = None
            failed = True
            non_finite_acceleration_indices.append(trajectory_index)

        true_conservation = _compute_conservation_metrics(
            true_positions,
            true_velocities,
            trajectory_masses,
            gravitational_constant,
            softening_epsilon,
        )
        if predictions_are_finite:
            predicted_conservation = _compute_conservation_metrics(
                predicted_positions,
                predicted_velocities,
                trajectory_masses,
                gravitational_constant,
                softening_epsilon,
            )
        else:
            predicted_conservation = {
                name: None for name in CONSERVATION_METRIC_NAMES
            }
        predicted_minus_true = {
            name: (
                predicted_conservation[name] - true_conservation[name]
                if predicted_conservation[name] is not None
                and true_conservation[name] is not None
                else None
            )
            for name in CONSERVATION_METRIC_NAMES
        }

        _record_conservation_metrics(
            conservation_accumulator,
            "predicted",
            predicted_conservation,
            trajectory_index,
        )
        _record_conservation_metrics(
            conservation_accumulator,
            "true_numerical",
            true_conservation,
            trajectory_index,
        )
        _record_conservation_metrics(
            conservation_accumulator,
            "predicted_minus_true",
            predicted_minus_true,
            trajectory_index,
        )

        if failed:
            failure_indices.append(trajectory_index)

        per_trajectory.append(
            {
                "trajectory_index": trajectory_index,
                "rollout_position_mse": position_mse,
                "rollout_position_rmse": rollout_rmse,
                "final_step_position_rmse": final_rmse,
                "max_step_position_rmse": max_step_rmse,
                "acceleration_mse_on_true_states": acceleration_mse,
                "acceleration_rmse_on_true_states": acceleration_rmse,
                "non_finite_prediction": not predictions_are_finite,
                "non_finite_acceleration_prediction": not accelerations_are_finite,
                "failed": failed,
                "conservation": {
                    "predicted": predicted_conservation,
                    "true_numerical": true_conservation,
                    "predicted_minus_true": predicted_minus_true,
                },
            }
        )

        if suite_index == 1 or suite_index % 10 == 0 or suite_index == num_trajectories:
            print(f"evaluated {suite_index}/{num_trajectories} trajectories")

    elapsed_seconds = time.monotonic() - start_time
    position_mse_summary = _summarize(
        position_mse_values,
        position_mse_value_indices,
    )
    rollout_summary = _summarize(rollout_values, rollout_value_indices)
    final_summary = _summarize(final_values, final_value_indices)
    max_step_summary = _summarize(max_step_values, max_step_value_indices)
    acceleration_mse_summary = _summarize(
        acceleration_mse_values,
        acceleration_mse_value_indices,
    )
    acceleration_rmse_summary = _summarize(
        acceleration_rmse_values,
        acceleration_rmse_value_indices,
    )
    conservation_summary = _summarize_conservation(conservation_accumulator)
    failure_count = len(failure_indices)
    failure_rate = failure_count / num_trajectories

    report = {
        "protocol": {
            "name": "fixed_manifest_prefix_rollout_suite",
            "version": 3,
            "model": model_name,
            "split": split_name,
            "selection": "first trajectories in saved split-manifest order",
            "trajectory_indices": trajectory_indices,
            "num_trajectories": num_trajectories,
            "rollout_steps": rollout_steps,
            "exclude_initial_state_from_error": True,
            "acceleration_evaluation": (
                "physical acceleration error on every saved true state from "
                "state 0 through rollout_steps"
            ),
            "failure_definition": (
                "non-finite position, velocity, or acceleration prediction; "
                "or max-step position RMSE greater than failure_threshold"
            ),
            "failure_threshold": failure_threshold,
            "conservation": {
                "gravitational_constant": gravitational_constant,
                "softening_epsilon": softening_epsilon,
                "energy_drift_is_absolute": True,
                "relative_energy_drift_is_absolute": True,
                "center_of_mass_drift_definition": (
                    "Euclidean distance from the trajectory's initial "
                    "center of mass"
                ),
                "comparison": (
                    "predicted_minus_true subtracts the saved numerical "
                    "trajectory drift from predicted drift per trajectory"
                ),
            },
            "checkpoint_path": str(checkpoint_path),
            "dataset_path": str(dataset_path),
            "split_path": str(split_path),
        },
        "summary": {
            "rollout_position_mse": position_mse_summary,
            "rollout_position_rmse": rollout_summary,
            "final_step_position_rmse": final_summary,
            "max_step_position_rmse": max_step_summary,
            "acceleration_mse_on_true_states": acceleration_mse_summary,
            "acceleration_rmse_on_true_states": acceleration_rmse_summary,
            "conservation": conservation_summary,
            "failure_count": failure_count,
            "failure_rate": failure_rate,
            "failure_trajectory_indices": failure_indices,
            "non_finite_count": len(non_finite_indices),
            "non_finite_trajectory_indices": non_finite_indices,
            "non_finite_acceleration_count": len(non_finite_acceleration_indices),
            "non_finite_acceleration_trajectory_indices": (
                non_finite_acceleration_indices
            ),
            "elapsed_seconds": elapsed_seconds,
        },
        "per_trajectory": per_trajectory,
    }

    print("\nFixed test-suite summary")
    _print_metric_summary("rollout position MSE", position_mse_summary)
    _print_metric_summary("rollout position RMSE", rollout_summary)
    _print_metric_summary("final-step position RMSE", final_summary)
    _print_metric_summary("max-step position RMSE", max_step_summary)
    _print_metric_summary(
        "acceleration MSE on true states",
        acceleration_mse_summary,
    )
    _print_metric_summary(
        "acceleration RMSE on true states",
        acceleration_rmse_summary,
    )
    print(
        f"failures: {failure_count}/{num_trajectories} "
        f"({100.0 * failure_rate:.2f}%)"
    )
    if non_finite_indices:
        print(f"non-finite trajectory IDs: {non_finite_indices}")

    print("\nConservation statistics")
    for metric_name in CONSERVATION_METRIC_NAMES:
        display_name = metric_name.replace("_", " ")
        print(display_name)
        _print_metric_summary(
            "  predicted",
            conservation_summary["predicted"][metric_name],
        )
        _print_metric_summary(
            "  true numerical",
            conservation_summary["true_numerical"][metric_name],
        )
        _print_metric_summary(
            "  predicted minus true",
            conservation_summary["predicted_minus_true"][metric_name],
        )
    print(f"suite elapsed time: {elapsed_seconds:.2f} seconds")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        print(f"Saved test-suite report: {output_path}")

    return report

"""Small MLflow helper for the training scripts.

The project already writes JSON configs, JSON metrics, and PyTorch checkpoints
under ``experiments/runs``. This module adds MLflow tracking beside that flow
without forcing every trainer to know MLflow's details.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping

try:
    import mlflow
except ImportError:  # pragma: no cover - depends on the local environment
    mlflow = None


Scalar = str | int | float | bool | None


def _as_mapping(data: Any) -> Mapping[str, Any]:
    if isinstance(data, Mapping):
        return data
    if hasattr(data, "__dict__"):
        return vars(data)
    raise TypeError(f"Expected a mapping or argparse-style object, got {type(data)!r}")


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={tuple(value.shape)})"
    return str(value)


def _format_param_value(value: Any) -> Scalar:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return json.dumps(value, default=_json_default, sort_keys=True)


def flatten_dict(
    data: Mapping[str, Any] | Any,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested mappings so MLflow can log them as params or metrics."""
    flattened: dict[str, Any] = {}

    for key, value in _as_mapping(data).items():
        full_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_dict(value, full_key, separator))
        else:
            flattened[full_key] = value

    return flattened


def prepare_params(data: Mapping[str, Any] | Any) -> dict[str, Scalar]:
    """Convert CLI args/config dictionaries into MLflow-safe param values."""
    return {
        key: _format_param_value(value)
        for key, value in flatten_dict(data).items()
        if not key.startswith("_")
    }


def prepare_metrics(data: Mapping[str, Any] | Any) -> dict[str, float]:
    """Keep only finite numeric values because MLflow metrics must be numbers."""
    metrics: dict[str, float] = {}

    for key, value in flatten_dict(data).items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                continue

        if isinstance(value, bool):
            value = int(value)

        if isinstance(value, int | float):
            metric_value = float(value)
            if math.isfinite(metric_value):
                metrics[key] = metric_value

    return metrics


def prepare_tags(data: Mapping[str, Any] | None) -> dict[str, str]:
    """Convert tag values to strings; MLflow tags are string metadata."""
    if data is None:
        return {}
    return {
        key: str(_format_param_value(value))
        for key, value in flatten_dict(data).items()
        if value is not None
    }


class MLflowLogger:
    """Context-managed MLflow run logger with a no-op fallback.

    Example:
        with MLflowLogger(PROJECT_ROOT, "gns_onestep", tags={"model": "GNS"}) as logger:
            logger.log_params(args)
            logger.log_metrics({"loss": avg_loss}, step=epoch)
            logger.log_artifact(output_path, artifact_path="checkpoints")
    """

    def __init__(
        self,
        project_root: Path,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: Mapping[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = (
            # this is not a normal folder path, sqlite:///relative/path.db is a relative database path where sqlite:/// is the URI prefix
            tracking_uri or f"sqlite:///{self.project_root / 'mlflow.db'}"
        )
        self.tags = prepare_tags(tags)
        self.requested_enabled = enabled
        self.enabled = enabled and mlflow is not None
        self.run = None

    def __enter__(self) -> "MLflowLogger":
        return self.start()

    def start(self) -> "MLflowLogger":
        if not self.enabled:
            if not self.requested_enabled:
                print("MLflow logging disabled for this run.")
            elif mlflow is None:
                print("MLflow logging disabled: install mlflow to enable tracking.")
            return self

        # mlflow.set_tracking_uri takes in a string which points to where your mlflow backend lives, for example in our case
        # if a value is not passed in then it defaults to sqlite:///.../REU2026/mlflow.db
        # So mlflow stores run metadata, paranms, metrics and etc.
        mlflow.set_tracking_uri(self.tracking_uri)

        # this line selects the ml flow experiment such as "gns_onestep" or "hnn_rollout". If the name does not exist yet, then ml_flow
        # creates it
        mlflow.set_experiment(self.experiment_name)

        # this creates a new indivisual ml flow run inside the selected experiment above and makes it the active run.
        # self.run lets the logger later access the run id and know whether there is an active run to close.
        # a run can store params, metrics, artifacts (files, checkpoints, etc.), tags(metadata), and status(FINISHED/FAILED).
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self

    def end(self, status: str = "FINISHED") -> None:
        if self.enabled and self.run is not None:
            mlflow.end_run(status=status)
            self.run = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end(status=status)

    # @property decorator allows us to treat the method like a attribute. Instead of running ml_logger.run_id() we use ml_looger.run_id
    @property
    def run_id(self) -> str | None:
        if self.run is None:
            return None
        return self.run.info.run_id

    def log_params(self, params: Mapping[str, Any] | Any) -> None:
        if self.enabled:
            mlflow.log_params(prepare_params(params))

    def log_metrics(
        self,
        metrics: Mapping[str, Any] | Any,
        step: int | None = None,
    ) -> None:
        prepared_metrics = prepare_metrics(metrics)
        if self.enabled and prepared_metrics:
            mlflow.log_metrics(prepared_metrics, step=step)

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        prepared_tags = prepare_tags(tags)
        if self.enabled and prepared_tags:
            mlflow.set_tags(prepared_tags)

    def log_description(self, description: str | None) -> None:
        if self.enabled and description:
            mlflow.set_tag("mlflow.note.content", description)

    def log_artifact(
        self,
        path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        artifact = Path(path)
        if self.enabled and artifact.exists():
            mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

    def log_artifacts(
        self,
        path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        artifact_dir = Path(path)
        if self.enabled and artifact_dir.exists():
            mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_path)

    def log_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.log_artifact(checkpoint_path, artifact_path="checkpoints")

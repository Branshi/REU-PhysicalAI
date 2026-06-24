import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(project_root: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = project_root / "experiments" / "runs" / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def args_to_dict(args: Any) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def prepare_run_outputs(
    project_root: Path,
    run_name: str,
    args: Any,
    output_path: str | None = None,
) -> tuple[Path, Path | None, Path | None]:
    if output_path is not None:
        checkpoint_path = Path(output_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_path, None, None

    run_dir = create_run_dir(project_root, run_name)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "checkpoint.pt"

    save_json(config_path, args_to_dict(args))

    return checkpoint_path, run_dir, metrics_path


def find_latest_run_checkpoint(project_root: Path, run_name: str) -> Path | None:
    runs_root = project_root / "experiments" / "runs"
    if not runs_root.exists():
        return None

    run_dirs = sorted(
        (
            path
            for path in runs_root.glob(f"{run_name}_*")
            if path.is_dir() and (path / "checkpoint.pt").exists()
        ),
        key=lambda path: path.name,
        reverse=True,
    )

    if not run_dirs:
        return None

    return run_dirs[0] / "checkpoint.pt"

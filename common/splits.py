"""Shared trajectory-split validation used by every model family."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path):
    """Make CLI and manifest paths independent of the current working directory."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_split_manifest(split_path, dataset_path, dataset_shape):
    """Load a split and verify that it partitions the supplied dataset exactly once."""
    split_path = resolve_project_path(split_path)
    dataset_path = resolve_project_path(dataset_path)

    with split_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Shape checks catch a common mistake: pairing valid trajectory IDs with a
    # different dataset that happens to live near the intended file.
    num_trajectories, num_steps, num_bodies, dim = dataset_shape
    expected_dimensions = {
        "num_trajectories": num_trajectories,
        "num_steps": num_steps,
        "num_bodies": num_bodies,
        "dim": dim,
    }
    for key, expected_value in expected_dimensions.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Split manifest {key}={manifest.get(key)!r} does not match "
                f"dataset {key}={expected_value}."
            )

    # The path check distinguishes datasets that have identical tensor shapes
    # but were generated with different physical parameters or random seeds.
    manifest_dataset_path = manifest.get("dataset_path")
    if manifest_dataset_path is None:
        raise ValueError("Split manifest is missing dataset_path.")
    if resolve_project_path(manifest_dataset_path) != dataset_path:
        raise ValueError(
            "Split manifest was created for a different dataset: "
            f"{manifest_dataset_path!r}. Loaded dataset: {str(dataset_path)!r}."
        )

    split = manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("Split manifest must contain a split object.")

    # Validate each list before indexing tensors so malformed manifests fail
    # with a useful error instead of silently duplicating or dropping data.
    split_indices = {}
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_indices"
        values = split.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Split manifest {key} must be a non-empty list.")
        if any(type(index) is not int for index in values):
            raise ValueError(f"Split manifest {key} must contain only integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"Split manifest {key} contains duplicate indices.")
        if min(values) < 0 or max(values) >= num_trajectories:
            raise ValueError(
                f"Split manifest {key} contains an index outside "
                f"[0, {num_trajectories - 1}]."
            )
        split_indices[split_name] = values

    # A fair split must be disjoint and exhaustive: every trajectory belongs to
    # exactly one of train, validation, or test.
    train_set = set(split_indices["train"])
    val_set = set(split_indices["val"])
    test_set = set(split_indices["test"])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train, validation, and test trajectory splits overlap.")

    all_indices = train_set | val_set | test_set
    if all_indices != set(range(num_trajectories)):
        raise ValueError(
            "Train, validation, and test splits must include every trajectory "
            "exactly once."
        )

    return manifest, split_indices


def validate_checkpoint_split(checkpoint, manifest):
    """Reject a checkpoint that records a different dataset partition."""
    if not isinstance(checkpoint, dict):
        return False

    # Older checkpoints did not store manifests, so they remain loadable. New
    # checkpoints are checked strictly to prevent accidental split relabeling.
    checkpoint_manifest = checkpoint.get("split_manifest")
    if checkpoint_manifest is None:
        return False

    if checkpoint_manifest.get("split") != manifest.get("split"):
        raise ValueError(
            "Checkpoint trajectory split does not match the supplied split manifest."
        )

    checkpoint_dataset_path = checkpoint_manifest.get("dataset_path")
    manifest_dataset_path = manifest.get("dataset_path")
    if checkpoint_dataset_path is None or manifest_dataset_path is None:
        raise ValueError("Checkpoint or supplied split manifest is missing dataset_path.")
    if resolve_project_path(checkpoint_dataset_path) != resolve_project_path(
        manifest_dataset_path
    ):
        raise ValueError(
            "Checkpoint and supplied split manifests refer to different datasets."
        )

    return True

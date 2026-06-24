"""Helpers for loading custom N-body rollout initial conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


def load_initial_conditions(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    with Path(path).open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Initial conditions file must contain a JSON object.")

    return data


def parse_json_tensor(
    value: str | list[Any] | None,
    *,
    device: torch.device,
    name: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    if value is None:
        return None

    parsed = json.loads(value) if isinstance(value, str) else value
    tensor = torch.tensor(parsed, dtype=dtype, device=device)

    if tensor.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")

    return tensor


def get_condition(
    conditions: Mapping[str, Any],
    cli_value: str | None,
    *names: str,
) -> str | list[Any] | None:
    if cli_value is not None:
        return cli_value

    for name in names:
        if name in conditions:
            return conditions[name]

    return None


def normalize_masses(masses: torch.Tensor, num_bodies: int) -> torch.Tensor:
    masses = masses.reshape(-1)

    if masses.shape[0] != num_bodies:
        raise ValueError(
            f"masses must contain {num_bodies} values. Got {masses.shape[0]}."
        )

    return masses.unsqueeze(-1)


def validate_body_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    num_bodies: int | None = None,
    dim: int | None = None,
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [num_bodies, dim]. Got {tensor.shape}.")

    if num_bodies is not None and tensor.shape[0] != num_bodies:
        raise ValueError(
            f"{name} must contain {num_bodies} bodies. Got {tensor.shape[0]}."
        )

    if dim is not None and tensor.shape[1] != dim:
        raise ValueError(f"{name} must have dim={dim}. Got {tensor.shape[1]}.")

    return tensor


def resolve_rollout_dt(
    checkpoint_dt: float,
    num_steps: int,
    *,
    dt_override: float | None = None,
    period: float | None = None,
) -> float:
    if dt_override is not None:
        return dt_override

    if period is not None:
        if num_steps <= 0:
            raise ValueError("rollout steps must be positive when period is provided.")
        return period / num_steps

    return checkpoint_dt

"""Plot comparable error and conservation metrics from fixed-suite JSON reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DARK_GREEN = "#18453B"
LIGHT_GREEN = "#E8F1EE"
YELLOW = "#FFCC00"
LIGHT_YELLOW = "#FFEDA6"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 17,
    }
)

DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT / "experiments" / "evaluations" / "one_step_4h"
)
DEFAULT_REPORTS = [
    DEFAULT_EVALUATION_DIR / f"{model}_test_suite.json"
    for model in ("gns", "egns", "setnode", "egnn_hnn", "hnn")
]

ERROR_METRICS = (
    ("rollout_position_mse", "Rollout position MSE"),
    ("acceleration_mse_on_true_states", "Acceleration MSE on true states"),
)

POSITION_RMSE_METRIC = ("rollout_position_rmse", "Rollout position RMSE")

RELATIVE_ENERGY_DRIFT_METRIC = (
    "final_absolute_relative_energy_drift",
    "Final |relative energy drift|",
)

CONSERVATION_METRICS = (
    (
        "final_absolute_energy_drift",
        "Final |energy drift|",
        "Energy units",
    ),
    (
        "final_absolute_relative_energy_drift",
        "Final |relative energy drift|",
        "Dimensionless",
    ),
    (
        "final_center_of_mass_drift",
        "Final center-of-mass drift",
        "Position units",
    ),
    (
        "max_absolute_energy_drift",
        "Maximum |energy drift|",
        "Energy units",
    ),
    (
        "max_absolute_relative_energy_drift",
        "Maximum |relative energy drift|",
        "Dimensionless",
    ),
    (
        "max_center_of_mass_drift",
        "Maximum center-of-mass drift",
        "Position units",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create cross-model plots from fixed rollout test-suite JSON reports."
        )
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        default=[str(path) for path in DEFAULT_REPORTS],
        help=(
            "One fixed-suite JSON report per model. Defaults to the five "
            "one_step_4h evaluation reports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVALUATION_DIR / "plots"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
        help="Figure formats to save. Defaults to PNG and PDF.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_reports(paths: list[str]) -> list[dict]:
    reports = []
    for value in paths:
        path = resolve_path(value)
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
        report["_report_path"] = str(path)
        reports.append(report)

    model_names = [report.get("protocol", {}).get("model") for report in reports]
    if any(name is None for name in model_names):
        raise ValueError("Every report must contain protocol.model.")
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"Report model names must be unique. Got {model_names}.")
    return reports


def validate_comparable_reports(reports: list[dict]) -> None:
    if not reports:
        raise ValueError("At least one report is required.")

    reference = reports[0]["protocol"]
    comparison_fields = (
        "split",
        "trajectory_indices",
        "rollout_steps",
        "failure_threshold",
    )
    for report in reports:
        protocol = report.get("protocol", {})
        if protocol.get("version", 0) < 3:
            raise ValueError(
                f"{report['_report_path']} predates acceleration MSE. "
                "Rerun the fixed test suite with the current evaluator."
            )
        for field in comparison_fields:
            if protocol.get(field) != reference.get(field):
                raise ValueError(
                    f"Reports are not comparable: protocol.{field} differs "
                    f"between {reports[0]['_report_path']} and "
                    f"{report['_report_path']}."
                )

        summary = report.get("summary", {})
        for metric_name, _ in ERROR_METRICS:
            if metric_name not in summary:
                raise ValueError(
                    f"{report['_report_path']} is missing summary.{metric_name}."
                )
        conservation = summary.get("conservation", {})
        for series_name in ("predicted", "true_numerical"):
            if series_name not in conservation:
                raise ValueError(
                    f"{report['_report_path']} is missing conservation "
                    f"series {series_name}."
                )
            for metric_name, _, _ in CONSERVATION_METRICS:
                if metric_name not in conservation[series_name]:
                    raise ValueError(
                        f"{report['_report_path']} is missing conservation "
                        f"metric {metric_name}."
                    )


def finite_mean_std(summary: dict, context: str) -> tuple[float, float]:
    mean = summary.get("mean")
    std = summary.get("std")
    if mean is None or std is None or not np.isfinite([mean, std]).all():
        raise ValueError(f"{context} does not have finite mean/std values.")
    return float(mean), float(std)


def positive_error_bars(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    # Keep lower error-bar endpoints above zero so they remain valid on log axes.
    lower = np.minimum(stds, np.maximum(means * 0.999, 0.0))
    return np.vstack([lower, stds])


def model_fill_colors(model_names: list[str]) -> list[str]:
    """Use the poster palette while highlighting SETNODE."""

    return [
        LIGHT_YELLOW if model_name.casefold() == "setnode" else LIGHT_GREEN
        for model_name in model_names
    ]


def suite_title(protocol: dict) -> str:
    """Describe both saved states and integration transitions unambiguously."""

    rollout_transitions = int(protocol["rollout_steps"])
    saved_states = rollout_transitions + 1
    return (
        f"Fixed {protocol['split']} suite: mean ± population standard deviation\n"
        f"{len(protocol['trajectory_indices'])} trajectories; "
        f"{saved_states} states each ({rollout_transitions} rollout transitions)"
    )


def style_axis(axis, *, ylabel: str | None = None) -> None:
    axis.set_yscale("log")
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.grid(
        axis="y",
        which="both",
        color=DARK_GREEN,
        alpha=0.16,
        linewidth=0.8,
    )
    axis.tick_params(axis="x", rotation=20)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(DARK_GREEN)
    axis.spines["bottom"].set_color(DARK_GREEN)


def save_figure(figure, output_dir: Path, stem: str, formats: list[str], dpi: int):
    saved_paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def plot_error_metrics(
    reports: list[dict], output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    model_names = [report["protocol"]["model"] for report in reports]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for axis, (metric_name, title) in zip(axes, ERROR_METRICS):
        values = [
            finite_mean_std(
                report["summary"][metric_name],
                f"{report['protocol']['model']} {metric_name}",
            )
            for report in reports
        ]
        means = np.asarray([value[0] for value in values])
        stds = np.asarray([value[1] for value in values])
        bars = axis.bar(
            x,
            means,
            color=model_colors,
            edgecolor=DARK_GREEN,
            linewidth=1.6,
            yerr=positive_error_bars(means, stds),
            capsize=3,
            error_kw={
                "ecolor": DARK_GREEN,
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )
        axis.set_title(title)
        axis.set_xticks(x, model_names)
        style_axis(axis, ylabel="Mean squared error")
        axis.bar_label(bars, fmt="%.1e", padding=4, fontsize=11, rotation=90)

    protocol = reports[0]["protocol"]
    figure.suptitle(suite_title(protocol))
    paths = save_figure(figure, output_dir, "error-metrics", formats, dpi)
    plt.close(figure)
    return paths


def plot_position_rmse(
    reports: list[dict], output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    """Plot aggregate rollout position RMSE for each model."""

    metric_name, title = POSITION_RMSE_METRIC
    model_names = [report["protocol"]["model"] for report in reports]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))
    values = [
        finite_mean_std(
            report["summary"][metric_name],
            f"{report['protocol']['model']} {metric_name}",
        )
        for report in reports
    ]
    means = np.asarray([value[0] for value in values])
    stds = np.asarray([value[1] for value in values])

    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    bars = axis.bar(
        x,
        means,
        color=model_colors,
        edgecolor=DARK_GREEN,
        linewidth=1.6,
        yerr=positive_error_bars(means, stds),
        capsize=3,
        error_kw={
            "ecolor": DARK_GREEN,
            "elinewidth": 1.2,
            "capthick": 1.2,
        },
    )
    axis.set_title(title)
    axis.set_xticks(x, model_names)
    style_axis(axis, ylabel="Position RMSE")
    axis.bar_label(bars, fmt="%.2e", padding=4, fontsize=11, rotation=90)

    protocol = reports[0]["protocol"]
    figure.suptitle(suite_title(protocol))
    paths = save_figure(figure, output_dir, "position-rmse", formats, dpi)
    plt.close(figure)
    return paths


def plot_relative_energy_drift(
    reports: list[dict], output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    """Compare final predicted and true numerical relative-energy drift."""

    metric_name, title = RELATIVE_ENERGY_DRIFT_METRIC
    model_names = [report["protocol"]["model"] for report in reports]
    x = np.arange(len(reports))
    width = 0.36
    series = (
        ("predicted", "Predicted", YELLOW),
        ("true_numerical", "True numerical", DARK_GREEN),
    )

    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    for series_index, (series_name, series_label, color) in enumerate(series):
        values = [
            finite_mean_std(
                report["summary"]["conservation"][series_name][metric_name],
                f"{report['protocol']['model']} {series_name} {metric_name}",
            )
            for report in reports
        ]
        means = np.asarray([value[0] for value in values])
        stds = np.asarray([value[1] for value in values])
        offset = (series_index - 0.5) * width
        bars = axis.bar(
            x + offset,
            means,
            width,
            label=series_label,
            color=color,
            edgecolor=DARK_GREEN,
            linewidth=1.2,
            yerr=positive_error_bars(means, stds),
            capsize=2.5,
            error_kw={
                "ecolor": DARK_GREEN,
                "elinewidth": 1.1,
                "capthick": 1.1,
            },
        )
        if series_name == "predicted":
            axis.bar_label(
                bars,
                fmt="%.1e",
                padding=4,
                fontsize=10,
                rotation=90,
            )
    axis.set_title(title)
    axis.set_xticks(x, model_names)
    style_axis(axis, ylabel="Absolute relative drift")

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=False,
    )
    protocol = reports[0]["protocol"]
    figure.suptitle(suite_title(protocol))
    paths = save_figure(
        figure,
        output_dir,
        "relative-energy-drift",
        formats,
        dpi,
    )
    plt.close(figure)
    return paths


def plot_conservation_metrics(
    reports: list[dict], output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    model_names = [report["protocol"]["model"] for report in reports]
    x = np.arange(len(reports))
    width = 0.36
    series = (
        ("predicted", "Predicted", YELLOW),
        ("true_numerical", "True numerical", DARK_GREEN),
    )

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
    for axis, (metric_name, title, ylabel) in zip(
        axes.flat,
        CONSERVATION_METRICS,
    ):
        for series_index, (series_name, series_label, color) in enumerate(series):
            values = [
                finite_mean_std(
                    report["summary"]["conservation"][series_name][metric_name],
                    f"{report['protocol']['model']} {series_name} {metric_name}",
                )
                for report in reports
            ]
            means = np.asarray([value[0] for value in values])
            stds = np.asarray([value[1] for value in values])
            offset = (series_index - 0.5) * width
            axis.bar(
                x + offset,
                means,
                width,
                label=series_label,
                color=color,
                edgecolor=DARK_GREEN,
                linewidth=1.2,
                yerr=positive_error_bars(means, stds),
                capsize=2.5,
                error_kw={
                    "ecolor": DARK_GREEN,
                    "elinewidth": 1.1,
                    "capthick": 1.1,
                },
            )
        axis.set_title(title)
        axis.set_xticks(x, model_names)
        style_axis(axis, ylabel=ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=False,
    )
    paths = save_figure(figure, output_dir, "conservation-metrics", formats, dpi)
    plt.close(figure)
    return paths


def main():
    args = parse_args()
    reports = load_reports(args.reports)
    validate_comparable_reports(reports)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    saved_paths.extend(plot_error_metrics(reports, output_dir, args.formats, args.dpi))
    saved_paths.extend(plot_position_rmse(reports, output_dir, args.formats, args.dpi))
    saved_paths.extend(
        plot_relative_energy_drift(reports, output_dir, args.formats, args.dpi)
    )
    saved_paths.extend(
        plot_conservation_metrics(reports, output_dir, args.formats, args.dpi)
    )
    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()

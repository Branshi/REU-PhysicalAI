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
LIGHT_YELLOW = "#FFEDA6"
MLP_BROWN = "#D8B28D"

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.titlesize": 21,
    }
)

DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT / "experiments" / "evaluations" / "one_step_4h"
)
DEFAULT_REPORTS = [
    DEFAULT_EVALUATION_DIR / f"{model}_test_suite.json"
    for model in ("gns", "egns", "setnode", "egnn_hnn", "hnn", "mlp")
]

ERROR_METRICS = (
    ("rollout_position_mse", "Rollout position MSE"),
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

FINAL_CONSERVATION_METRICS = (
    (
        "final_absolute_energy_drift",
        "Final absolute energy drift",
        "Energy units",
    ),
    CONSERVATION_METRICS[2],
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
            "One fixed-suite JSON report per model. Defaults to the six "
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


def finite_mean(summary: dict, context: str) -> float:
    mean = summary.get("mean")
    if mean is None or not np.isfinite(mean):
        raise ValueError(f"{context} does not have a finite mean value.")
    return float(mean)


def relative_to_setnode(
    values: np.ndarray,
    model_names: list[str],
) -> np.ndarray:
    baseline_indices = [
        index
        for index, model_name in enumerate(model_names)
        if model_name.casefold() == "setnode"
    ]
    if len(baseline_indices) != 1:
        raise ValueError(
            "Exactly one SETNODE report is required for relative labels. "
            f"Got model names {model_names}."
        )

    baseline = values[baseline_indices[0]]
    if baseline == 0.0:
        raise ValueError("SETNODE baseline must be nonzero for relative labels.")
    return 100.0 * (values / baseline - 1.0)


def format_relative_percentage(value: float) -> str:
    if abs(value) < 0.5:
        return "0%"
    if abs(value) >= 10_000:
        return f"{value:+.1e}%"
    return f"{value:+.0f}%"


def add_value_and_relative_labels(
    axis,
    bars,
    values: np.ndarray,
    model_names: list[str],
    *,
    value_format: str,
    fontsize: int,
    padding: float = 4,
) -> None:
    relative_percentages = relative_to_setnode(values, model_names)
    labels = [
        f"{value_format % value}\n{format_relative_percentage(relative)}"
        for value, relative in zip(values, relative_percentages)
    ]
    axis.bar_label(
        bars,
        labels=labels,
        padding=padding,
        fontsize=fontsize,
        rotation=0,
    )
    axis.margins(y=0.24)


def add_value_labels(
    axis,
    bars,
    values: np.ndarray,
    *,
    value_format: str,
    fontsize: int,
    padding: float = 4,
) -> None:
    """Label bars with raw metric values only."""

    axis.bar_label(
        bars,
        labels=[value_format % value for value in values],
        padding=padding,
        fontsize=fontsize,
        rotation=0,
    )
    axis.margins(y=0.18)


def shared_true_numerical_mean(
    reports: list[dict],
    metric_name: str,
) -> float:
    values = np.asarray([
        finite_mean(
            report["summary"]["conservation"]["true_numerical"][metric_name],
            f"{report['protocol']['model']} true_numerical {metric_name}",
        )
        for report in reports
    ])
    reference = float(np.median(values))
    if not np.allclose(values, reference, rtol=1e-2, atol=1e-12):
        raise ValueError(
            f"True numerical {metric_name} differs between model reports: "
            f"{values.tolist()}."
        )
    return reference


def add_true_numerical_line(
    axis,
    value: float,
    *,
    fontsize: int,
) -> None:
    axis.axhline(
        value,
        color=DARK_GREEN,
        linestyle="--",
        linewidth=1.8,
        zorder=3,
    )
    axis.annotate(
        f"Reference solver: {value:.1e}",
        xy=(0.99, value),
        xycoords=("axes fraction", "data"),
        xytext=(0, -5),
        textcoords="offset points",
        ha="right",
        va="top",
        color=DARK_GREEN,
        fontsize=fontsize,
    )


def display_model_name(model_name: str) -> str:
    """Return the publication-facing model name used on plot axes."""

    publication_names = {
        "egns": "EGNN",
    }
    return publication_names.get(model_name.casefold(), model_name)


def model_fill_colors(model_names: list[str]) -> list[str]:
    """Use the poster palette while highlighting SETNODE and the MLP baseline."""

    return [
        (
            LIGHT_YELLOW
            if model_name.casefold() == "setnode"
            else MLP_BROWN
            if model_name.casefold() == "mlp"
            else LIGHT_GREEN
        )
        for model_name in model_names
    ]


def suite_title(protocol: dict, *, relative_labels: bool = True) -> str:
    """Describe both saved states and integration transitions unambiguously."""

    rollout_transitions = int(protocol["rollout_steps"])
    saved_states = rollout_transitions + 1
    label_description = (
        "value and % difference from SETNODE"
        if relative_labels
        else "raw mean value"
    )
    return (
        f"Fixed {protocol['split']} suite: mean error; labels show "
        f"{label_description}\n"
        f"{len(protocol['trajectory_indices'])} trajectories; "
        f"{saved_states} states each ({rollout_transitions} rollout transitions)"
    )


def style_axis(axis, *, ylabel: str | None = None) -> None:
    axis.set_yscale("log")
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.tick_params(axis="x")
    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
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
    model_names = [
        display_model_name(report["protocol"]["model"]) for report in reports
    ]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))

    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    for metric_name, title in ERROR_METRICS:
        means = np.asarray([
            finite_mean(
                report["summary"][metric_name],
                f"{report['protocol']['model']} {metric_name}",
            )
            for report in reports
        ])
        bars = axis.bar(
            x,
            means,
            color=model_colors,
            edgecolor=DARK_GREEN,
            linewidth=1.6,
        )
        axis.set_title(title)
        axis.set_xticks(x, model_names)
        style_axis(axis, ylabel="Mean squared error")
        add_value_and_relative_labels(
            axis,
            bars,
            means,
            model_names,
            value_format="%.1e",
            fontsize=13,
        )

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
    model_names = [
        display_model_name(report["protocol"]["model"]) for report in reports
    ]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))
    means = np.asarray([
        finite_mean(
            report["summary"][metric_name],
            f"{report['protocol']['model']} {metric_name}",
        )
        for report in reports
    ])

    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    bars = axis.bar(
        x,
        means,
        color=model_colors,
        edgecolor=DARK_GREEN,
        linewidth=1.6,
    )
    axis.set_title(title)
    axis.set_xticks(x, model_names)
    style_axis(axis, ylabel="Position RMSE")
    add_value_and_relative_labels(
        axis,
        bars,
        means,
        model_names,
        value_format="%.2e",
        fontsize=10,
    )

    protocol = reports[0]["protocol"]
    figure.suptitle(suite_title(protocol))
    paths = save_figure(figure, output_dir, "position-rmse", formats, dpi)
    plt.close(figure)
    return paths


def plot_relative_energy_drift(
    reports: list[dict], output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    """Compare final predicted and reference-solver relative-energy drift."""

    metric_name, title = RELATIVE_ENERGY_DRIFT_METRIC
    model_names = [
        display_model_name(report["protocol"]["model"]) for report in reports
    ]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))

    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    means = np.asarray([
        finite_mean(
            report["summary"]["conservation"]["predicted"][metric_name],
            f"{report['protocol']['model']} predicted {metric_name}",
        )
        for report in reports
    ])
    bars = axis.bar(
        x,
        means,
        color=model_colors,
        edgecolor=DARK_GREEN,
        linewidth=1.2,
    )
    add_value_labels(
        axis,
        bars,
        means,
        value_format="%.1e",
        fontsize=9,
    )
    add_true_numerical_line(
        axis,
        shared_true_numerical_mean(reports, metric_name),
        fontsize=9,
    )
    axis.set_title(title)
    axis.set_xticks(x, model_names)
    style_axis(axis, ylabel="Absolute relative drift")

    protocol = reports[0]["protocol"]
    figure.suptitle(suite_title(protocol, relative_labels=False))
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
    model_names = [
        display_model_name(report["protocol"]["model"]) for report in reports
    ]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
    for axis, (metric_name, title, ylabel) in zip(
        axes.flat,
        CONSERVATION_METRICS,
    ):
        means = np.asarray([
            finite_mean(
                report["summary"]["conservation"]["predicted"][metric_name],
                f"{report['protocol']['model']} predicted {metric_name}",
            )
            for report in reports
        ])
        bars = axis.bar(
            x,
            means,
            color=model_colors,
            edgecolor=DARK_GREEN,
            linewidth=1.2,
        )
        add_value_labels(
            axis,
            bars,
            means,
            value_format="%.1e",
            fontsize=7,
        )
        add_true_numerical_line(
            axis,
            shared_true_numerical_mean(reports, metric_name),
            fontsize=7,
        )
        axis.set_title(title)
        axis.set_xticks(x, model_names)
        style_axis(axis, ylabel=ylabel)

    paths = save_figure(figure, output_dir, "conservation-metrics", formats, dpi)
    plt.close(figure)
    return paths


def plot_final_conservation_metrics(
    reports: list[dict], output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    """Plot final energy and center-of-mass drift side by side."""

    model_names = [
        display_model_name(report["protocol"]["model"]) for report in reports
    ]
    model_colors = model_fill_colors(model_names)
    x = np.arange(len(reports))

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True)
    for axis, (metric_name, title, ylabel) in zip(
        axes,
        FINAL_CONSERVATION_METRICS,
    ):
        means = np.asarray([
            finite_mean(
                report["summary"]["conservation"]["predicted"][metric_name],
                f"{report['protocol']['model']} predicted {metric_name}",
            )
            for report in reports
        ])
        bars = axis.bar(
            x,
            means,
            color=model_colors,
            edgecolor=DARK_GREEN,
            linewidth=1.2,
        )
        add_value_labels(
            axis,
            bars,
            means,
            value_format="%.1e",
            fontsize=14,
        )
        add_true_numerical_line(
            axis,
            shared_true_numerical_mean(reports, metric_name),
            fontsize=14,
        )
        axis.set_title(title)
        axis.set_xticks(x, model_names)
        style_axis(axis, ylabel=ylabel)
        axis.set_ylabel(ylabel, fontsize=20)
        axis.tick_params(axis="x", labelsize=16)
        axis.tick_params(axis="y", labelsize=17)

    paths = save_figure(
        figure,
        output_dir,
        "final-energy-and-center-of-mass-drift",
        formats,
        dpi,
    )
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
    saved_paths.extend(
        plot_final_conservation_metrics(
            reports,
            output_dir,
            args.formats,
            args.dpi,
        )
    )
    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()

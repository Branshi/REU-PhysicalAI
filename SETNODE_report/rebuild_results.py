"""Rebuild the report's descriptive results from the supplied evaluation JSON.

No model is trained or evaluated by this script. Run from any directory:
    python rebuild_results.py
    python rebuild_results.py --no-figure
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("GNS", "gns"), ("EGNN", "egns"), ("EGNN-HNN", "egnn_hnn"),
    ("HNN", "hnn"), ("MLP", "mlp"), ("SETNODE", "setnode"),
]


def load_records() -> dict[str, dict[str, Any]]:
    """Read and validate the saved, paired evaluation records."""
    records = {}
    split = json.loads((ROOT / "data" / "nbody_split_manifest.json").read_text())
    expected_ids = split["split"]["test_indices"][:50]
    reference_protocol = None
    for _, key in MODELS:
        path = ROOT / "data" / f"{key}_test_suite.json"
        record = json.loads(path.read_text())
        protocol = record["protocol"]
        comparable = {k: v for k, v in protocol.items() if k not in {
            "model", "checkpoint_path", "dataset_path", "split_path"
        }}
        if reference_protocol is None:
            reference_protocol = comparable
        if comparable != reference_protocol:
            raise ValueError(f"Evaluation protocol differs for {key}.")
        if protocol["trajectory_indices"] != expected_ids:
            raise ValueError(f"Test selection differs from the saved manifest: {key}.")
        rows = record["per_trajectory"]
        if len(rows) != 50 or [r["trajectory_index"] for r in rows] != expected_ids:
            raise ValueError(f"Missing or misordered trajectory rows: {key}.")
        values = np.array([r["rollout_position_mse"] for r in rows], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Unexpected position errors in {key}.")
        stats = record["summary"]["rollout_position_mse"]
        for name, computed in {
            "mean": values.mean(), "std": values.std(ddof=0),
            "median": np.median(values), "worst": values.max(),
        }.items():
            if not np.isclose(computed, stats[name], rtol=1e-12, atol=1e-15):
                raise ValueError(f"Summary {name} mismatch in {key}.")
        failures = sum(bool(r["failed"]) for r in rows)
        if failures != record["summary"]["failure_count"]:
            raise ValueError(f"Failure count mismatch in {key}.")
        records[key] = record
    return records


def sci(value: float, significant: int = 3) -> str:
    mantissa, exponent = f"{value:.{significant - 1}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def result_rows(records: dict[str, dict[str, Any]]) -> str:
    rows = []
    for label, key in MODELS:
        summary = records[key]["summary"]
        stats = summary["rollout_position_mse"]
        values = [f"${sci(stats[k])}$" for k in ("mean", "std", "median")]
        name = r"\textbf{SETNODE}" if key == "setnode" else label
        rows.append(name + " & " + " & ".join(values)
                    + f" & {summary['failure_count']}/50 " + r"\\")
    return "\n".join(rows)


def conservation_rows(records: dict[str, dict[str, Any]]) -> str:
    fields = ["final_absolute_energy_drift", "final_absolute_relative_energy_drift",
              "final_center_of_mass_drift"]
    rows = []
    for label, key in MODELS:
        stats = records[key]["summary"]["conservation"]["predicted"]
        rows.append(label + " & " + " & ".join(
            f"${sci(stats[k]['mean'])}$" for k in fields) + " " + r"\\")
    ref = records["setnode"]["summary"]["conservation"]["true_numerical"]
    rows.append(r"\midrule")
    rows.append("Numerical reference & " + " & ".join(
        f"${sci(ref[k]['mean'])}$" for k in fields) + " " + r"\\")
    return "\n".join(rows)


def update_tex(records: dict[str, dict[str, Any]]) -> None:
    path = ROOT / "SETNODE_Report.tex"
    text = path.read_text()
    for tag, rows in [("RESULTS_ROWS", result_rows(records)),
                      ("CONSERVATION_ROWS", conservation_rows(records))]:
        pattern = rf"% BEGIN {tag}\n.*?% END {tag}"
        replacement = f"% BEGIN {tag}\n{rows}\n% END {tag}"
        text, count = re.subn(pattern, lambda _: replacement, text, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected one marked table region: {tag}.")
    path.write_text(text)


def save_descriptive_summary(records: dict[str, dict[str, Any]]) -> None:
    errors = {key: np.array([r["rollout_position_mse"] for r in rec["per_trajectory"]])
              for key, rec in records.items()}
    base = errors["setnode"]
    summary = {
        "scope": "Descriptive reanalysis of the saved fixed suite; no new model evaluation.",
        "standard_deviation": "population standard deviation across 50 trajectories (ddof=0)",
        "models": {
            key: {
                "mean_position_mse": float(x.mean()),
                "sd_position_mse": float(x.std(ddof=0)),
                "median_position_mse": float(np.median(x)),
                "setnode_lower_mse_count": int(np.sum(base < x)) if key != "setnode" else None,
                "setnode_mean_mse_reduction_percent": float(100 * (1 - base.mean()/x.mean()))
                    if key != "setnode" else None,
            } for key, x in errors.items()
        },
    }
    (ROOT / "data" / "descriptive_summary.json").write_text(json.dumps(summary, indent=2))


def save_figure(records: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [np.array([r["rollout_position_mse"]
                        for r in records[key]["per_trajectory"]]) for _, key in MODELS]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.boxplot(values, tick_labels=[label for label, _ in MODELS],
               whis=(0, 100), showfliers=False, widths=0.48)
    rng = np.random.default_rng(2026)
    for i, x in enumerate(values, start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(x)), x, s=13, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Rollout position MSE")
    ax.set_xlabel("Each point represents one of the same 50 test trajectories")
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "rollout_error_distribution.pdf", bbox_inches="tight")
    fig.savefig(destination / "rollout_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figure", action="store_true", help="Validate data and update tables only.")
    args = parser.parse_args()
    records = load_records()
    update_tex(records)
    save_descriptive_summary(records)
    if not args.no_figure:
        save_figure(records)
    print("Validated all six paired 50-trajectory evaluation records and updated report tables.")


if __name__ == "__main__":
    main()

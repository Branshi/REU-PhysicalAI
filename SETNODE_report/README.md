# SETNODE technical report

**Trillyon Earl. _SETNODE: Spatial Equivariant Transformer Neural ODE for Physical Dynamics._** Unpublished NSF REU technical report, revised October 3, 2026.

[Read the report (PDF)](SETNODE_Report.pdf) · [LaTeX source](SETNODE_Report.tex) · [Project README](../README.md)

This directory contains the report, editable architecture diagrams, figures, and archived evaluation records. The report's original six-model comparison and its later extended-readout illustrations are separate experiments.

## Compile the report

Install a TeX distribution with `latexmk`, pdfLaTeX, `newtx`, TikZ, and the other packages listed in the report preamble. From this directory:

```sh
sh build.sh
```

Equivalently:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error SETNODE_Report.tex
```

For Overleaf, upload this directory with its `figures/` subdirectory and set `SETNODE_Report.tex` as the main document. References are embedded in a `thebibliography` environment; no external bibliography file is needed.

To compile the standalone architecture diagram:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error SETNODE_Diagram.tex
```

`SETNODE_Architecture_TikZ.tex` is a self-contained version of the diagram. The report and `SETNODE_Diagram.tex` use `figures/setnode_architecture.tex`.

## Rebuild the descriptive summaries

From this directory, in a Python environment:

```sh
python -m pip install -r requirements.txt
python rebuild_results.py --no-figure
sh build.sh
```

The script validates the six archived 50-trajectory evaluation records, updates the marked result tables in `SETNODE_Report.tex`, and writes `data/descriptive_summary.json`. Omit `--no-figure` to regenerate the rollout-error distribution figure as well. Recompile the report after updating its source or figures.

These commands reanalyze saved results. They do not train models or generate new rollout predictions. The trained checkpoint weights and raw trajectory datasets are not included here. For training and evaluation code, see the [repository quick start](../README.md#quick-start).

## Included files

- `SETNODE_Report.pdf` and `SETNODE_Report.tex`: compiled report and editable source.
- `figures/`: the report's architecture diagram, error distribution, and illustrative rollouts.
- `SETNODE_Diagram.tex` / `.pdf`: standalone architecture diagram.
- `SETNODE_Architecture_TikZ.tex` / `.pdf`: self-contained diagram variant.
- `data/`: six evaluation records, their split manifest, checkpoint configurations, and descriptive summary.
- `rebuild_results.py` and `requirements.txt`: descriptive-analysis script and dependencies.
- `build.sh`: report compilation command.

The original experiment record does not establish the exact historical training commit or preserve every training setting. Consult the report's experimental-record appendix before interpreting the current code defaults as reproduction instructions.

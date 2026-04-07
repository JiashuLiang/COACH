# COACH

COACH is a script-first workflow for training the COACH exchange-correlation functional from precomputed reaction data.

## Validation Status

The original COACH workflow used Q-Chem. The PySCF-based path in this repository replaces that original route, but it has not been validated as extensively as the earlier implementation. Treat generated data and fitted models as research artifacts that should be checked carefully before downstream use.

## Supported Workflow

The maintained baseline pipeline is:

1. Generate one PySCF text output per species.
2. Parse those outputs into `raw_data.pkl` and `reaction_data.pkl`.
3. Prepare training and test data from reaction data plus CSV metadata.
4. Run mixed-integer optimization.
5. Select manuscript-style grid-sensitivity constraints.
6. Run mixed-integer optimization again with grid-sensitivity constraints.
7. Analyze fitted coefficient sets.

For steps 1-2, see [`1_data_generation/README.md`](1_data_generation/README.md).

For steps 3-7, including the preprocessing, optimization, constraint-selection, and analysis commands, see [`2_optimization/README.md`](2_optimization/README.md).

## Requirements

- Python 3.10+
- `numpy`
- `scipy` and `pyscf` for the data-generation stage
- `gurobipy` plus a working Gurobi license for optimization
- `pandas`

## Metadata Setup

Project-specific metadata files are expected as CSVs. Start from the reference files in [`2_optimization/templates/`](2_optimization/templates/):

- [`2_optimization/templates/dataset_eval.csv`](2_optimization/templates/dataset_eval.csv)
- [`2_optimization/templates/training_weights.csv`](2_optimization/templates/training_weights.csv)
- [`2_optimization/templates/Standard_errors.csv`](2_optimization/templates/Standard_errors.csv)
- [`2_optimization/templates/dataset_info.csv`](2_optimization/templates/dataset_info.csv)

Populate copies of `dataset_eval.csv` and `training_weights.csv` with your actual reactions, dataset assignments, and weights before running the workflow. For required columns and downstream usage details, see [`2_optimization/README.md`](2_optimization/README.md).

## Example

A runnable reference example based on the AE18 atomic XYZ files lives under [`example/`](example/). See [`example/README.md`](example/README.md) for the exact commands and generated artifacts.

## Directory Guide

- [`1_data_generation/`](1_data_generation): PySCF data extraction and matrix generation
- [`2_optimization/`](2_optimization): preprocessing, optimization, grid-constraint selection, and analysis
- [`Optimization.md`](Optimization.md): manuscript-side description of the fitting procedure
- [`1_data_generation/qchem_codes_insert.C`](1_data_generation/qchem_codes_insert.C): reference-only Q-Chem/C++ source retained for comparison; it is not part of the maintained workflow

## Contact

For questions, issues, or suggestions regarding this workflow:

- **Primary contact**: Jiashu Liang (`jsliang@berkeley.edu`)
- **Research group**: Martin Head-Gordon Group, UC Berkeley
- **Issues**: Please use the GitHub issue tracker for bug reports and feature requests

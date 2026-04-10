# COACH

COACH is a protocol for developing carefully optimized and appropriately constrained density functionals within a chosen application domain.

The protocol combines exact-constraint enforcement, flexible functional forms, and modern optimization to support systematic functional development. When applied to the range-separated hybrid meta-GGA framework, it yields the COACH functional, which improves accuracy and transferability relative to leading RSH meta-GGAs while retaining the practical computational cost of its rung.

The approach is not limited to RSH mGGAs and can be extended to other rungs of Jacob's Ladder, including mGGAs and double hybrids, as well as to specialized functionals for particular applications such as nuclear magnetic resonance properties or solid-state systems.

## Requirements

- Base: Python 3.10+, `numpy`, `scipy`
- Workflow (`1_data_generation/`, `2_optimization/`): `pyscf`, `pandas`, `PyYAML`, `gurobipy` plus a working Gurobi license
- Functional use (`FunctionalCOACH/`): `pyscf`, `dftd4`, `basis-set-exchange`
- Tests: `pytest`

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

### Metadata Setup

Project-specific metadata files are expected as CSVs. Start from the reference files in [`2_optimization/templates/`](2_optimization/templates/):

- [`2_optimization/templates/dataset_eval.csv`](2_optimization/templates/dataset_eval.csv)
- [`2_optimization/templates/training_weights.csv`](2_optimization/templates/training_weights.csv)
- [`2_optimization/templates/Standard_errors.csv`](2_optimization/templates/Standard_errors.csv)
- [`2_optimization/templates/dataset_info.csv`](2_optimization/templates/dataset_info.csv)

Populate copies of `dataset_eval.csv` and `training_weights.csv` with your actual reactions, dataset assignments, and weights before running the workflow. For required columns and downstream usage details, see [`2_optimization/README.md`](2_optimization/README.md).

### Example

A runnable reference example based on the AE18 atomic XYZ files lives under [`example/`](example/). See [`example/README.md`](example/README.md) for the exact commands and generated artifacts.

## FunctionalCOACH

[`FunctionalCOACH/`](FunctionalCOACH) is the standalone PySCF implementation of the fitted COACH functional. It is separate from the training workflow above and is meant for running and checking COACH energies directly.

This folder contains:

- [`FunctionalCOACH/coach_pyscf.py`](FunctionalCOACH/coach_pyscf.py): the main entry point for building and running COACH from an XYZ input with metadata
- [`FunctionalCOACH/coach_x.py`](FunctionalCOACH/coach_x.py), [`FunctionalCOACH/coach_css.py`](FunctionalCOACH/coach_css.py), [`FunctionalCOACH/coach_cos.py`](FunctionalCOACH/coach_cos.py): semilocal COACH exchange and correlation kernels
- [`FunctionalCOACH/tests/test_coach_regression.py`](FunctionalCOACH/tests/test_coach_regression.py): regression tests for the standalone functional path

## Directory Guide

- [`1_data_generation/`](1_data_generation): PySCF data extraction and matrix generation
- [`2_optimization/`](2_optimization): preprocessing, optimization, grid-constraint selection, and analysis
- [`FunctionalCOACH/`](FunctionalCOACH): standalone PySCF implementation and regression fixtures for the fitted COACH functional
- [`Optimization.md`](Optimization.md): manuscript-side description of the fitting procedure
- [`1_data_generation/qchem_codes_insert.C`](1_data_generation/qchem_codes_insert.C): reference-only Q-Chem/C++ source retained for comparison; it is not part of the maintained workflow

## Citation

If you use this workflow in academic work, please cite:

```bibtex
@article{liang2026reaching,
  title={Reaching for the performance limit of hybrid density functional theory for molecular chemistry},
  author={Liang, Jiashu and Head-Gordon, Martin},
  journal={arXiv preprint arXiv:2603.23466},
  year={2026}
}
```

## Contact

For questions, issues, or suggestions regarding this workflow:

- **Primary contact**: Jiashu Liang (`jsliang@berkeley.edu`)
- **Research group**: Martin Head-Gordon Group, UC Berkeley
- **Issues**: Please use the GitHub issue tracker for bug reports and feature requests

# COACH

COACH is a script-first workflow for training the COACH exchange-correlation functional from precomputed reaction data.

## Validation Status

The original COACH workflow used Q-Chem. The PySCF-based path in this repository replaces that original route, but it has not been validated as extensively as the historical implementation. Treat generated data and fitted models as research artifacts that should be checked carefully before downstream use.

## Supported Workflow

The maintained baseline pipeline is:

1. Generate one PySCF text output per species with [`1_data_generation/pyscf_integrated_dv.py`](1_data_generation/pyscf_integrated_dv.py).
2. Parse those outputs into `raw_data.dict` and `reaction_data.dict` with [`1_data_generation/extract_data.py`](1_data_generation/extract_data.py).
3. Build legacy-compatible training arrays from reaction data plus CSV metadata with [`2_optimization/build_training_data.py`](2_optimization/build_training_data.py).
4. Run mixed-integer optimization in two passes with [`2_optimization/run_mio.py`](2_optimization/run_mio.py).
5. Select manuscript-style grid-sensitivity constraints with [`2_optimization/select_grid_constraints.py`](2_optimization/select_grid_constraints.py).
6. Analyze fitted coefficient sets with [`2_optimization/analyze_results.py`](2_optimization/analyze_results.py).

## Requirements

- Python 3.10+
- `numpy`
- `scipy` and `pyscf` for the data-generation stage
- `gurobipy` plus a working Gurobi license for optimization
- `pandas` only if you plan to pass Excel `DatasetEval` files to `extract_data.py`

## Metadata Setup

Project-specific metadata files are expected as CSVs. Start from the header templates in [`2_optimization/templates/`](2_optimization/templates/):

- [`2_optimization/templates/dataset_eval.csv`](2_optimization/templates/dataset_eval.csv)
- [`2_optimization/templates/training_weights.csv`](2_optimization/templates/training_weights.csv)
- [`2_optimization/templates/dataset_info.csv`](2_optimization/templates/dataset_info.csv)

Populate copies of those templates with your actual reactions, dataset assignments, and weights before running the workflow.

## End-to-End Example

### 1. Generate PySCF molecule outputs

Adapt [`1_data_generation/pyscf_integrated_dv.py`](1_data_generation/pyscf_integrated_dv.py) to your molecules, basis sets, grids, and batch logic. It should produce one text file per species containing the energy terms and integratedDV blocks expected by `extract_data.py`.

### 2. Extract reaction data

```bash
python3 1_data_generation/extract_data.py \
  --input_data_dir path/to/pyscf_outputs \
  --DatasetEval path/to/dataset_eval.csv \
  --output_dir processed/raw
```

This writes:

- `processed/raw/raw_data.dict`
- `processed/raw/reaction_data.dict`
- `processed/raw/failed_files.log` if parsing failures occur
- `processed/raw/failed_reactions.log` if reactions cannot be assembled

### 3. Build training artifacts

```bash
python3 2_optimization/build_training_data.py \
  --reaction-data default=processed/raw/reaction_data.dict \
  --analysis-source default \
  --dataset-eval path/to/dataset_eval.csv \
  --training-weights path/to/training_weights.csv \
  --output-dir processed_data
```

This writes the legacy-compatible core artifacts:

- `A_matrix.npy`
- `b_vec.npy`
- `weight_vec.npy`
- `name_list.npy`
- `A_matrix_dataset.dict`
- `b_vec_dataset.dict`
- `diff_99590.npy`
- `name_list_diff_99590.npy`

### 4. Pass 1 optimization

```bash
python3 2_optimization/run_mio.py \
  -n 24 32 40 48 \
  --input_dir processed_data \
  --out_dir runs/pass1 \
  --A_rows 64 153 166
```

### 5. Select pass 2 grid constraints

```bash
python3 2_optimization/select_grid_constraints.py \
  --diff-matrix processed_data/diff_99590.npy \
  --diff-names processed_data/name_list_diff_99590.npy \
  --run-dir runs/pass1 \
  --output-dir processed_data
```

This implements the manuscript rule:

- top 100 rows per candidate by `|(A - A') c|`
- plus top 200 rows globally by row `L1` norm
- then deduplicate and save the selected subset

### 6. Pass 2 optimization

```bash
python3 2_optimization/run_mio.py \
  -n 24 32 40 48 \
  --with_diff \
  --diff_name diff_constraint_99590.npy \
  --input_dir processed_data \
  --out_dir runs/pass2 \
  --warm_start_dir runs/pass1 \
  --A_rows 64 153 166
```

### 7. Analyze results

```bash
python3 2_optimization/analyze_results.py \
  --run-dir runs/pass2 \
  --processed-dir processed_data \
  --dataset-info path/to/dataset_info.csv
```

Analysis writes summary CSVs under `runs/pass2/analysis/` and selected coefficient files under `runs/pass2/best_models/`.

## Directory Guide

- [`1_data_generation/`](1_data_generation): PySCF data extraction and matrix generation
- [`2_optimization/`](2_optimization): preprocessing, optimization, grid-constraint selection, and analysis
- [`Optimization.md`](Optimization.md): manuscript-side description of the fitting procedure

## Contact

For questions, issues, or suggestions regarding this workflow:

- **Primary contact**: Jiashu Liang (`jsliang@berkeley.edu`)
- **Research group**: Martin Head-Gordon Group, UC Berkeley
- **Issues**: Please use the GitHub issue tracker for bug reports and feature requests

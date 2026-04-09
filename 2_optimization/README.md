# 2_optimization

This directory contains the maintained script-first COACH workflow for preprocessing, optimization, constraint selection, and analysis.

## Workflow

1. Prepare training and test data from `reaction_data.pkl` plus populated CSV metadata:

   ```bash
   python3 2_optimization/build_data.py \
     --reaction-data processed_data/reaction_data.pkl \
     --dataset-eval path/to/dataset_eval.csv \
     --training-weights path/to/training_weights.csv \
     --output-dir processed_data
   ```

   This prepares training and test data, including the per-dataset dictionaries used in downstream analysis and testing.

2. Run pass 1 without grid constraints:

   ```bash
   python3 2_optimization/run_mio.py \
     -n 24 32 40 48 \
     --input_dir processed_data \
     --out_dir runs/pass1 \
     --A_rows 64 153 166
   ```

3. Select manuscript-style grid constraints from pass 1:

   ```bash
   python3 2_optimization/select_grid_constraints.py \
     --diff-matrix processed_data/diff_99590.npy \
     --diff-names processed_data/name_list_diff_99590.txt \
     --run-dir runs/pass1 \
     --output-dir processed_data
   ```

   This implements the manuscript rule:

   - top 100 rows per candidate by `|(A - A') c|`
   - plus top 200 rows globally by row `L1` norm
   - then deduplicate and save the selected subset

   Add `--show-largesterror` if you want the script to print the 20 largest diff names for each beta candidate.

4. Run pass 2 with the selected constraints:

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

5. Analyze any run directory:

   ```bash
   python3 2_optimization/analyze_results.py \
     --run-dir runs/pass2 \
     --processed-dir processed_data \
     --standard-errors path/to/Standard_errors.csv \
     --dataset-info path/to/dataset_info.csv
   ```

   If `processed_data` contains both `diff_99590.npy` and `diff_75302.npy`, the analysis tables include metrics for both grids.

## Metadata Setup

Start from the reference files in [`templates/`](templates) and populate them with your project data:

- [`templates/dataset_eval.csv`](templates/dataset_eval.csv)
- [`templates/training_weights.csv`](templates/training_weights.csv)
- [`templates/Standard_errors.csv`](templates/Standard_errors.csv)
- [`templates/dataset_info.csv`](templates/dataset_info.csv)

`analyze_results.py` requires a standard-error CSV with at least `Dataset` and `RMSE` columns. It reads `dataset_info.csv` directly using the shipped `Name` and `Datatype` columns.

- `dataset_eval.csv`
  - Required columns: `Reaction,Dataset,Reference,Stoichiometry`
- `training_weights.csv`
  - Required columns: `Dataset,datapoints,weights`
  - `datapoints` supports `All` or a comma-separated reaction list.
  - `weights` supports a numeric value, `Shrink`, or `Shrink2`.
- `Standard_errors.csv`
  - Required columns for `analyze_results.py`: `Dataset,RMSE`
  - Extra columns such as `Metric` and `MAE` are ignored by the CLI.
- `dataset_info.csv`
  - `analyze_results.py` reads `Name` and `Datatype` from the shipped reference table.
  - Extra columns such as `Datatype_Short` and `Description` are ignored by the CLI.

## Outputs

Preprocessing writes:

- `A_matrix.npy`
- `b_vec.npy`
- `weight_vec.npy`
- `name_list_training.txt`
- `A_matrix_dataset.pkl`
- `b_vec_dataset.pkl`
- `diff_99590.npy`
- `name_list_diff_99590.txt`
- `diff_75302.npy`
- `name_list_diff_75302.txt`

Constraint selection writes:

- `diff_constraint_99590.npy`

Optimization writes one `betas_nonzero<N>.npy` file per sparsity plus `run_config.json`.

Analysis writes:

- `detailed_result.csv`
- `representative_scan.csv`

## Notes

- The cleaned baseline workflow is the manuscript baseline only: 289 parameters, `A_rows = [64, 153, 166]`, and one grid-sensitive second pass against the 99,590 grid.
- Preprocessing now writes diff matrices for both analysis grids (`99590` and `75302`) when those keys are present in `reaction_data.pkl`.
- Gurobi is required only for optimization. Preprocessing, constraint selection, and most analysis utilities do not require it.

## `run_mio.py` Options

The examples above show the minimal baseline commands. [`run_mio.py`](run_mio.py) also supports these optional flags:

- `--config_file`: load defaults from a JSON or YAML config file.
  Use [`templates/run_mio.yaml`](templates/run_mio.yaml) as the starting template. Values in the config file act as defaults, and explicit CLI arguments override them.
- `--nonzeros`, `--nthreads`, `--repeats`, `--time_limit`, `--random_seed`, `--verbose`: control sweep size and solver runtime behavior. `--repeats` is the number of solves to run for each warm start and defaults to `1`.
- `--input_dir`, `--out_dir`: choose where optimization reads inputs and writes outputs.
- `--A_rows`: override the three fitting rows used to define the 289-parameter baseline.
  In the constraint code, `row % 18` selects the basis-group index and `row // 18` selects the exchange mode.
- `--bvec_name`, `--Amatrix_name`, `--weight_name`, `--diff_name`: override input artifact filenames inside `--input_dir`.
- `--with_diff`, `--grid_thresh`: enable diff-matrix constraints and set the threshold in kcal/mol.
- `--warm_start_dir`: for each requested sparsity, load `betas_nonzero<N>.npy` from this directory after the built-in `simple` seed. This matches the pass-1 to pass-2 handoff when adding selected grid constraints.

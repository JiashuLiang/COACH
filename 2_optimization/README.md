# 2_optimization

This directory contains the maintained script-first COACH optimization workflow.

## Workflow

1. Prepare training and test data from `reaction_data.pkl` plus populated CSV metadata:

   ```bash
   python3 2_optimization/build_data.py \
     --reaction-data processed/raw/reaction_data.pkl \
     --dataset-eval path/to/dataset_eval.csv \
     --training-weights path/to/training_weights.csv \
     --output-dir processed_data
   ```

2. Run pass 1 without grid constraints:

   ```bash
   python3 2_optimization/run_mio.py \
     -n 24 32 40 \
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

4. Run pass 2 with the selected constraints:

   ```bash
   python3 2_optimization/run_mio.py \
     -n 24 32 40 \
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
     --dataset-info path/to/dataset_info.csv
   ```

## Metadata Contract

Start from the header templates in [`templates/`](templates) and populate them with your project data.

- `dataset_eval.csv`
  - Required columns: `Reaction,Dataset,Reference,Stoichiometry`
- `training_weights.csv`
  - Required columns: `Dataset,datapoints,weights`
  - `datapoints` supports `All` or a comma-separated reaction list.
  - `weights` supports a numeric value, `Shrink`, or `Shrink2`.
- `dataset_info.csv`
  - Required columns: `Dataset,Datatype`

## Outputs

Preprocessing writes:

- `A_matrix.npy`
- `b_vec.npy`
- `weight_vec.npy`
- `name_list.txt`
- `A_matrix_dataset.pkl`
- `b_vec_dataset.pkl`
- `diff_99590.npy`
- `name_list_diff_99590.txt`

Constraint selection writes:

- `diff_constraint_99590.npy`
- `name_list_diff_constraint_99590.txt`
- `diff_constraint_99590.json`

Optimization writes one `betas_nonzero<N>.npy` file per sparsity plus `run_config.json`.

Analysis writes:

- `analysis/summary.csv`
- `analysis/best_by_nonzeros.csv`
- `analysis/dataset_rmse.csv`
- `best_models/best_overall.npy`

## Notes

- The cleaned baseline workflow is the manuscript baseline only: 289 parameters, `A_rows = [64, 153, 166]`, and one grid-sensitive second pass against the 99,590 grid.
- Gurobi is required only for optimization. Preprocessing, constraint selection, and most analysis utilities do not require it.

## `run_mio.py` Options

The examples above show the minimal baseline commands. [`run_mio.py`](run_mio.py) also supports these optional flags:

- `--config_file`: load defaults from a JSON or YAML config file.
  Use [`template/run_mio.yaml`](template/run_mio.yaml) as the starting template. Values in the config file act as defaults, and explicit CLI arguments override them.
- `--nonzeros`, `--nthreads`, `--repeats`, `--time_limit`, `--random_seed`, `--verbose`: control sweep size and solver runtime behavior.
- `--input_dir`, `--out_dir`: choose where optimization reads inputs and writes outputs.
- `--A_rows`: override the three fitting rows used to define the 289-parameter baseline.
- `--bvec_name`, `--Amatrix_name`, `--weight_name`, `--diff_name`: override input artifact filenames inside `--input_dir`.
- `--with_diff`, `--grid_thresh`: enable diff-matrix constraints and set the threshold in kcal/mol.
- `--warm_start_dir`, `--warm_start_file`, `--no_reference_warm_starts`: control warm-start sources.

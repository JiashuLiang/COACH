# 2_optimization

This directory contains the maintained script-first COACH optimization workflow.

## Workflow

1. Build training artifacts from `reaction_data.dict` plus populated CSV metadata:

   ```bash
   python3 2_optimization/build_training_data.py \
     --reaction-data default=processed/raw/reaction_data.dict \
     --analysis-source default \
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
     --diff-names processed_data/name_list_diff_99590.npy \
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

## One-shot Runner

[`run_workflow.py`](run_workflow.py) orchestrates preprocessing, pass 1, row selection, pass 2, and analysis in one command:

```bash
python3 2_optimization/run_workflow.py \
  --reaction-data default=processed/raw/reaction_data.dict \
  --analysis-source default \
  --dataset-eval path/to/dataset_eval.csv \
  --training-weights path/to/training_weights.csv \
  --dataset-info path/to/dataset_info.csv \
  --processed-dir processed_data \
  --run-root runs/full \
  -n 24 32 40
```

## Metadata Contract

Start from the header templates in [`templates/`](templates) and populate them with your project data.

- `dataset_eval.csv`
  - Required columns: `Reaction,Dataset,Reference,Stoichiometry`
- `training_weights.csv`
  - Required columns: `Dataset,Density Source,datapoints,weights`
  - `datapoints` supports `All` or a comma-separated reaction list.
  - `weights` supports a numeric value, `Shrink`, or `Shrink2`.
- `dataset_info.csv`
  - Required columns: `Dataset,Datatype`

## Outputs

Preprocessing writes legacy-compatible artifacts:

- `A_matrix.npy`
- `b_vec.npy`
- `weight_vec.npy`
- `name_list.npy`
- `A_matrix_dataset.dict`
- `b_vec_dataset.dict`
- `diff_99590.npy`
- `name_list_diff_99590.npy`

Constraint selection writes:

- `diff_constraint_99590.npy`
- `name_list_diff_constraint_99590.npy`
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
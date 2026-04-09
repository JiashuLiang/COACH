# Example Workflow

This directory contains a runnable reference for the full workflow using the `AE18` atomic XYZ files as example.

The generated PySCF outputs and downstream preprocessing artifacts were created with:

```bash
python 1_data_generation/pyscf_integrated_dv.py \
  --xyz-dir example/xyzfiles \
  --output-dir example/pyscf_outputs

python 1_data_generation/extract_data.py \
  --input-data-dir example/pyscf_outputs \
  --dataset-eval 2_optimization/templates/dataset_eval.csv \
  --output-dir example/processed_data

python 2_optimization/build_data.py \
  --reaction-data example/processed_data/reaction_data.pkl \
  --dataset-eval 2_optimization/templates/dataset_eval.csv \
  --training-weights 2_optimization/templates/training_weights.csv \
  --output-dir example/processed_data

GRB_LICENSE_FILE=~/gurobi.lic python 2_optimization/run_mio.py \
  -n 6 8 10 \
  --time_limit 600 \
  --repeats 1 \
  --nthreads 4 \
  --input_dir example/processed_data \
  --out_dir example/runs/pass1

python 2_optimization/select_grid_constraints.py \
  --diff-matrix example/processed_data/diff_99590.npy \
  --diff-names example/processed_data/name_list_diff_99590.txt \
  --run-dir example/runs/pass1 \
  --output-dir example/processed_data \
  --top-per-beta 1 \
  --top-l1 2

GRB_LICENSE_FILE=~/gurobi.lic python 2_optimization/run_mio.py \
  -n 6 8 10 \
  --with_diff \
  --diff_name diff_constraint_99590.npy \
  --time_limit 600 \
  --repeats 2 \
  --nthreads 4 \
  --input_dir example/processed_data \
  --out_dir example/runs/pass2 \
  --warm_start_dir example/runs/pass1

python 2_optimization/analyze_results.py \
  --run-dir example/runs/pass1 \
  --processed-dir example/processed_data \
  --standard-errors 2_optimization/templates/Standard_errors.csv \
  --dataset-info 2_optimization/templates/dataset_info.csv

python 2_optimization/analyze_results.py \
  --run-dir example/runs/pass2 \
  --processed-dir example/processed_data \
  --standard-errors 2_optimization/templates/Standard_errors.csv \
  --dataset-info 2_optimization/templates/dataset_info.csv
```

Notes:

- The XYZ comment line supplies `charge` and `multiplicity`.
- The PySCF generator uses its maintained default basis (`def2-qzvppd`) unless `--use-xyz-basis` is requested.
- `example/pyscf_outputs/` stores one text output per species, matching the shape expected by `extract_data.py`.
- `example/processed_data/` now keeps both the raw pickles and the downstream preprocessing artifacts in one place.
- `example/runs/pass1/` stores the first optimization sweep for nonzeros `6`, `8`, and `10`, plus `detailed_result.csv` and `representative_scan.csv`.
- `example/processed_data/diff_constraint_99590.npy` stores the reduced diff-constraint matrix selected with `--top-per-beta 1 --top-l1 2`.
- `example/runs/pass2/` stores the second optimization sweep plus `detailed_result.csv` and `representative_scan.csv`.

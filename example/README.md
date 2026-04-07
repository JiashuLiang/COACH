# Example Workflow

This directory contains a runnable reference for README steps 1-3 using the shipped `AE18` atomic XYZ files.

The generated PySCF outputs and downstream preprocessing artifacts were created with:

```bash
~/.venv/bin/python 1_data_generation/pyscf_integrated_dv.py \
  --xyz-dir example/xyzfiles \
  --output-dir example/pyscf_outputs

~/.venv/bin/python 1_data_generation/extract_data.py \
  --input-data-dir example/pyscf_outputs \
  --dataset-eval 2_optimization/templates/dataset_eval.csv \
  --output-dir example/processed_data

~/.venv/bin/python 2_optimization/build_data.py \
  --reaction-data example/processed_data/reaction_data.pkl \
  --dataset-eval 2_optimization/templates/dataset_eval.csv \
  --training-weights 2_optimization/templates/training_weights.csv \
  --output-dir example/processed_data
```

Notes:

- The XYZ comment line supplies `charge` and `multiplicity`.
- The PySCF generator uses its maintained default basis (`def2-qzvppd`) unless `--use-xyz-basis` is requested.
- `example/pyscf_outputs/` stores one text output per species, matching the shape expected by `extract_data.py`.
- `example/processed_data/` now keeps both the raw pickles and the downstream preprocessing artifacts in one place.

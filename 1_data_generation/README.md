# 1_data_generation

This directory contains the maintained PySCF-side data-generation and extraction path used by the cleaned COACH workflow.

## Maintained Entry Points

- [`extract_data.py`](extract_data.py): parse one text output per species and assemble the training dictionaries used downstream
- [`pyscf_integrated_dv.py`](pyscf_integrated_dv.py): PySCF reference implementation for generating integratedDV matrices and related energy terms

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `pyscf`
- `pandas` 

## Expected Inputs

`extract_data.py` expects:

- one `.txt` file per species under `--input-data-dir`
- a populated CSV `dataset_eval` file describing reactions and stoichiometries
- each species file to contain the energy labels and integratedDV matrix blocks consumed by the parser

The maintained interface is CSV-only and expects `Reaction`, `Reference`, and `Stoichiometry` columns.

## Usage

```bash
python3 1_data_generation/extract_data.py \
  --input-data-dir path/to/pyscf_outputs \
  --dataset-eval path/to/dataset_eval.csv \
  --output-dir processed/raw
```

The script writes:

- `raw_data.dict`
- `reaction_data.dict`
- `failed_files.log` when parsing failures occur
- `failed_reactions.log` when a reaction cannot be assembled from the parsed species

## `reaction_data.dict` Structure

Each reaction entry stores:

- scalar energies such as `Tofit`, `Nofit`, and `DFT Non-Local Correlation`
- short-range and long-range exchange terms
- `Fitting`: the high-resolution fitting matrix used to build the optimization design matrix
- one diff matrix per additional grid, already shifted relative to the fitting grid

The downstream optimization pipeline assumes:

- `reaction["Fitting"]` has shape `(180, 96)`
- grid-difference entries such as `reaction["99000590"]` have the same shape
- `reaction["Tofit"]` is the target fitted by the linear model

## Reference Files

- [`SIE4x4_h2o.out`](SIE4x4_h2o.out): sample text output used for parser validation
- [`qchem_codes_insert.C`](qchem_codes_insert.C): original Q-Chem/C++ reference implementation retained for provenance
- [`pyscf_integratedDV_matrices.txt`](pyscf_integratedDV_matrices.txt): matrix notes and reference data used during the PySCF port

`extract_data.py` is the maintained extraction interface for the supported workflow.

## Matrix Layout

The integratedDV matrices preserve the original channel ordering used in the paper. Conceptually, the 180 rows come from 10 channel families built from repeated 18-function basis groups, while each selected row expands into 96 coefficients during optimization.

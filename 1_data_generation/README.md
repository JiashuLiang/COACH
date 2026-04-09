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

Step 1 expects either:

- a single molecule configured directly inside `pyscf_integrated_dv.py`
- or one `.xyz` file per species when using the maintained CLI (`--xyz` or `--xyz-dir`)

Step 2 (`extract_data.py`) expects:

- one `.txt` file per species under `--input-data-dir`
- a populated CSV `dataset_eval` file describing reactions and stoichiometries
- each species file to contain the energy labels and integratedDV matrix blocks consumed by the parser

The maintained interface is CSV-only and expects `Reaction`, `Reference`, and `Stoichiometry` columns.

The `Stoichiometry` field is a comma-separated sequence of alternating `coefficient,species_name` tokens. For example:

- `1,H2,-1,H,-1,H` for `H2 -> H + H`
- `2,product,-1,reactant_a,-1,reactant_b` for `2 product - reactant_a - reactant_b`

Species names must match the stem of the corresponding `.txt` file in `--input-data-dir`.

## Usage

### 1. Generate PySCF molecule outputs

Adapt [`pyscf_integrated_dv.py`](pyscf_integrated_dv.py) to your molecules, basis sets, grids, and batch logic, or point it at a directory of XYZ files. It should produce one text file per species containing the energy terms and integratedDV blocks expected by `extract_data.py`.

Batch XYZ example:

```bash
python3 1_data_generation/pyscf_integrated_dv.py \
  --xyz-dir path/to/xyzfiles \
  --output-dir path/to/pyscf_outputs
```

Single XYZ example:

```bash
python3 1_data_generation/pyscf_integrated_dv.py \
  --xyz path/to/species.xyz \
  --output-txt path/to/species.txt
```

### 2. Extract reaction data

```bash
python3 1_data_generation/extract_data.py \
  --input-data-dir path/to/pyscf_outputs \
  --dataset-eval path/to/dataset_eval.csv \
  --output-dir processed_data
```

The script writes:

- `raw_data.pkl`
- `reaction_data.pkl`
- `failed_files.log` when parsing failures occur
- `failed_reactions.log` when a reaction cannot be assembled from the parsed species

If you keep step-2 outputs in the same directory used by preprocessing, step 3 in [`../2_optimization/README.md`](../2_optimization/README.md) can consume `reaction_data.pkl` in place.

## `reaction_data.pkl` Structure

Each reaction entry stores:

- scalar energies such as `Tofit`, `Nofit`, and `DFT Non-Local Correlation`
- short-range and long-range exchange terms
- `Fitting`: the high-resolution fitting matrix used to build the optimization design matrix
- one diff matrix per additional grid, already shifted relative to the fitting grid

The downstream optimization pipeline assumes:

- `reaction["Fitting"]` has shape `(180, 96)`
- grid-difference entries such as `reaction["99590"]` have the same shape and store the difference between that grid and `reaction["Fitting"]`
- `reaction["Tofit"]` is the target fitted by the linear model

## Reference Files

- [`qchem_codes_insert.C`](qchem_codes_insert.C): reference-only Q-Chem/C++ implementation; it is not part of the maintained workflow
- [`pyscf_integratedDV_matrices.txt`](pyscf_integratedDV_matrices.txt): matrix notes and reference data used during the PySCF port

`extract_data.py` is the maintained extraction interface for the supported workflow.

## Matrix Layout

The integratedDV matrices preserve the original channel ordering used in the paper. Conceptually, the 180 rows come from 10 channel families built from repeated 18-function basis groups, while each selected row expands into 96 coefficients during optimization.

### Expansion Group Mapping

The 18 basis groups are indexed `0..17` and are built from tensor products of:

- one of three `u` polynomial families: linear, Legendre, or Chebyshev
- one companion variable family using either `w` or `beta_f`
- one of three companion polynomial families: linear, Legendre, or Chebyshev

The groups are ordered as:

- `0..8`: `w` with `u`
- `9..17`: `beta_f` with `u`

Within each block of 9:

- `0,1,2`: `u` linear with companion linear, Legendre, Chebyshev
- `3,4,5`: `u` Legendre with companion linear, Legendre, Chebyshev
- `6,7,8`: `u` Chebyshev with companion linear, Legendre, Chebyshev

This is the indexing used by `expansion_basis_batch()` in [`pyscf_integrated_dv.py`](pyscf_integrated_dv.py).

### IntegratedDV Group Definitions

The 180 integratedDV columns are arranged as 10 conceptual channel families:

- exchange mGGA, spin-summed: 18 columns
- exchange mGGA with nonuniform scaling, spin-summed: 18 columns
- exchange RSH mGGA, spin-summed: 18 columns
- exchange RSH mGGA with nonuniform scaling, spin-summed: 18 columns
- same-spin correlation, PW92-like: 18 columns
- same-spin correlation, PW92-like, with self-correlation correction factor: 18 columns
- opposite-spin correlation, PW92-like: 18 columns
- same-spin correlation, SCAN-like: 18 columns
- same-spin correlation, SCAN-like, with self-correlation correction factor: 18 columns
- opposite-spin correlation, SCAN-like: 18 columns

# Training data generation

## Why this folder exists
The original workflow used in the paper was implemented in Q-Chem/C++ (see `qchem_codes_insert.C`).
To help others train new exchange-correlation functionals without requiring Q-Chem internals, this repo provides a pure-Python/PySCF implementation:

- generator: `pyscf_integrated_dv.py`
- reference output: `SIE4x4_h2o.out`
- generated output: `pyscf_integratedDV_matrices.txt`

The goal is to keep the same equations and channel ordering while making the pipeline easier to run, inspect, and modify.

## How to use this for training
1. Run `pyscf_integrated_dv.py` to generate integratedDV matrices.
2. Compare against `SIE4x4_h2o.out` (via `compare_integrated_dv_tmp.py` if needed).
3. Use each `96 x 180` matrix as feature blocks for coefficient fitting / functional training.

## How to read the 96 x 180 matrix
Each point contributes an expansion basis of shape `96 x 18`.  
Different physical channels reuse this same 18-group basis and are concatenated into the 180 columns.

High-level channel groups (conceptual):
1. Exchange mGGA
2. Exchange mGGA with a multiplier to satisfy the nonuniform scaling
3. Exchange RSH mGGA
4. Exchange RSH mGGA with a multiplier (from SCAN) to satisfy the nonuniform scaling
5. Same-spin correlation from B97
6. Same-spin correlation from B97 with `2beta`
7. Opposite-spin correlation from B97
8. Same-spin correlation from SCAN UEG cases
9. Same-spin correlation from SCAN UEG cases with `2beta`
10. Opposite-spin correlation from SCAN UEG

In practice, these conceptual groups are packed into contiguous column slices in `pyscf_integrated_dv.py` (`accumulate_integrated_dv_block`, `accumulate_exchange_block`, `accumulate_correlation_block`).

## Expansion Group Mapping (0..17)
This mapping defines the 18 basis groups used inside each channel block.

| Group | `w` or `beta_f` source | GGA expansion (u-side) | mGGA expansion (w/beta_f-side) |
|---|---|---|---|
| 0 | `w` | Linear `u` | Linear |
| 1 | `w` | Linear `u` | Legendre |
| 2 | `w` | Linear `u` | Chebyshev |
| 3 | `w` | Legendre | Linear |
| 4 | `w` | Legendre | Legendre |
| 5 | `w` | Legendre | Chebyshev |
| 6 | `w` | Chebyshev | Linear |
| 7 | `w` | Chebyshev | Legendre |
| 8 | `w` | Chebyshev | Chebyshev |
| 9 | `beta_f` | Linear `u` | Linear |
| 10 | `beta_f` | Linear `u` | Legendre |
| 11 | `beta_f` | Linear `u` | Chebyshev |
| 12 | `beta_f` | Legendre | Linear |
| 13 | `beta_f` | Legendre | Legendre |
| 14 | `beta_f` | Legendre | Chebyshev |
| 15 | `beta_f` | Chebyshev | Linear |
| 16 | `beta_f` | Chebyshev | Legendre |
| 17 | `beta_f` | Chebyshev | Chebyshev |

### Notes on interpretation
- Groups `0..8` use `w`-based mGGA variables.
- Groups `9..17` use `beta_f`-based mGGA variables.
- The `u` side is always one of `{Linear, Legendre, Chebyshev}` expansions.
- The full 96 comes from `12 (w or beta_f terms) x 8 (u terms)`.

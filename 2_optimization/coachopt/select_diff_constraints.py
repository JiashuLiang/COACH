"""Helpers for selecting manuscript-style grid-sensitivity constraints."""

from __future__ import annotations

import numpy as np


def _largest_abs_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Return indices for the largest absolute entries in descending order."""
    if values.size == 0 or count <= 0:
        return np.asarray([], dtype=int)
    return np.argsort(np.abs(values))[::-1][: min(count, values.size)]


def print_largest_diff_names(
    diff_matrix: np.ndarray,
    diff_names: list[str],
    candidates,
    count: int = 20,
) -> None:
    """Print the largest absolute diff names for each beta candidate."""
    for candidate in candidates:
        deviations = np.asarray(diff_matrix @ candidate.coefficients, dtype=float).reshape(-1)
        print(f"{candidate.label} largest diffs:")
        for index in _largest_abs_indices(deviations, count):
            print(diff_names[index])


def select_diff_constraint_rows(
    diff_matrix: np.ndarray,
    betas: list[np.ndarray],
    top_per_beta: int = 100,
    top_l1: int = 200,
) -> np.ndarray:
    """Choose informative diff rows from candidate beta vectors plus an L1 fallback."""
    if diff_matrix.ndim != 2:
        raise ValueError("diff_matrix must be a 2D array")
    if not betas:
        raise ValueError("At least one beta vector is required to select constraint rows")

    chosen_indices: set[int] = set()
    for beta in betas:
        coeff = np.asarray(beta, dtype=float).reshape(-1)
        if coeff.shape[0] != diff_matrix.shape[1]:
            raise ValueError(
                f"Beta length {coeff.shape[0]} does not match diff matrix width {diff_matrix.shape[1]}"
            )
        chosen_indices.update(_largest_abs_indices(diff_matrix @ coeff, top_per_beta).tolist())

    remaining_indices = np.asarray(sorted(set(range(diff_matrix.shape[0])) - chosen_indices), dtype=int)
    if remaining_indices.size and top_l1 > 0:
        l1_norms = np.sum(np.abs(diff_matrix[remaining_indices]), axis=1)
        chosen_indices.update(remaining_indices[_largest_abs_indices(l1_norms, top_l1)].tolist())

    ordered_indices = np.asarray(sorted(chosen_indices), dtype=int)
    return diff_matrix[ordered_indices]

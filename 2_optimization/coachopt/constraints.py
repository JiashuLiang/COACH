"""Helpers for selecting manuscript-style grid-sensitivity constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstraintSelection:
    indices: np.ndarray
    rows: np.ndarray
    names: list[str]
    metadata: dict[str, int]


def _top_abs_indices(values: np.ndarray, count: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=int)
    count = min(count, values.size)
    if count == values.size:
        return np.arange(values.size, dtype=int)
    return np.argpartition(np.abs(values), -count)[-count:]


def select_diff_constraint_rows(
    diff_matrix: np.ndarray,
    diff_names: list[str],
    betas: list[np.ndarray],
    top_per_beta: int = 100,
    top_l1: int = 200,
) -> ConstraintSelection:
    if diff_matrix.ndim != 2:
        raise ValueError("diff_matrix must be a 2D array")
    if diff_matrix.shape[0] != len(diff_names):
        raise ValueError("diff_matrix row count does not match diff_names")
    if not betas:
        raise ValueError("At least one beta vector is required to select constraint rows")

    chosen_indices: set[int] = set()
    for beta in betas:
        coeff = np.asarray(beta, dtype=float).reshape(-1)
        if coeff.shape[0] != diff_matrix.shape[1]:
            raise ValueError(
                f"Beta length {coeff.shape[0]} does not match diff matrix width {diff_matrix.shape[1]}"
            )
        deviations = diff_matrix @ coeff
        chosen_indices.update(_top_abs_indices(deviations, top_per_beta).tolist())

    l1_norms = np.sum(np.abs(diff_matrix), axis=1)
    chosen_indices.update(_top_abs_indices(l1_norms, top_l1).tolist())

    ordered_indices = np.asarray(sorted(chosen_indices), dtype=int)
    selected_names = [diff_names[index] for index in ordered_indices]
    return ConstraintSelection(
        indices=ordered_indices,
        rows=diff_matrix[ordered_indices],
        names=selected_names,
        metadata={
            "candidate_count": len(betas),
            "selected_count": int(ordered_indices.size),
            "top_per_beta": top_per_beta,
            "top_l1": top_l1,
        },
    )

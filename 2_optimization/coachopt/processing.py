"""Data-artifact construction from reaction dictionaries plus CSV metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_A_ROWS,
    DEFAULT_GRID_KEY,
)
from .utils import ensure_directory, save_name_array, save_pickle, write_json


def feature_count(a_rows: tuple[int, ...]) -> int:
    """Return the 289-feature width for the requested fitting rows."""
    return len(a_rows) * 96 + 1


def _feature_vector(reaction: dict, a_rows: tuple[int, ...]) -> tuple[np.ndarray, float]:
    """Flatten selected fitting rows and append the short-range exchange feature."""
    fitting = np.asarray(reaction["Fitting"], dtype=float)
    if fitting.ndim != 2:
        raise ValueError("reaction['Fitting'] must be a 2D array")
    if max(a_rows) >= fitting.shape[0]:
        raise ValueError(
            f"A_rows {a_rows} exceed available fitting rows ({fitting.shape[0]} total rows)"
        )
    features = fitting[list(a_rows)].reshape(-1)
    b_value = float(reaction["Tofit"])
    sr_exchange = float(reaction["Alpha Short Range Exchange"]) + float(reaction["Beta Short Range Exchange"])
    features = np.concatenate([features, np.asarray([sr_exchange], dtype=float)])
    return features, b_value


def _weights_for_dataset(weight_spec: str, count: int) -> list[float]:
    """Expand a supported weight specification into one scalar per reaction."""
    if weight_spec == "Shrink":
        return (1.0 / np.sqrt(np.arange(1, count + 1))).tolist()
    if weight_spec == "Shrink2":
        return (1.0 / np.arange(1, count + 1)).tolist()
    value = float(weight_spec)
    return [value] * count


def artifact_grid_suffix(grid_key: str) -> str:
    """Convert raw grid ids like 99000590 into artifact names like 99590."""
    if not grid_key.isdigit() or len(grid_key) < 3:
        return grid_key
    prefix = grid_key[:2]
    remainder = str(int(grid_key[2:]))
    return f"{prefix}{remainder}"


def build_and_save_data(
    reaction_data: dict[str, dict],
    dataset_eval: pd.DataFrame,
    training_weight: pd.DataFrame,
    output_dir: str | Path,
    a_rows: tuple[int, ...] = DEFAULT_A_ROWS,
    diff_grid: str = DEFAULT_GRID_KEY,
) -> dict[str, str]:
    """Build all preprocessing artifacts expected by the cleaned optimization pipeline."""
    by_reaction = {
        row.Reaction: {"Dataset": row.Dataset}
        for row in dataset_eval.itertuples(index=False)
    }
    reactions_by_dataset: dict[str, list[str]] = {}
    a_matrix_dataset_rows: dict[str, list[np.ndarray]] = {}
    b_vec_dataset_rows: dict[str, list[float]] = {}
    diff_rows: list[np.ndarray] = []
    diff_names: list[str] = []
    a_matrix_rows: list[np.ndarray] = []
    b_vec_rows: list[float] = []
    weight_rows: list[float] = []
    name_list: list[str] = []

    for row in dataset_eval.itertuples(index=False):
        reaction_id = row.Reaction
        dataset = row.Dataset
        if reaction_id not in reaction_data:
            raise KeyError(f"Reaction {reaction_id!r} not found in reaction_data")

        reactions_by_dataset.setdefault(dataset, []).append(reaction_id)
        reaction = reaction_data[reaction_id]
        features, target = _feature_vector(reaction, a_rows)
        a_matrix_dataset_rows.setdefault(dataset, []).append(features)
        b_vec_dataset_rows.setdefault(dataset, []).append(target)

        if diff_grid in reaction:
            diff_features = np.asarray(reaction[diff_grid], dtype=float)
            selected = diff_features[list(a_rows)].reshape(-1)
            # The final feature is the SR-exchange scalar, which is not part of
            # grid-difference constraints, so the diff row gets a trailing zero.
            diff_rows.append(np.concatenate([selected, np.asarray([0.0], dtype=float)]))
            diff_names.append(reaction_id)

    for row in training_weight.itertuples(index=False):
        dataset = row.Dataset
        if dataset not in reactions_by_dataset:
            raise KeyError(f"Dataset {dataset!r} was not found in dataset_eval metadata")

        if row.datapoints == "All":
            reaction_ids = reactions_by_dataset[dataset]
        else:
            reaction_ids = [item.strip() for item in row.datapoints.split(",") if item.strip()]
            if not reaction_ids:
                raise ValueError(f"Dataset {dataset!r} has an empty datapoints field")
            for reaction_id in reaction_ids:
                if reaction_id not in by_reaction:
                    raise KeyError(f"Reaction {reaction_id!r} was not found in dataset_eval metadata")
                if by_reaction[reaction_id]["Dataset"] != dataset:
                    raise ValueError(
                        f"Reaction {reaction_id!r} belongs to dataset "
                        f"{by_reaction[reaction_id]['Dataset']!r}, not {dataset!r}"
                    )

        weights = _weights_for_dataset(row.weights, len(reaction_ids))
        for reaction_id in reaction_ids:
            features, target = _feature_vector(reaction_data[reaction_id], a_rows)
            a_matrix_rows.append(features)
            b_vec_rows.append(target)
            name_list.append(reaction_id)
        weight_rows.extend(weights)

    output_dir = ensure_directory(output_dir)
    a_matrix = np.asarray(a_matrix_rows, dtype=float)
    b_vec = np.asarray(b_vec_rows, dtype=float)
    weight_vec = np.asarray(weight_rows, dtype=float)
    a_matrix_dataset = {
        dataset: np.asarray(values, dtype=float) for dataset, values in a_matrix_dataset_rows.items()
    }
    b_vec_dataset = {
        dataset: np.asarray(values, dtype=float) for dataset, values in b_vec_dataset_rows.items()
    }
    diff_matrix = np.asarray(diff_rows, dtype=float)
    diff_suffix = artifact_grid_suffix(diff_grid)

    np.save(output_dir / "A_matrix.npy", a_matrix)
    np.save(output_dir / "b_vec.npy", b_vec)
    np.save(output_dir / "weight_vec.npy", weight_vec)
    save_name_array(output_dir / "name_list.npy", name_list)
    save_pickle(output_dir / "A_matrix_dataset.pkl", a_matrix_dataset)
    save_pickle(output_dir / "b_vec_dataset.pkl", b_vec_dataset)
    np.save(output_dir / f"diff_{diff_suffix}.npy", diff_matrix)
    save_name_array(output_dir / f"name_list_diff_{diff_suffix}.npy", diff_names)

    write_json(
        output_dir / "build_manifest.json",
        {
            "a_rows": list(a_rows),
            "feature_count": feature_count(a_rows),
            "training_rows": len(a_matrix_rows),
            "dataset_count": len(a_matrix_dataset),
            "diff_grid": diff_grid,
            "diff_suffix": diff_suffix,
            "diff_rows": len(diff_rows),
        },
    )

    return {
        "A_matrix": str(output_dir / "A_matrix.npy"),
        "b_vec": str(output_dir / "b_vec.npy"),
        "weight_vec": str(output_dir / "weight_vec.npy"),
        "name_list": str(output_dir / "name_list.npy"),
        "A_matrix_dataset": str(output_dir / "A_matrix_dataset.pkl"),
        "b_vec_dataset": str(output_dir / "b_vec_dataset.pkl"),
        "diff_matrix": str(output_dir / f"diff_{diff_suffix}.npy"),
        "diff_names": str(output_dir / f"name_list_diff_{diff_suffix}.npy"),
        "manifest": str(output_dir / "build_manifest.json"),
    }

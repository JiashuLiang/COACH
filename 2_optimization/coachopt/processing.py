"""Data-artifact construction from reaction dictionaries plus CSV metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_A_ROWS,
    DEFAULT_GRID_KEY,
)
from .utils import ensure_directory, save_names, save_pickle, write_json


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


def _selected_training_entries(
    dataset: str,
    datapoints: str,
    reaction_entries: list[tuple[str, np.ndarray, float]],
) -> list[tuple[str, np.ndarray, float]]:
    """Resolve the ordered training reactions for one dataset-weight row."""
    if datapoints == "All":
        return reaction_entries

    requested_ids = [item.strip() for item in datapoints.split(",") if item.strip()]
    if not requested_ids:
        raise ValueError(f"Dataset {dataset!r} has an empty datapoints field")

    by_reaction = {reaction_id: (reaction_id, features, target) for reaction_id, features, target in reaction_entries}
    selected_entries: list[tuple[str, np.ndarray, float]] = []
    for reaction_id in requested_ids:
        if reaction_id not in by_reaction:
            raise KeyError(f"Reaction {reaction_id!r} was not found in dataset_eval metadata for dataset {dataset!r}")
        selected_entries.append(by_reaction[reaction_id])
    return selected_entries


def build_and_save_data(
    reaction_data: dict[str, dict],
    dataset_eval: pd.DataFrame,
    training_weight: pd.DataFrame,
    output_dir: str | Path,
    a_rows: tuple[int, ...] = DEFAULT_A_ROWS,
    diff_grid: str = DEFAULT_GRID_KEY,
) -> dict[str, str]:
    """Build all preprocessing artifacts expected by the cleaned optimization pipeline."""
    reactions_by_dataset: dict[str, list[str]] = {}
    a_matrix_dataset_rows: dict[str, list[np.ndarray]] = {}
    b_vec_dataset_rows: dict[str, list[float]] = {}
    diff_rows: list[np.ndarray] = []
    diff_names: list[str] = []
    a_matrix_rows: list[np.ndarray] = []
    b_vec_rows: list[float] = []
    weight_rows: list[float] = []
    name_list: list[str] = []
    training_rows_by_dataset: dict[str, list] = {}
    for row in training_weight.itertuples(index=False):
        training_rows_by_dataset.setdefault(row.Dataset, []).append(row)

    for dataset, df_dataset in dataset_eval.groupby("Dataset", sort=False):
        reaction_entries: list[tuple[str, np.ndarray, float]] = []

        for row in df_dataset.itertuples(index=False):
            reaction_id = row.Reaction
            if reaction_id not in reaction_data:
                raise KeyError(f"Reaction {reaction_id!r} not found in reaction_data")

            reaction = reaction_data[reaction_id]
            features, target = _feature_vector(reaction, a_rows)
            reaction_entries.append((reaction_id, features, target))
            reactions_by_dataset.setdefault(dataset, []).append(reaction_id)
            a_matrix_dataset_rows.setdefault(dataset, []).append(features)
            b_vec_dataset_rows.setdefault(dataset, []).append(target)

            if diff_grid in reaction:
                diff_features = np.asarray(reaction[diff_grid], dtype=float)
                selected = diff_features[list(a_rows)].reshape(-1)
                # The final feature is the SR-exchange scalar, which is not part of
                # grid-difference constraints, so the diff row gets a trailing zero.
                diff_rows.append(np.concatenate([selected, np.asarray([0.0], dtype=float)]))
                diff_names.append(reaction_id)

        if dataset not in training_rows_by_dataset:
            continue

        for training_row in training_rows_by_dataset[dataset]:
            selected_entries = _selected_training_entries(dataset, training_row.datapoints, reaction_entries)
            weights = _weights_for_dataset(training_row.weights, len(selected_entries))
            for reaction_id, features, target in selected_entries:
                a_matrix_rows.append(features)
                b_vec_rows.append(target)
                name_list.append(reaction_id)
            weight_rows.extend(weights)

    for dataset in training_rows_by_dataset:
        if dataset not in reactions_by_dataset:
            raise KeyError(f"Dataset {dataset!r} was not found in dataset_eval metadata")

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
    np.save(output_dir / "A_matrix.npy", a_matrix)
    np.save(output_dir / "b_vec.npy", b_vec)
    np.save(output_dir / "weight_vec.npy", weight_vec)
    save_names(output_dir / "name_list_training.txt", name_list)
    save_pickle(output_dir / "A_matrix_dataset.pkl", a_matrix_dataset)
    save_pickle(output_dir / "b_vec_dataset.pkl", b_vec_dataset)
    np.save(output_dir / f"diff_{diff_grid}.npy", diff_matrix)
    save_names(output_dir / f"name_list_diff_{diff_grid}.txt", diff_names)

    write_json(
        output_dir / "build_manifest.json",
        {
            "a_rows": list(a_rows),
            "feature_count": feature_count(a_rows),
            "training_rows": len(a_matrix_rows),
            "dataset_count": len(a_matrix_dataset),
            "diff_grid": diff_grid,
            "diff_rows": len(diff_rows),
        },
    )

    return {
        "A_matrix": str(output_dir / "A_matrix.npy"),
        "b_vec": str(output_dir / "b_vec.npy"),
        "weight_vec": str(output_dir / "weight_vec.npy"),
        "name_list_training": str(output_dir / "name_list_training.txt"),
        "A_matrix_dataset": str(output_dir / "A_matrix_dataset.pkl"),
        "b_vec_dataset": str(output_dir / "b_vec_dataset.pkl"),
        "diff_matrix": str(output_dir / f"diff_{diff_grid}.npy"),
        "diff_names": str(output_dir / f"name_list_diff_{diff_grid}.txt"),
        "manifest": str(output_dir / "build_manifest.json"),
    }

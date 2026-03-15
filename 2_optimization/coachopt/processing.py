"""Training-data construction from reaction dictionaries plus CSV metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_A_ROWS, DEFAULT_GRID_KEY
from .utils import ensure_directory, save_name_array, save_pickle, write_json


@dataclass(frozen=True)
class FeatureSpec:
    """Controls how the 289-parameter baseline feature vector is assembled."""

    a_rows: tuple[int, int, int] = DEFAULT_A_ROWS

    @property
    def feature_count(self) -> int:
        return len(self.a_rows) * 96 + 1


def _reaction_lookup(dataset_eval_rows: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    by_reaction = {row["Reaction"]: row for row in dataset_eval_rows}
    by_dataset: dict[str, list[str]] = {}
    for row in dataset_eval_rows:
        by_dataset.setdefault(row["Dataset"], []).append(row["Reaction"])
    return by_reaction, by_dataset


def _feature_vector(reaction: dict, spec: FeatureSpec) -> tuple[np.ndarray, float]:
    fitting = np.asarray(reaction["Fitting"], dtype=float)
    if fitting.ndim != 2:
        raise ValueError("reaction['Fitting'] must be a 2D array")
    if max(spec.a_rows) >= fitting.shape[0]:
        raise ValueError(
            f"A_rows {spec.a_rows} exceed available fitting rows ({fitting.shape[0]} total rows)"
        )
    features = fitting[list(spec.a_rows)].reshape(-1)
    b_value = float(reaction["Tofit"])
    sr_exchange = float(reaction["Alpha Short Range Exchange"]) + float(reaction["Beta Short Range Exchange"])
    features = np.concatenate([features, np.asarray([sr_exchange], dtype=float)])
    return features, b_value


def build_dataset_matrices(
    reaction_data: dict[str, dict],
    dataset_eval_rows: list[dict],
    spec: FeatureSpec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, list[np.ndarray]] = {}
    targets: dict[str, list[float]] = {}
    for row in dataset_eval_rows:
        reaction_id = row["Reaction"]
        if reaction_id not in reaction_data:
            raise KeyError(f"Reaction {reaction_id!r} not found in reaction_data")
        features, target = _feature_vector(reaction_data[reaction_id], spec)
        matrices.setdefault(row["Dataset"], []).append(features)
        targets.setdefault(row["Dataset"], []).append(target)
    return (
        {dataset: np.asarray(values, dtype=float) for dataset, values in matrices.items()},
        {dataset: np.asarray(values, dtype=float) for dataset, values in targets.items()},
    )


def _weights_for_dataset(weight_spec: str, count: int) -> list[float]:
    if weight_spec == "Shrink":
        return (1.0 / np.sqrt(np.arange(1, count + 1))).tolist()
    if weight_spec == "Shrink2":
        return (1.0 / np.arange(1, count + 1)).tolist()
    value = float(weight_spec)
    return [value] * count


def build_training_arrays(
    reaction_sources: dict[str, dict[str, dict]],
    dataset_eval_rows: list[dict],
    training_weight_rows: list[dict],
    spec: FeatureSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    by_reaction, reactions_by_dataset = _reaction_lookup(dataset_eval_rows)
    b_vec: list[float] = []
    a_matrix: list[np.ndarray] = []
    weight_vec: list[float] = []
    name_list: list[str] = []

    for row in training_weight_rows:
        dataset = row["Dataset"]
        source = row["Density Source"]
        if source not in reaction_sources:
            available = ", ".join(sorted(reaction_sources))
            raise KeyError(f"Unknown Density Source {source!r}; available sources: {available}")
        if dataset not in reactions_by_dataset:
            raise KeyError(f"Dataset {dataset!r} was not found in dataset_eval metadata")

        if row["datapoints"] == "All":
            reaction_ids = reactions_by_dataset[dataset]
        else:
            reaction_ids = [item.strip() for item in row["datapoints"].split(",") if item.strip()]
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

        weights = _weights_for_dataset(row["weights"], len(reaction_ids))
        for reaction_id in reaction_ids:
            reaction_data = reaction_sources[source]
            if reaction_id not in reaction_data:
                raise KeyError(f"Reaction {reaction_id!r} not found in source {source!r}")
            features, target = _feature_vector(reaction_data[reaction_id], spec)
            a_matrix.append(features)
            b_vec.append(target)
            name_list.append(reaction_id)
        weight_vec.extend(weights)

    return (
        np.asarray(b_vec, dtype=float),
        np.asarray(a_matrix, dtype=float),
        np.asarray(weight_vec, dtype=float),
        name_list,
    )


def build_diff_matrix(
    reaction_data: dict[str, dict],
    dataset_eval_rows: list[dict],
    spec: FeatureSpec,
    diff_grid: str = DEFAULT_GRID_KEY,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    names: list[str] = []
    for row in dataset_eval_rows:
        reaction = reaction_data[row["Reaction"]]
        if diff_grid not in reaction:
            continue
        diff_features = np.asarray(reaction[diff_grid], dtype=float)
        selected = diff_features[list(spec.a_rows)].reshape(-1)
        selected = np.concatenate([selected, np.asarray([0.0], dtype=float)])
        rows.append(selected)
        names.append(row["Reaction"])
    return np.asarray(rows, dtype=float), names


def artifact_grid_suffix(grid_key: str) -> str:
    """Convert raw grid ids like 99000590 into legacy artifact names like 99590."""
    if not grid_key.isdigit() or len(grid_key) < 3:
        return grid_key
    prefix = grid_key[:2]
    remainder = str(int(grid_key[2:]))
    return f"{prefix}{remainder}"


def build_and_save_training_data(
    reaction_sources: dict[str, dict[str, dict]],
    analysis_source: str,
    dataset_eval_rows: list[dict],
    training_weight_rows: list[dict],
    output_dir: str | Path,
    spec: FeatureSpec,
    diff_grid: str = DEFAULT_GRID_KEY,
) -> dict[str, str]:
    if analysis_source not in reaction_sources:
        raise KeyError(f"Analysis source {analysis_source!r} was not supplied")

    output_dir = ensure_directory(output_dir)
    analysis_reactions = reaction_sources[analysis_source]
    a_matrix_dict, b_vec_dict = build_dataset_matrices(analysis_reactions, dataset_eval_rows, spec)
    b_vec, a_matrix, weight_vec, name_list = build_training_arrays(
        reaction_sources, dataset_eval_rows, training_weight_rows, spec
    )
    diff_matrix, diff_names = build_diff_matrix(analysis_reactions, dataset_eval_rows, spec, diff_grid=diff_grid)
    diff_suffix = artifact_grid_suffix(diff_grid)

    np.save(output_dir / "A_matrix.npy", a_matrix)
    np.save(output_dir / "b_vec.npy", b_vec)
    np.save(output_dir / "weight_vec.npy", weight_vec)
    save_name_array(output_dir / "name_list.npy", name_list)
    save_pickle(output_dir / "A_matrix_dataset.dict", a_matrix_dict)
    save_pickle(output_dir / "b_vec_dataset.dict", b_vec_dict)
    np.save(output_dir / f"diff_{diff_suffix}.npy", diff_matrix)
    save_name_array(output_dir / f"name_list_diff_{diff_suffix}.npy", diff_names)

    manifest = {
        "analysis_source": analysis_source,
        "a_rows": list(spec.a_rows),
        "feature_count": spec.feature_count,
        "training_rows": int(a_matrix.shape[0]),
        "dataset_count": len(a_matrix_dict),
        "diff_grid": diff_grid,
        "diff_suffix": diff_suffix,
        "diff_rows": int(diff_matrix.shape[0]),
    }
    write_json(output_dir / "build_manifest.json", manifest)

    return {
        "A_matrix": str(output_dir / "A_matrix.npy"),
        "b_vec": str(output_dir / "b_vec.npy"),
        "weight_vec": str(output_dir / "weight_vec.npy"),
        "name_list": str(output_dir / "name_list.npy"),
        "A_matrix_dataset": str(output_dir / "A_matrix_dataset.dict"),
        "b_vec_dataset": str(output_dir / "b_vec_dataset.dict"),
        "diff_matrix": str(output_dir / f"diff_{diff_suffix}.npy"),
        "diff_names": str(output_dir / f"name_list_diff_{diff_suffix}.npy"),
        "manifest": str(output_dir / "build_manifest.json"),
    }

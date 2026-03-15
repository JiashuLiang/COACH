"""CSV metadata loaders and validation helpers."""

from __future__ import annotations

import csv
from pathlib import Path


def _load_csv_rows(path: str | Path, required_columns: list[str]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            normalized = {key: (value or "").strip() for key, value in row.items()}
            for column in required_columns:
                if not normalized[column]:
                    raise ValueError(f"{path}: row {row_number} is missing {column}")
            rows.append(normalized)
    if not rows:
        raise ValueError(f"{path}: file contains no data rows")
    return rows


def load_dataset_eval_csv(path: str | Path) -> list[dict]:
    rows = _load_csv_rows(path, ["Reaction", "Dataset", "Reference", "Stoichiometry"])
    normalized = []
    seen_reactions: set[str] = set()
    for row in rows:
        reaction = row["Reaction"]
        if reaction in seen_reactions:
            raise ValueError(f"{path}: duplicate Reaction value {reaction!r}")
        seen_reactions.add(reaction)
        normalized.append(
            {
                "Reaction": reaction,
                "Dataset": row["Dataset"],
                "Reference": float(row["Reference"]),
                "Stoichiometry": row["Stoichiometry"],
            }
        )
    return normalized


def load_training_weights_csv(path: str | Path) -> list[dict]:
    rows = _load_csv_rows(path, ["Dataset", "Density Source", "datapoints", "weights"])
    normalized = []
    for row in rows:
        normalized.append(
            {
                "Dataset": row["Dataset"],
                "Density Source": row["Density Source"],
                "datapoints": row["datapoints"],
                "weights": row["weights"],
            }
        )
    return normalized


def load_dataset_info_csv(path: str | Path) -> dict[str, dict[str, str]]:
    rows = _load_csv_rows(path, ["Dataset", "Datatype"])
    dataset_info: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row["Dataset"]
        if dataset in dataset_info:
            raise ValueError(f"{path}: duplicate Dataset value {dataset!r}")
        dataset_info[dataset] = row
    return dataset_info


def load_baseline_error_csv(path: str | Path) -> dict[str, dict[str, float]]:
    rows = _load_csv_rows(path, ["Dataset"])
    baseline: dict[str, dict[str, float]] = {}
    numeric_columns: list[str] | None = None
    for row in rows:
        dataset = row["Dataset"]
        if numeric_columns is None:
            numeric_columns = [key for key in row if key != "Dataset"]
            if not numeric_columns:
                raise ValueError(f"{path}: expected at least one baseline metric column")
        baseline[dataset] = {key: float(row[key]) for key in numeric_columns}
    return baseline

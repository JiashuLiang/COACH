"""Low-level helpers used by the CLI scripts."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: str | Path, value) -> None:
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle)


def write_json(path: str | Path, payload) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_name_array(path: str | Path, values: list[str]) -> None:
    np.save(Path(path), np.asarray(values, dtype=str))


def format_float(value: float) -> float:
    """Round tiny floating-point noise before serializing to CSV/JSON."""
    return float(np.round(value, 12))

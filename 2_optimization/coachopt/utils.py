"""Low-level helpers used by the CLI scripts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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


def read_csv_frame(path: str | Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    required_columns = required_columns or []
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    return frame


def read_csv_rows(path: str | Path, required_columns: list[str]) -> list[dict]:
    frame = read_csv_frame(path, required_columns)
    return frame[required_columns].to_dict(orient="records")


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    frame = pd.DataFrame(list(rows), columns=fieldnames)
    frame.to_csv(Path(path), index=False)


def save_name_array(path: str | Path, values: list[str]) -> None:
    np.save(Path(path), np.asarray(values, dtype=str))


def format_float(value: float) -> float:
    """Round tiny floating-point noise before serializing to CSV/JSON."""
    return float(np.round(value, 12))

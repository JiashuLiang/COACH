"""Reusable analysis helpers for post-processing optimization runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ANALYSIS_DIFF_GRIDS, HARTREE_TO_KCAL_MOL
from .utils import ensure_directory, load_pickle, write_csv_rows


def rms(values: np.ndarray) -> float:
    """Compute an unweighted root-mean-square value."""
    return float(np.sqrt(np.mean(np.square(values))))


@dataclass(frozen=True)
class BetaCandidate:
    """One beta vector loaded from a ``betas_nonzero*.npy`` artifact."""

    label: str
    nonzeros: int
    candidate_index: int
    coefficients: np.ndarray
    source_file: Path


def load_beta_candidates(run_dir: str | Path) -> list[BetaCandidate]:
    """Load all beta candidates saved by the optimization sweep."""
    candidates: list[BetaCandidate] = []
    for path in sorted(Path(run_dir).glob("betas_nonzero*.npy")):
        suffix = path.stem.split("betas_nonzero", 1)[1]
        if not suffix.isdigit():
            continue
        nonzeros = int(suffix)
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for idx, coeff in enumerate(np.asarray(data, dtype=float)):
            candidates.append(
                BetaCandidate(
                    label=f"nz{nonzeros}_cand{idx}",
                    nonzeros=nonzeros,
                    candidate_index=idx,
                    coefficients=coeff,
                    source_file=path,
                )
            )
    return candidates


def _dataset_rmse_map(
    coeff: np.ndarray,
    a_matrix_dict: dict[str, np.ndarray],
    b_vec_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute per-dataset RMSE values for one beta candidate."""
    return {
        dataset: rms(np.asarray(b_vec_dict[dataset]) - np.asarray(a_matrix_dict[dataset]) @ coeff) * HARTREE_TO_KCAL_MOL
        for dataset in sorted(a_matrix_dict)
    }


def _load_diff_matrices(processed_dir: Path) -> dict[str, np.ndarray]:
    """Load all required analysis diff matrices from the processed-data directory."""
    matrices: dict[str, np.ndarray] = {}
    for diff_grid in ANALYSIS_DIFF_GRIDS:
        diff_path = processed_dir / f"diff_{diff_grid}.npy"
        if not diff_path.exists():
            raise ValueError(f"Missing required diff matrix: {diff_path}")
        matrices[diff_grid] = np.load(diff_path)
    return matrices


def _validate_standard_errors(
    standard_errors: dict[str, float],
    dataset_names: list[str],
) -> None:
    """Check that standard errors cover every analyzed dataset and are nonzero."""
    missing = [dataset for dataset in dataset_names if dataset not in standard_errors]
    if missing:
        raise ValueError(
            "standard_errors is missing datasets: " + ", ".join(missing)
        )

    zero_values = [dataset for dataset in dataset_names if standard_errors[dataset] == 0.0]
    if zero_values:
        raise ValueError(
            "standard_errors contains zero RMSE values for datasets: " + ", ".join(zero_values)
        )


def _datatype_groups(
    dataset_info: dict[str, dict[str, str]] | None,
    dataset_names: list[str],
) -> list[tuple[str, list[str]]]:
    """Collect datasets grouped by Datatype in a stable order."""
    if not dataset_info:
        return []

    datasets_by_datatype: dict[str, list[str]] = {}
    for dataset in dataset_names:
        datatype = dataset_info.get(dataset, {}).get("Datatype", "").strip()
        if datatype:
            datasets_by_datatype.setdefault(datatype, []).append(dataset)
    return [(datatype, datasets_by_datatype[datatype]) for datatype in sorted(datasets_by_datatype)]


def _grid_stats_by_label(coeff: np.ndarray, diff_matrices: dict[str, np.ndarray]) -> dict[str, float | str]:
    """Compute the requested grid-difference statistics for one beta candidate."""
    stats: dict[str, float | str] = {}

    diff99590 = np.abs(diff_matrices["99590"] @ coeff) * HARTREE_TO_KCAL_MOL
    stats["99590 max diff"] = float(np.max(diff99590, initial=0.0))
    stats["99590 Percentage diff > 0.015 kcal/mol"] = "{:.2f}%".format(
        np.sum(diff99590 > 0.015) / len(diff99590) * 100 if len(diff99590) else 0.0
    )
    stats["99590 count diff > 0.015 kcal/mol"] = int(np.sum(diff99590 > 0.015))
    stats["99590 count diff > 0.1 kcal/mol"] = int(np.sum(diff99590 > 0.1))
    stats["99590 count diff > 0.5 kcal/mol"] = int(np.sum(diff99590 > 0.5))

    diff75302 = np.abs(diff_matrices["75302"] @ coeff) * HARTREE_TO_KCAL_MOL
    stats["75302 max diff"] = float(np.max(diff75302, initial=0.0))
    stats["75302 median diff"] = float(np.median(diff75302)) if len(diff75302) else 0.0
    stats["75302 Percentage diff > 0.015 kcal/mol"] = "{:.2f}%".format(
        np.sum(diff75302 > 0.015) / len(diff75302) * 100 if len(diff75302) else 0.0
    )
    stats["75302 count diff > 0.015 kcal/mol"] = int(np.sum(diff75302 > 0.015))
    stats["75302 count diff > 0.1 kcal/mol"] = int(np.sum(diff75302 > 0.1))
    stats["75302 count diff > 0.5 kcal/mol"] = int(np.sum(diff75302 > 0.5))

    return stats


def _metric_row_order(
    datatype_groups: list[tuple[str, list[str]]],
    dataset_names: list[str],
) -> list[str]:
    """Return the fixed row order shared by both analysis CSVs."""
    datatype_rows = [f"{datatype} mean relative error" for datatype, _ in datatype_groups]
    return [
        "mean relative error",
        "median relative error",
        *datatype_rows,
        "99590 max diff",
        "99590 Percentage diff > 0.015 kcal/mol",
        "99590 count diff > 0.015 kcal/mol",
        "99590 count diff > 0.1 kcal/mol",
        "99590 count diff > 0.5 kcal/mol",
        "75302 max diff",
        "75302 median diff",
        "75302 Percentage diff > 0.015 kcal/mol",
        "75302 count diff > 0.015 kcal/mol",
        "75302 count diff > 0.1 kcal/mol",
        "75302 count diff > 0.5 kcal/mol",
        *dataset_names,
    ]


def _rows_for_columns(
    metric_order: list[str],
    candidate_labels: list[str],
    metrics_by_candidate: dict[str, dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Transpose candidate metric dictionaries into CSV row dictionaries."""
    rows: list[dict[str, float | str]] = []
    for metric_name in metric_order:
        row: dict[str, float | str] = {"Metric": metric_name}
        for candidate_label in candidate_labels:
            row[candidate_label] = metrics_by_candidate[candidate_label].get(metric_name, "")
        rows.append(row)
    return rows


def analyze_run_directory(
    run_dir: str | Path,
    processed_dir: str | Path,
    standard_errors: dict[str, float],
    dataset_info: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Analyze a run directory and write detailed and representative scan CSVs."""
    run_dir = Path(run_dir)
    processed_dir = Path(processed_dir)
    analysis_dir = ensure_directory(run_dir / "analysis")

    a_matrix_dict = load_pickle(processed_dir / "A_matrix_dataset.pkl")
    b_vec_dict = load_pickle(processed_dir / "b_vec_dataset.pkl")
    diff_matrices = _load_diff_matrices(processed_dir)

    candidates = load_beta_candidates(run_dir)
    if not candidates:
        raise ValueError(f"No beta files were found in {run_dir}")

    dataset_names = sorted(a_matrix_dict)
    _validate_standard_errors(standard_errors, dataset_names)
    datatype_groups = _datatype_groups(dataset_info, dataset_names)
    metric_order = _metric_row_order(datatype_groups, dataset_names)

    metrics_by_candidate: dict[str, dict[str, float | str]] = {}
    best_by_nonzero: dict[int, tuple[float, float, str]] = {}

    for candidate in candidates:
        dataset_rmse = _dataset_rmse_map(candidate.coefficients, a_matrix_dict, b_vec_dict)
        relative_errors = {
            dataset: dataset_rmse[dataset] / standard_errors[dataset]
            for dataset in dataset_names
        }

        metrics: dict[str, float | str] = {
            "mean relative error": float(np.mean(list(relative_errors.values()))),
            "median relative error": float(np.median(list(relative_errors.values()))),
        }
        for datatype, datatype_datasets in datatype_groups:
            metrics[f"{datatype} mean relative error"] = float(
                np.mean([relative_errors[dataset] for dataset in datatype_datasets])
            )
        metrics.update(_grid_stats_by_label(candidate.coefficients, diff_matrices))
        metrics.update(relative_errors)
        metrics_by_candidate[candidate.label] = metrics

        score = (
            float(metrics["mean relative error"]),
            float(metrics["median relative error"]),
            candidate.label,
        )
        current_best = best_by_nonzero.get(candidate.nonzeros)
        if current_best is None or score < current_best:
            best_by_nonzero[candidate.nonzeros] = score

    detailed_labels = [candidate.label for candidate in candidates]
    representative_labels = [
        best_by_nonzero[nonzeros][2] for nonzeros in sorted(best_by_nonzero)
    ]

    fieldnames = ["Metric"] + detailed_labels
    write_csv_rows(
        analysis_dir / "detailed_result.csv",
        fieldnames,
        _rows_for_columns(metric_order, detailed_labels, metrics_by_candidate),
    )

    representative_fieldnames = ["Metric"] + representative_labels
    write_csv_rows(
        analysis_dir / "representative_scan.csv",
        representative_fieldnames,
        _rows_for_columns(metric_order, representative_labels, metrics_by_candidate),
    )

    return {
        "analysis_dir": str(analysis_dir),
        "detailed_result_csv": str(analysis_dir / "detailed_result.csv"),
        "representative_scan_csv": str(analysis_dir / "representative_scan.csv"),
    }

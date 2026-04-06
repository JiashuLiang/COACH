"""Reusable analysis helpers for post-processing optimization runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DEFAULT_DIFF_MATRIX_NAME, DEFAULT_GRID_THRESHOLD, HARTREE_TO_KCAL_MOL
from .utils import ensure_directory, load_pickle, write_csv_rows, write_json


def rms(values: np.ndarray) -> float:
    """Compute an unweighted root-mean-square value."""
    return float(np.sqrt(np.mean(np.square(values))))


def wrms(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted root-mean-square value with per-row scaling."""
    return float(np.sqrt(np.mean(np.square(values) * weights)))


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


def _grid_stats(diff_matrix: np.ndarray | None, coeff: np.ndarray, threshold: float) -> dict[str, float]:
    """Summarize grid-difference magnitudes for one coefficient vector."""
    if diff_matrix is None:
        return {}
    values = np.abs(diff_matrix @ coeff) * HARTREE_TO_KCAL_MOL
    return {
        "grid_max_diff_kcal_mol": float(values.max(initial=0.0)),
        "grid_median_diff_kcal_mol": float(np.median(values)) if values.size else 0.0,
        "grid_count_diff_gt_threshold": int(np.sum(values > threshold)),
        "grid_count_diff_gt_0_1": int(np.sum(values > 0.1)),
        "grid_count_diff_gt_0_5": int(np.sum(values > 0.5)),
    }


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


def analyze_run_directory(
    run_dir: str | Path,
    processed_dir: str | Path,
    dataset_info: dict[str, dict[str, str]] | None = None,
    baseline_errors: dict[str, dict[str, float]] | None = None,
    diff_name: str = DEFAULT_DIFF_MATRIX_NAME,
    grid_threshold: float = DEFAULT_GRID_THRESHOLD,
) -> dict[str, str]:
    """Analyze a run directory and materialize summary CSVs plus best-model artifacts."""
    run_dir = Path(run_dir)
    processed_dir = Path(processed_dir)
    analysis_dir = ensure_directory(run_dir / "analysis")
    best_dir = ensure_directory(run_dir / "best_models")

    a_matrix = np.load(processed_dir / "A_matrix.npy")
    b_vec = np.load(processed_dir / "b_vec.npy")
    weight_vec = np.load(processed_dir / "weight_vec.npy")
    a_matrix_dict = load_pickle(processed_dir / "A_matrix_dataset.pkl")
    b_vec_dict = load_pickle(processed_dir / "b_vec_dataset.pkl")
    diff_path = processed_dir / diff_name
    diff_matrix = np.load(diff_path) if diff_path.exists() else None

    candidates = load_beta_candidates(run_dir)
    if not candidates:
        raise ValueError(f"No beta files were found in {run_dir}")

    summary_rows: list[dict] = []
    dataset_rmse_rows: dict[str, dict[str, float | str]] = {}
    best_by_nonzero: dict[int, tuple[BetaCandidate, dict]] = {}

    for candidate in candidates:
        residual = b_vec - a_matrix @ candidate.coefficients
        row = {
            "label": candidate.label,
            "nonzeros": candidate.nonzeros,
            "candidate_index": candidate.candidate_index,
            "source_file": str(candidate.source_file.name),
            "rmse_train_kcal_mol": rms(residual) * HARTREE_TO_KCAL_MOL,
            "wrmse_train_kcal_mol": wrms(residual, weight_vec) * HARTREE_TO_KCAL_MOL,
        }
        row.update(_grid_stats(diff_matrix, candidate.coefficients, grid_threshold))
        dataset_rmse = _dataset_rmse_map(candidate.coefficients, a_matrix_dict, b_vec_dict)

        if baseline_errors:
            for baseline_name in next(iter(baseline_errors.values())).keys():
                ratios = [
                    dataset_rmse[dataset] / baseline_errors[dataset][baseline_name]
                    for dataset in dataset_rmse
                    if dataset in baseline_errors and baseline_errors[dataset][baseline_name] != 0.0
                ]
                if ratios:
                    row[f"mean_relative_{baseline_name}"] = float(np.mean(ratios))
                    row[f"median_relative_{baseline_name}"] = float(np.median(ratios))

        summary_rows.append(row)

        for dataset, value in dataset_rmse.items():
            dataset_row = dataset_rmse_rows.setdefault(
                dataset,
                {
                    "Dataset": dataset,
                    "Datatype": (dataset_info or {}).get(dataset, {}).get("Datatype", ""),
                },
            )
            dataset_row[candidate.label] = value

        current_best = best_by_nonzero.get(candidate.nonzeros)
        if current_best is None or row["wrmse_train_kcal_mol"] < current_best[1]["wrmse_train_kcal_mol"]:
            best_by_nonzero[candidate.nonzeros] = (candidate, row)

    summary_rows.sort(key=lambda item: (item["nonzeros"], item["candidate_index"]))
    best_candidates = [item[0] for item in sorted(best_by_nonzero.values(), key=lambda pair: pair[0].nonzeros)]
    overall_best, overall_best_summary = min(
        best_by_nonzero.values(),
        key=lambda item: (item[1]["wrmse_train_kcal_mol"], item[1]["rmse_train_kcal_mol"]),
    )

    summary_fieldnames = list(summary_rows[0].keys())
    write_csv_rows(analysis_dir / "summary.csv", summary_fieldnames, summary_rows)

    best_summary_rows = [item[1] for item in sorted(best_by_nonzero.values(), key=lambda pair: pair[0].nonzeros)]
    write_csv_rows(analysis_dir / "best_by_nonzeros.csv", summary_fieldnames, best_summary_rows)

    dataset_labels = [candidate.label for candidate in best_candidates]
    if overall_best.label not in dataset_labels:
        dataset_labels.append(overall_best.label)
    dataset_fieldnames = ["Dataset", "Datatype"] + dataset_labels
    dataset_rows = []
    for dataset in sorted(dataset_rmse_rows):
        source_row = dataset_rmse_rows[dataset]
        row = {field: source_row.get(field, "") for field in dataset_fieldnames}
        dataset_rows.append(row)
    write_csv_rows(analysis_dir / "dataset_rmse.csv", dataset_fieldnames, dataset_rows)

    np.save(best_dir / "best_overall.npy", overall_best.coefficients)
    shutil.copy2(overall_best.source_file, best_dir / overall_best.source_file.name)
    for candidate in best_candidates:
        np.save(best_dir / f"best_nonzero{candidate.nonzeros}.npy", candidate.coefficients)

    write_json(
        analysis_dir / "analysis_manifest.json",
        {
            "run_dir": str(run_dir),
            "processed_dir": str(processed_dir),
            "candidate_count": len(candidates),
            "overall_best_label": overall_best.label,
            "overall_best_nonzeros": overall_best.nonzeros,
            "overall_best_wrmse_train_kcal_mol": overall_best_summary["wrmse_train_kcal_mol"],
        },
    )

    return {
        "analysis_dir": str(analysis_dir),
        "summary_csv": str(analysis_dir / "summary.csv"),
        "best_by_nonzeros_csv": str(analysis_dir / "best_by_nonzeros.csv"),
        "dataset_rmse_csv": str(analysis_dir / "dataset_rmse.csv"),
        "best_models_dir": str(best_dir),
    }

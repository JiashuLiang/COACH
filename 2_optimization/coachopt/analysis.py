"""Post-processing for optimization runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ensure_directory, load_pickle, write_csv_rows

HARTREE_TO_KCAL_MOL = 627.50947406


def analyze_run_directory(
    run_dir: str | Path,
    processed_dir: str | Path,
    standard_errors: dict[str, float],
    dataset_info: dict[str, dict[str, str]] | None = None,
) -> None:
    """Analyze one run directory and write the requested CSV summaries."""
    run_dir = Path(run_dir)
    processed_dir = Path(processed_dir)
    ensure_directory(run_dir)

    a_matrix_dict = load_pickle(processed_dir / "A_matrix_dataset.pkl")
    b_vec_dict = load_pickle(processed_dir / "b_vec_dataset.pkl")

    diff99590_path = processed_dir / "diff_99590.npy"
    diff75302_path = processed_dir / "diff_75302.npy"
    feature_count = next(iter(a_matrix_dict.values())).shape[1]

    try:
        diff99590_matrix = np.load(diff99590_path) if diff99590_path.exists() else None
    except Exception:
        diff99590_matrix = None
    try:
        diff75302_matrix = np.load(diff75302_path) if diff75302_path.exists() else None
    except Exception:
        diff75302_matrix = None
    if diff99590_matrix is not None and (
        diff99590_matrix.ndim != 2 or diff99590_matrix.shape[1] != feature_count
    ):
        diff99590_matrix = None
    if diff75302_matrix is not None and (
        diff75302_matrix.ndim != 2 or diff75302_matrix.shape[1] != feature_count
    ):
        diff75302_matrix = None

    if dataset_info:
        # Follow the dataset order from dataset_info and keep only datasets that
        # are available in every input needed for relative-error analysis.
        dataset_names = [
            dataset
            for dataset in dataset_info
            if dataset in a_matrix_dict and dataset in b_vec_dict and dataset in standard_errors and standard_errors[dataset] != 0.0
        ]
    else:
        dataset_names = [
            dataset
            for dataset in sorted(a_matrix_dict)
            if dataset in b_vec_dict and dataset in standard_errors and standard_errors[dataset] != 0.0
        ]

    if not dataset_names:
        raise ValueError("No matching datasets were found across processed data and standard_errors")

    datatype_groups: dict[str, list[str]] = {}
    if dataset_info:
        for dataset in dataset_names:
            datatype = dataset_info.get(dataset, {}).get("Datatype", "").strip()
            if datatype:
                datatype_groups.setdefault(datatype, []).append(dataset)

    metric_order = [
        "mean relative error",
        "median relative error",
        *[f"{datatype} mean relative error" for datatype in sorted(datatype_groups)],
    ]
    if diff99590_matrix is not None:
        metric_order.extend([
            "99590 max diff",
            "99590 Percentage diff > 0.015 kcal/mol",
            "99590 count diff > 0.015 kcal/mol",
            "99590 count diff > 0.1 kcal/mol",
            "99590 count diff > 0.5 kcal/mol",
        ])
    if diff75302_matrix is not None:
        metric_order.extend([
            "75302 max diff",
            "75302 median diff",
            "75302 Percentage diff > 0.015 kcal/mol",
            "75302 count diff > 0.015 kcal/mol",
            "75302 count diff > 0.1 kcal/mol",
            "75302 count diff > 0.5 kcal/mol",
        ])
    metric_order.extend(dataset_names)

    detailed_labels: list[str] = []
    metrics_by_label: dict[str, dict[str, float | str]] = {}
    best_by_nonzero: dict[int, tuple[float, float, str]] = {}

    for beta_path in sorted(run_dir.glob("betas_nonzero*.npy")):
        suffix = beta_path.stem.split("betas_nonzero", 1)[1]
        if not suffix.isdigit():
            continue

        nonzeros = int(suffix)
        beta_array = np.load(beta_path)
        if beta_array.ndim == 1:
            beta_array = beta_array.reshape(1, -1)

        for candidate_index, coefficients in enumerate(np.asarray(beta_array, dtype=float)):
            label = f"{beta_path.stem}_cand{candidate_index}"
            detailed_labels.append(label)

            relative_errors: dict[str, float] = {}
            for dataset in dataset_names:
                residual = np.asarray(b_vec_dict[dataset]) - np.asarray(a_matrix_dict[dataset]) @ coefficients
                dataset_rmse = float(np.sqrt(np.mean(np.square(residual))) * HARTREE_TO_KCAL_MOL)
                relative_errors[dataset] = dataset_rmse / standard_errors[dataset]

            metrics: dict[str, float | str] = {
                "mean relative error": float(np.mean(list(relative_errors.values()))),
                "median relative error": float(np.median(list(relative_errors.values()))),
            }
            for datatype in sorted(datatype_groups):
                metrics[f"{datatype} mean relative error"] = float(
                    np.mean([relative_errors[dataset] for dataset in datatype_groups[datatype]])
                )

            if diff99590_matrix is not None:
                diff99590 = np.abs(diff99590_matrix @ coefficients) * HARTREE_TO_KCAL_MOL
                metrics["99590 max diff"] = float(np.max(diff99590, initial=0.0))
                metrics["99590 Percentage diff > 0.015 kcal/mol"] = "{:.2f}%".format(
                    np.sum(diff99590 > 0.015) / len(diff99590) * 100 if len(diff99590) else 0.0
                )
                metrics["99590 count diff > 0.015 kcal/mol"] = int(np.sum(diff99590 > 0.015))
                metrics["99590 count diff > 0.1 kcal/mol"] = int(np.sum(diff99590 > 0.1))
                metrics["99590 count diff > 0.5 kcal/mol"] = int(np.sum(diff99590 > 0.5))

            if diff75302_matrix is not None:
                diff75302 = np.abs(diff75302_matrix @ coefficients) * HARTREE_TO_KCAL_MOL
                metrics["75302 max diff"] = float(np.max(diff75302, initial=0.0))
                metrics["75302 median diff"] = float(np.median(diff75302)) if len(diff75302) else 0.0
                metrics["75302 Percentage diff > 0.015 kcal/mol"] = "{:.2f}%".format(
                    np.sum(diff75302 > 0.015) / len(diff75302) * 100 if len(diff75302) else 0.0
                )
                metrics["75302 count diff > 0.015 kcal/mol"] = int(np.sum(diff75302 > 0.015))
                metrics["75302 count diff > 0.1 kcal/mol"] = int(np.sum(diff75302 > 0.1))
                metrics["75302 count diff > 0.5 kcal/mol"] = int(np.sum(diff75302 > 0.5))

            # The final block of rows is the per-dataset relative error table.
            metrics.update(relative_errors)
            metrics_by_label[label] = metrics

            # Representative scan keeps the beta with the lowest overall mean relative error
            # for each sparsity level, with median relative error as the tie-breaker.
            score = (
                float(metrics["mean relative error"]),
                float(metrics["median relative error"]),
                label,
            )
            current_best = best_by_nonzero.get(nonzeros)
            if current_best is None or score < current_best:
                best_by_nonzero[nonzeros] = score

    if not detailed_labels:
        raise ValueError(f"No beta files were found in {run_dir}")

    # The CSVs are written in transposed form: one metric per row and one beta per column.
    detailed_rows: list[dict[str, float | str]] = []
    for metric_name in metric_order:
        row: dict[str, float | str] = {"Metric": metric_name}
        for label in detailed_labels:
            row[label] = metrics_by_label[label].get(metric_name, "")
        detailed_rows.append(row)
    detailed_result_csv = run_dir / "detailed_result.csv"
    write_csv_rows(detailed_result_csv, ["Metric"] + detailed_labels, detailed_rows)

    representative_labels = [best_by_nonzero[nonzeros][2] for nonzeros in sorted(best_by_nonzero)]
    representative_rows: list[dict[str, float | str]] = []
    for metric_name in metric_order:
        row = {"Metric": metric_name}
        for label in representative_labels:
            row[label] = metrics_by_label[label].get(metric_name, "")
        representative_rows.append(row)
    representative_scan_csv = run_dir / "representative_scan.csv"
    write_csv_rows(
        representative_scan_csv,
        ["Metric"] + representative_labels,
        representative_rows,
    )

    print(f"Analysis written to {run_dir}")
    print(detailed_result_csv)
    print(representative_scan_csv)

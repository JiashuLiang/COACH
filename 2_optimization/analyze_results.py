#!/usr/bin/env python3
"""Analyze an optimization directory and write reproducible CSV summaries."""

from __future__ import annotations

import argparse

from coachopt.analysis import analyze_run_directory
from coachopt.utils import read_csv_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Optimization directory containing betas_nonzero*.npy")
    parser.add_argument("--processed-dir", required=True, help="Directory with A_matrix.npy and related artifacts")
    parser.add_argument("--dataset-info", help="Optional dataset_info.csv for Datatype annotations")
    parser.add_argument("--baseline-errors", help="Optional baseline error CSV for relative metrics")
    parser.add_argument("--diff-name", default="diff_99590.npy", help="Diff matrix filename inside processed-dir")
    parser.add_argument("--grid-threshold", type=float, default=0.015, help="Constraint threshold in kcal/mol")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_info = (
        read_csv_frame(args.dataset_info, ["Dataset", "Datatype"]).set_index("Dataset").to_dict(orient="index")
        if args.dataset_info
        else None
    )
    if args.baseline_errors:
        baseline_frame = read_csv_frame(args.baseline_errors, ["Dataset"])
        value_columns = [column for column in baseline_frame.columns if column != "Dataset"]
        if not value_columns:
            raise ValueError(f"{args.baseline_errors}: expected at least one baseline metric column")
        baseline_errors = baseline_frame.set_index("Dataset")[value_columns].to_dict(orient="index")
    else:
        baseline_errors = None
    outputs = analyze_run_directory(
        run_dir=args.run_dir,
        processed_dir=args.processed_dir,
        dataset_info=dataset_info,
        baseline_errors=baseline_errors,
        diff_name=args.diff_name,
        grid_threshold=args.grid_threshold,
    )

    print(f"Analysis written to {outputs['analysis_dir']}")
    print(outputs["summary_csv"])
    print(outputs["dataset_rmse_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

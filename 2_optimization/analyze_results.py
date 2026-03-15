#!/usr/bin/env python3
"""Analyze an optimization directory and write reproducible CSV summaries."""

from __future__ import annotations

import argparse

from coachopt.analysis import analyze_run_directory
from coachopt.metadata import load_baseline_error_csv, load_dataset_info_csv


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

    dataset_info = load_dataset_info_csv(args.dataset_info) if args.dataset_info else None
    baseline_errors = load_baseline_error_csv(args.baseline_errors) if args.baseline_errors else None
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

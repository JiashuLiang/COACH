#!/usr/bin/env python3
"""Analyze an optimization directory and write reproducible CSV summaries."""

from __future__ import annotations

import argparse

from coachopt.analysis import analyze_run_directory
from coachopt.utils import read_csv_frame


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for post-run analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Optimization directory containing betas_nonzero*.npy")
    parser.add_argument("--processed-dir", required=True, help="Directory with A_matrix.npy and related artifacts")
    parser.add_argument("--dataset-info", help="Optional dataset_info.csv for Datatype annotations")
    parser.add_argument("--standard-errors", required=True, help="Standard error CSV with Dataset and RMSE columns")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load optional metadata, analyze a run directory, and print artifact locations."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_info = (
        read_csv_frame(args.dataset_info, ["Dataset", "Datatype"]).set_index("Dataset").to_dict(orient="index")
        if args.dataset_info
        else None
    )
    standard_frame = read_csv_frame(args.standard_errors, ["Dataset", "RMSE"])
    standard_errors = standard_frame.set_index("Dataset")["RMSE"].to_dict()
    outputs = analyze_run_directory(
        run_dir=args.run_dir,
        processed_dir=args.processed_dir,
        standard_errors=standard_errors,
        dataset_info=dataset_info,
    )

    print(f"Analysis written to {outputs['analysis_dir']}")
    print(outputs["detailed_result_csv"])
    print(outputs["representative_scan_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

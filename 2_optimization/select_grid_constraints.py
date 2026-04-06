#!/usr/bin/env python3
"""Select manuscript-style grid constraints from pass-1 beta candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coachopt.analysis import load_beta_candidates
from coachopt.constants import (
    DEFAULT_DIFF_MATRIX_NAME,
    DEFAULT_DIFF_NAMES_NAME,
    DEFAULT_SELECTED_DIFF_PREFIX,
    DEFAULT_TOP_DIFF_PER_BETA,
    DEFAULT_TOP_L1_ROWS,
)
from coachopt.select_diff_constraints import print_largest_diff_names, select_diff_constraint_rows
from coachopt.utils import load_names


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for pass-2 grid-constraint selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff-matrix", required=True, help=f"Path to {DEFAULT_DIFF_MATRIX_NAME}")
    parser.add_argument("--diff-names", required=True, help=f"Path to {DEFAULT_DIFF_NAMES_NAME}")
    parser.add_argument("--run-dir", required=True, help="Pass-1 optimization directory with betas_nonzero*.npy files")
    parser.add_argument("--output-dir", required=True, help="Directory to write selected constraint artifacts")
    parser.add_argument(
        "--top-per-beta",
        type=int,
        default=DEFAULT_TOP_DIFF_PER_BETA,
        help="Rows selected per beta by |diff @ beta|",
    )
    parser.add_argument(
        "--top-l1",
        type=int,
        default=DEFAULT_TOP_L1_ROWS,
        help="Rows selected globally by row L1 norm",
    )
    parser.add_argument(
        "--show-largesterror",
        action="store_true",
        help="Print the 20 largest |diff @ beta| names for each beta candidate before saving the reduced matrix.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_SELECTED_DIFF_PREFIX,
        help="Output basename prefix. The script writes PREFIX.npy.",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """Select, save, and summarize the reduced diff-constraint matrix."""
    parser = build_parser()
    args = parser.parse_args(argv)

    diff_matrix = np.load(args.diff_matrix)
    diff_names = load_names(args.diff_names)
    candidates = load_beta_candidates(args.run_dir)
    betas = [candidate.coefficients for candidate in candidates]
    if args.show_largesterror:
        print_largest_diff_names(diff_matrix, diff_names, candidates)
    selected_rows = select_diff_constraint_rows(
        diff_matrix=diff_matrix,
        betas=betas,
        top_per_beta=args.top_per_beta,
        top_l1=args.top_l1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diff_path = output_dir / f"{args.output_prefix}.npy"
    np.save(diff_path, selected_rows)

    print(f"Selected {selected_rows.shape[0]} constraint rows")
    print(diff_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

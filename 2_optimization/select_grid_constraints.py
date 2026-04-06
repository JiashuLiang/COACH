#!/usr/bin/env python3
"""Select manuscript-style grid constraints from pass-1 beta candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coachopt.analysis import load_beta_candidates
from coachopt.constraints import select_diff_constraint_rows
from coachopt.utils import save_name_array, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for pass-2 grid-constraint selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff-matrix", required=True, help="Path to diff_99590.npy")
    parser.add_argument("--diff-names", required=True, help="Path to name_list_diff_99590.npy")
    parser.add_argument("--run-dir", required=True, help="Pass-1 optimization directory with betas_nonzero*.npy files")
    parser.add_argument("--output-dir", required=True, help="Directory to write selected constraint artifacts")
    parser.add_argument("--top-per-beta", type=int, default=100, help="Rows selected per beta by |diff @ beta|")
    parser.add_argument("--top-l1", type=int, default=200, help="Rows selected globally by row L1 norm")
    parser.add_argument(
        "--output-prefix",
        default="diff_constraint_99590",
        help="Output basename prefix. The script writes PREFIX.npy and name_list_PREFIX.npy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Select, save, and summarize the reduced diff-constraint matrix."""
    parser = build_parser()
    args = parser.parse_args(argv)

    diff_matrix = np.load(args.diff_matrix)
    diff_names = np.load(args.diff_names).astype(str).tolist()
    betas = [candidate.coefficients for candidate in load_beta_candidates(args.run_dir)]
    selection = select_diff_constraint_rows(
        diff_matrix=diff_matrix,
        diff_names=diff_names,
        betas=betas,
        top_per_beta=args.top_per_beta,
        top_l1=args.top_l1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diff_path = output_dir / f"{args.output_prefix}.npy"
    names_path = output_dir / f"name_list_{args.output_prefix}.npy"
    np.save(diff_path, selection.rows)
    save_name_array(names_path, selection.names)
    write_json(
        output_dir / f"{args.output_prefix}.json",
        {
            **selection.metadata,
            "indices": selection.indices.tolist(),
            "output_diff_matrix": str(diff_path),
            "output_names": str(names_path),
        },
    )

    print(f"Selected {selection.metadata['selected_count']} constraint rows")
    print(diff_path)
    print(names_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

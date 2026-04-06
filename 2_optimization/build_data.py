#!/usr/bin/env python3
"""Build legacy-compatible optimization artifacts from CSV metadata and reaction_data.dict."""

from __future__ import annotations

import argparse
from pathlib import Path

from coachopt.constants import REQUIRED_DATASET_EVAL_COLUMNS, REQUIRED_TRAINING_WEIGHT_COLUMNS, DEFAULT_A_ROWS, DEFAULT_GRID_KEY
from coachopt.processing import build_and_save_data
from coachopt.utils import load_pickle, read_csv_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reaction-data",
        required=True,
        help="Path to reaction_data.dict",
    )
    parser.add_argument("--dataset-eval", required=True, help="Path to dataset_eval.csv")
    parser.add_argument("--training-weights", required=True, help="Path to training_weights.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated NumPy and pickle artifacts")
    parser.add_argument("--a-rows", nargs=3, type=int, default=DEFAULT_A_ROWS, help="Three fitting-row indices for exchange, same-spin correlation, and opposite-spin correlation features")
    parser.add_argument("--diff-grid", default=DEFAULT_GRID_KEY, help="Grid key used for diff matrices")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reaction_data = load_pickle(args.reaction_data)
    dataset_eval = read_csv_frame(
        args.dataset_eval,
        REQUIRED_DATASET_EVAL_COLUMNS,
    )
    training_weight = read_csv_frame(
        args.training_weights,
        REQUIRED_TRAINING_WEIGHT_COLUMNS,
    )
    outputs = build_and_save_data(
        reaction_data=reaction_data,
        dataset_eval=dataset_eval,
        training_weight=training_weight,
        output_dir=Path(args.output_dir),
        a_rows=tuple(args.a_rows),
        diff_grid=args.diff_grid,
    )

    print(f"Wrote data artifacts to {args.output_dir}")
    print(f"A matrix: {outputs['A_matrix']}")
    print(f"b vector: {outputs['b_vec']}")
    print(f"weights: {outputs['weight_vec']}")
    print(f"diff matrix: {outputs['diff_matrix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build optimization artifacts from CSV metadata and reaction_data.pkl."""

from __future__ import annotations

import argparse
from pathlib import Path

from coachopt.constants import ANALYSIS_DIFF_GRIDS, REQUIRED_DATASET_EVAL_COLUMNS, REQUIRED_TRAINING_WEIGHT_COLUMNS, DEFAULT_A_ROWS
from coachopt.processing import build_and_save_data
from coachopt.utils import load_pickle, read_csv_frame


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for preprocessing optimization inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reaction-data",
        required=True,
        help="Path to reaction_data.pkl",
    )
    parser.add_argument("--dataset-eval", required=True, help="Path to dataset_eval.csv")
    parser.add_argument("--training-weights", required=True, help="Path to training_weights.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated NumPy and pickle artifacts")
    parser.add_argument("--a-rows", nargs=3, type=int, default=DEFAULT_A_ROWS, help="Three fitting-row indices for exchange, same-spin correlation, and opposite-spin correlation features")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate metadata inputs and write NumPy/pickle preprocessing artifacts."""
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
    )

    print(f"Wrote data artifacts to {args.output_dir}")
    print(f"A matrix: {outputs['A_matrix']}")
    print(f"b vector: {outputs['b_vec']}")
    print(f"weights: {outputs['weight_vec']}")
    print(f"a_rows: {outputs['a_rows']}")
    print(f"feature count: {outputs['feature_count']}")
    print(f"training rows: {outputs['training_rows']}")
    print(f"dataset count: {outputs['dataset_count']}")
    for diff_grid in ANALYSIS_DIFF_GRIDS:
        print(f"diff matrix {diff_grid}: {outputs[f'diff_matrix_{diff_grid}']}")
        print(f"diff rows {diff_grid}: {outputs['diff_rows_by_grid'][diff_grid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

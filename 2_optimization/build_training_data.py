#!/usr/bin/env python3
"""Build legacy-compatible training arrays from CSV metadata and reaction_data.dict."""

from __future__ import annotations

import argparse
from pathlib import Path

from coachopt.metadata import load_dataset_eval_csv, load_training_weights_csv
from coachopt.processing import FeatureSpec, build_and_save_training_data
from coachopt.utils import load_pickle


def _parse_source_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("reaction data must be provided as SOURCE=PATH")
    source, path = value.split("=", 1)
    source = source.strip()
    path = path.strip()
    if not source or not path:
        raise argparse.ArgumentTypeError("reaction data must be provided as SOURCE=PATH")
    return source, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reaction-data",
        action="append",
        required=True,
        type=_parse_source_assignment,
        help="Reaction data input in SOURCE=PATH form. Repeat this flag to provide multiple sources.",
    )
    parser.add_argument("--analysis-source", help="Source name used for per-dataset evaluation outputs.")
    parser.add_argument("--dataset-eval", required=True, help="Path to dataset_eval.csv")
    parser.add_argument("--training-weights", required=True, help="Path to training_weights.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated NumPy and pickle artifacts")
    parser.add_argument("--a-rows", nargs=3, type=int, default=(64, 153, 166), help="Three fitting-row indices")
    parser.add_argument("--diff-grid", default="99000590", help="Grid key used for diff matrices")
    parser.add_argument("--semilocal", action="store_true", help="Build semilocal features without the SR-HF term")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reaction_sources = {source: load_pickle(path) for source, path in args.reaction_data}
    analysis_source = args.analysis_source or next(iter(reaction_sources))
    dataset_eval_rows = load_dataset_eval_csv(args.dataset_eval)
    training_weight_rows = load_training_weights_csv(args.training_weights)
    spec = FeatureSpec(a_rows=tuple(args.a_rows), semilocal=args.semilocal)

    outputs = build_and_save_training_data(
        reaction_sources=reaction_sources,
        analysis_source=analysis_source,
        dataset_eval_rows=dataset_eval_rows,
        training_weight_rows=training_weight_rows,
        output_dir=Path(args.output_dir),
        spec=spec,
        diff_grid=args.diff_grid,
    )

    print(f"Wrote training artifacts to {args.output_dir}")
    print(f"A matrix: {outputs['A_matrix']}")
    print(f"b vector: {outputs['b_vec']}")
    print(f"weights: {outputs['weight_vec']}")
    print(f"diff matrix: {outputs['diff_matrix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

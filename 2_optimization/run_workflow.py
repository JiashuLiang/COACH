#!/usr/bin/env python3
"""Run the full baseline COACH workflow: preprocess, pass 1, select constraints, pass 2, analyze."""

from __future__ import annotations

import argparse
from pathlib import Path

from coachopt.analysis import analyze_run_directory
from coachopt.constraints import select_diff_constraint_rows
from coachopt.metadata import load_baseline_error_csv, load_dataset_eval_csv, load_dataset_info_csv, load_training_weights_csv
from coachopt.optimizer import OptimizationConfig, run_optimization_sweep
from coachopt.processing import FeatureSpec, artifact_grid_suffix, build_and_save_training_data
from coachopt.utils import load_pickle, save_name_array, write_json


def _parse_source_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("reaction data must be provided as SOURCE=PATH")
    source, path = value.split("=", 1)
    return source.strip(), path.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-data", action="append", required=True, type=_parse_source_assignment)
    parser.add_argument("--analysis-source", help="Source name used for dataset-level outputs")
    parser.add_argument("--dataset-eval", required=True, help="dataset_eval.csv")
    parser.add_argument("--training-weights", required=True, help="training_weights.csv")
    parser.add_argument("--dataset-info", required=True, help="dataset_info.csv")
    parser.add_argument("--baseline-errors", help="Optional baseline error CSV used during analysis")
    parser.add_argument("--processed-dir", required=True, help="Directory for training arrays")
    parser.add_argument("--run-root", required=True, help="Workflow output root")
    parser.add_argument("--nonzeros", "-n", nargs="+", type=int, required=True)
    parser.add_argument("--a-rows", nargs=3, type=int, default=(64, 153, 166))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--grid-threshold", type=float, default=0.015)
    parser.add_argument("--top-per-beta", type=int, default=100)
    parser.add_argument("--top-l1", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reaction_sources = {source: load_pickle(path) for source, path in args.reaction_data}
    analysis_source = args.analysis_source or next(iter(reaction_sources))
    dataset_eval_rows = load_dataset_eval_csv(args.dataset_eval)
    training_weight_rows = load_training_weights_csv(args.training_weights)
    dataset_info = load_dataset_info_csv(args.dataset_info)
    baseline_errors = load_baseline_error_csv(args.baseline_errors) if args.baseline_errors else None
    spec = FeatureSpec(a_rows=tuple(args.a_rows), semilocal=False)
    diff_suffix = artifact_grid_suffix("99000590")

    processed_dir = Path(args.processed_dir)
    run_root = Path(args.run_root)
    pass1_dir = run_root / "pass1"
    pass2_dir = run_root / "pass2"

    build_and_save_training_data(
        reaction_sources=reaction_sources,
        analysis_source=analysis_source,
        dataset_eval_rows=dataset_eval_rows,
        training_weight_rows=training_weight_rows,
        output_dir=processed_dir,
        spec=spec,
    )

    pass1_config = OptimizationConfig(
        nonzeros=args.nonzeros,
        a_rows=spec.a_rows,
        repeats=args.repeats,
        time_limit=args.time_limit,
        nthreads=args.nthreads,
        verbose=args.verbose,
        grid_threshold=args.grid_threshold,
        random_seed=args.random_seed,
        out_dir=str(pass1_dir),
        input_dir=str(processed_dir),
    )
    run_optimization_sweep(pass1_config)

    from coachopt.analysis import load_beta_candidates
    import numpy as np

    diff_matrix = np.load(processed_dir / f"diff_{diff_suffix}.npy")
    diff_names = np.load(processed_dir / f"name_list_diff_{diff_suffix}.npy").astype(str).tolist()
    pass1_betas = [candidate.coefficients for candidate in load_beta_candidates(pass1_dir)]
    selection = select_diff_constraint_rows(
        diff_matrix=diff_matrix,
        diff_names=diff_names,
        betas=pass1_betas,
        top_per_beta=args.top_per_beta,
        top_l1=args.top_l1,
    )
    np.save(processed_dir / "diff_constraint_99590.npy", selection.rows)
    save_name_array(processed_dir / "name_list_diff_constraint_99590.npy", selection.names)
    write_json(processed_dir / "diff_constraint_99590.json", selection.metadata)

    pass2_config = OptimizationConfig(
        nonzeros=args.nonzeros,
        a_rows=spec.a_rows,
        repeats=args.repeats,
        time_limit=args.time_limit,
        nthreads=args.nthreads,
        verbose=args.verbose,
        grid_threshold=args.grid_threshold,
        random_seed=args.random_seed,
        out_dir=str(pass2_dir),
        input_dir=str(processed_dir),
        diff_name="diff_constraint_99590.npy",
        warm_start_dir=str(pass1_dir),
    )
    run_optimization_sweep(pass2_config)

    analyze_run_directory(pass1_dir, processed_dir, dataset_info=dataset_info, baseline_errors=baseline_errors)
    analyze_run_directory(
        pass2_dir,
        processed_dir,
        dataset_info=dataset_info,
        baseline_errors=baseline_errors,
        diff_name="diff_constraint_99590.npy",
        grid_threshold=args.grid_threshold,
    )

    print(f"Processed data: {processed_dir}")
    print(f"Pass 1 results: {pass1_dir}")
    print(f"Pass 2 results: {pass2_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

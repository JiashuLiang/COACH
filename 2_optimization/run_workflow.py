#!/usr/bin/env python3
"""Run the full baseline COACH workflow: preprocess, pass 1, select constraints, pass 2, analyze."""

from __future__ import annotations

import argparse
from pathlib import Path

from coachopt.analysis import analyze_run_directory
from coachopt.constraints import select_diff_constraint_rows
from coachopt.optimizer import OptimizationConfig, run_optimization_sweep
from coachopt.processing import artifact_grid_suffix, build_and_save_training_data
from coachopt.utils import load_pickle, read_csv_frame, read_csv_rows, save_name_array, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-data", required=True, help="Path to reaction_data.dict")
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

    reaction_data = load_pickle(args.reaction_data)
    dataset_eval_rows = read_csv_rows(
        args.dataset_eval,
        ["Reaction", "Dataset", "Reference", "Stoichiometry"],
    )
    training_weight_rows = read_csv_rows(
        args.training_weights,
        ["Dataset", "datapoints", "weights"],
    )
    dataset_info = read_csv_frame(args.dataset_info, ["Dataset", "Datatype"]).set_index("Dataset").to_dict(
        orient="index"
    )
    if args.baseline_errors:
        baseline_frame = read_csv_frame(args.baseline_errors, ["Dataset"])
        value_columns = [column for column in baseline_frame.columns if column != "Dataset"]
        if not value_columns:
            raise ValueError(f"{args.baseline_errors}: expected at least one baseline metric column")
        baseline_errors = baseline_frame.set_index("Dataset")[value_columns].to_dict(orient="index")
    else:
        baseline_errors = None
    a_rows = tuple(args.a_rows)
    diff_suffix = artifact_grid_suffix("99000590")

    processed_dir = Path(args.processed_dir)
    run_root = Path(args.run_root)
    pass1_dir = run_root / "pass1"
    pass2_dir = run_root / "pass2"

    build_and_save_training_data(
        reaction_data=reaction_data,
        dataset_eval_rows=dataset_eval_rows,
        training_weight_rows=training_weight_rows,
        output_dir=processed_dir,
        a_rows=a_rows,
    )

    pass1_config = OptimizationConfig(
        nonzeros=args.nonzeros,
        a_rows=a_rows,
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
        a_rows=a_rows,
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

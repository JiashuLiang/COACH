#!/usr/bin/env python3
"""Run the 289-parameter COACH MIO optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coachopt.optimizer import OptimizationConfig, run_optimization_sweep


def _load_config_file(path: str | None) -> dict:
    """Load optional JSON or YAML defaults for the optimization CLI."""
    if not path:
        return {}
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config files require PyYAML to be installed") from exc
        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    raise ValueError(f"Unsupported config format for {path!r}")


def _preparse_config(argv: list[str] | None) -> str | None:
    """Read only ``--config_file`` so parser defaults can come from disk."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_file")
    namespace, _ = parser.parse_known_args(argv)
    return namespace.config_file


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    """Build the MIO CLI parser with optional config-file defaults."""
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", help="Optional JSON/YAML config file")
    parser.add_argument("--nonzeros", "-n", nargs="+", type=int, default=defaults.get("nonzeros"))
    parser.add_argument("--nthreads", "-t", type=int, default=defaults.get("nthreads", 16))
    parser.add_argument("--repeats", type=int, default=defaults.get("repeats", 3))
    parser.add_argument("--time_limit", type=int, default=defaults.get("time_limit", 3600))
    parser.add_argument("--with_diff", action="store_true", default=defaults.get("with_diff", False))
    parser.add_argument("--verbose", action="store_true", default=defaults.get("verbose", False))
    parser.add_argument("--input_dir", type=str, default=defaults.get("input_dir", "."))
    parser.add_argument("--out_dir", type=str, default=defaults.get("out_dir", "results"))
    parser.add_argument("--A_rows", type=int, nargs=3, default=defaults.get("A_rows", [64, 153, 166]))
    parser.add_argument("--bvec_name", type=str, default=defaults.get("bvec_name", "b_vec.npy"))
    parser.add_argument("--Amatrix_name", type=str, default=defaults.get("Amatrix_name", "A_matrix.npy"))
    parser.add_argument("--weight_name", type=str, default=defaults.get("weight_name", "weight_vec.npy"))
    parser.add_argument("--diff_name", type=str, default=defaults.get("diff_name", "diff_constraint_99590.npy"))
    parser.add_argument("--grid_thresh", type=float, default=defaults.get("grid_thresh", 0.015))
    parser.add_argument("--random_seed", type=int, default=defaults.get("random_seed", 0))
    parser.add_argument("--warm_start_dir", type=str, default=defaults.get("warm_start_dir"))
    parser.add_argument("--warm_start_file", action="append", default=defaults.get("warm_start_file", []))
    parser.add_argument(
        "--no_reference_warm_starts",
        action="store_true",
        default=defaults.get("no_reference_warm_starts", False),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI options and launch the optimization sweep."""
    config_file = _preparse_config(argv)
    config_defaults = _load_config_file(config_file)
    parser = build_parser(config_defaults)
    args = parser.parse_args(argv)

    if not args.nonzeros:
        parser.error("--nonzeros/-n is required")

    diff_name = args.diff_name if args.with_diff else None
    config = OptimizationConfig(
        nonzeros=args.nonzeros,
        a_rows=tuple(args.A_rows),
        repeats=args.repeats,
        time_limit=args.time_limit,
        nthreads=args.nthreads,
        verbose=args.verbose,
        grid_threshold=args.grid_thresh,
        random_seed=args.random_seed,
        out_dir=args.out_dir,
        input_dir=args.input_dir,
        a_matrix_name=args.Amatrix_name,
        b_vec_name=args.bvec_name,
        weight_name=args.weight_name,
        diff_name=diff_name,
        warm_start_dir=args.warm_start_dir,
        warm_start_files=args.warm_start_file,
        include_reference_warm_starts=not args.no_reference_warm_starts,
    )
    outputs = run_optimization_sweep(config)
    print(f"Wrote optimization outputs to {outputs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

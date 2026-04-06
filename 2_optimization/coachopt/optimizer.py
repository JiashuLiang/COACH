"""Gurobi-backed best-subset optimization for the 289-parameter baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import (
    DEFAULT_A_ROWS,
    DEFAULT_GRID_THRESHOLD,
    DEFAULT_MAX_COEFFICIENT,
    HARTREE_TO_KCAL_MOL,
)
from .physical_constraints import build_physical_constraints
from .utils import ensure_directory, write_json


def _vector_from_pairs(pairs: list[tuple[int, float]], size: int = 289) -> np.ndarray:
    """Expand sparse ``(index, value)`` pairs into a dense coefficient vector."""
    vector = np.zeros(size, dtype=float)
    for index, value in pairs:
        vector[index] = value
    return vector


SIMPLE_WARM_START = _vector_from_pairs([(0, 0.85), (1, 1.0), (96, 1.0), (192, 1.0), (288, 0.15)])


def _normalize_warm_start_files(value: str | Iterable[str] | None) -> list[str]:
    """Normalize one or many warm-start file paths into a simple list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(path) for path in value]


def _import_gurobi():
    """Import Gurobi lazily so non-solver utilities remain usable without it."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "Gurobi is required for optimization. Install gurobipy and ensure a valid license is available."
        ) from exc
    return gp, GRB


@dataclass
class OptimizationConfig:
    """Configuration for one optimization sweep over one or more sparsity levels."""

    nonzeros: list[int]
    a_rows: tuple[int, int, int] = DEFAULT_A_ROWS
    repeats: int = 3
    time_limit: int = 3600
    nthreads: int = 16
    verbose: bool = False
    grid_threshold: float = DEFAULT_GRID_THRESHOLD
    coefficient_bound: float = DEFAULT_MAX_COEFFICIENT
    random_seed: int = 0
    out_dir: str = "results"
    input_dir: str = "."
    a_matrix_name: str = "A_matrix.npy"
    b_vec_name: str = "b_vec.npy"
    weight_name: str = "weight_vec.npy"
    diff_name: str | None = None
    warm_start_files: str | list[str] | None = field(default_factory=list)

    def __post_init__(self) -> None:
        """Store warm-start filenames in a consistent list form."""
        self.warm_start_files = _normalize_warm_start_files(self.warm_start_files)


def _load_array(input_dir: str | Path, filename: str) -> np.ndarray:
    """Load a NumPy array from the configured input directory."""
    return np.load(Path(input_dir) / filename, allow_pickle=False)


def _load_warm_start_vectors(config: OptimizationConfig) -> dict[str, np.ndarray]:
    """Load named warm-start beta vectors keyed by source."""
    vectors: dict[str, np.ndarray] = {"simple": np.asarray(SIMPLE_WARM_START, dtype=float).copy()}

    for filename in config.warm_start_files:
        key = filename.replace("/", "_")
        data = np.load(filename, allow_pickle=False)
        array = np.asarray(data, dtype=float)
        if array.shape == (289,) or (array.ndim == 2 and array.shape[1] == 289):
            vectors[key] = array

    return vectors


def weighted_rmse(a_matrix: np.ndarray, b_vec: np.ndarray, weights: np.ndarray, coeff: np.ndarray) -> float:
    """Compute the training weighted RMSE in kcal/mol for one coefficient vector."""
    residual = b_vec - a_matrix @ coeff
    return float(np.sqrt(np.mean(np.square(residual) * weights)) * HARTREE_TO_KCAL_MOL)


def _solve_single_mio(
    x_matrix: np.ndarray,
    y_vec: np.ndarray,
    nonzeros: int,
    constraints: dict[str, np.ndarray],
    config: OptimizationConfig,
    diff_matrix: np.ndarray | None,
    warm_start: np.ndarray | None,
) -> np.ndarray:
    """Solve one mixed-integer quadratic program for a fixed sparsity level."""
    gp, GRB = _import_gurobi()
    dim = x_matrix.shape[1]
    quad = x_matrix.T @ x_matrix
    quad += 1e-10 * np.eye(dim)
    lin = y_vec.T @ x_matrix

    with gp.Env() as env, gp.Model("COACH_MIO", env=env) as model:
        model.Params.Threads = config.nthreads
        model.Params.TimeLimit = config.time_limit
        if not config.verbose:
            model.Params.OutputFlag = 0

        beta = model.addMVar(shape=dim, lb=-config.coefficient_bound, ub=config.coefficient_bound, name="beta")
        include = model.addMVar(shape=dim, vtype=GRB.BINARY, name="include")

        if warm_start is not None and warm_start.shape == (dim,):
            beta.Start = warm_start
            include.Start = (np.abs(warm_start) > 1e-8).astype(float)

        # Link binary support indicators to continuous coefficients via big-M bounds.
        model.addConstr(beta <= config.coefficient_bound * include, name="beta_upper_bound")
        model.addConstr(beta >= -config.coefficient_bound * include, name="beta_lower_bound")
        model.addConstr(include.sum() <= nonzeros, name="sparsity_limit")

        objective = gp.QuadExpr()
        for i, j in product(range(dim), repeat=2):
            objective += 0.5 * quad[i, j] * beta[i] * beta[j]
        for i in range(dim):
            objective -= lin[i] * beta[i]
        objective += 0.5 * float(y_vec @ y_vec)
        model.setObjective(objective, GRB.MINIMIZE)

        # The first 96 coefficients are exchange, the next 96 same-spin
        # correlation, the next 96 opposite-spin correlation, and the final
        # scalar is the SR-exchange mixing parameter.
        ex_rel = constraints["exchange00_relation"]
        model.addConstr(ex_rel @ beta[:96] + beta[288] == 1.0, name="ueg_exchange")

        for index, row in enumerate(constraints["exchange_bounds"]):
            model.addConstr((row - 2.2146 * ex_rel) @ beta[:96] <= 0.0, name=f"gx_max_{index}")
        for index, row in enumerate(constraints["same_spin_bounds"]):
            model.addConstr(row @ beta[96:192] <= 10.0, name=f"gcss_max_{index}")
            model.addConstr(row @ beta[96:192] >= -10.0, name=f"gcss_min_{index}")
        for index, row in enumerate(constraints["opposite_spin_bounds"]):
            model.addConstr(row @ beta[192:288] <= 10.0, name=f"gcos_max_{index}")
            model.addConstr(row @ beta[192:288] >= -10.0, name=f"gcos_min_{index}")
        for index, row in enumerate(constraints["constraint5"]):
            model.addConstr((row - 1.479 * ex_rel) @ beta[:96] <= 0.0, name=f"gx_alpha0_{index}")

        if diff_matrix is not None:
            threshold = config.grid_threshold / HARTREE_TO_KCAL_MOL
            for index, row in enumerate(diff_matrix):
                model.addConstr(row @ beta <= threshold, name=f"grid_diff_max_{index}")
                model.addConstr(row @ beta >= -threshold, name=f"grid_diff_min_{index}")

        model.optimize()
        if model.Status == GRB.Status.INFEASIBLE:
            raise RuntimeError("Optimization model is infeasible")
        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi stopped without a solution (status={model.Status})")
        return np.asarray(beta.X, dtype=float)


def run_optimization_sweep(config: OptimizationConfig) -> dict[str, str]:
    """Run the MIO solver for every requested sparsity level and save all candidates."""
    output_dir = ensure_directory(config.out_dir)
    a_matrix = _load_array(config.input_dir, config.a_matrix_name)
    b_vec = _load_array(config.input_dir, config.b_vec_name)
    weight_vec = _load_array(config.input_dir, config.weight_name)
    diff_matrix = None
    if config.diff_name:
        diff_matrix = _load_array(config.input_dir, config.diff_name)

    x_matrix = a_matrix * np.sqrt(weight_vec)[:, np.newaxis]
    y_vec = b_vec * np.sqrt(weight_vec)
    physical_constraints = build_physical_constraints(config.a_rows)
    rng = np.random.default_rng(config.random_seed)
    warm_start_pool = _load_warm_start_vectors(config)

    manifest = asdict(config)
    manifest["a_rows"] = list(config.a_rows)
    manifest["feature_count"] = int(a_matrix.shape[1])
    manifest["training_rows"] = int(a_matrix.shape[0])
    manifest["diff_rows"] = None if diff_matrix is None else int(diff_matrix.shape[0])
    write_json(output_dir / "run_config.json", manifest)

    for nonzeros in config.nonzeros:
        candidates: list[np.ndarray] = []
        seeds: list[np.ndarray | None] = []
        if warm_start_pool:
            for warm_start in warm_start_pool.values():
                if warm_start.ndim == 1:
                    seeds.append(warm_start)
                else:
                    seeds.extend(np.asarray(warm_start, dtype=float))
        while len(seeds) < max(config.repeats, 1):
            seeds.append(None)

        for repeat_index in range(max(config.repeats, 1)):
            seed = seeds[repeat_index]
            if seed is not None and repeat_index > 0:
                noise = rng.normal(scale=0.05, size=seed.shape[0])
                seed = seed + noise
            elif seed is None:
                seed = rng.normal(scale=0.1, size=a_matrix.shape[1])
            beta = _solve_single_mio(
                x_matrix=x_matrix,
                y_vec=y_vec,
                nonzeros=nonzeros,
                constraints=physical_constraints,
                config=config,
                diff_matrix=diff_matrix,
                warm_start=np.asarray(seed, dtype=float),
            )
            candidates.append(beta)

        beta_path = output_dir / f"betas_nonzero{nonzeros}.npy"
        np.save(beta_path, np.asarray(candidates, dtype=float))

        scores = [
            {
                "candidate_index": idx,
                "wrmse_train_kcal_mol": weighted_rmse(a_matrix, b_vec, weight_vec, candidate),
            }
            for idx, candidate in enumerate(candidates)
        ]
        write_json(output_dir / f"betas_nonzero{nonzeros}.json", scores)

    return {
        "output_dir": str(output_dir),
        "run_config": str(output_dir / "run_config.json"),
    }

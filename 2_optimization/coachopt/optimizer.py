"""Gurobi-backed best-subset optimization for the 289-parameter baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

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
    repeats: int = 1
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
    warm_start_dir: str | None = None


def _load_array(input_dir: str | Path, filename: str) -> np.ndarray:
    """Load a NumPy array from the configured input directory."""
    return np.load(Path(input_dir) / filename, allow_pickle=False)


def _load_warm_start_vectors(config: OptimizationConfig, nonzeros: int) -> dict[str, np.ndarray]:
    """Load the built-in simple seed plus the matching warm-start artifact for one sparsity."""
    vectors: dict[str, np.ndarray] = {"simple": np.asarray(SIMPLE_WARM_START, dtype=float).copy()}
    if not config.warm_start_dir:
        return vectors

    filename = Path(config.warm_start_dir) / f"betas_nonzero{nonzeros}.npy"
    if not filename.exists():
        return vectors

    key = str(filename).replace("/", "_")
    data = np.load(filename, allow_pickle=False)
    array = np.asarray(data, dtype=float)
    if array.shape == (289,):
        vectors[key] = array
    elif array.ndim == 2 and array.shape[1] == 289:
        if array.shape[0] == 1:
            vectors[key] = array[0]
        else:
            for idx, vector in enumerate(array):
                vectors[f"{key}_{idx}"] = np.asarray(vector, dtype=float)

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
        iszero = model.addVars(dim, vtype=GRB.BINARY, name="iszero") 

        if warm_start is not None and warm_start.shape == (dim,):
            beta.Start = warm_start
            for i in range(dim):
                iszero[i].start = (abs(warm_start[i]) < 1e-6)


        obj = sum(0.5 * quad[i,j] * beta[i] * beta[j]
                for i, j in product(range(dim), repeat=2))
        obj -= sum(lin[i] * beta[i] for i in range(dim))
        obj += 0.5 * float(y_vec @ y_vec)
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraint sets, we use dim-1 because we apply "exchange00_relation" to eliminate one degree of freedom.
        for i in range(dim-1):
            # If iszero[i]=1, then beta[i] = 0
            model.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
        model.addConstr(iszero.sum() == dim - nonzeros)


        # The first 96 coefficients are exchange, the next 96 same-spin
        # correlation, the next 96 opposite-spin correlation, and the final
        # scalar is the SR-exchange mixing parameter.
        ex_rel = constraints["exchange00_relation"]
        model.addConstr(ex_rel @ beta[:96] + beta[-1] == 1.0, name="UEG_exchange00_constraint")
        model.addConstr(beta[-1] <= 1.0, name="exchange00_max_constraint")
        model.addConstr(beta[-1] >= 0.0, name="exchange00_min_constraint")

        model.addConstr((constraints["exchange_bounds"] - 2.2146  * ex_rel) @ beta[:96] <= 0, name="gx_max_constraint")
        model.addConstr(constraints["exchange_bounds"] @ beta[:96] >= 0, name="gx_min_constraint")
        model.addConstr(constraints["same_spin_bounds"] @ beta[96:192] <= 10, name="gcss_max_constraint")
        model.addConstr(constraints["same_spin_bounds"] @ beta[96:192] >= -10, name="gcss_min_constraint")
        model.addConstr(constraints["opposite_spin_bounds"] @ beta[192:288] <= 10, name="gcos_max_constraint")
        model.addConstr(constraints["opposite_spin_bounds"] @ beta[192:288] >= -10, name="gcos_min_constraint")

        model.addConstr((constraints["constraint5"] - 1.479 * ex_rel) @ beta[:96] <= 0.0, name="gx_one_constraint")


        # Add diff constraints |diff * beta| <= 0.015 kcal/mol (99590 vs 250974)
        if diff_matrix is not None:
            threshold = config.grid_threshold / HARTREE_TO_KCAL_MOL  * 0.999 # Add a small safety margin to ensure the final model is below the threshold after rounding and unit conversion.
            model.addConstr(diff_matrix @ beta <= threshold, name='diff_max_constraint')
            model.addConstr(diff_matrix @ beta >= -threshold, name='diff_min_constraint')

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
    manifest = asdict(config)
    manifest["a_rows"] = list(config.a_rows)
    manifest["feature_count"] = int(a_matrix.shape[1])
    manifest["training_rows"] = int(a_matrix.shape[0])
    manifest["diff_rows"] = None if diff_matrix is None else int(diff_matrix.shape[0])
    write_json(output_dir / "run_config.json", manifest)

    for nonzeros in config.nonzeros:
        warm_start_pool = _load_warm_start_vectors(config, nonzeros)
        candidates: list[np.ndarray] = []
        score_labels: list[str] = []
        repeat_count = max(config.repeats, 1)
        for warm_start_key, warm_start in warm_start_pool.items():
            for repeat_index in range(repeat_count):
                label = warm_start_key
                seed = np.asarray(warm_start, dtype=float)
                if repeat_count > 1:
                    label = f"{label}_repeat_{repeat_index}"
                if repeat_index > 0:
                    noise = rng.normal(scale=0.05, size=seed.shape[0])
                    seed = seed + noise
                beta = _solve_single_mio(
                    x_matrix=x_matrix,
                    y_vec=y_vec,
                    nonzeros=nonzeros,
                    constraints=physical_constraints,
                    config=config,
                    diff_matrix=diff_matrix,
                    warm_start=seed,
                )
                candidates.append(beta)
                score_labels.append(label)

        beta_path = output_dir / f"betas_nonzero{nonzeros}.npy"
        np.save(beta_path, np.asarray(candidates, dtype=float))

        scores = [
            {"label": label, "wrmse_train_kcal_mol": weighted_rmse(a_matrix, b_vec, weight_vec, candidate)}
            for label, candidate in zip(score_labels, candidates, strict=True)
        ]
        write_json(output_dir / f"betas_nonzero{nonzeros}.json", scores)

    return {
        "output_dir": str(output_dir),
        "run_config": str(output_dir / "run_config.json"),
    }

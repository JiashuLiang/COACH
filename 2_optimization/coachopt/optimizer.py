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
    W_B97M_V_COEF_289,
    W_B97X_V_COEF_289,
)
from .utils import ensure_directory, write_json


def _vector_from_pairs(pairs: list[tuple[int, float]], size: int = 289) -> np.ndarray:
    """Expand sparse ``(index, value)`` pairs into a dense coefficient vector."""
    vector = np.zeros(size, dtype=float)
    for index, value in pairs:
        vector[index] = value
    return vector


DEFAULT_WARM_STARTS = {
    "simple": _vector_from_pairs([(0, 0.85), (1, 1.0), (96, 1.0), (192, 1.0), (288, 0.15)]),
    "wB97X-V": _vector_from_pairs(W_B97X_V_COEF_289),
    "wB97M-V": _vector_from_pairs(W_B97M_V_COEF_289),
}


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


def linear_expansion(values: np.ndarray, order: int) -> np.ndarray:
    """Build linear-polynomial features ``[1, x, x^2, ...]`` for each input value."""
    output = np.ones((order, values.shape[0]), dtype=float)
    for index in range(1, order):
        output[index] = output[index - 1] * values
    return output.T


def chebyshev_expansion(values: np.ndarray, order: int) -> np.ndarray:
    """Build Chebyshev polynomial features for each input value."""
    output = np.ones((order, values.shape[0]), dtype=float)
    output[1] = values
    for index in range(2, order):
        output[index] = 2.0 * values * output[index - 1] - output[index - 2]
    return output.T


def legendre_expansion(values: np.ndarray, order: int) -> np.ndarray:
    """Build Legendre polynomial features for each input value."""
    output = np.ones((order, values.shape[0]), dtype=float)
    output[1] = values
    for index in range(2, order):
        i_float = float(index)
        output[index] = (
            ((2.0 * i_float - 1.0) * values * output[index - 1]) - ((i_float - 1.0) * output[index - 2])
        ) / i_float
    return output.T


def u_to_s2(values: np.ndarray) -> np.ndarray:
    """Map bounded exchange variable ``u`` back to the reduced gradient ``s^2``."""
    return 250.0 * values / (1.0 - values)


def expansion_assist(us: np.ndarray, ws: np.ndarray, choice: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the requested polynomial families for the exchange/correlation basis."""
    if choice < 3:
        us_expanded = linear_expansion(us, 8)
    elif choice < 6:
        us_expanded = legendre_expansion(us, 8)
    else:
        us_expanded = chebyshev_expansion(us, 8)

    mgga_choice = choice % 3
    if mgga_choice == 0:
        ws_expanded = linear_expansion(ws, 12)
    elif mgga_choice == 1:
        ws_expanded = legendre_expansion(ws, 12)
    else:
        ws_expanded = chebyshev_expansion(ws, 12)
    return us_expanded, ws_expanded


def expansion(us: np.ndarray, ws: np.ndarray, choice: int, with_nonuniform_scaling: bool = False) -> np.ndarray:
    """Enumerate the 96-term tensor-product basis over all ``u``/``w`` pairs."""
    us_expanded, ws_expanded = expansion_assist(us, ws, choice)
    expansions = []
    for i in range(us.shape[0]):
        scale = 1.0
        if with_nonuniform_scaling:
            s2_value = u_to_s2(np.asarray([us[i]], dtype=float))[0]
            scale = 1.0 - np.exp(-13.815 / s2_value ** 0.25)
        for j in range(ws.shape[0]):
            expansions.append(scale * np.outer(ws_expanded[j], us_expanded[i]).reshape(-1))
    return np.asarray(expansions, dtype=float)


def build_physical_constraints(a_rows: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Construct the manuscript-style physical constraint matrices for one row triple."""
    exchange_choice = a_rows[0] % 18
    same_spin_choice = a_rows[1] % 18
    opposite_spin_choice = a_rows[2] % 18

    exchange00_relation = expansion(np.asarray([0.0]), np.asarray([0.0]), exchange_choice).reshape(-1)

    ws = np.asarray([-1.0, -0.7, -0.2, -0.08, -0.01, 0.01, 0.05, 0.1, 0.3, 0.5, 0.65, 0.7, 0.8, 0.95, 1.0])
    us = np.asarray([0.001, 0.01, 0.1, 0.15, 0.22, 0.3, 0.6, 0.7, 0.85, 0.95, 0.99])

    exchange_mode = a_rows[0] // 18
    with_scaling = exchange_mode in (1, 3)
    exchange_bounds = expansion(us, ws, exchange_choice, with_nonuniform_scaling=with_scaling)
    same_spin_bounds = expansion(us, ws, same_spin_choice)
    opposite_spin_bounds = expansion(us, ws, opposite_spin_choice)

    if exchange_choice // 9 == 0:
        s2_values = np.asarray(
            [
                0.3,
                0.6,
                1.2,
                1.8,
                2.4,
                2.7,
                3.0,
                3.3,
                3.6,
                3.9,
                4.2,
                4.5,
                4.8,
                5.2,
                5.6,
                6.0,
                6.6,
                7.2,
                7.8,
                8.4,
                9.0,
                10.2,
                11.4,
                12.0,
                13.2,
                15.0,
                18.0,
                24.0,
                36.0,
                60.0,
                120.0,
            ]
        )
        constant = 5.0 / 3.0 / 4.0 / (6.0 * np.pi * np.pi) ** (2.0 / 3.0)
        ux_array = 0.004 * s2_values / (1.0 + 0.004 * s2_values)
        ws_array = (1.0 - s2_values * constant) / (1.0 + s2_values * constant)
        us_expanded, ws_expanded = expansion_assist(ux_array, ws_array, exchange_choice)
        constraint5 = []
        for idx, s2_value in enumerate(s2_values):
            scale = 1.0
            if with_scaling:
                scale = 1.0 - np.exp(-13.815 / s2_value ** 0.25)
            constraint5.append(scale * np.outer(ws_expanded[idx], us_expanded[idx]).reshape(-1))
        constraint5_array = np.asarray(constraint5, dtype=float)
    else:
        constraint5_array = expansion(
            np.asarray([0.001, 0.01, 0.1, 0.15, 0.22, 0.3, 0.5, 0.6, 0.7, 0.85, 0.95, 0.99, 0.999]),
            np.asarray([-1.0]),
            exchange_choice,
            with_nonuniform_scaling=with_scaling,
        )

    return {
        "exchange00_relation": exchange00_relation,
        "exchange_bounds": exchange_bounds,
        "same_spin_bounds": same_spin_bounds,
        "opposite_spin_bounds": opposite_spin_bounds,
        "constraint5": constraint5_array,
    }


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
    warm_start_files: list[str] = field(default_factory=list)
    warm_start_dir: str | None = None
    include_reference_warm_starts: bool = True


def _load_array(input_dir: str | Path, filename: str) -> np.ndarray:
    """Load a NumPy array from the configured input directory."""
    return np.load(Path(input_dir) / filename, allow_pickle=False)


def _load_warm_start_vectors(config: OptimizationConfig) -> list[np.ndarray]:
    """Load and deduplicate optional warm-start beta vectors."""
    vectors: list[np.ndarray] = []
    if config.include_reference_warm_starts:
        vectors.extend(DEFAULT_WARM_STARTS.values())

    if config.warm_start_dir:
        for path in sorted(Path(config.warm_start_dir).glob("betas_nonzero*.npy")):
            data = np.load(path)
            if data.ndim == 1:
                vectors.append(data.astype(float))
            else:
                vectors.extend(np.asarray(data, dtype=float))

    for filename in config.warm_start_files:
        data = np.load(filename)
        if data.ndim == 1:
            vectors.append(data.astype(float))
        else:
            vectors.extend(np.asarray(data, dtype=float))

    unique_vectors: list[np.ndarray] = []
    seen: set[bytes] = set()
    for vector in vectors:
        if vector.shape != (289,):
            continue
        key = np.asarray(vector, dtype=float).round(12).tobytes()
        if key not in seen:
            seen.add(key)
            unique_vectors.append(np.asarray(vector, dtype=float))
    return unique_vectors


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
            seeds.extend(warm_start_pool)
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

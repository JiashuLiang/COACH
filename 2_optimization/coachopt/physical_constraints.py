"""Helpers for constructing manuscript-style physical-constraint matrices.

The optimizer works with a 289-parameter coefficient vector:

- indices `0:96`: exchange coefficients
- indices `96:192`: same-spin correlation coefficients
- indices `192:288`: opposite-spin correlation coefficients
- index `288`: short-range exact-exchange mixing scalar

`a_rows` chooses which integratedDV row family defines each 96-coefficient
block. For each selected row:

- `row % 18` picks the basis-group index within one 18-group family
- `row // 18` picks the exchange mode, where modes `1` and `3` apply the
  nonuniform scaling factor used in the manuscript baseline

The constraint matrices returned by `build_physical_constraints()` encode the mesh grids to enforce the following conditions:

- `exchange00_relation`: the normalization condition `exchange(0,0) + sr_ex = 1`
- `exchange_bounds`: sampled bounds enforcing `0 <= gx <= 2.2146`
- `same_spin_bounds`: sampled bounds enforcing `-10 <= gcss <= 10`
- `opposite_spin_bounds`: sampled bounds enforcing `-10 <= gcos <= 10`
- `gx_one_constraint`: the additional sampled bound `gx <= 1.479` on the one-electron limit
"""

from __future__ import annotations

import numpy as np


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


def ux_to_s2(values: np.ndarray) -> np.ndarray:
    """
    u = 0.004*s2/(1 + 0.004*s2)
    s2 = 250 * u/(1 - u)
    """
    return 250.0 * values / (1.0 - values)


def expansion_assist(us: np.ndarray, ws: np.ndarray, choice: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the polynomial families for one 96-term tensor-product basis.

    `choice` is an integer in `0..8`:

    - `0..2`: linear `u`
    - `3..5`: Legendre `u`
    - `6..8`: Chebyshev `u`

    In each block of three, `choice % 3` selects the companion family:

    - `0`: linear
    - `1`: Legendre
    - `2`: Chebyshev
    """
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
            # The coefficient is converted from SCANs's: (2(6\pi^2)^{1/3})^(1/2) * 4.9479 = 13.815
            s2_value = ux_to_s2(np.asarray([us[i]], dtype=float))[0]
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
        s2_values = np.asarray( [0.3, 0.6, 1.2, 1.8, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.2, 5.6, 6.0,\
        6.6, 7.2, 7.8, 8.4, 9.0, 10.2, 11.4, 12.0, 13.2, 15.0, 18, 24, 36, 60, 120], dtype=float)
        constant = 5.0 / 3.0 / 4.0 / (6.0 * np.pi * np.pi) ** (2.0 / 3.0)
        ux_array = 0.004 * s2_values / (1.0 + 0.004 * s2_values)
        ws_array = (1.0 - s2_values * constant) / (1.0 + s2_values * constant)
        us_expanded, ws_expanded = expansion_assist(ux_array, ws_array, exchange_choice)
        gx_one_constraint = []
        for idx, s2_value in enumerate(s2_values):
            scale = 1.0
            if with_scaling:
                scale = 1.0 - np.exp(-13.815 / s2_value ** 0.25)
            gx_one_constraint.append(scale * np.outer(ws_expanded[idx], us_expanded[idx]).reshape(-1))
        gx_one_constraint_array = np.asarray(gx_one_constraint, dtype=float)
    else:
        gx_one_constraint_array = expansion(
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
        "gx_one_constraint": gx_one_constraint_array,
    }

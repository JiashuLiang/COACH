"""Generate integratedDV matrices from PySCF using the cleaned COACH baseline layout."""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.special import erf
from pyscf import dft
from pyscf import gto


GEOMETRY = """
O   0.00000000   0.00000000   0.39393904
H  -0.75503878   0.00000000  -0.19696952
H   0.75503878   0.00000000  -0.19696952
"""
BASIS = "def2-qzvppd"
XC = "wb97xv"
CHARGE = 0
SPIN = 0
GRID_SETUPS = [(75, 302), (99, 590), (250, 974)]
BLOCK_SIZE = 20000
OUTPUT_TXT = "1_data_generation/pyscf_integratedDV_matrices.txt"
XC_GRID = (99, 590)
NL_GRID = (75, 302)
NELE_SERIES = 96
NSERIES = 18
NSERIES_X = 2 * NSERIES
NSERIES_SS = 2 * NSERIES
NSERIES_OS = NSERIES
TOL = 1e-14


def build_parser():
    """Create the CLI parser for single-species or batch XYZ generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", help="Path to one XYZ file to process.")
    parser.add_argument("--xyz-dir", help="Directory of XYZ files to process into one .txt output per file.")
    parser.add_argument("--output-txt", help="Output text path for a single generated species.")
    parser.add_argument("--output-dir", help="Directory for generated .txt files.")
    parser.add_argument("--basis", default=BASIS, help="PySCF basis for generated jobs unless --use-xyz-basis is set.")
    parser.add_argument("--use-xyz-basis", action="store_true", help="Read the basis=... field from the XYZ comment line.")
    parser.add_argument("--xc", default=XC, help="Exchange-correlation functional passed to PySCF.")
    parser.add_argument("--verbose", type=int, default=4, help="PySCF verbosity level.")
    return parser


def parse_xyz_metadata(comment_line):
    """Parse comma-separated key=value pairs from the XYZ comment line."""
    metadata = {}
    for field in comment_line.split(","):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def load_xyz_job(path, basis_override, use_xyz_basis):
    """Load geometry and simple metadata from an XYZ file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: expected at least 3 lines in XYZ file")

    natom = int(lines[0].strip())
    comment_line = lines[1].strip()
    atom_lines = [line.rstrip() for line in lines[2 : 2 + natom] if line.strip()]
    if len(atom_lines) != natom:
        raise ValueError(f"{path}: expected {natom} atom lines, found {len(atom_lines)}")

    metadata = parse_xyz_metadata(comment_line)
    charge = int(metadata.get("charge", "0"))
    multiplicity = int(metadata.get("multiplicity", "1"))
    if multiplicity < 1:
        raise ValueError(f"{path}: multiplicity must be >= 1")

    basis = basis_override
    if use_xyz_basis:
        basis = metadata.get("basis", basis_override)
        if not basis:
            raise ValueError(f"{path}: no basis found in XYZ comment line")

    return {
        "name": path.stem,
        "geometry": "\n".join(atom_lines),
        "charge": charge,
        "spin": multiplicity - 1,
        "basis": basis,
    }


def build_default_job():
    """Return the historical single-molecule defaults used by the original script."""
    return {
        "name": "pyscf_integratedDV_matrices",
        "geometry": GEOMETRY.strip(),
        "charge": CHARGE,
        "spin": SPIN,
        "basis": BASIS,
    }


def _linear_series_batch(x, n):
    """Return batched linear polynomial series [1, x, x^2, ...] up to order n.

    Args:
        x: 1D array of scalar inputs with shape (npoint,).
        n: Number of terms in the series.

    Returns:
        2D array with shape (npoint, n), where column k is x**k.
    """
    out = np.empty((x.size, n), dtype=np.float64)
    out[:, 0] = 1.0
    for i in range(1, n):
        out[:, i] = out[:, i - 1] * x
    return out


def _chebyshev_series_batch(x, n):
    """Return batched Chebyshev polynomial series T_k(x) for k=0..n-1.

    Args:
        x: 1D array of scalar inputs with shape (npoint,).
        n: Number of Chebyshev terms.

    Returns:
        2D array with shape (npoint, n), where column k is T_k(x).
    """
    out = np.empty((x.size, n), dtype=np.float64)
    out[:, 0] = 1.0
    if n > 1:
        out[:, 1] = x
    for i in range(2, n):
        out[:, i] = 2.0 * x * out[:, i - 1] - out[:, i - 2]
    return out


def _legendre_series_batch(x, n):
    """Return batched Legendre polynomial series P_k(x) for k=0..n-1.

    Args:
        x: 1D array of scalar inputs with shape (npoint,).
        n: Number of Legendre terms.

    Returns:
        2D array with shape (npoint, n), where column k is P_k(x).
    """
    out = np.empty((x.size, n), dtype=np.float64)
    out[:, 0] = 1.0
    if n > 1:
        out[:, 1] = x
    for i in range(2, n):
        i_d = float(i)
        out[:, i] = ((2.0 * i_d - 1.0) * x * out[:, i - 1] - (i_d - 1.0) * out[:, i - 2]) / i_d
    return out


def _kron_batch(a, b):
    """Compute row-wise Kronecker products between batched vectors a and b.

    Args:
        a: 2D array with shape (npoint, na).
        b: 2D array with shape (npoint, nb).

    Returns:
        2D array with shape (npoint, na * nb), row-wise Kronecker products.
    """
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)


def expansion_basis_batch(u, w, beta_f):
    """Build the 96x18 expansion basis per grid point following C++ Expansion().

    Args:
        u: 1D array of reduced-gradient-like variables.
        w: 1D array of kinetic-energy-ratio variables.
        beta_f: 1D array of beta_f variables.

    Returns:
        3D array with shape (npoint, 96, 18), one basis matrix per point.

    Notes:
        Group-to-basis mapping (0..17) is documented in
        ``1_data_generation/README.md`` under "Expansion Group Mapping".
    """
    u_linear = _linear_series_batch(u, 8)
    u_legendre = _legendre_series_batch(u, 8)
    u_chebyshev = _chebyshev_series_batch(u, 8)

    w_linear = _linear_series_batch(w, 12)
    w_legendre = _legendre_series_batch(w, 12)
    w_chebyshev = _chebyshev_series_batch(w, 12)

    beta_linear = _linear_series_batch(beta_f, 12)
    beta_legendre = _legendre_series_batch(beta_f, 12)
    beta_chebyshev = _chebyshev_series_batch(beta_f, 12)

    basis = np.empty((u.size, NELE_SERIES, NSERIES), dtype=np.float64)
    basis[:, :, 0] = _kron_batch(w_linear, u_linear)
    basis[:, :, 1] = _kron_batch(w_legendre, u_linear)
    basis[:, :, 2] = _kron_batch(w_chebyshev, u_linear)
    basis[:, :, 3] = _kron_batch(w_linear, u_legendre)
    basis[:, :, 4] = _kron_batch(w_legendre, u_legendre)
    basis[:, :, 5] = _kron_batch(w_chebyshev, u_legendre)
    basis[:, :, 6] = _kron_batch(w_linear, u_chebyshev)
    basis[:, :, 7] = _kron_batch(w_legendre, u_chebyshev)
    basis[:, :, 8] = _kron_batch(w_chebyshev, u_chebyshev)

    basis[:, :, 9] = _kron_batch(beta_linear, u_linear)
    basis[:, :, 10] = _kron_batch(beta_legendre, u_linear)
    basis[:, :, 11] = _kron_batch(beta_chebyshev, u_linear)
    basis[:, :, 12] = _kron_batch(beta_linear, u_legendre)
    basis[:, :, 13] = _kron_batch(beta_legendre, u_legendre)
    basis[:, :, 14] = _kron_batch(beta_chebyshev, u_legendre)
    basis[:, :, 15] = _kron_batch(beta_linear, u_chebyshev)
    basis[:, :, 16] = _kron_batch(beta_legendre, u_chebyshev)
    basis[:, :, 17] = _kron_batch(beta_chebyshev, u_chebyshev)
    return basis


def _g_func_arr(rs, a_const, alpha_1, beta_1, beta_2, beta_3, beta_4):
    """Vectorized PW92-like helper function used in correlation channels.

    Args:
        rs: Wigner-Seitz radius array.
        a_const: Parameter A in the analytic form.
        alpha_1: Parameter alpha_1.
        beta_1: Parameter beta_1.
        beta_2: Parameter beta_2.
        beta_3: Parameter beta_3.
        beta_4: Parameter beta_4.

    Returns:
        1D array of energy-per-particle values with the same shape as rs.
    """
    inner = 2.0 * a_const * ((beta_1 + beta_3 * rs) * np.sqrt(rs) + (beta_2 + beta_4 * rs) * rs)
    return -2.0 * a_const * (1.0 + alpha_1 * rs) * np.log1p(1.0 / inner)


def _h_func_arr(rs, s2, zeta, e_lda):
    """Vectorized SCAN correction term H(rs, s^2, zeta, e_LDA).

    Args:
        rs: Wigner-Seitz radius array.
        s2: Reduced-gradient-squared array.
        zeta: Spin polarization array (or scalar broadcastable to rs).
        e_lda: LDA correlation energy-per-particle array.

    Returns:
        1D array of SCAN correction values.
    """
    phi = ((1.0 - zeta) ** (2.0 / 3.0) + (1.0 + zeta) ** (2.0 / 3.0)) / 2.0
    gamma = (1.0 - math.log(2.0)) / (math.pi * math.pi)
    gammaphi3 = gamma * (phi**3)

    beta_rs = 0.066725 * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs)
    expo = np.clip(-e_lda / gammaphi3, -700.0, 700.0)
    w1 = np.expm1(expo)

    a_const = beta_rs / w1 / gamma
    t2 = s2 / (16.0 * (4.0 ** (1.0 / 3.0)) * rs * phi * phi)
    g_at2 = (1.0 + 4.0 * a_const * t2) ** (-0.25)
    return gammaphi3 * np.log1p(w1 * (1.0 - g_at2))


def _accumulate_channel(dst, coeff, basis):
    """Accumulate sum_i coeff[i] * basis[i] into destination channel matrix.

    Args:
        dst: 2D destination matrix of shape (96, nchannel), updated in place.
        coeff: 1D coefficient array of length npoint.
        basis: 3D basis array of shape (npoint, 96, nseries).

    Returns:
        None. ``dst`` is modified in place.
    """
    dst += np.tensordot(coeff, basis, axes=(0, 0))


def accumulate_exchange_block(weights, rho, rho1, tau, exchange_terms):
    """Accumulate exchange contributions for one spin channel over a grid block.

    Output layout (96 x 72):
    - cols  0:18  -> mGGA
    - cols 18:36  -> w-mGGA
    - cols 36:54  -> mGGA with nonuniform-scaling factor
    - cols 54:72  -> w-mGGA with nonuniform-scaling factor
    Args:
        weights: 1D quadrature weights for the current block.
        rho: 1D spin-density array.
        rho1: 2D density-gradient array with shape (npoint, 3).
        tau: 1D kinetic-energy-density array.
        exchange_terms: 2D output matrix (96, 72), updated in place.

    Returns:
        None. ``exchange_terms`` is modified in place.
    """
    exchange_mgga = exchange_terms[:, :NSERIES_X]
    exchange_wmgga = exchange_terms[:, NSERIES_X : 2 * NSERIES_X]

    c_lda = -(3.0 / 2.0) * ((3.0 / (4.0 * math.pi)) ** (1.0 / 3.0))
    tau_ueg_coeff = (3.0 / 5.0) * ((6.0 * math.pi * math.pi) ** (2.0 / 3.0))
    omega = 0.3
    gamma_x = 0.004
    sqrt_pi = math.sqrt(math.pi)
    kf_coeff = (6.0 * math.pi * math.pi) ** (1.0 / 3.0)

    # Match C++ defensive clipping for tiny negative numerical noise.
    ra = np.maximum(rho, 0.0)
    ga = np.einsum("ij,ij->i", rho1, rho1)
    ta = np.maximum(tau, 0.0)
    mask = (ra > TOL) & (ta > TOL)
    if not np.any(mask):
        return

    ra_m = ra[mask]
    ga_m = ga[mask]
    ta_m = ta[mask]
    wt_m = weights[mask]

    rho4o3 = ra_m ** (4.0 / 3.0)
    rho1o3 = ra_m ** (1.0 / 3.0)
    ex_ueg = c_lda * rho4o3

    a_val = omega / (kf_coeff * rho1o3)
    a3 = a_val**3
    am2 = a_val**-2
    ainv = a_val**-1
    fn = 1.0 - (2.0 / 3.0) * a_val * (
        a3 - 3.0 * a_val + (2.0 * a_val - a3) * np.exp(-am2) + 2.0 * sqrt_pi * erf(ainv)
    )

    s2 = ga_m / (ra_m ** (8.0 / 3.0))
    u = (gamma_x * s2) / (1.0 + gamma_x * s2)
    taua_ueg = tau_ueg_coeff * (ra_m ** (5.0 / 3.0))
    ts = taua_ueg / ta_m
    w_val = (-1.0 + ts) / (1.0 + ts)
    taua_w = ga_m / ra_m / 4.0
    beta_f = 2.0 * (ta_m - taua_w) / (ta_m + taua_ueg) - 1.0
    basis = expansion_basis_batch(u, w_val, beta_f)

    wex_ueg = ex_ueg * wt_m
    wfnex_ueg = wex_ueg * fn
    _accumulate_channel(exchange_mgga[:, :NSERIES], wex_ueg, basis)
    _accumulate_channel(exchange_wmgga[:, :NSERIES], wfnex_ueg, basis)

    nonuniform_scaling_factor = np.ones_like(s2)
    s2_mask = s2 > 0.0
    nonuniform_scaling_factor[s2_mask] = 1.0 - np.exp(-13.815 / (s2[s2_mask] ** 0.25))
    _accumulate_channel(exchange_mgga[:, NSERIES:2 * NSERIES], nonuniform_scaling_factor * wex_ueg, basis)
    _accumulate_channel(exchange_wmgga[:, NSERIES:2 * NSERIES], nonuniform_scaling_factor * wfnex_ueg, basis)


def accumulate_correlation_block(weights, rho_a, rho_b, rho_a1, rho_b1, tau_a, tau_b, correlation_terms):
    """Accumulate correlation channels (ss/os, PW92/SCAN) over one grid block.

    Output layout (96 x 108):
    - cols   0:36  -> corr_ss
    - cols  36:54  -> corr_os
    - cols  54:90  -> corr_ss_scan
    - cols  90:108 -> corr_os_scan
    Args:
        weights: 1D quadrature weights for the current block.
        rho_a: 1D alpha spin density.
        rho_b: 1D beta spin density.
        rho_a1: 2D alpha density-gradient array with shape (npoint, 3).
        rho_b1: 2D beta density-gradient array with shape (npoint, 3).
        tau_a: 1D alpha kinetic-energy density.
        tau_b: 1D beta kinetic-energy density.
        correlation_terms: 2D output matrix (96, 108), updated in place.

    Returns:
        None. ``correlation_terms`` is modified in place.
    """
    corr_ss = correlation_terms[:, :NSERIES_SS]
    corr_os = correlation_terms[:, NSERIES_SS : NSERIES_SS + NSERIES_OS]
    corr_ss_scan = correlation_terms[:, NSERIES_SS + NSERIES_OS : 2 * NSERIES_SS + NSERIES_OS]
    corr_os_scan = correlation_terms[:, 2 * NSERIES_SS + NSERIES_OS :]

    rs_coeff = (0.75 / math.pi) ** (1.0 / 3.0)
    tau_ueg_coeff = (3.0 / 5.0) * ((6.0 * math.pi * math.pi) ** (2.0 / 3.0))
    total_tau_ueg_coeff = (3.0 / 5.0) * ((3.0 * math.pi * math.pi) ** (2.0 / 3.0))
    gamma_ss = 0.2
    gamma_os = 0.006
    fppz = 4.0 / (9.0 * ((2.0 ** (1.0 / 3.0)) - 1.0))

    ra = np.maximum(rho_a, 0.0)
    rb = np.maximum(rho_b, 0.0)
    ga = np.einsum("ij,ij->i", rho_a1, rho_a1)
    gb = np.einsum("ij,ij->i", rho_b1, rho_b1)
    ta = np.maximum(tau_a, 0.0)
    tb = np.maximum(tau_b, 0.0)

    eps_c_pw92_a = np.zeros_like(ra)
    eps_c_scan_a = np.zeros_like(ra)
    eps_c_pw92_b = np.zeros_like(rb)
    eps_c_scan_b = np.zeros_like(rb)
    s2a = np.zeros_like(ra)
    s2b = np.zeros_like(rb)
    tsa = np.zeros_like(ra)
    tsb = np.zeros_like(rb)

    # Same-spin alpha branch.
    mask_a = (ra > TOL) & (ta > TOL)
    if np.any(mask_a):
        ra_m, ga_m, ta_m, wt_m = ra[mask_a], ga[mask_a], ta[mask_a], weights[mask_a]
        rs = rs_coeff * (ra_m ** (-1.0 / 3.0))
        eps_pw = _g_func_arr(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        s2 = ga_m / (ra_m ** (8.0 / 3.0))
        eps_scan = eps_pw + _h_func_arr(rs, s2, np.ones_like(rs), eps_pw)

        u = (gamma_ss * s2) / (1.0 + gamma_ss * s2)
        taua_ueg = tau_ueg_coeff * (ra_m ** (5.0 / 3.0))
        ts = taua_ueg / ta_m
        w_val = (-1.0 + ts) / (1.0 + ts)
        taua_w = ga_m / ra_m / 4.0
        beta = (ta_m - taua_w) / (ta_m + taua_ueg)
        beta_f = 2.0 * beta - 1.0
        basis = expansion_basis_batch(u, w_val, beta_f)

        we_pw = eps_pw * ra_m * wt_m
        we_scan = eps_scan * ra_m * wt_m
        _accumulate_channel(corr_ss[:, :NSERIES], we_pw, basis)
        _accumulate_channel(corr_ss_scan[:, :NSERIES], we_scan, basis)
        _accumulate_channel(corr_ss[:, NSERIES:2 * NSERIES], 2.0 * beta * we_pw, basis)
        _accumulate_channel(corr_ss_scan[:, NSERIES:2 * NSERIES], 2.0 * beta * we_scan, basis)

        eps_c_pw92_a[mask_a] = eps_pw
        eps_c_scan_a[mask_a] = eps_scan
        s2a[mask_a] = s2
        tsa[mask_a] = ts

    # Same-spin beta branch.
    mask_b = (rb > TOL) & (tb > TOL)
    if np.any(mask_b):
        rb_m, gb_m, tb_m, wt_m = rb[mask_b], gb[mask_b], tb[mask_b], weights[mask_b]
        rs = rs_coeff * (rb_m ** (-1.0 / 3.0))
        eps_pw = _g_func_arr(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        s2 = gb_m / (rb_m ** (8.0 / 3.0))
        eps_scan = eps_pw + _h_func_arr(rs, s2, -np.ones_like(rs), eps_pw)

        u = (gamma_ss * s2) / (1.0 + gamma_ss * s2)
        taub_ueg = tau_ueg_coeff * (rb_m ** (5.0 / 3.0))
        ts = taub_ueg / tb_m
        w_val = (-1.0 + ts) / (1.0 + ts)
        taub_w = gb_m / rb_m / 4.0
        beta = (tb_m - taub_w) / (tb_m + taub_ueg)
        beta_f = 2.0 * beta - 1.0
        basis = expansion_basis_batch(u, w_val, beta_f)

        we_pw = eps_pw * rb_m * wt_m
        we_scan = eps_scan * rb_m * wt_m
        _accumulate_channel(corr_ss[:, :NSERIES], we_pw, basis)
        _accumulate_channel(corr_ss_scan[:, :NSERIES], we_scan, basis)
        _accumulate_channel(corr_ss[:, NSERIES:2 * NSERIES], 2.0 * beta * we_pw, basis)
        _accumulate_channel(corr_ss_scan[:, NSERIES:2 * NSERIES], 2.0 * beta * we_scan, basis)

        eps_c_pw92_b[mask_b] = eps_pw
        eps_c_scan_b[mask_b] = eps_scan
        s2b[mask_b] = s2
        tsb[mask_b] = ts

    # Opposite-spin branch.
    mask_ab = (ra > TOL) & (rb > TOL) & (ta > TOL) & (tb > TOL)
    if np.any(mask_ab):
        ra_m, rb_m = ra[mask_ab], rb[mask_ab]
        ta_m, tb_m = ta[mask_ab], tb[mask_ab]
        r_tot = ra_m + rb_m
        grad_tot = rho_a1[mask_ab] + rho_b1[mask_ab]
        g_tot = np.einsum("ij,ij->i", grad_tot, grad_tot)
        t_tot = ta_m + tb_m

        rs = rs_coeff * (r_tot ** (-1.0 / 3.0))
        alpha_c = _g_func_arr(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
        gpw_0 = _g_func_arr(rs, 0.0310907, 0.2137, 7.5957, 3.5876, 1.6382, 0.49294)
        gpw_1 = _g_func_arr(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)

        zeta = (ra_m - rb_m) / r_tot
        zeta[np.abs(zeta) < TOL] = 0.0
        fpol = (
            -2.0 + (1.0 - zeta) ** (4.0 / 3.0) + (1.0 + zeta) ** (4.0 / 3.0)
        ) / (-2.0 + 2.0 * (2.0 ** (1.0 / 3.0)))
        zeta4 = zeta**4
        eps_c_pw92 = gpw_0 + (fpol * alpha_c * (zeta4 - 1.0)) / fppz + fpol * (gpw_1 - gpw_0) * zeta4

        wt_m = weights[mask_ab]
        eps_pw_a_m = eps_c_pw92_a[mask_ab]
        eps_pw_b_m = eps_c_pw92_b[mask_ab]
        eps_scan_a_m = eps_c_scan_a[mask_ab]
        eps_scan_b_m = eps_c_scan_b[mask_ab]

        we_pw_os = (eps_c_pw92 * r_tot - eps_pw_a_m * ra_m - eps_pw_b_m * rb_m) * wt_m
        s2 = g_tot / (r_tot ** (8.0 / 3.0))
        eps_c_scan = eps_c_pw92 + _h_func_arr(rs, s2, zeta, eps_c_pw92)
        we_scan_os = (eps_c_scan * r_tot - eps_scan_a_m * ra_m - eps_scan_b_m * rb_m) * wt_m

        s2_average = 0.5 * (s2a[mask_ab] + s2b[mask_ab])
        u = (gamma_os * s2_average) / (1.0 + gamma_os * s2_average)
        ts = 0.5 * (tsa[mask_ab] + tsb[mask_ab])
        w_val = (-1.0 + ts) / (1.0 + ts)
        tau_ueg = total_tau_ueg_coeff * (r_tot ** (5.0 / 3.0))
        tau_w = g_tot / r_tot / 4.0
        beta_f = 2.0 * (t_tot - tau_w) / (t_tot + tau_ueg) - 1.0
        basis = expansion_basis_batch(u, w_val, beta_f)

        _accumulate_channel(corr_os, we_pw_os, basis)
        _accumulate_channel(corr_os_scan, we_scan_os, basis)


def accumulate_integrated_dv_block(weights, rho_a, rho_b, rho_a1, rho_b1, tau_a, tau_b, integrated_dv):
    """Accumulate one grid block into the full 96x180 integratedDV matrix.

    Args:
        weights: 1D quadrature weights for the block.
        rho_a: 1D alpha spin density.
        rho_b: 1D beta spin density.
        rho_a1: 2D alpha density-gradient array with shape (npoint, 3).
        rho_b1: 2D beta density-gradient array with shape (npoint, 3).
        tau_a: 1D alpha kinetic-energy density.
        tau_b: 1D beta kinetic-energy density.
        integrated_dv: 2D matrix (96, 180), updated in place.

    Returns:
        None. ``integrated_dv`` is modified in place.

    Notes:
        The 10 conceptual integratedDV groups (exchange/correlation channels)
        are documented in ``1_data_generation/README.md`` under
        "IntegratedDV Group Definitions".
    """
    block_mat = np.zeros((NELE_SERIES, 180), dtype=np.float64)

    # 0:72 -> exchange_a (later merged with exchange_b), 72:180 -> correlation terms.
    exchange_a = block_mat[:, : 2 * NSERIES_X]
    exchange_b = block_mat[:, 2 * NSERIES_X : 4 * NSERIES_X]
    accumulate_exchange_block(weights, rho_a, rho_a1, tau_a, exchange_a)
    accumulate_exchange_block(weights, rho_b, rho_b1, tau_b, exchange_b)
    exchange_a += exchange_b

    exchange_b.fill(0.0)
    corr_terms = block_mat[:, 2 * NSERIES_X :]
    accumulate_correlation_block(weights, rho_a, rho_b, rho_a1, rho_b1, tau_a, tau_b, corr_terms)

    integrated_dv += block_mat


def build_mol_and_dm(geometry, basis, charge, spin, xc, verbose):
    """Build molecule, run UKS(wb97xv), and return converged spin density matrices.

    Args:
        geometry: Atomic geometry string accepted by PySCF.
        basis: PySCF basis name.
        charge: Total molecular charge.
        spin: PySCF spin value, equal to N_alpha - N_beta.
        xc: Exchange-correlation functional name.
        verbose: PySCF verbosity level.

    Returns:
        Tuple ``(mol, dm_a, dm_b, mf)``:
        - mol: PySCF Mole object.
        - dm_a: Alpha spin density matrix.
        - dm_b: Beta spin density matrix.
        - mf: Converged UKS object.
    """
    mol = gto.M(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        verbose=verbose,
    )
    mf = dft.UKS(mol)
    mf.xc = xc
    # Match the requested energy-breakdown grid setup.
    mf.grids.atom_grid = XC_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = None
    mf.nlcgrids.atom_grid = NL_GRID
    mf.nlcgrids.prune = None
    mf.nlcgrids.radii_adjust = None
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("UKS did not converge.")
    print(f"SCF converged. E = {energy:.12f}")

    dm = mf.make_rdm1()
    if isinstance(dm, np.ndarray) and dm.ndim == 3:
        dm_a, dm_b = dm[0], dm[1]
    else:
        dm_a, dm_b = dm
    return mol, dm_a, dm_b, mf


def build_grid(mol, radial, angular):
    """Build an unpruned PySCF atom-centered grid.

    Args:
        mol: PySCF Mole object.
        radial: Number of radial shells.
        angular: Number of angular points.

    Returns:
        Tuple ``(coords, weights, grid_id)``:
        - coords: 2D array of Cartesian grid coordinates.
        - weights: 1D quadrature weights.
        - grid_id: Integer shorthand id computed as ``int(f"{radial}{angular:03d}")``.
    """
    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (radial, angular)
    grids.prune = None
    grids.radii_adjust = None
    grids.build(with_non0tab=False)
    grid_id = int(f"{radial}{angular:03d}")
    return grids.coords, grids.weights, grid_id


def unpack_mgga_rho(rho_mgga):
    """Extract rho, grad(rho), tau from PySCF MGGA tensor.

    Args:
        rho_mgga: PySCF MGGA density tensor from ``numint.eval_rho``.

    Returns:
        Tuple ``(rho, grad, tau)``:
        - rho: 1D density array.
        - grad: 2D gradient array with shape (npoint, 3).
        - tau: 1D kinetic-energy density array in Q-Chem-compatible convention.
    """
    if rho_mgga.ndim != 2 or rho_mgga.shape[0] < 5:
        raise ValueError(f"Unexpected MGGA rho shape: {rho_mgga.shape}")
    rho = rho_mgga[0]
    grad = np.column_stack((rho_mgga[1], rho_mgga[2], rho_mgga[3]))
    tau_idx = 5 if rho_mgga.shape[0] >= 6 else 4
    # PySCF uses tau = 1/2 * sum_i |grad psi_i|^2; Q-Chem-side formulas use sum_i |grad psi_i|^2.
    tau = 2.0 * rho_mgga[tau_idx]
    return rho, grad, tau


def write_matrix_block(fh, grid_id, mat):
    """Write one grid matrix with the exact reference text format.

    Args:
        fh: Open writable text file handle.
        grid_id: Integer grid identifier.
        mat: 2D array with shape (96, 180).

    Returns:
        None. Data is written to ``fh``.
    """
    fh.write(f"In DFTenergy, GrdTyp = {grid_id}\n")
    fh.write("integratedDV\n")
    for row in mat:
        fh.write(" ".join(f"{val:.12e}" for val in row))
        fh.write("\n")
    fh.write("\n")


def resolve_output_path():
    """Resolve OUTPUT_TXT relative to repository root.

    Args:
        None.

    Returns:
        Absolute ``Path`` pointing to the output text file.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / OUTPUT_TXT


def format_energy_breakdown(energy_breakdown):
    """Format the energy breakdown text block for console and file output."""
    lines = [
        "Energy breakdown (PySCF @ grid (99, 590)):",
        f" Total energy in the final basis set =      {energy_breakdown['total']: .10f}",
        f"Nuclear Repulsion Energy =       {energy_breakdown['nuc']: .12f} hartrees",
        "",
        f" Alpha HF Exchange         Energy =  {energy_breakdown['alpha_hf_x']: .14f}",
        f" Beta HF Exchange          Energy =  {energy_breakdown['beta_hf_x']: .14f}",
        f" Alpha LR HF Exchange      Energy =  {energy_breakdown['alpha_lr_hf_x']: .14f}",
        f" Beta LR HF Exchange       Energy =  {energy_breakdown['beta_lr_hf_x']: .14f}",
        f" Alpha SR HF Exchange (w)  Energy =  {energy_breakdown['alpha_sr_hf_x']: .14f}",
        f" Beta SR HF Exchange (w)   Energy =  {energy_breakdown['beta_sr_hf_x']: .14f}",
        f" DFT XC (X+C, excl. NLC)   Energy =  {energy_breakdown['dft_xc_total']: .14f}",
        f" DFT Non-Local Correlation Energy =  {energy_breakdown['dft_nlc']: .14f}",
        f" One-Electron (alpha)      Energy = {energy_breakdown['one_e_alpha']: .14f}",
        f" One-Electron (beta)       Energy = {energy_breakdown['one_e_beta']: .14f}",
        f" Total Coulomb             Energy =  {energy_breakdown['coul']: .14f}",
    ]
    return "\n".join(lines) + "\n"


def compute_pyscf_energy_breakdown(mf, dm_a, dm_b):
    """Compute energy terms from converged PySCF UKS results.

    Args:
        mf: Converged UKS object.
        dm_a: Alpha density matrix.
        dm_b: Beta density matrix.

    Returns:
        Dict with PySCF-generated energy terms. For composite `wb97xv`,
        `dft_exchange` and `dft_correlation` are not separable via LibXC API,
        so they are set to None and `dft_xc_total` is provided.
    """
    dm = np.asarray([dm_a, dm_b])
    ni = mf._numint
    veff = mf.get_veff(mf.mol, dm)

    alpha_hf_x = -0.5 * np.einsum("ij,ji", dm_a, veff.vk[0]).real
    beta_hf_x = -0.5 * np.einsum("ij,ji", dm_b, veff.vk[1]).real

    omega, alpha, _ = ni.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
    alpha_lr_hf_x = None
    beta_lr_hf_x = None
    alpha_sr_hf_x = None
    beta_sr_hf_x = None
    if omega != 0 and alpha != 0:
        vk_lr = mf.get_k(mf.mol, dm, hermi=1, omega=omega)
        alpha_lr_hf_x = -0.5 * alpha * np.einsum("ij,ji", dm_a, vk_lr[0]).real
        beta_lr_hf_x = -0.5 * alpha * np.einsum("ij,ji", dm_b, vk_lr[1]).real
        alpha_sr_hf_x = alpha_hf_x - alpha_lr_hf_x
        beta_sr_hf_x = beta_hf_x - beta_lr_hf_x

    _, dft_xc_total, _ = ni.nr_uks(mf.mol, mf.grids, mf.xc, dm, max_memory=mf.max_memory)
    dft_nlc = 0.0
    if mf.do_nlc():
        xc_nlc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
        _, dft_nlc, _ = ni.nr_nlc_vxc(mf.mol, mf.nlcgrids, xc_nlc, dm_a + dm_b, max_memory=mf.max_memory)

    return {
        "total": float(mf.e_tot),
        "nuc": float(mf.energy_nuc()),
        "alpha_hf_x": float(alpha_hf_x),
        "beta_hf_x": float(beta_hf_x),
        "omega": float(omega),
        "alpha_lr_hf_x": None if alpha_lr_hf_x is None else float(alpha_lr_hf_x),
        "beta_lr_hf_x": None if beta_lr_hf_x is None else float(beta_lr_hf_x),
        "alpha_sr_hf_x": None if alpha_sr_hf_x is None else float(alpha_sr_hf_x),
        "beta_sr_hf_x": None if beta_sr_hf_x is None else float(beta_sr_hf_x),
        "dft_exchange": None,
        "dft_correlation": None,
        "dft_xc_total": float(dft_xc_total),
        "dft_nlc": float(dft_nlc),
        "one_e_alpha": float(mf.scf_summary["e1"] / 2.0),
        "one_e_beta": float(mf.scf_summary["e1"] / 2.0),
        "coul": float(mf.scf_summary["coul"]),
    }

def generate_species_output(job, output_path, xc, verbose):
    """Run the PySCF generation flow for one species and write one text output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {job['name']} with basis={job['basis']}, charge={job['charge']}, spin={job['spin']}, xc={xc}")

    mol, dm_a, dm_b, mf = build_mol_and_dm(
        geometry=job["geometry"],
        basis=job["basis"],
        charge=job["charge"],
        spin=job["spin"],
        xc=xc,
        verbose=verbose,
    )

    energy_breakdown = compute_pyscf_energy_breakdown(mf, dm_a, dm_b)
    energy_breakdown_text = format_energy_breakdown(energy_breakdown)
    print(energy_breakdown_text, end="")
    ni = dft.numint.NumInt()

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(energy_breakdown_text)
        fh.write("\n")
        for radial, angular in GRID_SETUPS:
            coords, weights, grid_id = build_grid(mol, radial, angular)
            npts = weights.size
            integrated_dv = np.zeros((NELE_SERIES, 180), dtype=np.float64)
            print(f"Grid {grid_id}: {npts} points")

            # Blocked processing keeps memory bounded for large grids.
            for i0 in range(0, npts, BLOCK_SIZE):
                i1 = min(i0 + BLOCK_SIZE, npts)
                coords_blk = coords[i0:i1]
                weights_blk = weights[i0:i1]

                ao = ni.eval_ao(mol, coords_blk, deriv=1)
                rho_a_mgga = ni.eval_rho(mol, ao, dm_a, xctype="MGGA", with_lapl=False)
                rho_b_mgga = ni.eval_rho(mol, ao, dm_b, xctype="MGGA", with_lapl=False)

                rho_a, rho_a1, tau_a = unpack_mgga_rho(rho_a_mgga)
                rho_b, rho_b1, tau_b = unpack_mgga_rho(rho_b_mgga)

                accumulate_integrated_dv_block(
                    weights_blk, rho_a, rho_b, rho_a1, rho_b1, tau_a, tau_b, integrated_dv
                )

                if i0 == 0 or i1 == npts or (i0 // BLOCK_SIZE) % 5 == 0:
                    print(f"  Processed points {i0}:{i1} / {npts}")

            if np.isnan(integrated_dv).any() or np.isinf(integrated_dv).any():
                raise FloatingPointError(f"NaN/Inf detected in integratedDV for grid {grid_id}")

            write_matrix_block(fh, grid_id, integrated_dv)
            print(f"Wrote matrix for grid {grid_id}")

    print(f"Done. Output written to: {output_path}")


def build_jobs(args):
    """Resolve CLI arguments into concrete generation jobs and output paths."""
    repo_root = Path(__file__).resolve().parents[1]
    if args.xyz and args.xyz_dir:
        raise ValueError("Use either --xyz or --xyz-dir, not both")
    if args.output_txt and args.output_dir:
        raise ValueError("Use either --output-txt or --output-dir, not both")

    if args.xyz_dir:
        xyz_dir = Path(args.xyz_dir).resolve()
        xyz_paths = sorted(xyz_dir.glob("*.xyz"))
        if not xyz_paths:
            raise ValueError(f"No .xyz files found in {xyz_dir}")
        output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "1_data_generation/pyscf_outputs")
        return [
            (
                load_xyz_job(path, args.basis, args.use_xyz_basis),
                output_dir / f"{path.stem}.txt",
            )
            for path in xyz_paths
        ]

    if args.xyz:
        xyz_path = Path(args.xyz).resolve()
        job = load_xyz_job(xyz_path, args.basis, args.use_xyz_basis)
        if args.output_dir:
            output_path = Path(args.output_dir).resolve() / f"{xyz_path.stem}.txt"
        elif args.output_txt:
            output_path = Path(args.output_txt).resolve()
        else:
            output_path = resolve_output_path()
        return [(job, output_path)]

    output_path = Path(args.output_txt).resolve() if args.output_txt else resolve_output_path()
    return [(build_default_job(), output_path)]


def main(argv=None):
    """Run full integratedDV generation and write matrices for all configured grids."""
    args = build_parser().parse_args(argv)
    jobs = build_jobs(args)
    for job, output_path in jobs:
        generate_species_output(job, output_path, xc=args.xc, verbose=args.verbose)


if __name__ == "__main__":
    main()

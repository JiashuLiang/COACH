from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dftd4.interface import DampingParam, DispersionModel
from pyscf import dft, gto, lib
from pyscf.gto import basis as basis_module

try:
    from FunctionalCOACH import coach_cos_d1, coach_css_d1, coach_x_d1
except ImportError:
    import coach_cos_d1
    import coach_css_d1
    import coach_x_d1


OMEGA = 0.27
CX_SR_HF = 0.22878980716640696
CX_LR_HF = 1.0
VV10_B = 5.5
VV10_C = 0.01
D4_PARAMS = dict(s6=0.0, s8=0.0, s9=1.0, a1=0.215, a2=5.8, alp=16.0)

def load_xyz_job(xyz_path: str | Path) -> dict[str, object]:
    xyz_path = Path(xyz_path).resolve()
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{xyz_path}: expected at least 3 lines")

    natom = int(lines[0].strip())
    metadata = {}
    for field in lines[1].strip().split(","):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        metadata[key.strip().lower()] = value.strip()
    atom_lines = tuple(line.strip() for line in lines[2 : 2 + natom] if line.strip())
    if len(atom_lines) != natom:
        raise ValueError(f"{xyz_path}: expected {natom} atoms, found {len(atom_lines)}")

    charge = int(metadata.get("charge", "0"))
    multiplicity = int(metadata.get("multiplicity", "1"))
    if multiplicity < 1:
        raise ValueError(f"{xyz_path}: multiplicity must be >= 1")

    basis = metadata["basis"]
    xc_grid = metadata.get("xc_grid", "000099000590")
    xc_grid_digits = xc_grid.strip()
    if len(xc_grid_digits) != 12 or not xc_grid_digits.isdigit():
        raise ValueError(f"Unsupported xc_grid format: {xc_grid!r}")
    max_cycle = int(metadata.get("max_scf_cycles", metadata.get("max_cycle", "200")))
    conv_tol = float(metadata.get("pyscf_conv_tol", metadata.get("conv_tol", "1e-7")))
    d4_only = metadata.get("coach_check", "scf").lower() == "d4_only"
    if basis.upper().startswith("AUG-CC-PC"):
        symbols = sorted({line.split()[0] for line in atom_lines})
        basis = {symbol: basis_module.load(basis, symbol) for symbol in symbols}

    return {
        "name": xyz_path.stem,
        "xyz_path": xyz_path,
        "atom": "\n".join(atom_lines),
        "atom_lines": atom_lines,
        "charge": charge,
        "spin": multiplicity - 1,
        "basis": basis,
        "atom_grid": (int(xc_grid_digits[:6]), int(xc_grid_digits[6:])),
        "conv_tol": conv_tol,
        "max_cycle": max_cycle,
        "d4_only": d4_only,
    }


def _prepare_spin_inputs(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    den = rho[0]
    grad = rho[1:4]
    sigma = np.einsum("xg,xg->g", grad, grad)
    tau_raw = 2.0 * rho[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_w = np.divide(sigma, 4.0 * den, out=np.zeros_like(sigma), where=den != 0.0)
    tau = np.maximum(tau_raw, tau_w)
    return den, sigma, tau


def evaluate_coach_terms(rho_a: np.ndarray, rho_b: np.ndarray):
    ra, gaa, ta = _prepare_spin_inputs(rho_a)
    rb, gbb, tb = _prepare_spin_inputs(rho_b)
    gab = np.einsum("xg,xg->g", rho_a[1:4], rho_b[1:4])

    x_a = coach_x_d1.evaluate(ra, OMEGA, gaa, ta)
    x_b = coach_x_d1.evaluate(rb, OMEGA, gbb, tb)
    css_a = coach_css_d1.evaluate(ra, gaa, ta)
    css_b = coach_css_d1.evaluate(rb, gbb, tb)
    cos = coach_cos_d1.evaluate(ra, rb, gaa, gab, gbb, ta, tb)
    return x_a, x_b, css_a, css_b, cos


def eval_coach_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    if deriv > 1:
        raise NotImplementedError("Only exc and vxc are implemented.")

    if spin == 0:
        rho_tot = rho
        rho_a = np.array(rho_tot, copy=True)
        rho_b = np.array(rho_tot, copy=True)
        rho_a[0:4] *= 0.5
        rho_b[0:4] *= 0.5
        rho_a[-1] *= 0.5
        rho_b[-1] *= 0.5
    elif spin == 1:
        rho_a, rho_b = rho
    else:
        raise NotImplementedError("Unsupported spin flag.")

    x_a, x_b, css_a, css_b, cos = evaluate_coach_terms(rho_a, rho_b)
    ex_a, vrho_x_a, vsigma_x_a, vtau_x_a = x_a
    ex_b, vrho_x_b, vsigma_x_b, vtau_x_b = x_b
    ec_ss_a, vrho_css_a, vsigma_css_a, vtau_css_a = css_a
    ec_ss_b, vrho_css_b, vsigma_css_b, vtau_css_b = css_b
    ec_os, vrho_os_a, vrho_os_b, vsigma_os_aa, vsigma_os_ab, vsigma_os_bb, vtau_os_a, vtau_os_b = cos

    f = ex_a + ex_b + ec_ss_a + ec_ss_b + ec_os
    if spin == 0:
        vrho = 0.5 * (vrho_x_a + vrho_css_a + vrho_os_a + vrho_x_b + vrho_css_b + vrho_os_b)
        vsigma = 0.25 * (vsigma_x_a + vsigma_css_a + vsigma_os_aa + vsigma_os_ab + vsigma_x_b + vsigma_css_b + vsigma_os_bb)
        vtau = vtau_x_a + vtau_css_a + vtau_os_a + vtau_x_b + vtau_css_b + vtau_os_b
        rho_scalar = rho_tot[0]
        exc = np.divide(f, rho_scalar, out=np.zeros_like(f), where=rho_scalar != 0.0)
        return exc, (vrho, vsigma, None, vtau), None, None

    vrho = np.stack([vrho_x_a + vrho_css_a + vrho_os_a, vrho_x_b + vrho_css_b + vrho_os_b], axis=1)
    vsigma = np.stack(
        [
            vsigma_x_a + vsigma_css_a + vsigma_os_aa,
            vsigma_os_ab,
            vsigma_x_b + vsigma_css_b + vsigma_os_bb,
        ],
        axis=1,
    )
    vtau = 2.0 * np.stack([vtau_x_a + vtau_css_a + vtau_os_a, vtau_x_b + vtau_css_b + vtau_os_b], axis=1)

    rho_scalar = rho_a[0] + rho_b[0]
    exc = np.divide(f, rho_scalar, out=np.zeros_like(f), where=rho_scalar != 0.0)
    return exc, (vrho, vsigma, None, vtau), None, None


def build_coach_mf(xyz_path: str | Path, verbose: int = 3, restricted: bool = False):
    job = load_xyz_job(xyz_path)
    if restricted and job["spin"] != 0:
        raise ValueError("Restricted COACH is only available for closed-shell inputs.")
    mol = gto.M(
        atom=job["atom"],
        charge=job["charge"],
        spin=job["spin"],
        basis=job["basis"],
        unit="Angstrom",
        verbose=verbose,
    )
    mf = dft.RKS(mol) if restricted else dft.UKS(mol)
    mf.conv_tol = job["conv_tol"]
    mf.max_cycle = job["max_cycle"]
    mf.grids.atom_grid = job["atom_grid"]
    mf.grids.prune = None
    mf.nlc = "vv10"
    mf.nlcgrids.atom_grid = (50, 194)
    mf.nlcgrids.prune = dft.gen_grid.sg1_prune
    mf = mf.define_xc_(
        eval_coach_xc,
        "MGGA",
        rsh=(OMEGA, CX_LR_HF, CX_SR_HF - CX_LR_HF),
    )
    mf.xc = f"RSH({OMEGA},{CX_LR_HF},{CX_SR_HF - CX_LR_HF})"
    mf._numint.nlc_coeff = lambda xc_code: (((VV10_B, VV10_C), 1.0),)
    return job, mf


def d4_atm_energy(mol: gto.Mole) -> float:
    model = DispersionModel(
        numbers=np.asarray(mol.atom_charges(), dtype=int),
        positions=np.asarray(mol.atom_coords(), dtype=float),
        charge=float(mol.charge),
    )
    params = DampingParam(**D4_PARAMS)
    return float(model.get_dispersion(params, grad=False)["energy"])


def run_coach_job(xyz_path: str | Path, verbose: int = 0, restricted: bool = False) -> dict[str, object]:
    job, mf = build_coach_mf(xyz_path, verbose=verbose, restricted=restricted)
    if job["d4_only"]:
        d4_energy = d4_atm_energy(mf.mol)
        return {
            "job": job,
            "mf": mf,
            "scf_energy": None,
            "d4_energy": d4_energy,
            "total_energy": d4_energy,
        }

    scf_energy = float(mf.kernel())
    d4_energy = d4_atm_energy(mf.mol)
    return {
        "job": job,
        "mf": mf,
        "scf_energy": scf_energy,
        "d4_energy": d4_energy,
        "total_energy": scf_energy + d4_energy,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the COACH PySCF workflow for one XYZ input.")
    parser.add_argument("xyz", type=Path, help="XYZ file with COACH metadata in the comment line.")
    parser.add_argument("--rks", action="store_true", help="Run restricted KS for closed-shell inputs.")
    parser.add_argument("--verbose", type=int, default=3, help="PySCF verbosity level.")
    args = parser.parse_args()

    result = run_coach_job(args.xyz, verbose=args.verbose, restricted=args.rks)
    job = result["job"]

    print(f"\n=== {job['name']} ===")
    if job["d4_only"]:
        print(f"D4-ATM correction: {result['d4_energy']: .10f} Ha")
        return

    print(f"PySCF SCF energy (semilocal + HF + VV10): {result['scf_energy']: .10f} Ha")
    print(f"D4-ATM correction:                        {result['d4_energy']: .10f} Ha")
    print(f"PySCF + D4 total:                       {result['total_energy']: .10f} Ha")


if __name__ == "__main__":
    main()

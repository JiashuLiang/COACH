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

def parse_xyz_metadata(comment_line: str) -> dict[str, str]:
    metadata = {}
    for field in comment_line.split(","):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def parse_xc_grid(spec: str) -> tuple[int, int]:
    digits = spec.strip()
    if len(digits) != 12 or not digits.isdigit():
        raise ValueError(f"Unsupported xc_grid format: {spec!r}")
    return int(digits[:6]), int(digits[6:])


def resolve_basis(atom_lines: tuple[str, ...], basis_name: str):
    if basis_name.upper().startswith("AUG-CC-PC"):
        symbols = sorted({line.split()[0] for line in atom_lines})
        return {symbol: basis_module.load(basis_name, symbol) for symbol in symbols}
    return basis_name


def load_xyz_job(xyz_path: str | Path) -> dict[str, object]:
    xyz_path = Path(xyz_path).resolve()
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{xyz_path}: expected at least 3 lines")

    natom = int(lines[0].strip())
    metadata = parse_xyz_metadata(lines[1].strip())
    atom_lines = tuple(line.strip() for line in lines[2 : 2 + natom] if line.strip())
    if len(atom_lines) != natom:
        raise ValueError(f"{xyz_path}: expected {natom} atoms, found {len(atom_lines)}")

    charge = int(metadata.get("charge", "0"))
    multiplicity = int(metadata.get("multiplicity", "1"))
    if multiplicity < 1:
        raise ValueError(f"{xyz_path}: multiplicity must be >= 1")

    basis = metadata["basis"]
    xc_grid = metadata.get("xc_grid", "000099000590")
    max_cycle = int(metadata.get("max_scf_cycles", metadata.get("max_cycle", "200")))
    conv_tol = float(metadata.get("pyscf_conv_tol", metadata.get("conv_tol", "1e-7")))
    d4_only = metadata.get("coach_check", "scf").lower() == "d4_only"

    return {
        "name": xyz_path.stem,
        "xyz_path": xyz_path,
        "atom": "\n".join(atom_lines),
        "atom_lines": atom_lines,
        "charge": charge,
        "spin": multiplicity - 1,
        "basis": basis,
        "atom_grid": parse_xc_grid(xc_grid),
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


def evaluate_coach_terms(rho_a: np.ndarray, rho_b: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    ra, gaa, ta = _prepare_spin_inputs(rho_a)
    rb, gbb, tb = _prepare_spin_inputs(rho_b)
    gab = np.einsum("xg,xg->g", rho_a[1:4], rho_b[1:4])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        x_a = coach_x_d1.alpha_d1(ra, OMEGA, gaa, ta)
        x_b = coach_x_d1.beta_d1(rb, OMEGA, gbb, tb)
        css_a = coach_css_d1.alpha_d1(ra, gaa, ta)
        css_b = coach_css_d1.beta_d1(rb, gbb, tb)
        cos = coach_cos_d1.os_d1(ra, rb, gaa, gab, gbb, ta, tb)
    return {"x_a": x_a, "x_b": x_b, "css_a": css_a, "css_b": css_b, "cos": cos}


def eval_coach_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    if spin != 1:
        raise NotImplementedError("This COACH wrapper is written for UKS only.")
    if deriv > 1:
        raise NotImplementedError("Only exc and vxc are implemented.")

    rho_a, rho_b = rho
    terms = evaluate_coach_terms(rho_a, rho_b)

    x_a = terms["x_a"]
    x_b = terms["x_b"]
    css_a = terms["css_a"]
    css_b = terms["css_b"]
    cos = terms["cos"]

    f = x_a["Ex"] + x_b["Ex"] + css_a["Ex"] + css_b["Ex"] + cos["Ec"]
    vrho = np.stack([x_a["V_RA"] + css_a["V_RA"] + cos["V_RA"], x_b["V_RB"] + css_b["V_RB"] + cos["V_RB"]], axis=1)
    vsigma = np.stack(
        [
            x_a["V_GAA"] + css_a["V_GAA"] + cos["V_GAA"],
            cos["V_GAB"],
            x_b["V_GBB"] + css_b["V_GBB"] + cos["V_GBB"],
        ],
        axis=1,
    )
    vtau = 2.0 * np.stack([x_a["V_TA"] + css_a["V_TA"] + cos["V_TA"], x_b["V_TB"] + css_b["V_TB"] + cos["V_TB"]], axis=1)

    rho_tot = rho_a[0] + rho_b[0]
    exc = np.divide(f, rho_tot, out=np.zeros_like(f), where=rho_tot != 0.0)
    return exc, (vrho, vsigma, None, vtau), None, None


def build_coach_mf(xyz_path: str | Path, verbose: int = 3):
    job = load_xyz_job(xyz_path)
    mol = gto.M(
        atom=job["atom"],
        charge=job["charge"],
        spin=job["spin"],
        basis=resolve_basis(job["atom_lines"], job["basis"]),
        unit="Angstrom",
        verbose=verbose,
    )
    mf = dft.UKS(mol)
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


def run_coach_job(xyz_path: str | Path, verbose: int = 0) -> dict[str, object]:
    job = load_xyz_job(xyz_path)
    if job["d4_only"]:
        mol = gto.M(atom=job["atom"], charge=job["charge"], spin=job["spin"], unit="Angstrom", verbose=0)
        d4_energy = d4_atm_energy(mol)
        return {
            "job": job,
            "mf": None,
            "scf_energy": None,
            "d4_energy": d4_energy,
            "total_energy": d4_energy,
        }

    _, mf = build_coach_mf(xyz_path, verbose=verbose)
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
    parser.add_argument("--verbose", type=int, default=3, help="PySCF verbosity level.")
    args = parser.parse_args()

    result = run_coach_job(args.xyz, verbose=args.verbose)
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

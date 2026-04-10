from __future__ import annotations

from pathlib import Path

import numpy as np
from pyscf import dft, lib

from FunctionalCOACH.coach_pyscf import evaluate_coach_terms, run_coach_job


ROOT = Path(__file__).resolve().parent
XYZ_DIR = ROOT / "xyzfiles"
QCHEM_DIR = ROOT / "qchem_outputs"

QCHEM_BREAKDOWN_LABELS = {
    "Alpha HF Exchange Energy": "alpha_hf_x",
    "Beta HF Exchange Energy": "beta_hf_x",
    "DFT Correlation Energy": "dft_correlation",
    "DFT Exchange Energy": "dft_exchange",
    "DFT Non-Local Correlation Energy": "dft_nlc",
    "One-Electron (alpha) Energy": "one_e_alpha",
    "One-Electron (beta) Energy": "one_e_beta",
    "Total Coulomb Energy": "coul",
    "SCF energy in the final basis set": "total",
    "Nuclear Repulsion Energy": "nuc",
}


def load_qchem_reference(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        if "-D4 energy including 3-body term" in line:
            out["d4"] = float(line.split("=", 1)[1].split()[0])
            continue
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        key = QCHEM_BREAKDOWN_LABELS.get(" ".join(lhs.split()))
        if key is not None:
            out[key] = float(rhs.strip().split()[0].replace("D", "E").replace("d", "e"))
    return out


def compute_energy_breakdown(mf) -> dict[str, float]:
    dm = np.asarray(mf.make_rdm1())
    dm_a, dm_b = dm
    ni = mf._numint
    mol = mf.mol

    hcore = mf.get_hcore(mol)
    one_e_alpha = float(np.einsum("ij,ji", hcore, dm_a).real)
    one_e_beta = float(np.einsum("ij,ji", hcore, dm_b).real)

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if omega == 0:
        vk = mf.get_k(mol, dm, hermi=1)
        vk *= hyb
    elif alpha == 0:
        vk = mf.get_k(mol, dm, hermi=1, omega=-omega)
        vk *= hyb
    elif hyb == 0:
        vk = mf.get_k(mol, dm, hermi=1, omega=omega)
        vk *= alpha
    else:
        vk = mf.get_k(mol, dm, hermi=1)
        vk *= hyb
        vklr = mf.get_k(mol, dm, hermi=1, omega=omega)
        vklr *= alpha - hyb
        vk += vklr

    alpha_hf_x = float(-0.5 * np.einsum("ij,ji", dm_a, vk[0]).real)
    beta_hf_x = float(-0.5 * np.einsum("ij,ji", dm_b, vk[1]).real)
    coul = float(mf.scf_summary["coul"])

    dma, dmb = dft.numint._format_uks_dm(dm)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, 1, False, mf.grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, 1, False, mf.grids)[0]
    if nset != 1:
        raise NotImplementedError("Only single-density decomposition is implemented.")

    dft_exchange = 0.0
    dft_correlation = 0.0
    max_memory = mf.max_memory - lib.current_memory()[0]
    for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, 1, max_memory=max_memory):
        rho_a = make_rhoa(0, ao, mask, "MGGA")
        rho_b = make_rhob(0, ao, mask, "MGGA")
        x_a, x_b, css_a, css_b, cos = evaluate_coach_terms(rho_a, rho_b)
        dft_exchange += float(np.dot(weight, x_a[0] + x_b[0]))
        dft_correlation += float(np.dot(weight, css_a[0] + css_b[0] + cos[0]))

    dft_nlc = 0.0
    if mf.nlc:
        xc_nlc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
        _, dft_nlc, _ = ni.nr_nlc_vxc(mol, mf.nlcgrids, xc_nlc, dm_a + dm_b, max_memory=mf.max_memory)
        dft_nlc = float(dft_nlc)

    return {
        "total": float(mf.e_tot),
        "alpha_hf_x": alpha_hf_x,
        "beta_hf_x": beta_hf_x,
        "dft_exchange": dft_exchange,
        "dft_correlation": dft_correlation,
        "dft_nlc": dft_nlc,
        "one_e_alpha": one_e_alpha,
        "one_e_beta": one_e_beta,
        "coul": coul,
        "nuc": float(mf.energy_nuc()),
    }


def test_sie4x4_h2o_matches_qchem():
    xyz_path = XYZ_DIR / "SIE4x4_h2o.xyz"
    qchem_path = QCHEM_DIR / "SIE4x4_h2o.out"
    result = run_coach_job(xyz_path, verbose=0)
    ref = load_qchem_reference(qchem_path)

    assert abs(result["total_energy"] - ref["total"]) < 2.0e-6
    breakdown = compute_energy_breakdown(result["mf"])
    assert abs(breakdown["alpha_hf_x"] - ref["alpha_hf_x"]) < 5.0e-6
    assert abs(breakdown["beta_hf_x"] - ref["beta_hf_x"]) < 5.0e-6
    assert abs(breakdown["dft_exchange"] - ref["dft_exchange"]) < 5.0e-6
    assert abs(breakdown["dft_correlation"] - ref["dft_correlation"]) < 5.0e-6
    assert abs(breakdown["dft_nlc"] - ref["dft_nlc"]) < 5.0e-6
    assert abs(breakdown["coul"] - ref["coul"]) < 5.0e-5


def test_sie4x4_h2o_rks_matches_qchem():
    xyz_path = XYZ_DIR / "SIE4x4_h2o.xyz"
    qchem_path = QCHEM_DIR / "SIE4x4_h2o.out"
    result = run_coach_job(xyz_path, verbose=0, restricted=True)
    ref = load_qchem_reference(qchem_path)

    assert abs(result["total_energy"] - ref["total"]) < 2.0e-6


def test_16_c_ae18_matches_qchem():
    xyz_path = XYZ_DIR / "16_C_AE18.xyz"
    qchem_path = QCHEM_DIR / "16_C_AE18.out"
    result = run_coach_job(xyz_path, verbose=0)
    ref = load_qchem_reference(qchem_path)

    assert abs(result["total_energy"] - ref["total"]) < 2.0e-6
    breakdown = compute_energy_breakdown(result["mf"])
    assert abs(breakdown["alpha_hf_x"] - ref["alpha_hf_x"]) < 5.0e-6
    assert abs(breakdown["beta_hf_x"] - ref["beta_hf_x"]) < 5.0e-6
    assert abs(breakdown["dft_exchange"] - ref["dft_exchange"]) < 5.0e-6
    assert abs(breakdown["dft_correlation"] - ref["dft_correlation"]) < 5.0e-6
    assert abs(breakdown["dft_nlc"] - ref["dft_nlc"]) < 5.0e-6
    assert abs(breakdown["coul"] - ref["coul"]) < 5.0e-5


def test_l14_2a_d4_matches_qchem():
    xyz_path = XYZ_DIR / "L14_2a.xyz"
    qchem_path = QCHEM_DIR / "L14_2a.out"
    result = run_coach_job(xyz_path, verbose=0)
    ref = load_qchem_reference(qchem_path)

    assert result["job"]["d4_only"]
    assert abs(result["d4_energy"] - ref["d4"]) < 1.0e-10

#!/home/jsliang/.venv/bin/python
from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from pyscf import dft, gto
from pyscf.gto import basis as basis_module
from dftd4.interface import DampingParam, DispersionModel


OMEGA = 0.27
CX_SR_HF = 0.22878980716640696
CX_LR_HF = 1.0
VV10_B = 5.5
VV10_C = 0.01
D4_PARAMS = dict(s6=0.0, s8=0.0, s9=1.0, a1=0.215, a2=5.8, alp=16.0)
TOL = 1.0e-12


@dataclass(frozen=True)
class Case:
    name: str
    atom: str
    charge: int
    spin: int
    basis: str
    atom_grid: tuple[int, int]
    conv_tol: float
    max_cycle: int
    reference_energy: float
    reference_d4: float
    d4_only: bool = False


CASES = {
    "16_C_AE18": Case(
        name="16_C_AE18",
        atom="C 0.0000000000 0.0000000000 0.0000000000",
        charge=0,
        spin=2,
        basis="augccpcv5z",
        atom_grid=(500, 974),
        conv_tol=1.0e-8,
        max_cycle=200,
        reference_energy=-37.8454603141,
        reference_d4=0.0,
    ),
    "L14_2a": Case(
        name="L14_2a",
        atom="""
C -0.144956 3.27465 -0.631433
C -1.3036 3.05772 -1.37824
C -2.48399 2.856 -0.662715
C -2.49717 2.82548 0.742857
C -1.32996 2.99323 1.4882
C -0.157622 3.242 0.773277
C -3.92845 2.61765 -1.11888
C -4.23882 1.17442 -0.715348
C -4.25313 1.14508 0.691795
C -3.95026 2.56969 1.16119
C -4.34602 0.00957863 -1.47349
C -4.45383 -1.18699 -0.765815
C -4.46973 -1.21599 0.640013
C -4.37687 -0.0498705 1.39878
C -4.67247 3.36573 0.0303717
C 1.26488 3.70065 -1.04697
C 1.24481 3.64631 1.23324
C 2.24449 2.58016 0.77859
C 2.25699 2.61391 -0.626393
C 3.129 1.77546 1.4977
C 4.058 1.04335 0.757234
C 4.0708 1.07757 -0.647357
C 3.15461 1.84485 -1.36783
C 1.54104 4.71035 0.12038
C 5.22804 0.180989 -1.09861
C 4.84791 -1.24632 -0.699519
C 4.83429 -1.28096 0.708636
C 5.20672 0.124642 1.1849
C 4.52448 -2.37025 -1.44444
C 4.18089 -3.5458 -0.755673
C 4.16684 -3.58001 0.63849
C 4.49643 -2.43992 1.39084
C 6.22992 0.490001 0.0608929
C -4.40151 -2.68962 1.04828
C -2.9982 -3.12914 0.61869
C -2.98152 -3.10006 -0.788422
C -4.37469 -2.64263 -1.23261
C -1.85173 -3.42884 1.33766
C -0.674652 -3.72059 0.626961
C -0.658035 -3.69181 -0.765609
C -1.81794 -3.37034 -1.49152
C -5.22652 -3.31832 -0.115468
H -1.29804 3.0913 -2.46506
H -1.34427 2.97793 2.57537
H -4.15945 2.89316 -2.14791
H -4.20057 2.80221 2.19629
H -4.29572 0.0301384 -2.55913
H -4.34927 -0.0743018 2.48511
H -4.44784 4.43628 0.0549276
H -5.7529 3.19226 0.0164583
H 1.36832 4.0675 -2.06892
H 1.3305 3.96383 2.27319
H 3.13292 1.75974 2.58482
H 3.17792 1.88214 -2.45416
H 0.835088 5.54666 0.134187
H 2.57566 5.06721 0.137959
H 5.56939 0.318553 -2.12449
H 5.52902 0.210975 2.22248
H 4.52615 -2.34567 -2.53072
H 3.92763 -4.44062 -1.31756
H 3.90275 -4.50118 1.15054
H 4.47685 -2.46884 2.47689
H 6.53271 1.54096 0.0897446
H 7.10292 -0.170004 0.052685
H -4.69164 -2.92552 2.07229
H -4.64037 -2.83628 -2.27202
H -1.85516 -3.43955 2.42477
H 0.235481 -3.96304 1.16865
H 0.264995 -3.91188 -1.29481
H -1.79526 -3.33603 -2.57793
H -6.26843 -2.98496 -0.120913
H -5.16504 -4.41055 -0.137193
C 0.239321 -0.447581 -1.41575
C 1.41734 -0.814101 -0.680882
C 1.40268 -0.84375 0.674305
C 0.208809 -0.510436 1.39941
C -0.985926 -0.208328 0.663194
C -0.971053 -0.17798 -0.692573
C 0.270837 -0.358367 -2.80773
C 1.46795 -0.590987 -3.53403
N 2.46487 -0.775335 -4.10746
C -0.888129 -0.00941652 -3.54867
N -1.85146 0.279961 -4.13587
C 0.210801 -0.484643 2.79431
C 1.39285 -0.750179 3.53381
N 2.37847 -0.961306 4.11742
C -0.964405 -0.172232 3.5259
N -1.94089 0.0877713 4.10507
H 2.31861 -1.07674 -1.2197
H 2.29227 -1.12989 1.2204
H -1.90127 0.00218481 1.20005
H -1.87449 0.0569473 -1.23938
""".strip(),
        charge=0,
        spin=0,
        basis="def2-TZVPPD",
        atom_grid=(99, 590),
        conv_tol=1.0e-6,
        max_cycle=100,
        reference_energy=-2296.93857494,
        reference_d4=0.0049518824,
        d4_only=True,
    ),
    "SIE4x4_h2o": Case(
        name="SIE4x4_h2o",
        atom="""
O 0.00000000 0.00000000 0.39393904
H -0.75503878 0.00000000 -0.19696952
H 0.75503878 0.00000000 -0.19696952
""".strip(),
        charge=0,
        spin=0,
        basis="def2-QZVPPD",
        atom_grid=(99, 590),
        conv_tol=1.0e-7,
        max_cycle=200,
        reference_energy=-76.4335179439,
        reference_d4=0.0,
    ),
}


POSITION_MAP = {
    "POS_RA": "V_RA",
    "POS_RB": "V_RB",
    "POS_GAA": "V_GAA",
    "POS_GAB": "V_GAB",
    "POS_GBB": "V_GBB",
    "POS_TA": "V_TA",
    "POS_TB": "V_TB",
}


def _max(a, b):
    return np.maximum(a, b)


def _translate_c_line(line: str) -> str | None:
    line = line.strip()
    if not line or line in {"{", "}"} or line.startswith("if "):
        return None
    if line.startswith("double "):
        line = line[7:]
    line = line.rstrip(";")
    line = line.replace("std::pow(", "np.power(")
    line = line.replace("std::max(", "_max(")
    line = line.replace("std::erf(", "erf(")
    line = line.replace("std::sqrt(", "np.sqrt(")
    line = line.replace("std::log(", "np.log(")
    line = line.replace("M_PI", "np.pi")
    line = line.replace("M_E", "np.e")
    line = re.sub(r"double\(([^()]+)\)", r"\1", line)
    line = line.replace("F[i]", "F")
    for pos, repl in POSITION_MAP.items():
        line = line.replace(f"D1F[{pos}][i]", repl)
    return line


def _extract_between(lines: list[str], start_marker: str, end_marker: str) -> list[str]:
    start = next(i for i, line in enumerate(lines) if start_marker in line) + 1
    end = next(i for i, line in enumerate(lines[start:], start) if end_marker in line)
    return lines[start:end]


def _compile_c_section(path: pathlib.Path, prelude_start: str, energy_end: str, d1_start: str, d1_end: str):
    lines = path.read_text().splitlines()
    loop_idx = next(i for i, line in enumerate(lines) if "for(size_t i = 0; i < NGrid; i++) {" in line)
    constants = [
        _translate_c_line(line)
        for line in lines[:loop_idx]
        if line.strip().startswith("double ") and line.strip().endswith(";")
    ]
    constants = "\n".join(line for line in constants if line)

    prelude = _extract_between(lines, prelude_start, energy_end)
    prelude = "\n".join(line for line in (_translate_c_line(x) for x in prelude) if line)

    d1 = _extract_between(lines, d1_start, d1_end)
    d1 = [_translate_c_line(line) for line in d1]
    d1 = "\n".join(line for line in d1 if line)

    const_env = {"np": np, "erf": erf, "_max": _max}
    exec(compile(constants, str(path), "exec"), const_env)
    const_env = {k: v for k, v in const_env.items() if k not in {"np", "erf", "_max", "__builtins__"}}
    return const_env, compile(prelude, f"{path.name}:prelude", "exec"), compile(prelude + "\n" + d1, f"{path.name}:d1", "exec")


class CoachSemilocal:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.x_consts, self.x_prelude, self.x_d1 = _compile_c_section(
            root / "coach_x.C",
            "if ( RA > Tol && TA > Tol && RA > 1e-112) {",
            "F[i] += Ex;",
            "if ((ssquare > 1e-180 && RA > 1e-81 && TA > 1e-81) && (D1F != NULL)) {",
            "D1F[POS_TA][i] += I69;",
        )
        self.css_consts, self.css_prelude, self.css_d1 = _compile_c_section(
            root / "coach_css.C",
            "if ( RA > Tol && TA > Tol && RA > 1e-112) {",
            "F[i] += Ex;",
            "if ((RA > 1e-81) && (D1F != NULL)) {",
            "D1F[POS_TA][i] += I90;",
        )
        self.cos_consts, self.cos_prelude, self.cos_d1 = _compile_c_section(
            root / "coach_cos.C",
            "if ( RA > Tol && RB > Tol && TA> Tol && TB> Tol && RA > 1e-128 && RB > 1e-128) {",
            "F[i] += Ec;",
            "if ((1.0+zeta > 0.0 && 1.0-zeta > 0.0 && RA > 1e-90 && RB > 1e-90) && (D1F != NULL)) {",
            "D1F[POS_TB][i] += I212;",
        )

    @staticmethod
    def _safe_tau(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        den = np.maximum(rho[0], 0.0)
        grad = rho[1:4]
        tau_raw = rho[-1]
        sigma = np.einsum("xg,xg->g", grad, grad)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_w = np.divide(sigma, den, out=np.zeros_like(sigma), where=den > 0.0) / 4.0
        tau = np.maximum(np.maximum(tau_raw, 0.0), tau_w)
        return sigma, tau

    @staticmethod
    def _scatter(target: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        if values.size:
            target[mask] += values

    def _single_spin(self, rho: np.ndarray, consts: dict, prelude, d1code, energy_name: str, d1_mask_fn):
        ra = np.maximum(rho[0], 0.0)
        ga, ta = self._safe_tau(rho)
        f = np.zeros_like(ra)
        v_ra = np.zeros_like(ra)
        v_ga = np.zeros_like(ra)
        v_ta = np.zeros_like(ra)

        emask = (ra > TOL) & (ta > TOL) & (ra > 1.0e-112)
        if emask.any():
            env = dict(consts)
            env.update(
                np=np,
                erf=erf,
                _max=_max,
                Tol=TOL,
                omega=OMEGA,
                F=np.zeros(emask.sum()),
                RA=ra[emask],
                GA=ga[emask],
                TA=ta[emask],
            )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
                exec(prelude, env)
            self._scatter(f, emask, env[energy_name])

        dmask = d1_mask_fn(ra, ga, ta)
        if dmask.any():
            env = dict(consts)
            env.update(
                np=np,
                erf=erf,
                _max=_max,
                Tol=TOL,
                omega=OMEGA,
                RA=ra[dmask],
                GA=ga[dmask],
                TA=ta[dmask],
                V_RA=np.zeros(dmask.sum()),
                V_GAA=np.zeros(dmask.sum()),
                V_TA=np.zeros(dmask.sum()),
            )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
                exec(d1code, env)
            self._scatter(v_ra, dmask, env["V_RA"])
            self._scatter(v_ga, dmask, env["V_GAA"])
            self._scatter(v_ta, dmask, env["V_TA"])

        return f, v_ra, v_ga, v_ta

    def _opposite_spin(self, rho_a: np.ndarray, rho_b: np.ndarray):
        ra = np.maximum(rho_a[0], 0.0)
        rb = np.maximum(rho_b[0], 0.0)
        ga, ta = self._safe_tau(rho_a)
        gb, tb = self._safe_tau(rho_b)
        gab = np.einsum("xg,xg->g", rho_a[1:4], rho_b[1:4])

        f = np.zeros_like(ra)
        derivs = {name: np.zeros_like(ra) for name in POSITION_MAP.values()}

        emask = (ra > TOL) & (rb > TOL) & (ta > TOL) & (tb > TOL) & (ra > 1.0e-128) & (rb > 1.0e-128)
        if emask.any():
            env = dict(self.cos_consts)
            env.update(
                np=np,
                erf=erf,
                _max=_max,
                Tol=TOL,
                F=np.zeros(emask.sum()),
                RA=ra[emask],
                RB=rb[emask],
                GA=ga[emask],
                GB=gb[emask],
                GAB=gab[emask],
                TA=ta[emask],
                TB=tb[emask],
            )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
                exec(self.cos_prelude, env)
            self._scatter(f, emask, env["Ec"])

        dmask = emask & (ra > 1.0e-90) & (rb > 1.0e-90)
        if dmask.any():
            env = dict(self.cos_consts)
            env.update(
                np=np,
                erf=erf,
                _max=_max,
                Tol=TOL,
                RA=ra[dmask],
                RB=rb[dmask],
                GA=ga[dmask],
                GB=gb[dmask],
                GAB=gab[dmask],
                TA=ta[dmask],
                TB=tb[dmask],
            )
            for name in POSITION_MAP.values():
                env[name] = np.zeros(dmask.sum())
            with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
                exec(self.cos_d1, env)
            for name in POSITION_MAP.values():
                self._scatter(derivs[name], dmask, env[name])

        return f, derivs

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        if spin != 1:
            raise NotImplementedError("This script is written for UKS only.")
        if deriv > 1:
            raise NotImplementedError("Only exc and vxc are implemented.")

        rho_a, rho_b = rho
        fx_a, vxa_ra, vxa_ga, vxa_ta = self._single_spin(
            rho_a,
            self.x_consts,
            self.x_prelude,
            self.x_d1,
            "Ex",
            lambda ra, ga, ta: (ra > TOL) & (ta > TOL) & (ra > 1.0e-81) & (ga > 0.0) & ((ga / np.power(ra, 8.0 / 3.0)) > 1.0e-180),
        )
        fx_b, vxb_rb, vxb_gb, vxb_tb = self._single_spin(
            rho_b,
            self.x_consts,
            self.x_prelude,
            self.x_d1,
            "Ex",
            lambda rb, gb, tb: (rb > TOL) & (tb > TOL) & (rb > 1.0e-81) & (gb > 0.0) & ((gb / np.power(rb, 8.0 / 3.0)) > 1.0e-180),
        )
        fcss_a, vcss_ra, vcss_ga, vcss_ta = self._single_spin(
            rho_a,
            self.css_consts,
            self.css_prelude,
            self.css_d1,
            "Ex",
            lambda ra, ga, ta: (ra > TOL) & (ta > TOL) & (ra > 1.0e-81),
        )
        fcss_b, vcss_rb, vcss_gb, vcss_tb = self._single_spin(
            rho_b,
            self.css_consts,
            self.css_prelude,
            self.css_d1,
            "Ex",
            lambda rb, gb, tb: (rb > TOL) & (tb > TOL) & (rb > 1.0e-81),
        )
        fcos, vcos = self._opposite_spin(rho_a, rho_b)

        f = fx_a + fx_b + fcss_a + fcss_b + fcos
        vrho = np.stack([vxa_ra + vcss_ra + vcos["V_RA"], vxb_rb + vcss_rb + vcos["V_RB"]], axis=1)
        vsigma = np.stack(
            [
                vxa_ga + vcss_ga + vcos["V_GAA"],
                vcos["V_GAB"],
                vxb_gb + vcss_gb + vcos["V_GBB"],
            ],
            axis=1,
        )
        vtau = np.stack([vxa_ta + vcss_ta + vcos["V_TA"], vxb_tb + vcss_tb + vcos["V_TB"]], axis=1)

        rho_tot = np.maximum(rho_a[0], 0.0) + np.maximum(rho_b[0], 0.0)
        exc = np.divide(f, rho_tot, out=np.zeros_like(f), where=rho_tot > 0.0)
        return exc, (vrho, vsigma, None, vtau), None, None


def build_mf(case: Case, coach: CoachSemilocal):
    basis = case.basis
    if case.name == "16_C_AE18":
        basis = {"C": basis_module.load("aug-cc-pCV5Z", "C")}

    mol = gto.M(
        atom=case.atom,
        charge=case.charge,
        spin=case.spin,
        basis=basis,
        unit="Angstrom",
        verbose=3,
    )
    mf = dft.UKS(mol)
    mf.conv_tol = case.conv_tol
    mf.max_cycle = case.max_cycle
    mf.grids.atom_grid = case.atom_grid
    mf.grids.prune = None

    mf.nlc = "VV10"
    mf.nlcgrids.atom_grid = (50, 194)
    mf.nlcgrids.prune = dft.gen_grid.sg1_prune

    dft.libxc.define_xc_(
        mf._numint,
        coach.eval_xc,
        xctype="MGGA",
        hyb=CX_SR_HF,
        rsh=(OMEGA, CX_LR_HF, CX_SR_HF - CX_LR_HF),
    )
    mf.xc = f"RSH({OMEGA},{CX_LR_HF},{CX_SR_HF - CX_LR_HF})"
    mf._numint.nlc_coeff = lambda xc_code: (((VV10_B, VV10_C), 1.0),)
    return mf


def d4_atm_energy(mol: gto.Mole) -> float:
    model = DispersionModel(
        numbers=np.asarray(mol.atom_charges(), dtype=int),
        positions=np.asarray(mol.atom_coords(), dtype=float),
        charge=float(mol.charge),
    )
    params = DampingParam(**D4_PARAMS)
    return float(model.get_dispersion(params, grad=False)["energy"])


def run_case(case: Case, coach: CoachSemilocal):
    print(f"\n=== {case.name} ===")
    if case.d4_only:
        mol = gto.M(atom=case.atom, charge=case.charge, spin=case.spin, unit="Angstrom", verbose=0)
        d4_energy = d4_atm_energy(mol)
        print(f"D4-ATM correction: {d4_energy: .10f} Ha")
        print(f"Reference D4:      {case.reference_d4: .10f} Ha")
        print(f"Delta D4:          {d4_energy - case.reference_d4: .10e} Ha")
        return

    mf = build_mf(case, coach)
    energy = mf.kernel()
    d4_energy = d4_atm_energy(mf.mol)
    total = energy + d4_energy

    print(f"PySCF SCF energy (semilocal + HF + VV10): {energy: .10f} Ha")
    print(f"D4-ATM correction:                      {d4_energy: .10f} Ha")
    print(f"PySCF + D4 total:                       {total: .10f} Ha")
    print(f"Reference D4:                           {case.reference_d4: .10f} Ha")
    print(f"Reference total:                        {case.reference_energy: .10f} Ha")
    print(f"Delta total:                            {total - case.reference_energy: .10e} Ha")


def main():
    parser = argparse.ArgumentParser(description="Reproduce COACH PySCF energies for the shipped Q-Chem outputs.")
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="Case name to run. Repeat to run multiple cases. Default: run both.",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    coach = CoachSemilocal(root)

    names = args.case or ["16_C_AE18", "SIE4x4_h2o", "L14_2a"]
    for name in names:
        run_case(CASES[name], coach)


if __name__ == "__main__":
    main()

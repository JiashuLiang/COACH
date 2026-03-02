import re
import sys
from pathlib import Path

import numpy as np


REF_OUT = "1_data_generation/SIE4x4_h2o.out"
GEN_OUT = "1_data_generation/pyscf_integratedDV_matrices.txt"
GRID_IDS = [75000302, 99000590, 250000974]
NROW = 96
NCOL = 180
NELEM = NROW * NCOL
TOP_N = 10


def resolve_path(rel_path):
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / rel_path


def parse_matrices(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    mats = {}

    i = 0
    while i < len(lines):
        m = re.search(r"In DFTenergy, GrdTyp =\s*(\d+)", lines[i])
        if not m:
            i += 1
            continue

        grid_id = int(m.group(1))
        i += 1
        while i < len(lines) and lines[i].strip() != "integratedDV":
            i += 1
        if i >= len(lines):
            break
        i += 1

        values = []
        while i < len(lines) and len(values) < NELEM:
            line = lines[i].strip()
            if line:
                values.extend(float(tok) for tok in line.split())
            i += 1

        if len(values) < NELEM:
            raise ValueError(f"Grid {grid_id} has only {len(values)} values (< {NELEM}).")
        mats[grid_id] = np.array(values[:NELEM], dtype=np.float64).reshape(NROW, NCOL)

    return mats


def top_diff_entries(diff, ref_mat, gen_mat, top_n):
    flat = diff.ravel()
    top_n = min(top_n, flat.size)
    if top_n == 0:
        return []
    idx = np.argpartition(flat, -top_n)[-top_n:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    rows, cols = np.unravel_index(idx, diff.shape)
    result = []
    for r, c in zip(rows, cols):
        result.append((int(r), int(c), float(diff[r, c]), float(ref_mat[r, c]), float(gen_mat[r, c])))
    return result


def main():
    ref_path = resolve_path(REF_OUT)
    gen_path = resolve_path(GEN_OUT)

    if not ref_path.exists():
        print(f"Missing reference file: {ref_path}")
        sys.exit(1)
    if not gen_path.exists():
        print(f"Missing generated file: {gen_path}")
        sys.exit(1)

    ref_mats = parse_matrices(ref_path)
    gen_mats = parse_matrices(gen_path)

    failed = False
    for gid in GRID_IDS:
        print(f"\nGrid {gid}")
        if gid not in ref_mats:
            print("  ERROR: missing in reference output.")
            failed = True
            continue
        if gid not in gen_mats:
            print("  ERROR: missing in generated output.")
            failed = True
            continue

        ref_mat = ref_mats[gid]
        gen_mat = gen_mats[gid]
        if ref_mat.shape != (NROW, NCOL) or gen_mat.shape != (NROW, NCOL):
            print(f"  ERROR: shape mismatch. ref={ref_mat.shape}, gen={gen_mat.shape}")
            failed = True
            continue

        diff = np.abs(gen_mat - ref_mat)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        fro = float(np.linalg.norm(gen_mat - ref_mat))
        print(f"  shape: {gen_mat.shape}")
        print(f"  max_abs_diff : {max_abs:.12e}")
        print(f"  mean_abs_diff: {mean_abs:.12e}")
        print(f"  fro_norm_diff: {fro:.12e}")

        top_entries = top_diff_entries(diff, ref_mat, gen_mat, TOP_N)
        print(f"  top-{len(top_entries)} |diff| entries (row, col, |diff|, ref, gen):")
        for r, c, dval, rval, gval in top_entries:
            print(f"    ({r:3d}, {c:3d})  {dval:.12e}  {rval:.12e}  {gval:.12e}")

    if failed:
        sys.exit(1)
    print("\nComparison complete.")


if __name__ == "__main__":
    main()

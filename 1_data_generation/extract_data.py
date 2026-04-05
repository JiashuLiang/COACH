import argparse
import csv
import pickle
from io import StringIO
from pathlib import Path

import numpy as np


SR_HF_COEF = 0.167
FITTING_GRID_IDS = {"250000974", "500000974"}
ENERGY_LABEL_MAP = {
    "Total energy in the final basis set": "Total energy",
    "Nuclear Repulsion Energy": "Nuclear Repulsion Energy",
    "Alpha HF Exchange Energy": "Alpha HF Exchange",
    "Beta HF Exchange Energy": "Beta HF Exchange",
    "Alpha LR HF Exchange Energy": "Alpha Long Range Exchange",
    "Beta LR HF Exchange Energy": "Beta Long Range Exchange",
    "Alpha SR HF Exchange (w) Energy": "Alpha SR HF Exchange (w)",
    "Beta SR HF Exchange (w) Energy": "Beta SR HF Exchange (w)",
    "DFT XC (X+C, excl. NLC) Energy": "DFT XC (X+C, excl. NLC)",
    "DFT Non-Local Correlation Energy": "DFT Non-Local Correlation",
    "One-Electron (alpha) Energy": "One-Electron (alpha)",
    "One-Electron (beta) Energy": "One-Electron (beta)",
    "Total Coulomb Energy": "Total Coulomb",
}
REQUIRED_ENERGY_KEYS = {
    "Total energy",
    "Nuclear Repulsion Energy",
    "Alpha HF Exchange",
    "Beta HF Exchange",
    "Alpha Long Range Exchange",
    "Beta Long Range Exchange",
    "Alpha SR HF Exchange (w)",
    "Beta SR HF Exchange (w)",
    "DFT XC (X+C, excl. NLC)",
    "DFT Non-Local Correlation",
    "One-Electron (alpha)",
    "One-Electron (beta)",
    "Total Coulomb",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Extract training data from PySCF one-file text outputs.")
    parser.add_argument("--input-data-dir", required=True, help="Directory containing one PySCF text file per molecule.")
    parser.add_argument("--dataset-eval", required=True, help="Dataset evaluation CSV file.")
    parser.add_argument("--output-dir", default=".", help="Output directory for raw_data.dict and reaction_data.dict.")
    return parser


def _parse_float_after_equals(line):
    return float(line.split("=", 1)[1].strip().split()[0])


def parse_pyscf_output(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    info = {
        "Spin_consistent": None,
        "Energy_consistent": True,
    }

    for line in lines:
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        label = " ".join(stripped.split("=", 1)[0].split())
        mapped = ENERGY_LABEL_MAP.get(label)
        if mapped is not None:
            info[mapped] = _parse_float_after_equals(stripped)

    missing_keys = sorted(REQUIRED_ENERGY_KEYS - info.keys())
    if missing_keys:
        raise ValueError(f"{path}: missing energy terms: {', '.join(missing_keys)}")

    info["Alpha Short Range Exchange"] = info["Alpha SR HF Exchange (w)"] / SR_HF_COEF
    info["Beta Short Range Exchange"] = info["Beta SR HF Exchange (w)"] / SR_HF_COEF
    info["Nofit"] = (
        info["Total energy"]
        - info["DFT XC (X+C, excl. NLC)"]
        - info["Alpha SR HF Exchange (w)"]
        - info["Beta SR HF Exchange (w)"]
    )

    i = 0
    grid_labels = []
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("In DFTenergy, GrdTyp ="):
            i += 1
            continue

        grid_id = line.split("=")[-1].strip()
        grid_label = "Fitting" if grid_id in FITTING_GRID_IDS else grid_id
        if grid_label == "Fitting" and "Fitting" in info:
            raise ValueError(f"{path}: multiple fitting grids found")
        if i + 1 >= len(lines) or lines[i + 1].strip() != "integratedDV":
            raise ValueError(f"{path}: malformed integratedDV block for grid {grid_id}")

        matrix_lines = lines[i + 2 : i + 98]
        if len(matrix_lines) != 96:
            raise ValueError(f"{path}: grid {grid_id} does not contain 96 matrix rows")
        matrix = np.loadtxt(StringIO("\n".join(matrix_lines))).T
        info[grid_label] = matrix
        grid_labels.append(grid_label)
        i += 98

    if "Fitting" not in info:
        raise ValueError(f"{path}: fitting grid {sorted(FITTING_GRID_IDS)} not found")

    for grid_label in grid_labels:
        if grid_label != "Fitting":
            info[grid_label] = info[grid_label] - info["Fitting"]

    return info


def load_dataset_eval(path):
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{path}: dataset_eval must be a CSV file")

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        required_columns = ["Reaction", "Reference", "Stoichiometry"]
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)

    normalized_rows = []
    for idx, row in enumerate(rows):
        reaction = row.get("Reaction")
        reference = row.get("Reference")
        stoichiometry = row.get("Stoichiometry")
        if not reaction or reference is None or stoichiometry is None:
            raise ValueError(f"{path}: row {idx} is missing Reaction, Reference, or Stoichiometry")
        normalized_rows.append(
            {
                "Reaction": str(reaction),
                "Reference": float(reference),
                "Stoichiometry": str(stoichiometry),
            }
        )
    return normalized_rows


def is_supported_value(value):
    return isinstance(value, np.ndarray) or (isinstance(value, (int, float, np.floating)) and not isinstance(value, bool))


def calculate_reaction(row, raw_data):
    tokens = [token.strip() for token in row["Stoichiometry"].split(",") if token.strip()]
    if len(tokens) % 2 != 0:
        raise ValueError(f"{row['Reaction']}: invalid Stoichiometry field")

    num_species = len(tokens) // 2
    species_names = [tokens[2 * i + 1] for i in range(num_species)]
    missing_species = [name for name in species_names if name not in raw_data]
    if missing_species:
        raise KeyError(f"{row['Reaction']}: missing species: {', '.join(missing_species)}")

    species_data = [raw_data[name] for name in species_names]
    reaction = {}
    for key, value in species_data[0].items():
        if all(key in species for species in species_data) and is_supported_value(value):
            total = 0.0
            for i in range(num_species):
                total += float(tokens[2 * i]) * species_data[i][key]
            reaction[key] = total

    reaction["Ref"] = row["Reference"]
    reaction["Tofit"] = reaction["Ref"] - reaction["Nofit"]
    return reaction


def write_pickle(path, obj):
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def write_log(path, messages):
    if messages:
        path.write_text("\n".join(messages) + "\n", encoding="utf-8")


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_data_dir = Path(args.input_data_dir).resolve()
    dataset_eval_path = Path(args.dataset_eval).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = {}
    failed_files = []
    for path in sorted(input_data_dir.glob("*.txt")):
        try:
            raw_data[path.stem] = parse_pyscf_output(path)
        except Exception as exc:
            failed_files.append(f"{path.name}: {exc}")

    if not raw_data:
        raise RuntimeError(f"No valid .txt files were parsed from {input_data_dir}")

    dataset_rows = load_dataset_eval(dataset_eval_path)
    reactions = {}
    failed_reactions = []
    for row in dataset_rows:
        try:
            reactions[row["Reaction"]] = calculate_reaction(row, raw_data)
        except Exception as exc:
            failed_reactions.append(str(exc))

    write_pickle(output_dir / "raw_data.dict", raw_data)
    write_pickle(output_dir / "reaction_data.dict", reactions)
    write_log(output_dir / "failed_files.log", failed_files)
    write_log(output_dir / "failed_reactions.log", failed_reactions)

    print(f"Parsed molecules: {len(raw_data)}")
    print(f"Built reactions: {len(reactions)}")
    print(f"Wrote: {output_dir / 'raw_data.dict'}")
    print(f"Wrote: {output_dir / 'reaction_data.dict'}")
    if failed_files:
        print(f"Failed files logged to: {output_dir / 'failed_files.log'}")
    if failed_reactions:
        print(f"Failed reactions logged to: {output_dir / 'failed_reactions.log'}")


if __name__ == "__main__":
    main()

"""Smoke tests for the preprocessing plus analysis workflow without Gurobi."""

import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.analysis import analyze_run_directory
from coachopt.constraints import select_diff_constraint_rows
from coachopt.processing import build_and_save_data
from coachopt.utils import read_csv_frame, save_name_array, write_json


def synthetic_reaction(offset: float) -> dict:
    """Create a minimal reaction payload for workflow-level integration tests."""
    fitting = np.zeros((180, 96), dtype=float)
    fitting[64] = offset + np.arange(96) * 0.01
    fitting[153] = offset + np.arange(96) * 0.02
    fitting[166] = offset + np.arange(96) * 0.03
    diff = np.zeros_like(fitting)
    diff[64] = 0.001 + offset * 0.0001
    diff[153] = 0.002 + offset * 0.0001
    diff[166] = 0.003 + offset * 0.0001
    return {
        "Fitting": fitting,
        "99000590": diff,
        "Tofit": 0.5 + offset,
        "Alpha Short Range Exchange": 0.1,
        "Beta Short Range Exchange": 0.2,
        "Alpha Long Range Exchange": 0.3,
        "Beta Long Range Exchange": 0.4,
    }


class WorkflowSmokeTests(unittest.TestCase):
    """Exercise the cleaned workflow around stored beta vectors and analysis."""

    def test_smoke_path_without_solver(self):
        """Run the pass-1/pass-2 artifact flow without invoking the optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reaction_data = {"R1": synthetic_reaction(0.0), "R2": synthetic_reaction(1.0)}
            processed_dir = root / "processed"
            pass1_dir = root / "runs" / "pass1"
            pass2_dir = root / "runs" / "pass2"
            pass1_dir.mkdir(parents=True)
            pass2_dir.mkdir(parents=True)

            dataset_eval_path = root / "dataset_eval.csv"
            with dataset_eval_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Reaction", "Dataset", "Reference", "Stoichiometry"]
                )
                writer.writeheader()
                writer.writerow({"Reaction": "R1", "Dataset": "DS1", "Reference": "1.0", "Stoichiometry": "1,A"})
                writer.writerow({"Reaction": "R2", "Dataset": "DS2", "Reference": "2.0", "Stoichiometry": "1,B"})

            training_weights_path = root / "training_weights.csv"
            with training_weights_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Dataset", "datapoints", "weights"])
                writer.writeheader()
                writer.writerow({"Dataset": "DS1", "datapoints": "All", "weights": "1.0"})
                writer.writerow({"Dataset": "DS2", "datapoints": "All", "weights": "1.0"})

            build_and_save_data(
                reaction_data=reaction_data,
                dataset_eval=read_csv_frame(
                    dataset_eval_path,
                    ["Reaction", "Dataset", "Reference", "Stoichiometry"],
                ),
                training_weight=read_csv_frame(
                    training_weights_path,
                    ["Dataset", "datapoints", "weights"],
                ),
                output_dir=processed_dir,
            )

            a_matrix = np.load(processed_dir / "A_matrix.npy")
            beta = np.zeros(a_matrix.shape[1], dtype=float)
            beta[0] = 1.0
            np.save(pass1_dir / "betas_nonzero1.npy", np.asarray([beta]))
            write_json(pass2_dir / "run_config.json", {"nonzeros": [1], "diff_name": "diff_constraint_99590.npy"})

            diff_matrix = np.load(processed_dir / "diff_99590.npy")
            diff_names = np.load(processed_dir / "name_list_diff_99590.npy").astype(str).tolist()
            selection = select_diff_constraint_rows(diff_matrix, diff_names, [beta], top_per_beta=1, top_l1=1)
            np.save(processed_dir / "diff_constraint_99590.npy", selection.rows)
            save_name_array(processed_dir / "name_list_diff_constraint_99590.npy", selection.names)

            with (processed_dir / "A_matrix_dataset.pkl").open("rb") as handle:
                a_dict = pickle.load(handle)
            with (processed_dir / "b_vec_dataset.pkl").open("rb") as handle:
                b_dict = pickle.load(handle)
            self.assertEqual(set(a_dict), {"DS1", "DS2"})
            self.assertEqual(set(b_dict), {"DS1", "DS2"})

            outputs = analyze_run_directory(
                run_dir=pass1_dir,
                processed_dir=processed_dir,
                dataset_info={"DS1": {"Datatype": "TypeA"}, "DS2": {"Datatype": "TypeB"}},
                diff_name="diff_constraint_99590.npy",
            )

            self.assertTrue((processed_dir / "diff_constraint_99590.npy").exists())
            self.assertTrue((processed_dir / "name_list_diff_constraint_99590.npy").exists())
            self.assertTrue((pass2_dir / "run_config.json").exists())
            self.assertTrue(Path(outputs["summary_csv"]).exists())


if __name__ == "__main__":
    unittest.main()

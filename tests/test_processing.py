import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.processing import FeatureSpec, build_and_save_training_data
from coachopt.utils import read_csv_rows


def synthetic_reaction(offset: float) -> dict:
    fitting = np.zeros((180, 96), dtype=float)
    fitting[64] = offset + np.arange(96) * 0.01
    fitting[153] = offset * 2.0 + np.arange(96) * 0.02
    fitting[166] = offset * 3.0 + np.arange(96) * 0.03
    diff = np.zeros_like(fitting)
    diff[64] = 0.001
    diff[153] = 0.002
    diff[166] = 0.003
    return {
        "Fitting": fitting,
        "99000590": diff,
        "Tofit": 1.0 + offset,
        "Alpha Short Range Exchange": 0.1 + offset,
        "Beta Short Range Exchange": 0.2 + offset,
        "Alpha Long Range Exchange": 0.3,
        "Beta Long Range Exchange": 0.4,
    }


class ProcessingTests(unittest.TestCase):
    def test_build_and_save_training_data_preserves_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reaction_path = root / "reaction_data.dict"
            dataset_eval_path = root / "dataset_eval.csv"
            weights_path = root / "training_weights.csv"
            out_dir = root / "processed"

            reaction_data = {
                "R1": synthetic_reaction(0.0),
                "R2": synthetic_reaction(1.0),
                "R3": synthetic_reaction(2.0),
            }
            with reaction_path.open("wb") as handle:
                pickle.dump(reaction_data, handle)

            with dataset_eval_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Reaction", "Dataset", "Reference", "Stoichiometry"]
                )
                writer.writeheader()
                writer.writerow({"Reaction": "R1", "Dataset": "DS1", "Reference": "1.0", "Stoichiometry": "1,A"})
                writer.writerow({"Reaction": "R2", "Dataset": "DS1", "Reference": "2.0", "Stoichiometry": "1,B"})
                writer.writerow({"Reaction": "R3", "Dataset": "DS2", "Reference": "3.0", "Stoichiometry": "1,C"})

            with weights_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Dataset", "datapoints", "weights"])
                writer.writeheader()
                writer.writerow({"Dataset": "DS1", "datapoints": "All", "weights": "2.0"})
                writer.writerow({"Dataset": "DS2", "datapoints": "R3", "weights": "Shrink"})

            outputs = build_and_save_training_data(
                reaction_data=reaction_data,
                dataset_eval_rows=read_csv_rows(
                    dataset_eval_path,
                    ["Reaction", "Dataset", "Reference", "Stoichiometry"],
                ),
                training_weight_rows=read_csv_rows(
                    weights_path,
                    ["Dataset", "datapoints", "weights"],
                ),
                output_dir=out_dir,
                spec=FeatureSpec(),
            )

            a_matrix = np.load(out_dir / "A_matrix.npy")
            b_vec = np.load(out_dir / "b_vec.npy")
            weight_vec = np.load(out_dir / "weight_vec.npy")
            diff_matrix = np.load(out_dir / "diff_99590.npy")

            self.assertEqual(a_matrix.shape, (3, 289))
            self.assertEqual(b_vec.shape, (3,))
            self.assertEqual(weight_vec.tolist(), [2.0, 2.0, 1.0])
            self.assertEqual(diff_matrix.shape, (3, 289))
            self.assertTrue(np.allclose(diff_matrix[:, -1], 0.0))
            self.assertTrue(Path(outputs["diff_matrix"]).name == "diff_99590.npy")
            self.assertTrue((out_dir / "A_matrix_dataset.dict").exists())
            self.assertTrue((out_dir / "b_vec_dataset.dict").exists())


if __name__ == "__main__":
    unittest.main()

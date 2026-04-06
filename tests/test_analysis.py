"""Regression tests for optimization-result analysis artifacts."""

import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.analysis import analyze_run_directory


class AnalysisTests(unittest.TestCase):
    """Validate summary-table generation and best-model selection."""

    def test_analysis_writes_summary_and_best_models(self):
        """Prefer the lower-error candidate and emit all expected reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            run_dir = root / "run"
            processed.mkdir()
            run_dir.mkdir()

            a_matrix = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=float)
            b_vec = np.asarray([1.0, 1.0], dtype=float)
            weights = np.asarray([1.0, 1.0], dtype=float)
            diff = np.asarray([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=float)
            np.save(processed / "A_matrix.npy", a_matrix)
            np.save(processed / "b_vec.npy", b_vec)
            np.save(processed / "weight_vec.npy", weights)
            np.save(processed / "diff_99590.npy", diff)
            with (processed / "A_matrix_dataset.dict").open("wb") as handle:
                pickle.dump({"DS1": a_matrix[:1], "DS2": a_matrix[1:]}, handle)
            with (processed / "b_vec_dataset.dict").open("wb") as handle:
                pickle.dump({"DS1": b_vec[:1], "DS2": b_vec[1:]}, handle)

            good = np.asarray([1.0, 1.0, 0.0], dtype=float)
            bad = np.asarray([0.0, 0.0, 0.0], dtype=float)
            np.save(run_dir / "betas_nonzero2.npy", np.asarray([bad, good]))

            outputs = analyze_run_directory(
                run_dir=run_dir,
                processed_dir=processed,
                dataset_info={"DS1": {"Datatype": "TypeA"}, "DS2": {"Datatype": "TypeB"}},
                baseline_errors={"DS1": {"Standard": 1.0}, "DS2": {"Standard": 1.0}},
            )

            self.assertTrue(Path(outputs["summary_csv"]).exists())
            self.assertTrue(Path(outputs["dataset_rmse_csv"]).exists())
            best_overall = np.load(Path(outputs["best_models_dir"]) / "best_overall.npy")
            self.assertTrue(np.allclose(best_overall, good))

            with Path(outputs["summary_csv"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("mean_relative_Standard", rows[0])


if __name__ == "__main__":
    unittest.main()

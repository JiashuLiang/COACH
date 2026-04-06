"""Tests for the pass-2 grid-constraint selection CLI."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.constants import (
    DEFAULT_DIFF_MATRIX_NAME,
    DEFAULT_DIFF_NAMES_NAME,
    DEFAULT_SELECTED_DIFF_NAME,
    DEFAULT_SELECTED_DIFF_PREFIX,
)
from select_grid_constraints import main

from coachopt.utils import save_names


class SelectGridConstraintsTests(unittest.TestCase):
    """Verify the CLI writes only the reduced matrix and optional reporting."""

    def test_cli_writes_only_diff_matrix_and_reports_largest_errors(self):
        """The selection script should not emit names or JSON sidecar artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed_dir = root / "processed"
            run_dir = root / "runs" / "pass1"
            processed_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            diff_matrix = np.asarray(
                [
                    [5.0, 0.0, 0.0],
                    [0.0, -4.0, 0.0],
                    [0.0, 0.0, 3.0],
                    [1.0, 1.0, 1.0],
                ]
            )
            diff_names = ["row_a", "row_b", "row_c", "row_d"]
            np.save(processed_dir / DEFAULT_DIFF_MATRIX_NAME, diff_matrix)
            save_names(processed_dir / DEFAULT_DIFF_NAMES_NAME, diff_names)
            np.save(
                run_dir / "betas_nonzero1.npy",
                np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ]
                ),
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--diff-matrix",
                        str(processed_dir / DEFAULT_DIFF_MATRIX_NAME),
                        "--diff-names",
                        str(processed_dir / DEFAULT_DIFF_NAMES_NAME),
                        "--run-dir",
                        str(run_dir),
                        "--output-dir",
                        str(processed_dir),
                        "--top-per-beta",
                        "1",
                        "--top-l1",
                        "1",
                        "--show-largesterror",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((processed_dir / DEFAULT_SELECTED_DIFF_NAME).exists())
            self.assertFalse((processed_dir / f"name_list_{DEFAULT_SELECTED_DIFF_PREFIX}.txt").exists())
            self.assertFalse((processed_dir / f"{DEFAULT_SELECTED_DIFF_PREFIX}.json").exists())
            np.testing.assert_array_equal(
                np.load(processed_dir / DEFAULT_SELECTED_DIFF_NAME),
                diff_matrix[[0, 1, 3]],
            )

            output = stdout.getvalue()
            self.assertIn("nz1_cand0 largest diffs:", output)
            self.assertIn("nz1_cand1 largest diffs:", output)
            self.assertIn("row_a", output)
            self.assertIn("row_b", output)
            self.assertIn("row_c", output)


if __name__ == "__main__":
    unittest.main()

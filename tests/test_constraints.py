"""Unit tests for diff-constraint row selection heuristics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.select_diff_constraints import select_diff_constraint_rows


class ConstraintSelectionTests(unittest.TestCase):
    """Verify the union of per-beta and global diff-row heuristics."""

    def test_selection_uses_top_deviation_and_top_l1_union(self):
        """Keep rows that are extreme for any beta or globally large in L1 norm."""
        diff_matrix = np.asarray(
            [
                [0.1, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, -4.0, 0.0],
                [0.0, 0.0, 0.2],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        betas = [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])]

        selection = select_diff_constraint_rows(diff_matrix, betas, top_per_beta=1, top_l1=1)

        np.testing.assert_array_equal(selection, diff_matrix[[1, 2, 4]])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.constraints import select_diff_constraint_rows


class ConstraintSelectionTests(unittest.TestCase):
    def test_selection_uses_top_deviation_and_top_l1_union(self):
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
        diff_names = [f"R{i}" for i in range(diff_matrix.shape[0])]
        betas = [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])]

        selection = select_diff_constraint_rows(diff_matrix, diff_names, betas, top_per_beta=1, top_l1=1)

        self.assertEqual(selection.indices.tolist(), [1, 2, 4])
        self.assertEqual(selection.names, ["R1", "R2", "R4"])
        self.assertEqual(selection.rows.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()

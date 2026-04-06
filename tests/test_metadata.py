"""Regression tests for preprocessing metadata validation helpers."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.utils import read_csv_frame


class MetadataTests(unittest.TestCase):
    """Check required-column enforcement for CSV metadata."""

    def test_missing_required_column_raises(self):
        """Raise a helpful error when dataset metadata omits required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset_eval.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Reaction", "Reference", "Stoichiometry"])
                writer.writeheader()
                writer.writerow({"Reaction": "R1", "Reference": "1.0", "Stoichiometry": "1,A"})

            with self.assertRaisesRegex(ValueError, "missing required columns: Dataset"):
                read_csv_frame(path, ["Reaction", "Dataset", "Reference", "Stoichiometry"])


if __name__ == "__main__":
    unittest.main()

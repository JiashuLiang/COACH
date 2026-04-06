"""Regression tests for CSV validation in the extraction helpers."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_generation"))

from extract_data import load_dataset_eval


class ExtractDataTests(unittest.TestCase):
    """Cover dataset-evaluation input validation for ``extract_data.py``."""

    def test_load_dataset_eval_requires_csv(self):
        """Reject non-CSV metadata inputs early."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset_eval.tsv"
            path.write_text("Reaction\tReference\tStoichiometry\nR1\t1.0\t1,A\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a CSV file"):
                load_dataset_eval(path)

    def test_load_dataset_eval_requires_maintained_columns(self):
        """Require the maintained ``Reference`` column name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset_eval.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Reaction", "Value", "Stoichiometry"])
                writer.writeheader()
                writer.writerow({"Reaction": "R1", "Value": "1.0", "Stoichiometry": "1,A"})

            with self.assertRaisesRegex(ValueError, "missing required columns: Reference"):
                load_dataset_eval(path)


if __name__ == "__main__":
    unittest.main()

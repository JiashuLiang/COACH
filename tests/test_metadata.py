import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.metadata import load_dataset_eval_csv


class MetadataTests(unittest.TestCase):
    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset_eval.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Reaction", "Reference", "Stoichiometry"])
                writer.writeheader()
                writer.writerow({"Reaction": "R1", "Reference": "1.0", "Stoichiometry": "1,A"})

            with self.assertRaisesRegex(ValueError, "missing required columns: Dataset"):
                load_dataset_eval_csv(path)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for run_mio warm-start argument handling."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

from coachopt.optimizer import OptimizationConfig, SIMPLE_WARM_START, _load_warm_start_vectors
from run_mio import build_parser


class RunMioWarmStartTests(unittest.TestCase):
    """Validate the simplified warm-start input flow."""

    def test_parser_accepts_warm_start_files_list(self):
        """Parse one CLI flag followed by one or more warm-start files."""
        args = build_parser().parse_args(["-n", "24", "--warm_start_files", "a.npy", "b.npy"])

        self.assertEqual(args.warm_start_files, ["a.npy", "b.npy"])

    def test_optimizer_returns_named_warm_starts(self):
        """Keep the built-in simple seed and key file warm starts by filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            single_path = root / "single.npy"
            batch_path = root / "batch.npy"

            extra = np.zeros(289, dtype=float)
            extra[10] = 2.5
            np.save(single_path, extra)
            np.save(batch_path, np.asarray([SIMPLE_WARM_START, extra]))

            single_config = OptimizationConfig(nonzeros=[24], warm_start_files=str(single_path))
            self.assertEqual(single_config.warm_start_files, [str(single_path)])

            config = OptimizationConfig(nonzeros=[24], warm_start_files=[str(single_path), str(batch_path)])
            warm_starts = _load_warm_start_vectors(config)

            self.assertEqual(list(warm_starts), ["simple", str(single_path).replace("/", "_"), str(batch_path).replace("/", "_")])
            self.assertTrue(np.allclose(warm_starts["simple"], SIMPLE_WARM_START))
            self.assertTrue(np.allclose(warm_starts[str(single_path).replace("/", "_")], extra))
            self.assertEqual(warm_starts[str(batch_path).replace("/", "_")].shape, (2, 289))
            self.assertTrue(np.allclose(warm_starts[str(batch_path).replace("/", "_")][0], SIMPLE_WARM_START))
            self.assertTrue(np.allclose(warm_starts[str(batch_path).replace("/", "_")][1], extra))


if __name__ == "__main__":
    unittest.main()

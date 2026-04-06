"""Unit tests for run_mio warm-start argument handling."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_optimization"))

import coachopt.optimizer as optimizer_module
from coachopt.optimizer import OptimizationConfig, SIMPLE_WARM_START, _load_warm_start_vectors, run_optimization_sweep
from run_mio import build_parser


class RunMioWarmStartTests(unittest.TestCase):
    """Validate the simplified warm-start input flow."""

    def test_parser_accepts_warm_start_dir_and_default_repeats(self):
        """Parse the directory flag and keep the per-warm-start repeat default."""
        args = build_parser().parse_args(["-n", "24", "--warm_start_dir", "runs/pass1"])

        self.assertEqual(args.warm_start_dir, "runs/pass1")
        self.assertEqual(args.repeats, 1)

    def test_optimizer_returns_simple_plus_matching_nonzero_file(self):
        """Load only betas_nonzero<N>.npy for the requested sparsity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            warm_start_dir = root / "pass1"
            warm_start_dir.mkdir()

            extra = np.zeros(289, dtype=float)
            extra[10] = 2.5
            other = np.zeros(289, dtype=float)
            other[12] = 3.0
            np.save(warm_start_dir / "betas_nonzero24.npy", np.asarray([extra]))
            np.save(warm_start_dir / "betas_nonzero32.npy", np.asarray([other]))

            config = OptimizationConfig(nonzeros=[24, 32], warm_start_dir=str(warm_start_dir))
            warm_starts_24 = _load_warm_start_vectors(config, 24)
            warm_starts_32 = _load_warm_start_vectors(config, 32)
            warm_starts_40 = _load_warm_start_vectors(config, 40)

            key_24 = str(warm_start_dir / "betas_nonzero24.npy").replace("/", "_")
            key_32 = str(warm_start_dir / "betas_nonzero32.npy").replace("/", "_")
            self.assertEqual(list(warm_starts_24), ["simple", key_24])
            self.assertTrue(np.allclose(warm_starts_24["simple"], SIMPLE_WARM_START))
            self.assertEqual(warm_starts_24[key_24].shape, (289,))
            self.assertTrue(np.allclose(warm_starts_24[key_24], extra))
            self.assertEqual(list(warm_starts_32), ["simple", key_32])
            self.assertTrue(np.allclose(warm_starts_32[key_32], other))
            self.assertEqual(list(warm_starts_40), ["simple"])

    def test_score_labels_repeat_each_warm_start(self):
        """Repeat each warm start instead of padding the run with unrelated random seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            warm_start_dir = root / "pass1"
            input_dir.mkdir()
            warm_start_dir.mkdir()

            np.save(input_dir / "A_matrix.npy", np.zeros((1, 289), dtype=float))
            np.save(input_dir / "b_vec.npy", np.zeros(1, dtype=float))
            np.save(input_dir / "weight_vec.npy", np.ones(1, dtype=float))

            seed_a = np.zeros(289, dtype=float)
            seed_a[5] = 1.25
            seed_b = np.zeros(289, dtype=float)
            seed_b[6] = 2.0
            np.save(warm_start_dir / "betas_nonzero1.npy", np.asarray([seed_a, seed_b]))

            original_solve = optimizer_module._solve_single_mio
            original_constraints = optimizer_module.build_physical_constraints
            try:
                optimizer_module.build_physical_constraints = lambda a_rows: {}
                optimizer_module._solve_single_mio = lambda **kwargs: np.asarray(kwargs["warm_start"], dtype=float)

                run_optimization_sweep(
                    OptimizationConfig(
                        nonzeros=[1],
                        repeats=2,
                        input_dir=str(input_dir),
                        out_dir=str(output_dir),
                        warm_start_dir=str(warm_start_dir),
                    )
                )
            finally:
                optimizer_module._solve_single_mio = original_solve
                optimizer_module.build_physical_constraints = original_constraints

            with (output_dir / "betas_nonzero1.json").open("r", encoding="utf-8") as handle:
                scores = json.load(handle)

            source_key = str(warm_start_dir / "betas_nonzero1.npy").replace("/", "_")
            self.assertEqual(
                [row["label"] for row in scores],
                [
                    "simple_repeat_0",
                    "simple_repeat_1",
                    f"{source_key}_0_repeat_0",
                    f"{source_key}_0_repeat_1",
                    f"{source_key}_1_repeat_0",
                    f"{source_key}_1_repeat_1",
                ],
            )


if __name__ == "__main__":
    unittest.main()

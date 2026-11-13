"""Synthetic fit invariants and cross-codec software checks; no GPU gain claims."""
import json
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class HeadPolicyTests(unittest.TestCase):
    def test_learned_map_recovers_synthetic_cross_head_signal(self):
        from kv_compare.head_spread import fit_head_map
        rng = np.random.default_rng(8)
        def record(count):
            result = np.empty((3, 4, count), np.int32)
            result[0] = rng.integers(10000, 200000, (4, count))
            for layer in (1, 2):
                result[layer] = (0.8 * np.roll(result[layer - 1], 1, axis=0) + 15000
                    + rng.normal(0, 100, (4, count))).astype(np.int32)
            return result
        train, test = record(500), record(200)
        fitted = fit_head_map([train], 16)
        x = test[0].T / (16 * 4096)
        y = test[1].T / (16 * 4096)
        predicted = x @ fitted.weights[1].T + fitted.bias[1]
        global_prediction = x.mean(1)[:, None]
        self.assertLess(np.square(predicted - y).mean(), np.square(global_prediction - y).mean() / 100)
        frozen = fitted.digest()
        test[:] = 0
        self.assertEqual(fitted.digest(), frozen)

    def test_global_and_zero_blend_have_identical_priors(self):
        from kv_compare.head_spread import Policy, prior_reference
        sums = np.arange(64, dtype=np.int32).reshape(4, 16) * 1729
        a = prior_reference(sums, Policy(), 16)
        b = prior_reference(sums, Policy(1, 0.0), 16)
        np.testing.assert_array_equal(a, b)

    def test_policy_checks_and_charged_arrays(self):
        from kv_compare.head_spread import Policy
        with self.assertRaises(ValueError):
            Policy(1, 2.0).validate(3, 4)
        with self.assertRaises(ValueError):
            Policy(2, 0.5).validate(3, 4)
        p = Policy(2, 0.5, 0, np.zeros((3, 4, 4), np.float32), np.ones((3, 4), np.float32))
        p.validate(3, 4)
        self.assertEqual(p.array_bytes, (3 * 4 * 4 + 3 * 4) * 4)
        self.assertEqual(p.description()["paid_token_metadata_bytes"], 0)

    def test_original_streams_and_compact_packets_in_interpreter(self):
        env = dict(os.environ, TRITON_INTERPRET="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-m", "kv_compare.head_spread_checks", "--device", "cpu", "--interpret"],
            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr[-6000:])
        report = json.loads(result.stdout.splitlines()[-1])
        self.assertFalse(report["gpu_executed"])
        self.assertEqual(len(report["rows"]), 12)
        self.assertTrue(all(row["exact"] for row in report["rows"]))

    def test_cuda_entry_rejects_interpreter(self):
        env = dict(os.environ, TRITON_INTERPRET="1")
        result = subprocess.run([sys.executable, "-m", "kv_compare.head_spread_checks", "--device", "cuda"],
            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unset TRITON_INTERPRET", result.stderr)

    def test_frozen_state_rejects_changed_predictor_or_policy(self):
        import torch
        from kv_compare.head_spread import Policy
        from kv_compare.head_spread_wire import consts_digest, save_state, load_state
        constants = {"Aq": torch.zeros((3, 64, 64), dtype=torch.int8)}
        policy = Policy(2, 0.5, 0, np.zeros((3, 4, 4), np.float32), np.ones((3, 4), np.float32))
        digest = consts_digest(constants, 32, 16, 8, "uniform")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            save_state(path, constants, 32, 16, 8, "uniform", policy, digest)
            self.assertEqual(load_state(path, "cpu")["loaded_policy"].digest(), policy.digest())
            original = path.read_bytes()
            for target in ("predictor", "policy"):
                path.write_bytes(original)
                changed = torch.load(path, map_location="cpu", weights_only=True)
                if target == "predictor":
                    changed["consts"]["Aq"][0, 0, 0] = 1
                else:
                    changed["policy"]["weights"][1, 0, 0] = 1
                torch.save(changed, path)
                with self.assertRaisesRegex(ValueError, target + " checksum mismatch"):
                    load_state(path, "cpu")

    def test_runner_rejects_nonfinite_budgets_and_duplicate_modes(self):
        from scripts.bench_head_spread import main
        for extra in (["--ridge", "nan"], ["--max-decode-ratio", "inf"], ["--modes", "dense,dense"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--captures", "unused", "--output", "unused-output"] + extra)
            self.assertEqual(error.exception.code, 2)

    def test_original_t_source_files_are_pinned_and_unchanged(self):
        from scripts.bench_head_spread import check_base
        self.assertEqual(len(check_base()), 6)


if __name__ == "__main__":
    unittest.main()

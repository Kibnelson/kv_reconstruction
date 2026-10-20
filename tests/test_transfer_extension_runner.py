"""Tests of the controlled experiment's data separation and metadata accounting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("bench_transfer_extensions",
    Path(__file__).resolve().parents[1] / "scripts/bench_transfer_extensions.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(seed=1, tokens=32):
    random = np.random.default_rng(seed)
    values = random.normal(size=(2, tokens, 8)).astype(np.float32)
    bits = ((values.view(np.uint32) + 0x7fff + ((values.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
    return {"K_bits": bits[..., :4], "V_bits": bits[..., 4:],
            "H": random.normal(size=(2, tokens, 6)).astype(np.float32),
            "ids": np.arange(tokens), "Hk": np.array(1), "hd": np.array(4), "base": np.array(10000.)}


class TransferRunnerTests(unittest.TestCase):
    def test_split_preserves_positions_and_values(self):
        act = fixture()
        subset = runner.slice_act(act, 16, 32)
        np.testing.assert_array_equal(subset["positions"], np.arange(16, 32))
        np.testing.assert_array_equal(subset["K_bits"], act["K_bits"][:, 16:])
        np.testing.assert_array_equal(subset["H"], act["H"][:, 16:])

    def test_hf_key_permutation_is_per_head(self):
        keys = np.arange(16).reshape(1, 1, 16)
        actual = runner.adjacent_key_order(keys, 8)
        np.testing.assert_array_equal(actual.ravel(), [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15])
        with self.assertRaises(ValueError):
            runner.adjacent_key_order(keys, 3)

    def test_metadata_frozen_and_charged(self):
        training = [fixture(1), fixture(2)]
        projector = runner.fit_projector(training, "hidden", 2)
        state = {name: projector[name].copy() for name in ("mean", "basis", "scale")}
        codes, diagnostic = runner.project_codes(projector, fixture(3))
        self.assertEqual(codes.dtype, np.int8)
        self.assertEqual(codes.shape, (2, 32, 2))
        self.assertEqual(codes.nbytes, 128)
        for name in state:
            np.testing.assert_array_equal(state[name], projector[name])
        shuffled, _ = runner.project_codes(projector, fixture(3), shuffled=True)
        np.testing.assert_array_equal(shuffled, np.roll(codes, 16, axis=1))
        with tempfile.TemporaryDirectory() as directory:
            report = runner.save_projector(projector, Path(directory) / "encoder.npz")
            self.assertGreater(report["serialized_bytes"], report["array_bytes"])
            self.assertEqual(report["array_bytes"], sum(value.nbytes for value in state.values()))

    def test_combined_words_preserve_bits(self):
        act = fixture()
        words = runner.combined_words(act)
        self.assertEqual(words.shape, (2, 32, 8))
        np.testing.assert_array_equal(words[..., :4], act["K_bits"])
        np.testing.assert_array_equal(words[..., 4:], act["V_bits"])

    def test_bad_rank_or_missing_hidden_rejected(self):
        with self.assertRaises(ValueError):
            runner.fit_projector([fixture()], "hidden", 20)
        act = fixture(); act.pop("H")
        with self.assertRaises(ValueError):
            runner.source_features(act, "hidden")

    def test_partial_hidden_selection_never_aliases_layer_zero(self):
        act = fixture()
        act["H"] = act["H"][[1]]
        act["sel"] = np.array([1])
        selected = runner.limit_act_layers(act, 1)
        self.assertEqual(selected["K_bits"].shape[0], 1)
        self.assertNotIn("H", selected)
        self.assertNotIn("sel", selected)
        self.assertNotIn("H", runner.limit_act_layers(act))
        self.assertIn("H", act)

    def test_layer_prefix_remaps_hidden_selection_and_checks_bounds(self):
        act = fixture()
        expected = act["H"].copy()
        act["H"] = act["H"][[1, 0]]
        act["sel"] = np.array([1, 0])
        normalized = runner.limit_act_layers(act)
        np.testing.assert_array_equal(normalized["H"], expected)
        np.testing.assert_array_equal(normalized["sel"], [0, 1])
        prefix = runner.limit_act_layers(act, 1)
        np.testing.assert_array_equal(prefix["H"], expected[:1])
        for count in (0, 3):
            with self.assertRaises(ValueError):
                runner.limit_act_layers(act, count)

    def test_hidden_source_rejects_noncanonical_layer_indices(self):
        act = fixture()
        act["sel"] = np.array([1, 0])
        with self.assertRaises(ValueError):
            runner.source_features(act, "hidden")

    def test_dev_shuffle_stays_inside_selection_half(self):
        act = fixture(tokens=32)
        projector = runner.fit_projector([fixture(2)], "hidden", 1)
        spec = {"projector": projector, "shuffled": True}
        _, diagnostic = runner.metadata_for(spec, act, 16, 4, (16, 32))
        self.assertTrue(all(16 <= index < 32 for index in diagnostic["source_token_indices"]))
        full, _ = runner.project_codes(projector, runner.slice_act(act, 16, 32), shuffled=True)
        selected, _ = runner.metadata_for(spec, act, 16, 4, (16, 32))
        np.testing.assert_array_equal(selected, full[:, :4])

    @unittest.skipUnless(importlib.util.find_spec("constriction"), "optional constriction dependency")
    def test_complete_runner_packet_only_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "act"; source.mkdir()
            for name, seed in [("train0", 1), ("train1", 2), ("tables", 3), ("test_text", 4)]:
                np.savez(source / (name + ".npz"), **fixture(seed))
            with mock.patch("builtins.print"):
                status = runner.main(["--act-dir", str(source), "--train-cases", "2", "--test-cases", "1",
                    "--block-size", "2", "--metadata-ranks", "1", "--coset-bits", "1", "--page-tokens", "2",
                    "--output", str(root / "output")])
            self.assertEqual(status, 0)
            import json
            report = json.loads((root / "output/results.json").read_text())
            self.assertTrue(report["test_rows"])
            self.assertTrue(all(row["packet_only_subprocess"] and row["bit_exact"] for row in report["test_rows"]))
            for row in report["test_rows"]:
                self.assertEqual(len(row["decoder_inputs"]), 2)
                self.assertAlmostEqual(row["encode_wall_seconds"], row["metadata_prepare_seconds"] + row["encode_codec_wall_seconds"])
                self.assertEqual(row["shared_state_bytes"], row["resident_decoder_state_bytes"] + row["resident_encoder_projector_bytes"])


if __name__ == "__main__":
    unittest.main()

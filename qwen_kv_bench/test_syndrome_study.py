"""Real packet/forward integration, selection provenance and decision ledgers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import syndrome_study as study, syndrome_codec as codec
from qwen_kv_bench.test_fixtures import FakeTokenizer


def record(layer=0, start=0, tokens=32):
    rng = np.random.default_rng(71 + layer + start)
    values = rng.integers(0, 65536, (tokens, 8), dtype=np.uint16)
    return dict(layer=layer, start=start, tokens=tokens, head_dim=4,
        k=values[:, :4].copy(), v=values[:, 4:].copy(), hidden=values.copy(),
        k_pre_rope=values[:, :4].copy(), k_proj=values[:, :4].copy(), k_norm_eps=1e-6)


class SyndromeStudyTests(unittest.TestCase):
    def test_defaults_and_input_preflight_fail_before_any_fit(self):
        args = study._parser().parse_args([]); study._defaults(args)
        self.assertEqual(args.steps, 1000)
        self.assertEqual(args.tile_tokens, 64)
        self.assertEqual(args.dev_tokens, 128)
        self.assertEqual(args.modes, ["none", "kv", "kv_canonical", "hidden", "hidden_norm"])
        args.dev_tokens = 16
        with self.assertRaisesRegex(ValueError, "--dev-tokens=16"):
            study.previous._preflight_inputs(FakeTokenizer(), args)
        tiny = study._parser().parse_args(["--tiny"]); study._defaults(tiny)
        self.assertEqual(tiny.device, "cpu")
        self.assertGreaterEqual(tiny.tile_tokens, 31)

    def test_lightweight_field_control_reopens_without_neural_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            centers = np.full((2, 8), 127, np.uint8)
            prepared, state = study._field_state(centers, output, 4)
            self.assertEqual(state["resident_bytes"], centers.nbytes)
            self.assertEqual(state["serialized_bytes"], (output / state["file"]).stat().st_size)
            self.assertEqual(state["sha256"], hashlib.sha256((output / state["file"]).read_bytes()).hexdigest())
            self.assertLess(state["charged_bytes"], 1024)
            rows = [record(layer, start) for layer in range(2) for start in (0, 32)]
            args = study._parser().parse_args(["--tiny"]); study._defaults(args)
            geometry = dict(layers=2, tokens=64, channels=4)
            wire, stats = study._encode(rows, prepared, geometry, args, None, 7, "syndrome")
            self.assertEqual(sum(stats["component_bytes"].values()), len(wire))
            restored, measured = study.restore_pages(wire, prepared, require_full=True, seed=2)
            self.assertTrue(measured["order_is_shuffled"])
            self.assertEqual(measured["raw_prerequisite_pages"], 0)
            self.assertEqual(measured["returned_output_bytes"], 2 * 2 * 64 * 4 * 2)
            for row in rows:
                for kind, index in (("k", 0), ("v", 1)):
                    np.testing.assert_array_equal(restored[row["layer"]][index][row["start"]:row["start"] + 32], row[kind])
            self.assertEqual(stats["coding_summary"]["totals"]["pages_using_short_correction"], 0)

    def test_restore_rejects_missing_or_duplicate_addresses(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, _ = study._field_state(np.full((2, 8), 127, np.uint8), Path(directory), 4)
            page, _ = codec.encode_page(record(), prepared, residue_bits=7)
            incomplete = study.base.pack_pages([page], layers=2, tokens=64, channels=4, tile_tokens=32)
            with self.assertRaisesRegex(ValueError, "every layer/token"):
                study.restore_pages(incomplete, prepared, require_full=True)
            duplicate = study.base.pack_pages([page, page], layers=2, tokens=64, channels=4, tile_tokens=32)
            with self.assertRaisesRegex(ValueError, "address"):
                study.restore_pages(duplicate, prepared)

    def test_input_hash_includes_actual_projection_and_normalizer_eps(self):
        rows = [record()]
        before = study._producer_inputs_hash(rows)
        changed = copy.deepcopy(rows)
        changed[0]["k_proj"][0, 0] ^= 1
        self.assertEqual(study.base._records_hash(rows), study.base._records_hash(changed))
        self.assertNotEqual(before, study._producer_inputs_hash(changed))
        changed = copy.deepcopy(rows); changed[0]["k_norm_eps"] *= 2
        self.assertNotEqual(before, study._producer_inputs_hash(changed))

    def test_code_diagnostics_and_ablations_preserve_budget(self):
        codes = {(0, 0): np.array([[0, 0x3f80], [0x4000, 0x3f80], [0x4040, 0]], np.uint16)}
        summary = study._feature_summary(codes)
        self.assertEqual(summary["bytes"], 12)
        self.assertEqual(summary["distinct_words_by_feature"], [3, 2])
        self.assertEqual(summary["nonfinite_words"], 0)
        for altered in study._ablation_codes(codes, 71).values():
            self.assertEqual(altered[(0, 0)].shape, codes[(0, 0)].shape)
            self.assertEqual(altered[(0, 0)].nbytes, codes[(0, 0)].nbytes)

    def test_tiny_all_modes_real_forward_packet_selection_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(study, "_note"):
            output = Path(directory) / "run"
            result = study.main(["--tiny", "--steps", "1", "--residue-bits", "2", "6",
                "--generation-steps", "1", "--repeats", "1", "--max-pairs", "4",
                "--threads", "2", "--output", str(output)])
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(result, 0, report.get("traceback"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["generation_checks"]), 5)
            self.assertEqual(len(report["models"]), 5)
            self.assertEqual(len(report["selection"]), 5)
            for mode, model in report["models"].items():
                self.assertTrue(model["reloaded_from_calibration"])
                self.assertTrue(all(row["file"].startswith("calibration/") for row in model["files"]))
                self.assertEqual(sum(row["bytes"] for row in model["files"]), model["serialized_state_bytes"])
                candidates = report["development_grid"][mode]
                best = min(candidates, key=lambda row: (row["packet_bytes"], row["residue_bits"]))
                self.assertEqual(report["selection"][mode], best["residue_bits"])
                self.assertTrue(all(row["roundtrip_exact"] for row in candidates))
                if mode != "none":
                    self.assertEqual({row["kind"] for row in report["development_code_ablations"][mode]}, {"zero", "shuffled"})
            freeze = json.loads((output / "frozen_manifest.json").read_text())
            self.assertEqual(freeze["residue_selection"], report["selection"])
            self.assertEqual(freeze["selection_provenance"]["split"], "dev")
            self.assertEqual(freeze["training_hashes"], [row["producer_inputs_sha256"] for row in report["training_captures"]])
            for check in report["generation_checks"]:
                self.assertTrue(study.previous._generation_passed(check["result"]))
                self.assertEqual(check["complete_prefix_tokens"], 32)
                self.assertIsInstance(check["syndrome_recovery_exercised"], bool)
            case = report["cases"][0]
            self.assertEqual(case["coverage"]["measured_pages"], 4)
            self.assertEqual(case["common_residue_bits"], report["selection"]["none"])
            self.assertEqual(set(case["common_residue_methods"]), set(report["models"]))
            for pair in case["syndrome_positions"].values():
                self.assertEqual(pair["packet_byte_difference"], 0)
            for row in case["results"]:
                if row["name"] == "frozen_exponent_control":
                    continue
                measured = row["measurement"]
                self.assertEqual(sum(measured["component_bytes"].values()), measured["packet_bytes"])
                self.assertTrue(measured["roundtrip_exact"])
                self.assertTrue(measured["all_repeats_exact"])
                self.assertEqual(measured["maximum_dependency_depth"], 0)
                self.assertEqual(measured["raw_prerequisite_pages"], 0)
                self.assertEqual(row["accounting"]["sample"]["packet_bytes"], measured["packet_bytes"])
                self.assertEqual(len(measured["records"]), 4)
                self.assertEqual(set(measured["by_layer"]), {"0", "1"})
                if row["name"] != "field_baseline":
                    self.assertTrue(measured["inline_codes_used"])
                    self.assertTrue(all(page["feature_words_supplied"] for page in measured["records"]))
            decision = json.loads((output / "decision_summary.json").read_text())
            self.assertEqual(decision["status"], "completed")
            self.assertTrue(decision["cases"][0]["common_residue_comparisons"])

    def test_controlled_tiny_model_exercises_short_syndrome_in_generation(self):
        # Random untrained tiny-model predictions can legitimately escape every
        # group. A deliberately predictable real forward checks the retained
        # syndrome path too; this is not a pretrained rate/quality result.
        from qwen_kv_bench import producer_study
        original = producer_study._tiny
        def predictable(device, args):
            model, tokenizer, identity = original(device, args)
            with torch.no_grad():
                for layer in model.model.layers:
                    layer.self_attn.k_proj.weight.zero_()
                    layer.self_attn.v_proj.weight.zero_()
            identity = dict(identity, controlled_zero_KV=True,
                scope="Deliberately zero KV projections; actual forward and generation correctness only, no useful prediction/rate claim")
            return model, tokenizer, identity
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(study, "_note"), \
                mock.patch.object(producer_study, "_tiny", side_effect=predictable):
            output = Path(directory) / "run"
            result = study.main(["--tiny", "--steps", "1", "--residue-bits", "6", "--modes", "none", "hidden",
                "--generation-steps", "1", "--repeats", "1", "--max-pairs", "4",
                "--threads", "2", "--output", str(output)])
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(result, 0, report.get("traceback"))
            self.assertTrue(report["model_identity"]["controlled_zero_KV"])
            self.assertTrue(all(check["syndrome_recovery_exercised"] for check in report["generation_checks"]))
            self.assertTrue(all(study.previous._generation_passed(check["result"]) for check in report["generation_checks"]))
            for row in report["cases"][0]["results"]:
                if row["name"].startswith("coset_"):
                    self.assertGreater(row["measurement"]["coding_summary"]["totals"]["retained_short_groups"], 0)


if __name__ == "__main__":
    unittest.main()

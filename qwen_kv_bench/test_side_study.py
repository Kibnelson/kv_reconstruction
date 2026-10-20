"""Independent restoration gates, accounting and actual inline code provenance."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import side_study as study
from qwen_kv_bench.fixtures import DEV
from qwen_kv_bench.test_fixtures import FakeTokenizer


class SideStudyTests(unittest.TestCase):
    def test_complete_container_roundtrip_and_corruption_rejection(self):
        wire = study.pack_pages([b"first", b"second"], layers=2, tokens=16, channels=4, tile_tokens=8)
        meta, pages = study.unpack_pages(wire)
        self.assertEqual(pages, (b"first", b"second"))
        self.assertEqual(meta["layers"], 2)
        self.assertEqual(len(wire), study._HEADER.size + len(study._json(meta)) + 3 * 8 + 11)
        for damaged in (wire[:-1], wire[:8], wire[:-1] + bytes([wire[-1] ^ 1])):
            with self.assertRaises(ValueError):
                study.unpack_pages(damaged)
        with self.assertRaises(ValueError):
            study.pack_pages([], layers=2, tokens=16, channels=4, tile_tokens=8)

    def test_restore_shuffles_and_requires_only_own_packet_and_state(self):
        from qwen_kv_bench import side_codec as codec
        pages = (b"a", b"b", b"c", b"d")
        addresses = {page: (layer, start) for page, (layer, start) in
                     zip(pages, ((0, 0), (0, 2), (1, 0), (1, 2)))}
        state = SimpleNamespace(device=torch.device("cpu"))
        def metadata(page):
            layer, start = addresses[page]
            return dict(layer=layer, start=start, tokens=2, channels=3)
        calls = []
        def decode(page, prepared, *, return_stats):
            self.assertIs(prepared, state); self.assertTrue(return_stats)
            calls.append(page)
            layer, start = addresses[page]
            words = np.full((2, 6), 100 * layer + start, np.uint16)
            return words, dict(prediction_seconds=.01, entropy_decode_seconds=.02,
                              decode_seconds=.03, feature_bytes=4, dependency_depth=0)
        wire = study.pack_pages(pages, layers=2, tokens=4, channels=3, tile_tokens=2)
        with mock.patch.object(codec, "page_metadata", side_effect=metadata), \
             mock.patch.object(codec, "decode_page", side_effect=decode):
            restored, metrics = study.restore_pages(wire, state, require_full=True, seed=2)
        self.assertNotEqual(calls, list(pages)); self.assertCountEqual(calls, pages)
        self.assertEqual(metrics["maximum_dependency_depth"], 0)
        self.assertEqual(metrics["raw_prerequisite_pages"], 0)
        self.assertEqual(metrics["returned_output_bytes"], 2 * 2 * 4 * 3 * 2)
        np.testing.assert_array_equal(restored[1][1][2:], np.full((2, 3), 102, np.uint16))

    def test_restore_rejects_missing_full_coverage_and_duplicate_addresses(self):
        from qwen_kv_bench import side_codec as codec
        state = SimpleNamespace(device=torch.device("cpu"))
        meta = dict(layer=0, start=0, tokens=2, channels=3)
        for pages, full in (([b"a"], True), ([b"a", b"b"], False)):
            wire = study.pack_pages(pages, layers=2, tokens=4, channels=3, tile_tokens=2)
            with mock.patch.object(codec, "page_metadata", return_value=meta), \
                 mock.patch.object(codec, "decode_page") as decoder:
                with self.assertRaises(ValueError):
                    study.restore_pages(wire, state, require_full=full)
                decoder.assert_not_called()

    def test_defaults_match_shared_total_updates_and_preserve_prompt_preflight(self):
        args = study._parser().parse_args([]); study._defaults(args)
        self.assertEqual(args.steps, args.shared_steps)
        self.assertEqual(args.width, 128); self.assertEqual(args.shared_width, 512)
        self.assertEqual(args.dev_tokens, 128)
        args.dev_tokens = 16
        with self.assertRaisesRegex(ValueError, "--dev-tokens=16"):
            study.previous._preflight_inputs(FakeTokenizer(), args)
        args = study._parser().parse_args(["--tiny"]); study._defaults(args)
        self.assertEqual(args.device, "cpu"); self.assertEqual(args.tile_tokens, 8)
        self.assertEqual(args.width, args.shared_width)

    def test_code_is_emitted_inside_actual_model_forward(self):
        # Real tiny Qwen hooks call the frozen encoder while the root forward
        # is active. The retained words survive after prepared state release.
        from qwen_kv_bench.producer_study import _tiny
        torch.set_num_threads(2); torch.manual_seed(71)
        args = study._parser().parse_args(["--tiny"]); study._defaults(args)
        model, tokenizer, _ = _tiny("cpu", args)
        active = {"forward": False}
        seen = []
        def pre(*_):
            active["forward"] = True
        def post(*_):
            active["forward"] = False
        handles = [model.register_forward_pre_hook(pre), model.register_forward_hook(post)]
        class Prepared:
            device = torch.device("cpu")
            def encode_features(self, record):
                seen.append(active["forward"])
                return np.zeros((record["tokens"], 2), np.uint16)
        fitted = SimpleNamespace(prepare=lambda device: Prepared())
        inline = study.InlineCodes({"hidden": fitted}, "cpu")
        try:
            capture, records, _ = study._capture(model, tokenizer, DEV[0], 8, "cpu", args, inline=inline)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(seen); self.assertTrue(all(seen))
        self.assertFalse(active["forward"])
        self.assertEqual(inline.report()["methods"]["hidden"]["calls"], len(records))
        self.assertTrue(capture["inline_side_codes"]["observed_inside_attention_forward"])
        inline.release_prepared()
        self.assertFalse(inline.prepared)
        self.assertEqual(len(inline.codes["hidden"]), len(records))
        self.assertTrue(all(not array.flags.writeable for array in inline.codes["hidden"].values()))

    def test_tiny_main_all_variants_actual_packets_and_generation(self):
        # One actual forward/training/packet/generation integration protects
        # the column geometry, callback code reuse and reporting interfaces.
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(study, "_note"):
            output = Path(directory) / "run"
            status = study.main(["--tiny", "--device", "cpu", "--threads", "2",
                "--steps", "1", "--shared-steps", "1", "--lengths", "16",
                "--cases", "1", "--generation-steps", "1", "--max-pairs", "4",
                "--output", str(output)])
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(status, 0, report.get("traceback"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["generation_checks"]), 5)
            for check in report["generation_checks"]:
                self.assertEqual(check["status"], "ok")
                self.assertTrue(study.previous._generation_passed(check["result"]))
            self.assertEqual(len(report["cases"]), 1)
            case = report["cases"][0]
            self.assertEqual({r["name"] for r in case["results"]},
                {"frozen_exponent_control", "previous_layer_shared", "side_none",
                 "side_kv", "side_kv_canonical", "side_hidden"})
            self.assertTrue(case["capture"]["inline_side_codes"]["observed_inside_attention_forward"])
            for row in case["results"]:
                if not row["name"].startswith("side_"):
                    continue
                measure = row["measurement"]
                self.assertTrue(measure["inline_codes_used"])
                self.assertTrue(measure["roundtrip_exact"])
                self.assertEqual(measure["maximum_dependency_depth"], 0)
                self.assertEqual(sum(measure["component_bytes"].values()), measure["packet_bytes"])
                self.assertEqual(row["accounting"]["sample"]["packet_bytes"], measure["packet_bytes"])
                self.assertTrue(measure["independent_packet_decode"]["order_is_shuffled"])
                self.assertTrue(all(r["feature_words_supplied"] for r in measure["records"]))

    def test_sample_comparison_requires_every_requested_address(self):
        words = np.zeros((4, 3), np.uint16)
        matrices = [(words, words), (words, words)]
        sample = [dict(layer=layer, start=start, tokens=2,
                       k=words[start:start + 2], v=words[start:start + 2])
                  for layer in range(2) for start in (0, 2)]
        study._check_sample(sample, matrices, [0, 2], 2)
        with self.assertRaises(AssertionError):
            study._check_sample(sample[:-1], matrices, [0, 2], 2)
        with self.assertRaises(AssertionError):
            study._check_sample(sample + sample[:1], matrices, [0, 2], 2)


if __name__ == "__main__":
    unittest.main()

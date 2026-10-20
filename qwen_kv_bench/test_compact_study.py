"""Real tiny forwards, frozen storage selection and independent page assembly."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from qwen_kv_bench import compact_study as study, compact_models as models
from qwen_kv_bench import compact_codec as codec
from qwen_kv_bench.test_compact_codec import Prepared, HintPredictor
from qwen_kv_bench.test_fixtures import FakeTokenizer


def selection_row(name, total, *, packet=100, reduction=10):
    return dict(name=name, status="ok", measurement=dict(packet_bytes=packet),
        accounting=dict(full_cache_estimate=dict(estimated_total_bytes=total, net_reduction_pct=reduction)))


class CompactStudyTests(unittest.TestCase):
    def test_budgets_and_missing_matched_controls_fail_before_fit(self):
        parser = study._parser()
        args = parser.parse_args([]); study._defaults(args)
        self.assertEqual(args.steps, 4096)
        self.assertEqual(args.lengths, [2048, 8192])
        self.assertEqual(args.dev_tokens, 128)
        self.assertEqual(len(study._validate(args, parser, models.default_specs())), 14)
        args.dev_tokens = 16
        with self.assertRaisesRegex(ValueError, "--dev-tokens=16"):
            study.previous._preflight_inputs(FakeTokenizer(), args, test_fixtures=study.FRESH_TEST)
        for requested in (["lsb_pool_hidden2"], ["msb_pool_hidden2", "lsb_pool_none"],
                          ["lsb_pool_none", "lsb_pool_none"], ["unknown"]):
            args = parser.parse_args(["--variants", *requested]); study._defaults(args)
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                study._validate(args, parser, models.default_specs())

    def test_storage_selection_includes_state_and_test_cannot_replace_it(self):
        # A small packet does not win when its complete state costs more.
        rows = [selection_row("tiny_packet_large_state", 1100, packet=100),
                selection_row("field_baseline", 250, packet=200)]
        self.assertEqual(study._select(rows), "field_baseline")
        self.assertEqual(study._select([selection_row("b", 100), selection_row("a", 100)]), "a")
        report = dict(status="completed", selection=dict(winner="lsb_pool_hidden2"), specs=models.default_specs(),
            cases=[dict(name="heldout", tokens=8192, results=[
                selection_row("lsb_pool_hidden2", 600, packet=500, reduction=25),
                selection_row("field_baseline", 500, packet=400, reduction=35),
                selection_row("frozen_exponent_control", 520, packet=410, reduction=34)])])
        decision = study._decisions(report)
        self.assertEqual(decision["selection"]["winner"], "lsb_pool_hidden2")
        self.assertFalse(decision["cases"][0]["frozen_winner_beats_storage_controls"])
        comparison = decision["cases"][0]["comparisons"][0]
        self.assertEqual(comparison["packet_bytes_delta"], 100)
        self.assertEqual(comparison["estimated_net_savings_pp_delta"], -10)

    def test_shuffled_real_page_container_checks_completeness_and_duplicates(self):
        p = Prepared(source="none", side_bits=0)
        p.device = "cpu"
        records = []
        matrices = [(np.empty((8, 4), np.uint16), np.empty((8, 4), np.uint16)) for _ in range(2)]
        for layer in range(2):
            for start in (0, 4):
                words = np.random.default_rng(71 + layer * 10 + start).integers(0, 65536, (4, 8), dtype=np.uint16)
                matrices[layer][0][start:start + 4] = words[:, :4]
                matrices[layer][1][start:start + 4] = words[:, 4:]
                records.append(dict(layer=layer, start=start, tokens=4, head_dim=4,
                    k=words[:, :4].copy(), v=words[:, 4:].copy()))
        args = study._parser().parse_args(["--tiny", "--tile-tokens", "4"]); study._defaults(args)
        wire, measured = study._encode(records, p, dict(layers=2, tokens=8, channels=4), args)
        restored, decoded = study.restore_pages(wire, p, require_full=True)
        self.assertTrue(decoded["order_is_shuffled"])
        self.assertFalse(decoded["source_arrays_used"])
        self.assertEqual(study.previous._matrices_hash(restored), study.previous._matrices_hash(matrices))
        self.assertEqual(sum(measured["component_bytes"].values()), len(wire))
        geometry, pages = study.base.unpack_pages(wire)
        missing = study.base.pack_pages(pages[:-1], layers=2, tokens=8, channels=4, tile_tokens=4)
        with self.assertRaisesRegex(ValueError, "requires all"):
            study.restore_pages(missing, p, require_full=True)
        duplicate = study.base.pack_pages([*pages, pages[0]], layers=2, tokens=8, channels=4, tile_tokens=4)
        with self.assertRaisesRegex(ValueError, "address"):
            study.restore_pages(duplicate, p)

    def test_header_only_packet_improvement_is_not_producer_payload_gain(self):
        current = selection_row("lsb_pool_hidden2", 1400, packet=992, reduction=20)
        control = selection_row("lsb_pool_none", 1100, packet=1000, reduction=22)
        current["measurement"]["component_bytes"] = dict(stage0_bytes=300, stage1_bytes=300,
            base_bytes=200, hint_bytes=8, framing_bytes=180, container_framing_bytes=4)
        control["measurement"]["component_bytes"] = dict(stage0_bytes=300, stage1_bytes=300,
            base_bytes=200, hint_bytes=0, framing_bytes=196, container_framing_bytes=4)
        current["accounting"]["state"] = dict(total_charged_bytes=400)
        control["accounting"]["state"] = dict(total_charged_bytes=100)
        specs = {row["name"]: row for row in models.default_specs()}
        row = study._comparison(current, control, specs[current["name"]], specs[control["name"]])
        self.assertEqual(row["packet_bytes_delta"], -8)
        self.assertEqual(row["packet_partition_delta"], dict(main_payload_bytes=0,
            hint_payload_bytes=8, framing_bytes=-16, learned_stage_payload_bytes=0))
        self.assertEqual(row["charged_state_bytes_delta"], 300)
        self.assertTrue(row["packet_gain_without_main_payload_reduction"])
        self.assertTrue(row["producer_gain_requires_coded_payload_reduction"])
        self.assertFalse(row["producer_specific_payload_gain_observed"])
        current["measurement"]["component_bytes"]["stage0_bytes"] -= 10
        current["measurement"]["packet_bytes"] -= 10
        row = study._comparison(current, control, specs[current["name"]], specs[control["name"]])
        self.assertTrue(row["producer_specific_payload_gain_observed"])
        self.assertEqual(row["packet_partition_delta"]["learned_stage_payload_bytes"], -10)

    def test_dev_hint_ablations_mask_unknown_bits_and_handle_page_sharing(self):
        hints = np.array([1, 2, 3, 0, 2, 1, 0, 3], np.uint8)
        words = np.full((len(hints), 8), 0x3f80, np.uint16)
        words |= hints[:, None].astype(np.uint16) << 2
        words |= hints[:, None].astype(np.uint16) << 4
        record = dict(layer=0, start=0, tokens=len(hints), head_dim=4,
            k=words[:, :4], v=words[:, 4:])
        p = HintPredictor()
        result = study._dev_hint_ablations([record], p, {(0, 0): hints}, maximum=1, seed=71)
        by_kind = {row["kind"]: row for row in result["rows"]}
        self.assertLess(by_kind["original"]["rate"]["integer_cdf_bits"], by_kind["zero_id"]["rate"]["integer_cdf_bits"])
        self.assertGreater(by_kind["zero_id"]["NLL_delta_vs_original_bits_per_KV_value"], 0)
        self.assertEqual(set(by_kind["original"]["rate"]["by_kind"]), {"K", "V"})
        self.assertFalse(by_kind["original"]["rate"]["entropy_streams_constructed"])
        # HintPredictor checks all future mantissa fields are absent at every call.
        self.assertEqual(len(p.calls), 6)
        page = HintPredictor(page_shared=True)
        result = study._dev_hint_ablations([record], page, {(0, 0): np.array([2], np.uint8)}, maximum=1, seed=71)
        shuffled = result["rows"][-1]
        self.assertEqual(shuffled["kind"], "shuffled_pages_within_layer")
        self.assertEqual(shuffled["status"], "not_applicable")
        self.assertNotIn("rate", shuffled)
        codes = {(0, 0): np.array([1], np.uint8), (0, 8): np.array([3], np.uint8), (1, 0): np.array([2], np.uint8)}
        swapped = study._hint_ablations(codes, page.spec, seed=71)[-1]
        self.assertEqual(swapped["eligible_pages"], 2)
        self.assertEqual(int(swapped["codes"][(0, 0)][0]), 3)
        self.assertEqual(int(swapped["codes"][(1, 0)][0]), 2)

    def test_all_fourteen_variants_real_tiny_forward_and_metadata_accounting(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(study, "_note"):
            output = Path(directory) / "run"
            result = study.main(["--tiny", "--steps", "1", "--cases", "1", "--max-pairs", "4",
                "--repeats", "1", "--generation-steps", "1", "--threads", "2", "--output", str(output)])
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(result, 0, report.get("traceback"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(set(report["models"]), {s["name"] for s in models.default_specs()})
            freeze = json.loads((output / "frozen_manifest.json").read_text())
            self.assertEqual(freeze["selection"], report["selection"])
            self.assertEqual(report["selection"]["split"], "dev")
            self.assertEqual(report["selection"]["selection_reference_cache_tokens"], 64)
            expected = study._select(report["development_results"])
            self.assertEqual(report["selection"]["winner"], expected)
            for row in report["development_results"]:
                if row["name"] != "frozen_exponent_control":
                    self.assertTrue(row["measurement"]["roundtrip_exact"])
            checked = {row["name"] for row in report["generation_checks"]}
            self.assertTrue({"lsb_pool_hidden2", "msb_pool_hidden2", "lsb_kfirst_hidden2", "field_baseline"} <= checked)
            self.assertEqual(len(report["development_hint_ablations"]), 10)
            for diagnostic in report["development_hint_ablations"].values():
                self.assertTrue(diagnostic["sampled_addresses"])
                self.assertEqual(diagnostic["rows"][0]["kind"], "original")
            for check in report["generation_checks"]:
                self.assertTrue(study.previous._generation_passed(check["result"]))
            case = report["cases"][0]
            self.assertEqual(case["coverage"]["measured_pages"], 4)
            self.assertEqual(len(case["results"]), 16)
            for row in case["results"]:
                if row["name"] == "frozen_exponent_control":
                    continue
                measured = row["measurement"]
                self.assertTrue(measured["all_repeats_exact"])
                self.assertTrue(measured["roundtrip_exact"])
                self.assertEqual(sum(measured["component_bytes"].values()), measured["packet_bytes"])
                self.assertEqual(row["accounting"]["sample"]["packet_bytes"], measured["packet_bytes"])
                self.assertTrue(measured["independent_packet_decode"]["order_is_shuffled"])
                if row["name"] in report["models"]:
                    self.assertTrue(measured["inline_hints_used"])
                    spec = report["models"][row["name"]]["spec"]
                    for page in measured["records"]:
                        expected_ids = 0 if spec["source"] == "none" else 1 if spec["page_shared"] else page["tokens"]
                        self.assertEqual(page["hint_bytes"], (expected_ids * spec["side_bits"] + 7) // 8)
                    artifact = measured["hint_array"]
                    with np.load(output / artifact["file"], allow_pickle=False) as hints:
                        self.assertEqual(len(hints.files), 4)
                        self.assertTrue(all(hints[name].dtype == np.uint8 for name in hints.files))
                    self.assertEqual(len(report["training_hint_arrays"][row["name"]]), 3)
            self.assertTrue((output / "decision_summary.json").is_file())
            self.assertTrue((output / "source_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

"""Independent accounting checks for sampled conditional experiment summaries."""
import copy
import unittest

from qwen_kv_bench.conditional_reporting import analyze, protocol


def fixture():
    traces = [dict(layer=0, start=0, tokens=2, raw_bytes=64, serialized_bytes=40,
                   recipe_diagnostics=dict(mode="coded", error_hist=[3, 1], feature_bytes=4)),
              dict(layer=1, start=0, tokens=2, raw_bytes=64, serialized_bytes=52,
                   recipe_diagnostics=dict(mode="raw", error_hist=[2, 2], feature_bytes=4))]
    measurement = dict(size_is_estimated=True, estimated_full_cache_serialized_bytes=256,
        serialized_bytes=148, full_raw_payload_bytes=256, raw_payload_bytes=128,
        records_total=4, records_measured=2, decoder_state_bytes=10, state_id="state",
        component_bytes=dict(header=20, exponent=32, lower=40, cache_header=32, pair_offset_table=24),
        sample_decode=dict(median_seconds=0.1, payload_bytes=128),
        sample_packet_transfer_then_decode=dict(median_seconds=0.2, payload_bytes=128),
        encode_seconds_total=0.3, record_diagnostics=traces)
    specs = [dict(name="control", family="control"), dict(name="candidate", family="anchor")]
    return dict(status="completed", candidates=specs, experiment_protocol=protocol(specs),
                cases=[dict(name="test", split="test", tokens=4, results=[
                    dict(name=spec["name"], spec=spec, status="ok", measurement=copy.deepcopy(measurement))
                    for spec in specs])])


class ConditionalReportingTests(unittest.TestCase):
    def test_running_checkpoint_before_fit_or_freeze_is_reportable(self):
        report = analyze(dict(status="running", source_manifest=None, freeze=None,
                              calibration=None, cases=[], generation_checks=[]))
        self.assertEqual(report["status"], "running")
        self.assertIsNone(report["frozen_sha256"])
        self.assertEqual(report["cases"], [])

    def test_full_framing_and_state_are_not_extrapolated_twice(self):
        report = analyze(fixture())
        method = report["cases"][0]["methods"][0]
        self.assertEqual(method["effective_bits_per_word"], 16 * 266 / 256)
        self.assertEqual(sum(method["component_bits_per_word"].values()), 16 * 266 / 256)
        self.assertEqual(method["component_bits_per_word"]["outer_cache_framing"], 16 * 72 / 256)
        self.assertFalse(method["target_40_percent_met"])
        self.assertTrue(method["size_is_estimated"])
        self.assertEqual(report["test_case_averages"][0]["cases"], 1)

    def test_histograms_modes_and_matching_controls_remain_visible(self):
        report = analyze(fixture())
        method = report["cases"][0]["methods"][1]
        self.assertEqual(method["diagnostics"]["numeric"]["error_hist[0]"]["sum"], 5)
        self.assertEqual(method["diagnostics"]["categorical"]["mode"], {"coded": 1, "raw": 1})
        self.assertEqual(method["matched_control_deltas"]["control"]["savings_percentage_points"], 0)
        self.assertEqual(len(method["per_layer_record_bits_per_word"]), 2)

    def test_sample_with_short_tail_uses_record_extrapolation_contract(self):
        source = fixture()
        # Selected records may include a shorter tail. The benchmark estimates
        # inner bytes by record count, not by measured/full raw-byte ratio.
        for row in source["cases"][0]["results"]:
            row["measurement"]["raw_payload_bytes"] = 96
            row["measurement"]["record_diagnostics"][0]["raw_bytes"] = 32
        report = analyze(source)
        self.assertEqual(report["cases"][0]["methods"][0]["effective_bits_per_word"], 16 * 266 / 256)
        self.assertEqual(sum(report["cases"][0]["methods"][0]["component_bits_per_word"].values()), 16 * 266 / 256)

    def test_unmatched_addresses_do_not_receive_a_matched_comparison(self):
        source = fixture()
        source["cases"][0]["results"][1]["measurement"]["record_diagnostics"][0]["start"] = 2
        report = analyze(source)
        self.assertFalse(report["cases"][0]["all_successful_methods_use_same_ordered_records"])
        self.assertEqual(report["cases"][0]["methods"][1]["matched_control_deltas"], {})

    def test_errors_are_preserved_and_dev_is_excluded_from_test_means(self):
        source = fixture()
        source["cases"][0]["split"] = "dev"
        source["cases"][0]["results"].append(dict(name="failed", status="error", error="deliberate"))
        report = analyze(source)
        self.assertEqual(report["test_case_averages"], [])
        self.assertEqual(report["cases"][0]["methods"][-1]["error"], "deliberate")


if __name__ == "__main__":
    unittest.main()

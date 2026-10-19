"""Independent arithmetic checks for matched recipe and size-oracle reports."""
import copy
import json
import unittest

from qwen_kv_bench.recipe_analysis import analyze_case


def _row(name, sizes, *, state=0, method="prefill", options=None, total=None,
         raw_sizes=None, kinds=None, model=None):
    count = len(sizes)
    total = count if total is None else total
    raw_sizes = raw_sizes or [1024] * count
    kinds = kinds or ["K", "V"][:count]
    if len(kinds) < count:
        kinds = ["K" if index % 2 == 0 else "V" for index in range(count)]
    records = [dict(ordinal=index, layer=index // 2, kind=kinds[index], start=0,
        tokens=16, raw_bytes=raw_sizes[index], serialized_bytes=size,
        component_bytes=dict(header=10, correction=size-10),
        encode_seconds=.001, decode_seconds=.002)
        for index, size in enumerate(sizes)]
    raw = sum(raw_sizes)
    outer = 32 + 8 * (count + 1)
    opts = dict(predictor="affine", correction="residual", layout="channel", group_size=8)
    opts.update(options or {})
    spec = dict(name=name, method=method, options=opts)
    if model is not None:
        spec["recipe_model"] = model
    return dict(name=name, track="native_packet", status="ok", spec=spec,
        measurement=dict(records_measured=count, records_total=total,
            raw_payload_bytes=raw, full_raw_payload_bytes=raw * total // count,
            serialized_bytes=sum(sizes) + outer, actual_record_payload_bytes=sum(sizes),
            decoder_state_bytes=state, native_state_physical_bytes=state,
            state_id="no_learned_state" if state == 0 else "frozen_state_id",
            exact_verified=True, roundtrip_exact=True,
            record_diagnostics=records, encode_seconds_total=.02,
            decode_seconds_total=.04, execution="tensor_native_cpu_functional", device="cpu",
            sample_decode=dict(median_seconds=.003, payload_bytes=1024),
            sample_packet_transfer_then_decode=dict(median_seconds=.005, payload_bytes=2048)))


class RecipeAnalysisTests(unittest.TestCase):
    def case(self, *rows):
        return dict(name="test_fixture", split="test", tokens=2048, results=list(rows))

    def test_stateless_oracle_counts_all_inner_mode_and_framing_bytes(self):
        a = _row("channel", [500, 800])
        b = _row("time", [650, 600], options=dict(layout="time"))
        result = analyze_case(self.case(a, b))
        oracle = result["stateless_size_oracle"]
        self.assertEqual(oracle["status"], "ok")
        self.assertEqual(oracle["chosen_record_payload_bytes"], 1100)
        self.assertEqual(oracle["mode_id_bytes"], 2)
        self.assertEqual(oracle["measured_outer_framing_bytes"], 56)
        self.assertEqual(oracle["measured_total_bytes"], 1158)
        self.assertEqual(oracle["best_fixed_stateless_candidate"], "time")
        self.assertEqual(oracle["best_fixed_measured_total_bytes"], 1306)
        self.assertEqual(oracle["measured_bytes_saved_over_best_fixed"], 148)
        self.assertEqual(oracle["mode_counts"], dict(channel=1, time=1))
        self.assertEqual(oracle["mode_counts_by_kind"], dict(K=dict(channel=1), V=dict(time=1)))
        self.assertFalse(oracle["full_cache_size_is_estimated"])
        self.assertAlmostEqual(oracle["measured_savings_percent"], 100 * (1 - 1158 / 2048))
        self.assertIn("SIZE-ONLY", oracle["scope"])
        self.assertIn("not serialized", oracle["accounting"])

    def test_sampled_oracle_scales_inner_and_selectors_before_full_framing(self):
        a = _row("channel", [500, 800], total=20)
        b = _row("time", [650, 600], options=dict(layout="time"), total=20)
        oracle = analyze_case(self.case(a, b))["stateless_size_oracle"]
        self.assertEqual(oracle["full_cache_total_bytes"], 11020 + 32 + 8 * 21)
        self.assertEqual(oracle["best_fixed_full_cache_total_bytes"], 12500 + 32 + 8 * 21)
        self.assertEqual(oracle["full_cache_bytes_saved_over_best_fixed"], 1480)
        self.assertTrue(oracle["full_cache_size_is_estimated"])
        self.assertAlmostEqual(oracle["full_cache_savings_gain_over_best_fixed_percentage_points"], 100 * 1480 / 20480)

    def test_stateful_dictionary_excluded_and_state_charged_once_in_pair(self):
        plain = _row("affine", [700, 700], total=20)
        learned = _row("dictionary", [600, 600], state=128, method="prefill_dictionary", total=20, model="TRAIN-only")
        result = analyze_case(self.case(plain, learned))
        oracle = result["stateless_size_oracle"]
        self.assertEqual(oracle["eligible_candidates"], ["affine"])
        self.assertEqual(oracle["excluded_candidates"], [dict(name="dictionary", reason="stateful candidate", decoder_state_bytes=128)])
        self.assertEqual(oracle["mode_id_bytes"], 2)
        comparison = result["comparisons"][0]
        self.assertEqual(comparison["comparison"], "add_frozen_residual_dictionary")
        self.assertEqual(comparison["measured_bytes_saved"], 72)
        self.assertEqual(comparison["decoder_state_increment_bytes"], 128)
        self.assertEqual(comparison["full_cache_bytes_saved"], 1872)
        self.assertEqual(comparison["record_component_bytes_saved"], dict(header=0, correction=200))
        self.assertEqual(comparison["per_kind"]["K"]["record_payload_bytes_saved"], 100)
        self.assertEqual(comparison["per_kind"]["V"]["record_payload_bytes_saved"], 100)
        self.assertEqual(comparison["record_wins"], 2)
        self.assertFalse(comparison["isolated_option_change"])

    def test_record_mismatch_rejects_portfolio_and_pair_without_aggregation(self):
        a = _row("channel", [500, 800])
        b = _row("time", [500, 800], options=dict(layout="time"))
        b["measurement"]["record_diagnostics"][0]["start"] = 16
        result = analyze_case(self.case(a, b))
        self.assertEqual(result["comparisons"], [])
        self.assertEqual(len(result["skipped_comparisons"]), 1)
        self.assertIn("identities differ", result["skipped_comparisons"][0]["reason"])
        self.assertEqual(result["stateless_size_oracle"]["status"], "unavailable")
        self.assertIn("identities differ", result["stateless_size_oracle"]["reason"])

    def test_record_order_and_raw_sizes_are_part_of_match(self):
        for change in ("order", "raw"):
            a = _row("channel", [500, 800], raw_sizes=[1024, 2048])
            b = _row("time", [500, 800], raw_sizes=[1024, 2048], options=dict(layout="time"))
            if change == "order":
                b["measurement"]["record_diagnostics"].reverse()
            else:
                b["measurement"]["record_diagnostics"][0]["raw_bytes"] = 2048
                b["measurement"]["record_diagnostics"][1]["raw_bytes"] = 1024
            result = analyze_case(self.case(a, b))
            self.assertEqual(result["comparisons"], [])
            self.assertEqual(result["stateless_size_oracle"]["status"], "unavailable")

    def test_missing_or_invalid_traces_give_reasons(self):
        for invalid in ("missing", "components", "raw_total", "framing", "duplicates"):
            a = _row("channel", [500, 800])
            if invalid == "missing":
                del a["measurement"]["record_diagnostics"]
            elif invalid == "components":
                a["measurement"]["record_diagnostics"][0]["component_bytes"]["header"] += 1
            elif invalid == "raw_total":
                a["measurement"]["raw_payload_bytes"] += 2
            elif invalid == "framing":
                a["measurement"]["serialized_bytes"] += 1
            else:
                a["measurement"]["record_diagnostics"][1] = copy.deepcopy(a["measurement"]["record_diagnostics"][0])
            result = analyze_case(self.case(a))
            self.assertFalse(result["candidates"][0]["matched_records_valid"])
            self.assertTrue(result["candidates"][0]["record_validation_error"])
            self.assertEqual(result["stateless_size_oracle"]["status"], "unavailable")

    def test_only_same_case_standard_native_success_rows_used(self):
        a = _row("native", [700, 700])
        cpu = _row("cpu", [20, 20]); cpu["track"] = "cpu_reference"
        observer = _row("observer", [10, 10]); observer["track"] = "prefill_observer"
        failed = _row("failed", [1, 1]); failed["status"] = "error"
        case = self.case(a, cpu, observer, failed)
        case["prefill_observer"] = dict(results=[_row("nested", [1, 1])])
        original = copy.deepcopy(case)
        result = analyze_case(case)
        self.assertEqual([row["name"] for row in result["candidates"]], ["native"])
        self.assertEqual(result["split"], "test")
        self.assertEqual(case, original)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_timing_normalizes_by_declared_raw_sample_not_record_set(self):
        result = analyze_case(self.case(_row("native", [700, 700])))
        timing = result["candidates"][0]["timing"]
        self.assertAlmostEqual(timing["repeated_decode_ms_per_MiB"], 3072)
        self.assertAlmostEqual(timing["packet_transfer_then_decode_ms_per_MiB"], 2560)
        self.assertAlmostEqual(timing["encode_ms_per_MiB"], 10240)
        self.assertEqual(timing["execution"], "tensor_native_cpu_functional")
        self.assertIn("functionality", timing["scope"])

    def test_layout_and_group_changes_are_separate_matched_increments(self):
        channel8 = _row("channel8", [700, 700])
        time8 = _row("time8", [600, 600], options=dict(layout="time"))
        time64 = _row("time64", [550, 550], options=dict(layout="time", group_size=64))
        result = analyze_case(self.case(channel8, time8, time64))
        pairs = {(row["baseline"], row["candidate"]): row for row in result["comparisons"]}
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[("channel8", "time8")]["comparison"], "matched_layout")
        self.assertEqual(pairs[("time8", "time64")]["comparison"], "matched_group_size")
        self.assertTrue(all(row["isolated_option_change"] for row in pairs.values()))
        self.assertAlmostEqual(pairs[("channel8", "time8")]["measured_savings_gain_percentage_points"], 100 * 200 / 2048)

    def test_exponent_fixed127_control_pairs_with_metadata_predictor(self):
        fixed = _row("fixed", [700, 700], method="prefill_exponent", options=dict(predictor="fixed127", layout="time"))
        median = _row("median", [600, 600], method="prefill_exponent", options=dict(predictor="median", layout="time"))
        comparison = analyze_case(self.case(fixed, median))["comparisons"][0]
        self.assertEqual(comparison["comparison"], "matched_predictor")
        self.assertEqual(comparison["baseline"], "fixed")
        self.assertEqual(comparison["candidate"], "median")

    def test_exponent_layout_notes_do_not_claim_entropy_gain(self):
        channel = _row("channel", [700, 700], method="prefill_exponent", options=dict(predictor="median", layout="channel"))
        time = _row("time", [697, 697], method="prefill_exponent", options=dict(predictor="median", layout="time"))
        comparison = analyze_case(self.case(channel, time))["comparisons"][0]
        self.assertEqual(comparison["comparison"], "matched_layout")
        self.assertIn("not evidence", comparison["interpretation_notes"][0])
        self.assertIn("header", comparison["interpretation_notes"][0])

    def test_dictionary_assignment_pair_requires_same_frozen_model(self):
        a = _row("l1", [700, 700], state=128, method="prefill_dictionary", model="same", options=dict(assignment="l1"))
        b = _row("bitcost", [600, 600], state=128, method="prefill_dictionary", model="same", options=dict(assignment="bitcost"))
        result = analyze_case(self.case(a, b))
        self.assertEqual(result["comparisons"][0]["comparison"], "matched_assignment")
        self.assertEqual(result["comparisons"][0]["decoder_state_increment_bytes"], 0)
        b["spec"]["recipe_model"] = "different"
        self.assertEqual(analyze_case(self.case(a, b))["comparisons"], [])

    def test_dictionary_capacity_comparison_is_coupled_and_counts_state(self):
        a = _row("dictionary16", [700, 700], state=128, method="prefill_dictionary", model="sixteen")
        b = _row("dictionary64", [600, 600], state=512, method="prefill_dictionary", model="sixtyfour")
        a["spec"]["dictionary_size"] = 16
        b["spec"]["dictionary_size"] = 64
        comparison = analyze_case(self.case(b, a))["comparisons"][0]
        self.assertEqual(comparison["comparison"], "dictionary_capacity_comparison")
        self.assertEqual(comparison["baseline"], "dictionary16")
        self.assertEqual(comparison["candidate"], "dictionary64")
        self.assertFalse(comparison["isolated_option_change"])
        self.assertEqual(comparison["measured_bytes_saved"], -184)
        self.assertEqual(comparison["decoder_state_increment_bytes"], 384)
        self.assertIn("coupled", comparison["interpretation_notes"][0])

    def test_mode_overhead_can_lose_and_ties_are_name_stable(self):
        a = _row("aaa", [700, 700])
        z = _row("zzz", [700, 700], options=dict(layout="time"))
        first = analyze_case(self.case(z, a))["stateless_size_oracle"]
        second = analyze_case(self.case(a, z))["stateless_size_oracle"]
        self.assertEqual(first["mode_counts"], {"aaa": 2})
        self.assertEqual(first["mode_counts"], second["mode_counts"])
        self.assertEqual(first["measured_bytes_saved_over_best_fixed"], -2)

    def test_duplicate_candidate_names_and_unknown_state_are_not_assumed_safe(self):
        a = _row("same", [700, 700])
        self.assertIn("duplicate", analyze_case(self.case(a, copy.deepcopy(a)))["analysis_error"])
        del a["measurement"]["decoder_state_bytes"]
        result = analyze_case(self.case(a))
        self.assertEqual(result["stateless_size_oracle"]["status"], "unavailable")
        self.assertIn("invalid", result["stateless_size_oracle"]["excluded_candidates"][0]["reason"])

    def test_unverified_exactness_and_geometry_mismatch_block_oracle(self):
        a = _row("a", [700, 700])
        a["measurement"]["exact_verified"] = False
        result = analyze_case(self.case(a))
        self.assertIn("exact roundtrip", result["stateless_size_oracle"]["reason"])
        a = _row("a", [700, 700], total=20)
        b = _row("b", [700, 700], total=40, options=dict(layout="time"))
        result = analyze_case(self.case(a, b))
        self.assertEqual(result["comparisons"], [])
        self.assertIn("geometry", result["stateless_size_oracle"]["reason"])

    def test_mode_byte_capacity_has_explicit_limit(self):
        rows = [_row(f"candidate_{index}", [700, 700]) for index in range(257)]
        oracle = analyze_case(self.case(*rows))["stateless_size_oracle"]
        self.assertEqual(oracle["status"], "unavailable")
        self.assertIn("256", oracle["reason"])


if __name__ == "__main__":
    unittest.main()

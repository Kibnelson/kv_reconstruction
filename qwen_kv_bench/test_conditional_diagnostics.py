"""Independent per-head count/correlation oracles and immutable-source checks."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
import warnings

import numpy as np

from qwen_kv_bench import conditional_diagnostics as d


def _words(exponent, lower):
    exponent, lower = np.broadcast_arrays(np.asarray(exponent), np.asarray(lower))
    return ((exponent.astype(np.uint16) << 7) | (lower.astype(np.uint16) & 127)
            | ((lower.astype(np.uint16) & 128) << 8)).astype(np.uint16)


def _bf16(value):
    return (np.asarray(value, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _models(layers=(0,), columns=4):
    return {name: SimpleNamespace(params={f"layer_{layer:02d}":
        dict(base_K=np.full(columns, 120 + i, np.uint8), base_V=np.full(columns, 123 + i, np.uint8))
        for layer in layers}) for i, name in enumerate(("cost_final", "cost_pre_rope", "cost_kproj"))}


def _fingerprint(records, models):
    digest = hashlib.sha256()
    for record in records:
        for name in ("k", "v", "k_pre_rope", "k_proj"):
            digest.update(record[name].tobytes())
    for model in models.values():
        for params in model.params.values():
            for value in params.values():
                digest.update(value.tobytes())
    return digest.hexdigest()


class ConditionalDiagnosticsTests(unittest.TestCase):
    def source(self):
        rng = np.random.default_rng(21)
        return [dict(layer=layer, start=start, tokens=tokens, head_dim=2,
            **{name: rng.integers(0, 65536, (tokens, 4), dtype=np.uint16)
               for name in ("k", "v", "k_pre_rope", "k_proj")})
            for layer in (0, 3) for start, tokens in ((0, 3), (100, 2), (200, 0))]

    def test_source_specific_tier_bits_and_lower_entropy_independent_oracles(self):
        records, models = self.source(), _models(layers=(0, 3))
        before = _fingerprint(records, models)
        report = d.analyze(records, models)
        self.assertEqual([row["layer"] for row in report["layers"]], [0, 3])
        fields = {"cost_final": "k", "cost_pre_rope": "k_pre_rope", "cost_kproj": "k_proj"}
        for layer in report["layers"]:
            self.assertEqual(layer["selected_token_rows"], 5)
            self.assertEqual(layer["supplied_records"], 3)
            selected = [record for record in records if record["layer"] == layer["layer"]]
            for head in layer["heads"]:
                section = slice(2 * head["head"], 2 * head["head"] + 2)
                for name, field in fields.items():
                    params = models[name].params[f"layer_{layer['layer']:02d}"]
                    for kind, actual_field in (("K", field), ("V", "v")):
                        errors = [int((int(word) >> 7) & 255) - int(base)
                            for record in selected for row in record[actual_field][:, section]
                            for word, base in zip(row, params["base_" + kind][section])]
                        zz = [2 * value if value >= 0 else -2 * value - 1 for value in errors]
                        expected = sum(2 + 3 * (value >= 3) + 9 * (value >= 10) for value in zz)
                        result = head["frozen_exponent_tiers"][name][kind]
                        self.assertEqual(result["payload_bits"], expected)
                        self.assertEqual(result["values"], 10)
                        self.assertEqual(result["secondary_escapes"], sum(value >= 3 for value in zz))
                for name, field in (("final_K", "k"), ("final_V", "v"),
                                    ("pre_rope_K", "k_pre_rope"), ("kproj_K", "k_proj")):
                    values = [((int(word) >> 8) & 128) | (int(word) & 127)
                              for record in selected for word in record[field][:, section].flat]
                    counts = np.bincount(values, minlength=256)
                    probabilities = counts[counts > 0] / len(values)
                    entropy = head["domains"][name]["lower8_marginal_entropy"]
                    self.assertAlmostEqual(entropy["empirical_bits_per_value"],
                                           float(-(probabilities * np.log2(probabilities)).sum()))
                    self.assertEqual(entropy["observations"], 10)
                    self.assertEqual(entropy["occupied_symbols"], int(np.count_nonzero(counts)))
                    self.assertEqual(entropy["singleton_symbols"], int(np.count_nonzero(counts == 1)))
                    self.assertFalse(entropy["heuristic_adequate_samples"])
                    self.assertIn("neither a lower bound", entropy["reliability_warning"])
        self.assertEqual(_fingerprint(records, models), before)
        self.assertFalse(report["fitting_performed"])
        self.assertFalse(report["selector_implemented"])
        self.assertFalse(report["raw_arrays_included"])
        self.assertIn("Excludes byte padding", report["exponent_scope"])
        json.dumps(report, allow_nan=False)

    def test_sign_transitions_reset_at_contiguous_and_gapped_record_boundaries(self):
        records = [dict(layer=0, start=start, tokens=3, head_dim=2,
            **{name: _words(np.full((3, 4), 123), np.full((3, 4), lower))
               for name in ("k", "v", "k_pre_rope", "k_proj")})
            for start, lower in ((0, 0), (3, 255), (100, 0))]
        report = d.analyze(records, _models())
        for head in report["layers"][0]["heads"]:
            for domain in head["domains"].values():
                self.assertEqual(domain["sign_transitions"], dict(changed=0, eligible=12, rate=0.))

    def test_numeric_matched_channel_pearson_matches_direct_finite_oracle(self):
        x = np.array([[-3, 5, 1, np.nan], [-1, 5, 2, np.inf],
                      [1, 5, 3, np.nan], [3, 5, 4, np.inf]], np.float32)
        y = np.array([[6, 1, 8, 1], [2, 2, 6, 2], [2, 3, 4, 3], [6, 4, 2, 4]], np.float32)
        records = [dict(layer=0, start=start, tokens=2, head_dim=2,
            k=_bf16(x[part]), v=_bf16(y[part]), k_pre_rope=_bf16(x[part]), k_proj=_bf16(x[part]))
            for start, part in ((0, slice(0, 2)), (100, slice(2, 4)))]
        report = d.analyze(records, _models())
        for head in report["layers"][0]["heads"]:
            first = head["head"] * 2
            corr = head["matched_channel_K_pre_rope_V"]
            for key, absolute_values in (("absolute_signed_pearson", False), ("pearson_of_absolute_values", True)):
                column = first  # The other channel is constant or has no finite pairs.
                a, b = x[:, column], y[:, column]
                expected = float(np.corrcoef(np.abs(a) if absolute_values else a,
                                             np.abs(b) if absolute_values else b)[0, 1])
                if not absolute_values:
                    expected = abs(expected)
                self.assertEqual(corr[key]["valid_channels"], 1)
                for statistic in ("median", "p90", "maximum"):
                    self.assertAlmostEqual(corr[key][statistic], expected)
            self.assertEqual(corr["absolute_signed_pearson"]["constant_channels"], 1 if first == 0 else 0)
            self.assertEqual(corr["absolute_signed_pearson"]["insufficient_finite_channels"], 0 if first == 0 else 1)
        json.dumps(report, allow_nan=False)

    def test_all_empty_records_report_null_metrics_and_strict_json(self):
        record = dict(layer=0, start=0, tokens=0, head_dim=2,
            **{name: np.empty((0, 4), np.uint16) for name in ("k", "v", "k_pre_rope", "k_proj")})
        report = d.analyze([record], _models())
        head = report["layers"][0]["heads"][0]
        self.assertIsNone(head["domains"]["final_K"]["sign_transitions"]["rate"])
        self.assertIsNone(head["domains"]["final_K"]["lower8_marginal_entropy"]["empirical_bits_per_value"])
        self.assertIsNone(head["frozen_exponent_tiers"]["cost_final"]["K"]["bits_per_value"])
        self.assertEqual(head["matched_channel_K_pre_rope_V"]["absolute_signed_pearson"]["valid_channels"], 0)
        json.dumps(report, allow_nan=False)

    def test_every_BF16_word_including_signaling_nan_is_filtered_without_warnings(self):
        words = np.arange(65536, dtype=np.uint16).reshape(512, 128)
        record = dict(layer=0, start=0, tokens=512, head_dim=128,
            **{name: words for name in ("k", "v", "k_pre_rope", "k_proj")})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = d.analyze([record], _models(columns=128))
        statistics = report["layers"][0]["heads"][0]["matched_channel_K_pre_rope_V"]
        self.assertEqual(statistics["absolute_signed_pearson"]["finite_matched_values"], 65280)
        self.assertEqual(statistics["absolute_signed_pearson"]["valid_channels"], 128)
        self.assertAlmostEqual(statistics["absolute_signed_pearson"]["median"], 1.)
        json.dumps(report, allow_nan=False)

    def test_reject_invalid_heads_sources_bases_and_duplicate_observations(self):
        for change in ("head_dim", "tokens", "shape", "dtype", "missing", "overlap", "base"):
            records, models = copy.deepcopy(self.source()), _models(layers=(0, 3))
            if change == "head_dim": records[0]["head_dim"] = 3
            elif change == "tokens": records[0]["tokens"] = 4
            elif change == "shape": records[0]["v"] = records[0]["v"][:, :2]
            elif change == "dtype": records[0]["k"] = records[0]["k"].astype(np.int32)
            elif change == "missing": del records[0]["k_pre_rope"]
            elif change == "overlap": records[1]["start"] = 1
            else: models["cost_kproj"].params["layer_00"]["base_K"] = np.zeros(3, np.uint8)
            with self.subTest(change=change), self.assertRaises(ValueError):
                d.analyze(records, models)
        with self.assertRaises(ValueError):
            d.analyze([], _models())


if __name__ == "__main__":
    unittest.main()

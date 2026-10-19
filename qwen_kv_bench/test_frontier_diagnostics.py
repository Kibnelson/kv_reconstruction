"""Independent entropy/count oracles and complete CPU control roundtrips."""
from collections import Counter
import hashlib
import json
import math
import unittest

import numpy as np

from qwen_kv_bench import frontier_diagnostics as d


def words(exponent, lower):
    exponent, lower = np.broadcast_arrays(np.asarray(exponent), np.asarray(lower))
    return ((exponent.astype(np.uint16) << 7) | (lower.astype(np.uint16) & 127)
            | ((lower.astype(np.uint16) & 128) << 8)).astype(np.uint16)


def conditional_oracle(context, target):
    joint = Counter(zip(context, target))
    totals = Counter(context)
    return sum(count * math.log2(totals[key[0]] / count) for key, count in joint.items()) / len(target)


class FrontierDiagnosticsTests(unittest.TestCase):
    def source(self):
        rng = np.random.default_rng(81)
        return [dict(layer=layer, start=start, tokens=11,
            k=rng.integers(0, 65536, (11, 9), dtype=np.uint16),
            v=rng.integers(0, 65536, (11, 9), dtype=np.uint16),
            k_pre_rope=rng.integers(0, 65536, (11, 9), dtype=np.uint16),
            k_proj=rng.integers(0, 65536, (11, 9), dtype=np.uint16))
            for layer in (0, 1) for start in (0, 30)]

    def test_entire_BF16_domain_marginals_and_sparse_conditioning_warning(self):
        source = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        report = d.analyze([dict(layer=0, k=source, v=source)])
        domain = report["domains"]["final_K"]
        for field, expected in (("exponent8", 8.), ("lower8", 8.), ("sign", 1.), ("mantissa7", 7.)):
            self.assertAlmostEqual(domain["entropy"][field]["empirical_bits_per_value"], expected)
            self.assertEqual(domain["entropy"][field]["observations"], 65536)
        conditional = domain["conditional_lower8_given_exponent"]
        self.assertAlmostEqual(conditional["global_pooled"]["empirical_conditional_bits_per_value"], 8.)
        sparse = conditional["pooled_by_layer_channel"]
        # This ordered enumeration makes channel/exponent identify every bit
        # except sign. Two observations per context are not a source model.
        self.assertAlmostEqual(sparse["empirical_conditional_bits_per_value"], 1.)
        self.assertFalse(sparse["reliability"]["adequate_for_descriptive_estimate"])
        self.assertIn("Sparse", sparse["reliability"]["warning"])
        self.assertEqual(sparse["mathematical_alphabet_range_bits"], [0., 8.])
        self.assertIn("not a", domain["entropy"]["lower8"]["scope"])

    def test_independent_global_layer_channel_conditional_oracles(self):
        records = self.source()
        report = d.analyze(records)
        targets, exponents, layers, channels = [], [], [], []
        for record in records:
            value = record["k"]
            for row in range(len(value)):
                for channel in range(value.shape[1]):
                    symbol = int(value[row, channel])
                    targets.append(((symbol >> 8) & 128) | (symbol & 127))
                    exponents.append((symbol >> 7) & 255)
                    layers.append(record["layer"])
                    channels.append(channel)
        conditional = report["domains"]["final_K"]["conditional_lower8_given_exponent"]
        contexts = dict(global_pooled=exponents,
            pooled_by_layer=list(zip(layers, exponents)),
            pooled_by_layer_channel=list(zip(layers, channels, exponents)))
        for name, context in contexts.items():
            row = conditional[name]
            self.assertAlmostEqual(row["empirical_conditional_bits_per_value"], conditional_oracle(context, targets))
            self.assertEqual(row["occupied_contexts"], len(set(context)))
            expected_correction = (len(set(zip(context, targets))) - len(set(context))) / (2 * len(targets) * math.log(2))
            self.assertAlmostEqual(row["miller_madow_uncapped_bits_per_value"],
                row["empirical_conditional_bits_per_value"] + expected_correction)

    def test_temporal_transitions_never_bridge_tiles_or_gaps(self):
        records = [dict(layer=0, start=start, k=words(np.full((64, 8), 125), np.full((64, 8), lower)),
                       v=words(np.full((64, 8), 125), np.full((64, 8), lower)))
                   for start, lower in ((0, 0), (64, 255), (200, 0))]
        domain = d.analyze(records)["domains"]["final_K"]
        temporal = domain["temporal_lower8"]
        self.assertEqual(temporal["transition_values"], 3 * 63 * 8)
        self.assertEqual(temporal["zero_xor_fraction"], 1.)
        self.assertEqual(temporal["xor_bitplane_one_fractions"], [0.] * 8)
        self.assertEqual(domain["lower8_bitplane_one_fractions"], [1 / 3] * 8)
        self.assertEqual(sum(domain["histograms"]["temporal_xor_lower8"]), temporal["transition_values"])
        self.assertTrue(temporal["conditional_on_previous_lower8"]["reliability"]["adequate_for_descriptive_estimate"])

    def test_actual_CPU_records_all_modes_seeds_framing_and_corruption(self):
        rng = np.random.default_rng(17)
        for shape in ((1, 1), (7, 9), (64, 1024)):
            lower = rng.integers(0, 256, shape, dtype=np.uint8)
            for mode in d._MODES:
                with self.subTest(shape=shape, mode=mode):
                    blob = d._encode_lower_control(lower, mode)
                    np.testing.assert_array_equal(d._decode_lower_control(blob), lower)
                    fields = d._CONTROL.unpack_from(blob)
                    self.assertEqual(d._CONTROL.size, 32)
                    self.assertEqual(len(blob), 32 + fields[6] + fields[7])
                    self.assertEqual(fields[6], shape[1] if mode != "raw_channel_major" else 0)
                    with self.assertRaises(ValueError):
                        d._decode_lower_control(blob[:-1])
                    with self.assertRaises(ValueError):
                        d._decode_lower_control(blob + b"trailing")
                    bad = bytearray(blob); bad[28] ^= 1  # Original lower-plane CRC.
                    with self.assertRaises(ValueError):
                        d._decode_lower_control(bad)
                    if mode != "raw_channel_major":
                        bad = bytearray(blob); bad[32] ^= 1  # Charged raw first-row seed.
                        with self.assertRaises(ValueError):
                            d._decode_lower_control(bad)

    def test_aggregate_accounting_optional_sources_and_no_mutation(self):
        records = self.source()
        before = [[hashlib.sha256(record[key].tobytes()).hexdigest() for key in ("k", "v", "k_pre_rope", "k_proj")] for record in records]
        report = d.analyze(records)
        self.assertFalse(report["raw_arrays_included"])
        self.assertFalse(report["fitting_performed"])
        json.dumps(report, allow_nan=False)
        self.assertEqual(set(report["domains"]), {"final_K", "final_V", "pre_rope_K", "kproj_K"})
        for domain in report["domains"].values():
            self.assertEqual(domain["observations"], 4 * 11 * 9)
            self.assertEqual(domain["observations_per_layer_channel"], dict(minimum=22, median=22., maximum=22))
            for mode, row in domain["cpu_lower8_controls"]["methods"].items():
                self.assertEqual(row["roundtrip_verified_records"], 4)
                self.assertEqual(row["record_header_bytes"], 4 * 32)
                self.assertEqual(row["seed_bytes"], 0 if mode == "raw_channel_major" else 4 * 9)
                self.assertEqual(row["actual_record_bytes"], row["record_header_bytes"] + row["seed_bytes"] + row["compressed_payload_bytes"])
                self.assertEqual(row["total_bytes_including_calculated_outer"], row["actual_record_bytes"] + 32 + 8 * 5)
                self.assertEqual(row["bits_per_original_BF16_value"], 8 * row["total_bytes_including_calculated_outer"] / domain["observations"])
        after = [[hashlib.sha256(record[key].tobytes()).hexdigest() for key in ("k", "v", "k_pre_rope", "k_proj")] for record in records]
        self.assertEqual(before, after)
        targets_only = [{key: value for key, value in record.items() if key not in ("k_pre_rope", "k_proj")} for record in records]
        only = d.analyze(targets_only)
        self.assertEqual(only["domains"]["pre_rope_K"]["status"], "unavailable")
        self.assertEqual(only["domains"]["kproj_K"]["status"], "unavailable")
        self.assertEqual(only["target_ledger"]["max_total_bits_per_BF16_value"], 9.6)
        self.assertEqual(only["target_ledger"]["old_raw_lower8_plus_two_bit_primary_max_savings_percent"], 37.5)
        self.assertIn("no universal", only["target_ledger"]["scope"])

    def test_reject_duplicate_overlap_invalid_sources_and_geometry(self):
        for change in ("duplicate", "overlap", "dtype", "shape", "missing_k", "bad_layer", "bad_tokens"):
            with self.subTest(change=change):
                records = self.source()
                if change == "duplicate": records.append(records[0])
                elif change == "overlap": records[1]["start"] = 5
                elif change == "dtype": records[0]["k_proj"] = records[0]["k_proj"].astype(np.float32)
                elif change == "shape": records[0]["v"] = records[0]["v"][:, :-1]
                elif change == "missing_k": del records[0]["k"]
                elif change == "bad_layer": records[0]["layer"] = True
                else: records[0]["tokens"] = 12
                with self.assertRaises(ValueError): d.analyze(records)
        with self.assertRaises(ValueError): d.analyze([])


if __name__ == "__main__":
    unittest.main()

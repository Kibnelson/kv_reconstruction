"""Exact arithmetic checks for measured-versus-estimated neural accounting."""
from fractions import Fraction
import json
import unittest

from qwen_kv_bench.neural_accounting import account, break_even_contexts


class NeuralAccountingTests(unittest.TestCase):
    def test_full_neural_and_anchor_states_charged_once(self):
        report = account(100, 50, 1000, 700, 600, 20, 100)
        self.assertEqual(report["sample"]["payload_reduction_pct"], 50.)
        self.assertFalse(report["sample"]["state_amortized_over_sample"])
        self.assertEqual(report["state"]["neural"]["charged_bytes"], 700)
        self.assertEqual(report["state"]["anchor"]["charged_bytes"], 100)
        self.assertEqual(report["state"]["total_charged_bytes"], 800)
        full = report["full_cache_estimate"]
        self.assertEqual(full["estimated_packet_bytes"], 500)
        self.assertEqual(full["estimated_total_bytes"], 1300)
        self.assertEqual(full["net_reduction_pct"], -30.)
        rows = report["concurrent_shared_state_estimates"]
        self.assertEqual([row["contexts"] for row in rows], [1, 4, 16, 64])
        self.assertEqual([row["net_reduction_pct"] for row in rows], [-30., 30., 45., 48.75])
        for row in rows:
            n = row["contexts"]
            self.assertEqual(row["estimated_total_bytes"], n * 500 + 800)
            self.assertEqual(row["charged_state_bytes"], 800)
        self.assertEqual(report["break_even"]["positive_reduction_contexts"], 2)
        self.assertEqual(report["break_even"]["target_reduction_contexts"], 8)
        json.dumps(report, allow_nan=False)

    def test_state_charged_per_family_not_max_of_combined_sizes(self):
        report = account(100, 60, 1000, 900, 100, 10, 800)
        self.assertEqual(report["state"]["total_charged_bytes"], 1700)
        self.assertEqual(report["full_cache_estimate"]["estimated_total_bytes"], 2300)

    def test_positive_is_strict_and_target_is_inclusive(self):
        report = break_even_contexts(1000, 500, 1000)
        # Two contexts merely break even: 2*500 + 1000 == 2*1000.
        self.assertEqual(report["positive_reduction_contexts"], 3)
        # Ten contexts reach exactly 40% net reduction.
        self.assertEqual(report["target_reduction_contexts"], 10)
        self.assertEqual(break_even_contexts(1000, 500, 1001)["target_reduction_contexts"], 11)

    def test_target_equals_payload_asymptote(self):
        self.assertEqual(break_even_contexts(1000, 600, 0)["target_reduction_contexts"], 1)
        self.assertIsNone(break_even_contexts(1000, 600, 1)["target_reduction_contexts"])
        self.assertIsNone(break_even_contexts(1000, 601, 0)["target_reduction_contexts"])

    def test_payload_expansion_is_reported_and_impossible_to_amortize(self):
        report = account(100, 110, 1000, 0, 0, 0, 0)
        self.assertEqual(report["sample"]["payload_reduction_pct"], -10.)
        self.assertEqual(report["full_cache_estimate"]["net_reduction_pct"], -10.)
        self.assertIsNone(report["break_even"]["positive_reduction_contexts"])
        self.assertIsNone(report["break_even"]["target_reduction_contexts"])
        self.assertIsNone(break_even_contexts(1000, 1000, 0)["positive_reduction_contexts"])

    def test_fractional_extrapolation_is_never_rounded_before_boundaries(self):
        report = account(3, 1, 10, 1, 1, 1, 1, concurrencies=(3,))
        self.assertEqual(report["sample"]["payload_fraction_of_raw_exact"],
                         {"numerator": 1, "denominator": 3})
        self.assertEqual(report["full_cache_estimate"]["estimated_packet_bytes_exact"],
                         {"numerator": 10, "denominator": 3})
        self.assertEqual(report["concurrent_shared_state_estimates"][0]["estimated_total_bytes"], 12)
        # 60% of 5 is 3: target margin is exactly 1/3 byte/context.
        self.assertEqual(break_even_contexts(5, Fraction(8, 3), 1)["target_reduction_contexts"], 3)
        huge = 10**30
        self.assertEqual(break_even_contexts(5, Fraction(8, 3), huge)["target_reduction_contexts"], 3 * huge)
        self.assertEqual(break_even_contexts(3, 2, huge)["positive_reduction_contexts"], huge + 1)

    def test_empty_payload_and_target_endpoints(self):
        report = account(1, 0, 10, 0, 0, 0, 0)
        self.assertEqual(report["full_cache_estimate"]["net_reduction_pct"], 100.)
        self.assertEqual(report["break_even"]["positive_reduction_contexts"], 1)
        self.assertEqual(break_even_contexts(10, 0, 0, target_reduction=1)["target_reduction_contexts"], 1)
        self.assertIsNone(break_even_contexts(10, 0, 1, target_reduction=1)["target_reduction_contexts"])
        self.assertEqual(break_even_contexts(10, 10, 0, target_reduction=0)["target_reduction_contexts"], 1)
        self.assertIsNone(break_even_contexts(10, 10, 0, target_reduction=0)["positive_reduction_contexts"])

    def test_inputs_reject_ambiguous_or_invalid_byte_counts(self):
        args = [100, 50, 1000, 7, 8, 9, 10]
        for index in range(len(args)):
            for invalid in (-1, 1.5, True):
                changed = list(args)
                changed[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises((TypeError, ValueError)):
                    account(*changed)
        for index in (0, 2):
            changed = list(args)
            changed[index] = 0
            with self.assertRaises(ValueError):
                account(*changed)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(concurrency=invalid), self.assertRaises((TypeError, ValueError)):
                account(*args, concurrencies=(invalid,))
        for target in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(target=target), self.assertRaises((TypeError, ValueError)):
                break_even_contexts(100, 50, 0, target_reduction=target)


if __name__ == "__main__":
    unittest.main()

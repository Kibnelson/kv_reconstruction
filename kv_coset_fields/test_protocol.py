"""Independent audit and accounting gates for the combined experiment."""

from dataclasses import dataclass
import unittest

import numpy as np

from kv_coset_fields import combined
from kv_research import container
from kv_stacked import huffman_codec
from kv_stacked.run_eval import aggregate


@dataclass
class TrialLower:
    data: bytes
    stats: dict


class ProtocolTests(unittest.TestCase):
    def test_rejected_trials_are_audited_and_baseline_bytes_are_identical(self):
        rng = np.random.default_rng(901)
        matrices = [
            tuple(rng.integers(0, 65536, (33, 16), dtype=np.uint16) for _ in range(2))
            for _ in range(2)
        ]
        calls = []

        def lower_encoder(lower, exponents):
            payload = lower.tobytes() + b"deliberately expensive trial"
            return TrialLower(payload, {"component_bytes": {"trial": len(payload)}})

        def lower_decoder(payload, exponents):
            calls.append(exponents.shape)
            return np.frombuffer(payload[:exponents.size], dtype=np.uint8).copy().reshape(exponents.shape)

        def encoder(words, layer, kind):
            record = combined.encode_matrix(words, lower_encoder, audit_decoder=lower_decoder)
            self.assertEqual(record.data, huffman_codec.encode_matrix(words).data)
            self.assertEqual(sum(record.stats["component_bytes"].values()), len(record.data))
            return record

        encoded = container.encode_cache(matrices, encoder, tile_tokens=16)
        expected_records = 2 * 2 * 3
        self.assertEqual(len(calls), expected_records)
        self.assertEqual(encoded.stats["diagnostics"]["audit"]["records_verified"], expected_records)
        self.assertEqual(
            encoded.stats["diagnostics"]["record_modes"]["original_huffman_fallback"],
            expected_records,
        )
        restored = container.decode_cache(encoded.data, lambda b, l, k: combined.decode_matrix(b, lower_decoder))
        # Decoding the stored KVHU fallback does not invoke a lower decoder.
        self.assertEqual(len(calls), expected_records)
        for original_pair, recovered_pair in zip(matrices, restored):
            for original, recovered in zip(original_pair, recovered_pair):
                np.testing.assert_array_equal(original, recovered)
        self.assertEqual(combined.HEADER.size, huffman_codec._HEADER.size)

    def test_broken_rejected_trial_is_not_hidden_by_fallback(self):
        words = np.arange(128, dtype=np.uint16).reshape(8, 16)

        def oversized(lower, exponents):
            payload = lower.tobytes() + b"overhead"
            return TrialLower(payload, {})

        def incorrect_decoder(payload, exponents):
            return np.zeros(exponents.shape, dtype=np.uint8)

        with self.assertRaises(AssertionError):
            combined.encode_matrix(words, oversized, audit_decoder=incorrect_decoder)

    def test_shared_state_charged_even_when_fallback_payloads_match(self):
        cases = []
        for raw_bytes, encoded_bytes in ((2048, 1400), (8192, 5600)):
            row = {
                "raw_payload_bytes": raw_bytes,
                "serialized_bytes": encoded_bytes,
                "roundtrip_exact": True,
                "encode_seconds": 0.0,
                "decode_seconds": 0.0,
            }
            cases.append({"codecs": {"baseline": dict(row), "learned": dict(row)}})
        state = 311
        result = aggregate(cases, {"baseline": 0, "learned": state}, ["baseline", "learned"])
        total_raw = 10240
        for count in (1, 4, 16):
            actual_difference = (
                result["baseline"]["savings_including_state_percent"][str(count)]
                - result["learned"]["savings_including_state_percent"][str(count)]
            )
            self.assertAlmostEqual(actual_difference, 100 * len(cases) * state / count / total_raw)


if __name__ == "__main__":
    unittest.main()

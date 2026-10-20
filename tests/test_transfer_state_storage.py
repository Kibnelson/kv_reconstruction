"""Exact coefficient storage must preserve both range and accumulation dtype."""
import copy
import importlib.util
import unittest

import numpy as np

from qwen_kv_bench import transfer_reference as reference
from qwen_kv_bench.test_transfer_reference import capture
from scripts.probe_transfer_state_storage import compact_coefficients, compare_predictions, storage, verify_packet_compaction


class TransferStateStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = reference.fit([capture(1), capture(2)], capture(3), block_size=4)

    def test_exact_cast_preserves_causal_predictions(self):
        candidate = compact_coefficients(self.model)
        checked = compare_predictions(self.model, candidate, capture(7), tokens=16)
        self.assertTrue(checked["predictions_bit_identical"])
        self.assertEqual(checked["accumulation_dtypes"], ["int64"])
        self.assertEqual(checked["checked_values"], 3 * 16 * 16)
        before, after = storage(self.model), storage(candidate)
        self.assertEqual(after["coefficient_bytes"] * 2, before["coefficient_bytes"])
        self.assertEqual(self.model.predictors[0][1]["A"].dtype, np.int64)
        self.assertNotEqual(candidate.state_sha256, self.model.state_sha256)

    def test_rejects_overflow_instead_of_wrapping(self):
        model = copy.deepcopy(self.model)
        model.predictors[0][1]["A"][0, 0] = 1 << 31
        with self.assertRaisesRegex(ValueError, "exactly as INT32"):
            compact_coefficients(model)
        model.predictors[0][1]["A"][0, 0] = -(1 << 31) - 1
        with self.assertRaisesRegex(ValueError, "exactly as INT32"):
            compact_coefficients(model)

    def test_integer_matrix_product_promotes_to_int64(self):
        x = np.array([[1 << 20, -(1 << 20)]], dtype=np.int64)
        a = np.array([[1 << 20, 1 << 19]], dtype=np.int32)
        result = x @ a.T
        self.assertEqual(result.dtype, np.int64)
        self.assertEqual(int(result[0, 0]), 1 << 39)

    def test_mixed_dtype_storage_candidate_counts_exact_block_ranges(self):
        model = copy.deepcopy(self.model)
        # Force both signed ranges in two nonempty matrices.
        model.predictors[0][1]["A"][:] = 100
        model.predictors[0][2]["A"][:] = 1 << 20
        result = storage(model)
        expected = sum(block["A"].size * (2 if not block["A"].size or
                       (int(block["A"].min()) >= -(1 << 15) and int(block["A"].max()) < 1 << 15) else 4)
                       for blocks in model.predictors for block in blocks)
        self.assertEqual(result["INT16_or_INT32_candidate_coefficient_bytes"], expected)
        self.assertEqual(result["INT16_or_INT32_candidate_all_frozen_array_bytes"],
                         result["all_frozen_array_bytes"] - result["coefficient_bytes"] + expected)
        self.assertIn("int16", result["minimum_exact_signed_dtype_block_counts"])
        self.assertIn("int32", result["minimum_exact_signed_dtype_block_counts"])

    @unittest.skipUnless(importlib.util.find_spec("constriction"), "constriction is optional")
    def test_packet_payload_unchanged_with_explicit_state_identity_update(self):
        candidate = compact_coefficients(self.model)
        expected = reference.words(capture(9, 16))
        packet, _ = reference.encode(self.model, expected)
        with self.assertRaisesRegex(ValueError, "different frozen state"):
            reference.decode(candidate, packet)
        reframed, report = verify_packet_compaction(self.model, candidate, packet, expected)
        self.assertEqual(report["changed_header_fields"], ["state_sha256"])
        self.assertTrue(report["payload_bytes_unchanged"])
        self.assertTrue(report["exact_decoder_checksum_verified"])
        self.assertTrue(report["equals_original_decoded_array"])
        self.assertNotEqual(packet, reframed)


if __name__ == "__main__":
    unittest.main()

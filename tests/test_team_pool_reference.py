"""Original pool round-trip and independent size-accounting checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import bench_team_pool_reference as reference


SOURCE = Path(__file__).resolve().parents[1] / "experiments/team_transfer_original/combined.py"


class TeamPoolReferenceTests(unittest.TestCase):
    def test_ast_loader_does_not_execute_original_module_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "original.py"
            source.write_text("raise RuntimeError('top level must not execute')\n" + SOURCE.read_text())
            functions = reference.load_reference(source, [], 16)
        self.assertTrue(callable(functions["pack_block"]))

    def test_arbitrary_bf16_patterns_and_escape_formula(self):
        rng = np.random.default_rng(91)
        words = rng.integers(0, 65536, (16, 16), dtype=np.uint16)
        # Include signs, zero/subnormal, infinities and distinct NaN payloads.
        words[0, :8] = [0, 0x8000, 1, 0x807F, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF]
        original = reference.load_reference(SOURCE, [], 16)
        ref = np.full(16, 127, np.int64)
        widths = np.tile(np.arange(1, 9), 2)
        packed = original["pack_block"](words, ref, widths)
        decoded = original["unpack_block"](packed, ref, widths)
        np.testing.assert_array_equal(decoded, words)
        layout = reference.exponent_layout(words, ref, widths)
        self.assertGreater(layout["escape_count"], 0)
        self.assertEqual(int(packed[0]) + 256 * int(packed[1]), layout["escape_count"])
        self.assertEqual(len(packed), 2 + words.size + layout["dense_exponent_bytes"] + layout["escape_count"])

    def test_group_min_zero_span_still_charges_metadata_and_raw_byte(self):
        words = np.full((16, 16), np.uint16(127 << 7))
        self.assertEqual(reference.group_min_formula_bits(words), words.size * 8 + words.size // 8 * 12)

    def test_fit_uses_train_only_and_record_exactness(self):
        train = {"K_bits": np.full((1, 16, 8), np.uint16(127 << 7)),
                 "V_bits": np.full((1, 16, 8), np.uint16(128 << 7))}
        original = reference.load_reference(SOURCE, [train], 16)
        ref, widths = original["fit_pool"](0)
        np.testing.assert_array_equal(ref, [127] * 8 + [128] * 8)
        np.testing.assert_array_equal(widths, np.ones(16, dtype=np.int64))
        test = original["words"](train, 0).copy()
        test[3, 0] = np.uint16(255 << 7)
        record = original["pack_block"](test, ref, widths)
        np.testing.assert_array_equal(original["unpack_block"](record, ref, widths), test)
        self.assertEqual(int(record[0]), 1)


if __name__ == "__main__":
    unittest.main()

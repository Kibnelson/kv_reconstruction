"""Exact prefill-derived recipes; CPU tests exercise the GPU tensor code path.

CUDA checks are explicit skips on a CPU-only host.  These are functional tests,
not evidence of GPU speed or serving-capacity improvement.
"""
import json
import unittest
import warnings

import numpy as np
import torch

from qwen_kv_bench import gpu_ops as g


def _numpy_e8_twice(values):
    """Independent NumPy nearest D8 / shifted D8 construction."""
    def d8(x):
        integers = np.rint(x).astype(np.int64)
        for row in range(len(x)):
            if int(integers[row].sum()) % 2:
                error = x[row] - integers[row]
                index = int(np.argmax(np.abs(error)))
                integers[row, index] += 1 if error[index] >= 0 else -1
        return integers
    integral = 2 * d8(values)
    shifted = 2 * d8(values - 0.5) + 1
    di = ((values - integral / 2.0) ** 2).sum(axis=1)
    ds = ((values - shifted / 2.0) ** 2).sum(axis=1)
    return np.where((ds < di)[:, None], shifted, integral)


def _numpy_prediction(words, predictor, step_log2=6):
    magnitude = words.astype(np.int64) & 32767
    tokens, channels = magnitude.shape
    if predictor == "zero":
        return np.zeros_like(magnitude)
    if predictor == "affine":
        return magnitude[0:1] + (magnitude[-1:] - magnitude[0:1]) * np.arange(tokens)[:, None] // max(tokens - 1, 1)
    template = (magnitude.min(axis=0) + magnitude.max(axis=0)) // 2
    if predictor == "e8":
        padded_channels = ((channels + 7) // 8) * 8
        padded = np.pad(template, (0, padded_channels - channels), mode="edge")
        minimum = np.pad(padded, (0, (-padded_channels) % 64), constant_values=32767).reshape(-1, 64).min(axis=1)
        base = np.repeat(minimum, 64)[:padded_channels]
        points2 = _numpy_e8_twice((padded - base).reshape(-1, 8) / float(1 << step_log2))
        template = np.clip(base + points2.reshape(-1) * (1 << (step_log2 - 1)), 0, 32767)[:channels]
    return np.broadcast_to(template, (tokens, channels))


class PrefillNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, packet, words):
        recovered = g.decode(packet)
        self.assertTrue(torch.equal(recovered.to(torch.int32), words.to(torch.int32)))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))

    def test_eight_variants_are_in_both_native_presets(self):
        for preset in ("main", "all"):
            methods = [entry for entry in g.native_options(preset) if entry["method"] == "prefill"]
            self.assertEqual(len(methods), 8)
            self.assertEqual({(row["options"]["predictor"], row["options"]["correction"]) for row in methods},
                {(p, c) for p in ("zero", "midpoint", "affine", "e8") for c in ("coset", "residual")})
            self.assertTrue(all(row["options"]["group_size"] == 8 and row["options"]["step_log2"] == 6 for row in methods))

    def test_all_65536_patterns_and_matching_coset_residual_payloads(self):
        words = torch.arange(65536, dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for predictor in ("zero", "midpoint", "affine", "e8"):
            packets = [g.encode(words, method="prefill", predictor=predictor, correction=correction,
                group_size=8, step_log2=6) for correction in ("coset", "residual")]
            for packet in packets:
                self.assert_exact(packet, words)
            self.assertTrue(torch.equal(packets[0].fields["prefill_correction_widths"], packets[1].fields["prefill_correction_widths"]))
            self.assertEqual(packets[0].fields["prefill_correction"].numel(), packets[1].fields["prefill_correction"].numel())
            self.assertEqual(packets[0].metadata["descriptions"]["prefill_correction"]["nbits"],
                packets[1].metadata["descriptions"]["prefill_correction"]["nbits"])

    def test_numpy_predictor_and_remainder_oracle_with_channel_tails(self):
        rng = np.random.default_rng(731)
        for channels in (1, 7, 9, 63, 65, 73):
            words_np = rng.integers(0, 65536, size=(5, channels), dtype=np.uint16)
            words = torch.from_numpy(words_np.copy())
            for predictor in ("zero", "midpoint", "affine", "e8"):
                packet = g.encode(words, method="prefill", predictor=predictor, group_size=8, step_log2=6)
                prediction = g._restore_prefill_prediction(packet.metadata, packet.fields).numpy()
                expected = _numpy_prediction(words_np, predictor)
                np.testing.assert_array_equal(prediction, expected)
                residual = (((words_np.astype(np.int64) & 32767) - expected + 16384) & 32767) - 16384
                zigzag = np.where(residual.reshape(-1) >= 0, 2 * residual.reshape(-1), -2 * residual.reshape(-1) - 1)
                padded = np.pad(zigzag, (0, (-zigzag.size) % 8))
                expected_widths = np.array([int(v).bit_length() for v in padded.reshape(-1, 8).max(axis=1)], dtype=np.uint8)
                np.testing.assert_array_equal(g._restore_widths(packet.fields, packet.metadata["descriptions"],
                    "prefill_correction").numpy(), expected_widths)
                self.assertEqual(packet.metadata["descriptions"]["prefill_correction"]["width_storage_bits"], 4)
                self.assertEqual(packet.fields["prefill_correction_widths"].numel(), (len(expected_widths) + 1) // 2)
                if len(expected_widths) % 2:
                    self.assertEqual(int(packet.fields["prefill_correction_widths"][-1]) >> 4, 0)
                stored = g._restore(packet.fields, packet.metadata["descriptions"], "prefill_correction").numpy()
                expected_moduli = 1 << np.repeat(expected_widths.astype(np.int64), 8)[:words_np.size]
                np.testing.assert_array_equal(stored, (words_np.astype(np.int64).reshape(-1) & 32767) % expected_moduli)
                self.assert_exact(packet, words)

    def test_coset_negative_ties_full_domain_and_zero_width(self):
        # Width 1 may encode error -1; error +1 requires width 2.  At width
        # 15, even the circular tie -16384 and the 32767/0 boundary are exact.
        symbols = [0, 1, 2, 16383, 16384, 16385, 32766, 32767]
        words = torch.tensor(symbols, dtype=torch.uint16).reshape(1, -1)
        packet = g.encode(words, method="prefill", predictor="zero", group_size=1)
        self.assertEqual(g._restore_widths(packet.fields, packet.metadata["descriptions"],
            "prefill_correction").tolist(), [0, 2, 3, 15, 15, 15, 2, 1])
        self.assert_exact(packet, words)
        zero = g.encode(torch.zeros((3, 11), dtype=torch.uint16), method="prefill", predictor="zero", group_size=8)
        self.assertEqual(zero.fields["prefill_correction"].numel(), 0)
        self.assertEqual(zero.metadata["descriptions"]["prefill_correction"]["nbits"], 0)
        self.assert_exact(zero, torch.zeros((3, 11), dtype=torch.uint16))

    def test_singletons_empty_signs_and_nonfinite_bf16_bits(self):
        # Includes +/- zero, infinities, NaN payloads, and subnormals.  The codec
        # manipulates words and therefore preserves all these bit patterns.
        special = [0, 32768, 0x7f80, 0xff80, 0x7fc1, 0xffc2, 1, 32769, 65535]
        for predictor in ("zero", "midpoint", "affine", "e8"):
            for correction in ("coset", "residual"):
                for shape in ((1, 9), (9, 1), (3, 3)):
                    words = torch.tensor(special, dtype=torch.uint16).reshape(shape)
                    self.assert_exact(g.encode(words, method="prefill", predictor=predictor,
                        correction=correction, group_size=8, step_log2=6), words)
                for shape in ((0, 5), (3, 0), (0, 0)):
                    words = torch.empty(shape, dtype=torch.uint16)
                    self.assert_exact(g.encode(words, method="prefill", predictor=predictor,
                        correction=correction, group_size=8, step_log2=6), words)

    def test_self_contained_wire_restore_and_component_accounting(self):
        words = (torch.arange(13 * 65, dtype=torch.int64) * 997 % 65536).to(torch.uint16).reshape(13, 65)
        unrelated = g.make_state({"other": torch.ones((1, 8), dtype=torch.int32)})
        for predictor in ("zero", "midpoint", "affine", "e8"):
            packet = g.encode(words, state=unrelated, method="prefill", predictor=predictor, group_size=8, step_log2=6)
            self.assertEqual(packet.metadata["state_id"], "no_learned_state")
            self.assert_exact(packet, words)
            wire = g.packet_to_cpu(packet)
            restored = g.packet_from_cpu_parts(wire.header.clone(), {name: value.clone() for name, value in wire.fields.items()})
            self.assert_exact(restored, words)
            self.assertEqual(json.loads(bytes(wire.header.tolist())), wire.metadata)
            recipe = packet.stats["prefill_recipe"]
            self.assertEqual(recipe["external_state_bytes"], 0)
            self.assertEqual(sum(recipe[name] for name in ("predictor_payload_bytes", "correction_payload_bytes",
                "correction_width_bytes", "original_sign_bytes", "header_bytes")), packet.physical_nbytes)
            self.assertTrue(torch.equal(g.decode(packet, unrelated), words))

    def test_parameter_validation_and_readonly_dictionary_copy(self):
        words = torch.ones((2, 3), dtype=torch.uint16)
        for options in ({"predictor": "unknown"}, {"correction": "unknown"}, {"predictor": "e8", "step_log2": 0}):
            with self.assertRaises(ValueError):
                g.encode(words, method="prefill", **options)
        source = np.ones((2, 8), dtype=np.uint16)
        source.setflags(write=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state = g.make_state({"readonly": source})
        self.assertEqual(state.physical_nbytes, 2 * 8 * 4)
        self.assertFalse(any("not writable" in str(entry.message) for entry in caught))

    @unittest.skipUnless(torch.cuda.is_available(), "No local CUDA hardware; requires actual CUDA validation")
    def test_cuda_all_predictors_wire_transfer_and_exact_words(self):
        words = torch.arange(65536, device="cuda", dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for predictor in ("zero", "midpoint", "affine", "e8"):
            for correction in ("coset", "residual"):
                packet = g.encode(words, method="prefill", predictor=predictor, correction=correction,
                    group_size=8, step_log2=6)
                self.assert_exact(packet, words)
                host = g.packet_to_cpu(packet)
                restored = g.packet_from_cpu_parts(host.header, host.fields)
                self.assert_exact(g.packet_to_device(restored, "cuda"), words)


if __name__ == "__main__":
    unittest.main()

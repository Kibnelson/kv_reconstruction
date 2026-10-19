"""Correction-layout ablations on exactly the same prefill predictors.

NumPy computes independent grouping/cost oracles.  CPU exercises tensor code;
the explicit CUDA test requires a real device and otherwise reports a skip.
"""
import hashlib
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import gpu_ops as g


def _oracle(words, predictor, layout, group_size):
    magnitude = words.astype(np.int64) & 32767
    tokens, channels = magnitude.shape
    if not magnitude.size:
        prediction = np.zeros_like(magnitude)
    elif predictor == "affine":
        prediction = magnitude[0:1] + ((magnitude[-1:] - magnitude[0:1]) *
            np.arange(tokens)[:, None] // max(tokens - 1, 1))
    elif predictor == "midpoint":
        prediction = np.broadcast_to((magnitude.min(axis=0) + magnitude.max(axis=0)) // 2,
            magnitude.shape)
    else:
        prediction = np.zeros_like(magnitude)
    if layout == "time":
        magnitude, prediction = magnitude.T, prediction.T
    residual = (((magnitude - prediction + 16384) & 32767) - 16384).reshape(-1)
    zigzag = np.where(residual >= 0, residual * 2, -residual * 2 - 1)
    padded = np.pad(zigzag, (0, (-zigzag.size) % group_size))
    widths = np.array([int(v).bit_length() for v in
        padded.reshape(-1, group_size).max(axis=1)], dtype=np.uint8)
    expanded = np.repeat(widths.astype(np.int64), group_size)[:magnitude.size]
    return magnitude.reshape(-1), residual, widths, expanded


class PrefillLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, packet, words):
        self.assertTrue(torch.equal(g.decode(packet).to(torch.int32), words.to(torch.int32)))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))

    def test_six_incremental_options_do_not_change_native_presets(self):
        options = g.insight_layout_options()
        self.assertEqual(len(options), 6)
        self.assertEqual(len({row["name"] for row in options}), 6)
        for row in options:
            option = row["options"]
            self.assertEqual(row["name"], f"native_prefill_{option['predictor']}_{option['layout']}_g{option['group_size']}_residual")
            self.assertEqual(row["method"], "prefill")
            self.assertEqual(option["correction"], "residual")
            self.assertIn((option["layout"], option["group_size"]),
                (("time", 8), ("channel", 64), ("time", 64)))
        for preset in ("main", "all"):
            names = {row["name"] for row in g.native_options(preset)}
            self.assertFalse(names & {row["name"] for row in options})

    def test_channel_default_preserves_preextension_wire_bytes(self):
        # Captured before adding layout support.  Check full header and every
        # named byte stream, rather than only comparing recovered words.
        golden = {
            ("midpoint", "coset"): "f1669a83b362a42cc504df2819d53c97c5c2b8bf934bc3de209b5eeb98b623c9",
            ("midpoint", "residual"): "4a4e0d7e1726a471f555a5cca6fcf87ea04a0c5af215c725e4dce120d48cadb6",
            ("affine", "coset"): "7927e0a2c9cb92e2ab22c8f9c245318dfa45a7a66a48bae479c6460341fca3b1",
            ("affine", "residual"): "965d75f30a6808e766258960dabd781b103fcd18cc2953c738825b93af061b57",
        }
        words = (torch.arange(45, dtype=torch.int64) * 997 % 65536).to(torch.uint16).reshape(5, 9)
        for (predictor, correction), expected in golden.items():
            packet = g.encode(words, method="prefill", predictor=predictor,
                correction=correction, group_size=8, step_log2=6)
            explicit = g.encode(words, method="prefill", predictor=predictor,
                correction=correction, group_size=8, step_log2=6, layout="channel")
            self.assertNotIn("layout", packet.metadata)
            self.assertEqual(packet.metadata, explicit.metadata)
            self.assertTrue(torch.equal(packet.header, explicit.header))
            digest = hashlib.sha256(bytes(packet.header.tolist()))
            for name, field in sorted(packet.fields.items()):
                digest.update(name.encode() + b"\0" + bytes(field.tolist()))
                self.assertTrue(torch.equal(field, explicit.fields[name]))
            self.assertEqual(digest.hexdigest(), expected)

    def test_numpy_layout_width_and_correction_oracle_with_tails(self):
        rng = np.random.default_rng(939)
        for shape in ((1, 1), (1, 9), (9, 1), (5, 7), (11, 13), (17, 65), (0, 7), (5, 0), (0, 0)):
            source = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            words = torch.from_numpy(source.copy())
            for predictor in ("zero", "midpoint", "affine"):
                for layout in ("channel", "time"):
                    for size in (1, 8, 64):
                        magnitude, residual, widths, expanded = _oracle(source, predictor, layout, size)
                        packets = []
                        for correction in ("coset", "residual"):
                            packet = g.encode(words, method="prefill", predictor=predictor,
                                correction=correction, layout=layout, group_size=size, step_log2=6)
                            packets.append(packet)
                            descriptions = packet.metadata["descriptions"]
                            np.testing.assert_array_equal(g._restore_widths(packet.fields,
                                descriptions, "prefill_correction").numpy(), widths)
                            stored = g._restore(packet.fields, descriptions, "prefill_correction").numpy()
                            expected = magnitude % (1 << expanded) if correction == "coset" else residual
                            np.testing.assert_array_equal(stored, expected)
                            self.assertEqual(descriptions["prefill_correction"]["nbits"], int(expanded.sum()))
                            self.assertEqual(packet.fields["prefill_correction_widths"].numel(), (len(widths) + 1) // 2)
                            stats = packet.stats["prefill_recipe"]
                            self.assertEqual(stats["correction_data_bits"], int(expanded.sum()))
                            self.assertEqual(stats["correction_width_bits"], len(widths) * 4)
                            self.assertEqual(stats["correction_data_padding_bits"], (-int(expanded.sum())) % 8)
                            self.assertEqual(stats["correction_width_padding_bits"], (-len(widths) * 4) % 8)
                            self.assert_exact(packet, words)
                        # Coset and residual are matched controls: layout
                        # changes grouping but cannot confer a coset bit gain.
                        self.assertTrue(torch.equal(packets[0].fields["prefill_correction_widths"],
                            packets[1].fields["prefill_correction_widths"]))
                        self.assertEqual(packets[0].fields["prefill_correction"].numel(),
                            packets[1].fields["prefill_correction"].numel())

    def test_every_bf16_pattern_all_predictors_and_both_layouts(self):
        words = torch.arange(65536, dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for predictor in ("zero", "midpoint", "affine", "e8"):
            for layout in ("channel", "time"):
                for correction in ("coset", "residual"):
                    for size in (8, 64):
                        packet = g.encode(words, method="prefill", predictor=predictor,
                            correction=correction, layout=layout, group_size=size, step_log2=6)
                        self.assert_exact(packet, words)

    def test_predictor_and_sign_streams_stay_in_canonical_order(self):
        words = (torch.arange(13 * 73, dtype=torch.int64) * 997 % 65536).to(torch.uint16).reshape(13, 73)
        for predictor in ("zero", "midpoint", "affine", "e8"):
            packets = [g.encode(words, method="prefill", predictor=predictor,
                correction="residual", layout=layout, group_size=8, step_log2=6)
                for layout in ("channel", "time")]
            for name in packets[0].fields:
                if not name.startswith("prefill_correction"):
                    self.assertTrue(torch.equal(packets[0].fields[name], packets[1].fields[name]))
            wire = g.packet_to_cpu(packets[1])
            reread = g.packet_from_cpu_parts(wire.header.clone(),
                {name: value.clone() for name, value in wire.fields.items()})
            self.assert_exact(reread, words)

    def test_time_layout_can_reduce_heterogeneous_channel_widths(self):
        amplitudes = torch.tensor([0, 1, 3, 7, 15, 31, 63, 127], dtype=torch.int64)
        direction = (torch.arange(64, dtype=torch.int64) % 2) * 2 - 1
        words = (16000 + direction[:, None] * amplitudes[None, :]).to(torch.uint16)
        packets = [g.encode(words, method="prefill", predictor="midpoint", correction="residual",
            layout=layout, group_size=8) for layout in ("channel", "time")]
        self.assertLess(packets[1].stats["prefill_recipe"]["correction_data_bits"],
            packets[0].stats["prefill_recipe"]["correction_data_bits"])
        self.assertLess(packets[1].physical_nbytes, packets[0].physical_nbytes)
        for packet in packets:
            self.assert_exact(packet, words)

    def test_encode_decode_stats_do_not_read_device_values_on_host(self):
        words = torch.arange(117, dtype=torch.int64).to(torch.uint16).reshape(13, 9)
        forbidden = AssertionError("unexpected tensor host-value read")
        with mock.patch.object(torch.Tensor, "cpu", side_effect=forbidden), \
             mock.patch.object(torch.Tensor, "item", side_effect=forbidden), \
             mock.patch.object(torch.Tensor, "numpy", side_effect=forbidden), \
             mock.patch.object(torch.Tensor, "tolist", side_effect=forbidden):
            packet = g.encode(words, method="prefill", predictor="midpoint",
                correction="residual", layout="time", group_size=8)
            restored = g.decode(packet)
            stats = packet.stats
        self.assertTrue(torch.equal(restored.to(torch.int32), words.to(torch.int32)))
        self.assertEqual(stats["prefill_recipe"]["layout"], "time")

    def test_reject_unknown_or_inapplicable_layout(self):
        words = torch.ones((3, 7), dtype=torch.uint16)
        for options in ({"method": "prefill", "layout": "unknown"},
                {"method": "raw", "layout": "time"}):
            with self.assertRaisesRegex(ValueError, "layout"):
                g.encode(words, **options)

    @unittest.skipUnless(torch.cuda.is_available(), "No local CUDA hardware; requires actual CUDA validation")
    def test_cuda_layout_full_domain_and_nonmultiple_shape(self):
        for shape in ((256, 256), (13, 73), (0, 7)):
            words = (torch.arange(np.prod(shape), device="cuda", dtype=torch.int64) % 65536).to(torch.uint16).reshape(shape)
            for row in g.insight_layout_options():
                packet = g.encode(words, method=row["method"], **row["options"])
                self.assert_exact(packet, words)
                wire = g.packet_to_cpu(packet)
                reread = g.packet_from_cpu_parts(wire.header, wire.fields)
                self.assert_exact(g.packet_to_device(reread, "cuda"), words)


if __name__ == "__main__":
    unittest.main()

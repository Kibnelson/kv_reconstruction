"""Independent NumPy byte oracle for exact prefill exponent tier packets."""
import unittest

import numpy as np
import torch

from qwen_kv_bench import exponent_recipe as e
from qwen_kv_bench import gpu_ops as g


def _pack_numpy(values, width):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    bits = ((values[:, None] >> np.arange(width, dtype=np.int64)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def _read_range_numpy(payload, start, count, width):
    """Read only the bytes touching the requested independent symbol range."""
    first_bit, stop_bit = start * width, (start + count) * width
    first_byte, stop_byte = first_bit // 8, (stop_bit + 7) // 8
    local = np.frombuffer(payload[first_byte:stop_byte], dtype=np.uint8)
    bits = np.unpackbits(local, bitorder="little")[first_bit % 8:first_bit % 8 + count * width]
    return (bits.reshape(count, width).astype(np.int64) * (1 << np.arange(width))).sum(axis=1)


def _oracle(words, predictor, layout, segment_size):
    integer = words.astype(np.int64)
    exponent = (integer >> 7) & 255
    tokens, channels = words.shape
    fields = {"sign_mantissa": (((integer >> 8) & 128) | (integer & 127)).astype(np.uint8).tobytes()}
    prediction = np.zeros_like(exponent)
    if predictor == "fixed127":
        prediction = np.full_like(exponent, 127)
    if words.size and predictor == "median":
        anchor = np.median(exponent, axis=0).astype(np.uint8)
        fields["exponent_anchor"] = anchor.tobytes()
        prediction = np.broadcast_to(anchor, words.shape)
    if words.size and predictor == "affine":
        first, last = exponent[0], exponent[-1]
        fields["exponent_first"] = first.astype(np.uint8).tobytes()
        fields["exponent_last"] = last.astype(np.uint8).tobytes()
        prediction = first[None, :] + (last - first)[None, :] * np.arange(tokens)[:, None] // max(tokens - 1, 1)
    error = exponent - prediction
    flat = (error if layout == "channel" else error.T).reshape(-1)
    zigzag = np.where(flat >= 0, 2 * flat, -2 * flat - 1)
    fields["exponent_primary"] = _pack_numpy(np.minimum(zigzag, 3), 2)
    fields["exponent_secondary"] = _pack_numpy(np.minimum(zigzag[zigzag >= 3] - 3, 7), 3)
    fields["exponent_rare"] = _pack_numpy(zigzag[zigzag >= 10], 9)
    for name, threshold in (("secondary_offsets", 3), ("rare_offsets", 10)):
        offsets = [0]
        for start in range(0, len(zigzag), segment_size):
            offsets.append(offsets[-1] + int(np.count_nonzero(zigzag[start:start + segment_size] >= threshold)))
        fields[name] = np.asarray(offsets, dtype="<u4").tobytes()
    return fields


class ExponentRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_exact(self, words, predictor, layout, segment_size=256):
        packet = e.encode(words, predictor=predictor, layout=layout, segment_size=segment_size)
        self.assertTrue(torch.equal(e.decode(packet).to(torch.int32), words.to(torch.int32)))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))
        description = e.describe(packet)
        components = ("predictor_payload_bytes", "sign_mantissa_bytes", "correction_payload_bytes",
                      "segment_offset_bytes", "header_bytes")
        self.assertEqual(sum(description[name] for name in components), packet.physical_nbytes)
        return packet

    def test_every_bf16_bit_pattern(self):
        words = torch.arange(65536, dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for option in e.options():
            self.assert_exact(words, **option["options"])

    def test_numpy_byte_oracle_random_tails_singletons_and_empty(self):
        rng = np.random.default_rng(120398)
        for shape in ((1, 1), (1, 9), (9, 1), (6, 7), (17, 65), (0, 7), (7, 0), (0, 0)):
            original = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            words = torch.from_numpy(original.copy())
            for predictor in ("zero", "fixed127", "median", "affine"):
                for layout in ("channel", "time"):
                    for segment in (1, 7, 256):
                        packet = self.assert_exact(words, predictor, layout, segment)
                        expected = _oracle(original, predictor, layout, segment)
                        self.assertEqual({name: value.numpy().tobytes() for name, value in packet.fields.items()}, expected)

    def test_tier_boundaries_and_segment_prefixes(self):
        # With zero prediction z=2*exponent, exercising both sides of each
        # primary/secondary escape threshold and the largest exponent.
        exponent = torch.tensor([0, 1, 2, 4, 5, 6, 127, 254, 255], dtype=torch.int64)
        words = ((exponent << 7) | 0x807f).to(torch.uint16).reshape(3, 3)
        packet = self.assert_exact(words, "zero", "channel", 4)
        self.assertEqual(packet.metadata["secondary_count"], 7)
        self.assertEqual(packet.metadata["rare_count"], 5)
        self.assertEqual(packet.fields["secondary_offsets"].view(torch.uint32).tolist(), [0, 2, 6, 7])
        self.assertEqual(packet.fields["rare_offsets"].view(torch.uint32).tolist(), [0, 0, 4, 5])

    def test_stored_offsets_allow_independent_reverse_segment_reads(self):
        rng = np.random.default_rng(152)
        words_np = rng.integers(0, 65536, size=(17, 79), dtype=np.uint16)
        words = torch.from_numpy(words_np.copy())
        for option in e.options():
            packet = self.assert_exact(words, **option["options"])
            fields = {name: value.numpy().tobytes() for name, value in packet.fields.items()}
            second_offsets = np.frombuffer(fields["secondary_offsets"], dtype="<u4")
            rare_offsets = np.frombuffer(fields["rare_offsets"], dtype="<u4")
            prediction = e._restore_prediction(packet.metadata, packet.fields).numpy()
            error = ((words_np.astype(np.int64) >> 7) & 255) - prediction
            if option["options"]["layout"] == "time":
                error = error.T
            error = error.reshape(-1)
            expected = np.where(error >= 0, 2 * error, -2 * error - 1)
            segment_size = option["options"]["segment_size"]
            # Every segment starts from its offsets and fixed primary position;
            # preceding segments do not need decoding or scanning.
            for segment in reversed(range(packet.metadata["segment_count"])):
                start = segment * segment_size
                count = min(segment_size, words.numel() - start)
                z = _read_range_numpy(fields["exponent_primary"], start, count, 2)
                second = _read_range_numpy(fields["exponent_secondary"], int(second_offsets[segment]),
                    int(second_offsets[segment + 1] - second_offsets[segment]), 3)
                rare = _read_range_numpy(fields["exponent_rare"], int(rare_offsets[segment]),
                    int(rare_offsets[segment + 1] - rare_offsets[segment]), 9)
                self.assertEqual(len(second), int(np.count_nonzero(z == 3)))
                self.assertEqual(len(rare), int(np.count_nonzero(second == 7)))
                correction = second + 3
                correction[second == 7] = rare
                z[z == 3] = correction
                np.testing.assert_array_equal(z, expected[start:start + count])

    def test_median_floor_average_even_and_affine_negative_floor(self):
        exponents = torch.tensor([[100, 104], [101, 100]], dtype=torch.int64)
        words = (exponents << 7).to(torch.uint16)
        median = self.assert_exact(words, "median", "time", 3)
        self.assertEqual(median.fields["exponent_anchor"].tolist(), [100, 102])
        exponents = torch.tensor([[12], [8], [7], [2]], dtype=torch.int64)
        affine = self.assert_exact((exponents << 7).to(torch.uint16), "affine", "time", 3)
        self.assertEqual(e._restore_prediction(affine.metadata, affine.fields).reshape(-1).tolist(), [12, 8, 5, 2])

    def test_self_contained_wire_roundtrip_and_metadata_validation(self):
        words = (torch.arange(1027, dtype=torch.int64) * 1999 % 65536).to(torch.uint16).reshape(13, 79)
        unrelated = g.make_state({"other": torch.ones((2, 8), dtype=torch.int32)})
        for option in e.options():
            packet = self.assert_exact(words, **option["options"])
            generic = g.encode(words, unrelated, method=option["method"], **option["options"])
            self.assertTrue(torch.equal(g.decode(generic, unrelated), words))
            host = g.packet_to_cpu(packet)
            restored = g.packet_from_cpu_parts(host.header.clone(), {name: field.clone() for name, field in host.fields.items()})
            self.assertTrue(torch.equal(e.decode(restored), words))
            self.assertEqual(restored.metadata["state_id"], "no_learned_state")
        for arguments in ({"predictor": "invalid"}, {"layout": "invalid"}, {"segment_size": 0},
                          {"segment_size": True}, {"segment_size": 65537}):
            with self.assertRaises(ValueError):
                e.encode(words, **arguments)
        with self.assertRaises(ValueError):
            e.encode(words.to(torch.int32))
        with self.assertRaises(ValueError):
            e.encode(words.reshape(-1))

    @unittest.skipUnless(torch.cuda.is_available(), "No local CUDA hardware; actual CUDA validation required")
    def test_cuda_all_patterns_and_packet_transfer(self):
        words = torch.arange(65536, device="cuda", dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for option in e.options():
            packet = self.assert_exact(words, **option["options"])
            host = g.packet_to_cpu(packet)
            restored = g.packet_from_cpu_parts(host.header, host.fields)
            self.assertTrue(torch.equal(e.decode(g.packet_to_device(restored, "cuda")), words))


if __name__ == "__main__":
    unittest.main()

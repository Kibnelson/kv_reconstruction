"""Independent byte-wire and exactness tests for device lower-bit coding."""
import copy
import json
import unittest
from unittest import mock
import warnings

import numpy as np
import torch

from qwen_kv_bench import frontier_lower as f


def _pack_cpu(values, width):
    bits = []
    for value in values:
        bits.extend((int(value) >> bit) & 1 for bit in range(width))
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little")


def _oracle_fields(source, mode):
    """Scalar CPU loops produce the independent exact expected wire streams."""
    fields = {}
    if mode == "plane_raw":
        residual = source
    else:
        fields["lower_first_row"] = source[0].copy()
        residual = source[1:] ^ (source[:1] if mode == "plane_first" else source[:-1])
    ordered = residual.T.reshape(-1)
    modes, raw, counts, values, positions = [], [], [], [], []
    mode_counts = {"raw": 0, "zero": 0, "one": 0, "minority": 0}
    for plane in range(8):
        bits = [(int(value) >> plane) & 1 for value in ordered]
        for start in range(0, len(bits), 256):
            group = bits[start:start + 256]
            one_count = sum(group)
            zero_count = len(group) - one_count
            minority = min(one_count, zero_count)
            if one_count == 0:
                modes.append(1)
                mode_counts["zero"] += 1
            elif zero_count == 0:
                modes.append(2)
                mode_counts["one"] += 1
            elif 10 + 8 * minority < len(group):
                modes.append(3)
                mode_counts["minority"] += 1
                value = int(one_count <= zero_count)
                counts.append(minority)
                values.append(value)
                positions.extend(index for index, bit in enumerate(group) if bit == value)
            else:
                modes.append(0)
                mode_counts["raw"] += 1
                raw.extend(group)
    fields.update(lower_modes=_pack_cpu(modes, 2), lower_raw_bits=_pack_cpu(raw, 1),
        lower_minority_counts=_pack_cpu(counts, 9), lower_minority_values=_pack_cpu(values, 1),
        lower_minority_positions=np.asarray(positions, dtype=np.uint8))
    return fields, dict(group_count=len(modes), raw_bit_count=len(raw),
        minority_group_count=len(counts), minority_position_count=len(positions), mode_counts=mode_counts)


def _mixed_planes(mode, rows=258, channels=17):
    """Residual bitplanes exercise zero, one, sparse-one/zero and raw groups."""
    residual_rows = rows if mode == "plane_raw" else rows - 1
    count = residual_rows * channels
    index = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(715)
    planes = [np.zeros(count, np.uint8), np.ones(count, np.uint8),
        (index % 256 == 7).astype(np.uint8), (index % 256 != 23).astype(np.uint8),
        (index & 1).astype(np.uint8), rng.integers(0, 2, count, dtype=np.uint8),
        np.zeros(count, np.uint8), np.ones(count, np.uint8)]
    ordered = sum(plane.astype(np.uint16) << bit for bit, plane in enumerate(planes)).astype(np.uint8)
    residual = ordered.reshape(channels, residual_rows).T.copy()
    if mode == "plane_raw":
        return residual
    first = ((np.arange(channels) * 13) % 256).astype(np.uint8)[None, :]
    rest = residual ^ first if mode == "plane_first" else np.bitwise_xor.accumulate(residual, axis=0) ^ first
    return np.concatenate((first, rest))


class FrontierLowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_roundtrip(self, source, mode):
        tensor = source if isinstance(source, torch.Tensor) else torch.from_numpy(source.copy())
        before = tensor.clone()
        fields, meta = f.encode(tensor, mode)
        restored = f.decode(fields, json.loads(json.dumps(meta)))
        self.assertTrue(torch.equal(restored, tensor), (tuple(tensor.shape), mode))
        self.assertTrue(torch.equal(tensor, before))
        self.assertEqual(tuple(restored.shape), tuple(tensor.shape))
        self.assertEqual(restored.dtype, torch.uint8)
        self.assertEqual(restored.device, tensor.device)
        self.assertTrue(all(value.dtype == torch.uint8 and value.ndim == 1 and
            value.is_contiguous() and value.device == tensor.device for value in fields.values()))
        for value in fields.values():
            self.assertEqual(value.untyped_storage().nbytes(), value.numel())
        return fields, meta

    def test_all_byte_values_and_lower_bytes_from_all_bf16_words(self):
        complete = torch.arange(256, dtype=torch.int64).to(torch.uint8).reshape(16, 16)
        words = torch.arange(65536, dtype=torch.int64).reshape(256, 256)
        lower = (((words >> 8) & 128) | (words & 127)).to(torch.uint8)
        self.assertEqual(torch.unique(lower).numel(), 256)
        for mode in f.MODES:
            self.assert_roundtrip(complete, mode)
            self.assert_roundtrip(lower, mode)

    def test_independent_wire_oracle_every_group_mode_and_tail(self):
        for mode in ("plane_raw", "plane_first", "plane_prev"):
            source = _mixed_planes(mode)
            fields, meta = self.assert_roundtrip(source, mode)
            self.assertEqual(meta["storage"], "adaptive_bitplanes")
            expected, statistics = _oracle_fields(source, mode)
            self.assertEqual(set(fields), set(expected))
            for name in fields:
                np.testing.assert_array_equal(fields[name].numpy(), expected[name], err_msg=(mode, name))
            for name, value in statistics.items():
                self.assertEqual(meta[name], value)
            self.assertTrue(all(count > 0 for count in meta["mode_counts"].values()))
            self.assertEqual(sum(meta["mode_counts"].values()), meta["group_count"])
            # Each stored stream is packed once, including terminal padding;
            # no full 256-bit payload padding is charged for a short tail.
            self.assertEqual(fields["lower_modes"].numel(), (meta["group_count"] * 2 + 7) // 8)
            self.assertEqual(fields["lower_raw_bits"].numel(), (meta["raw_bit_count"] + 7) // 8)
            self.assertEqual(fields["lower_minority_counts"].numel(), (meta["minority_group_count"] * 9 + 7) // 8)
            self.assertEqual(fields["lower_minority_values"].numel(), (meta["minority_group_count"] + 7) // 8)
            self.assertEqual(fields["lower_minority_positions"].numel(), meta["minority_position_count"])

    def test_random_tails_singletons_empty_and_noncontiguous(self):
        rng = np.random.default_rng(605)
        for shape in ((0, 0), (0, 7), (3, 0), (1, 1), (1, 17), (17, 1),
                (3, 7), (17, 33), (33, 17), (257, 7), (7, 257)):
            source = rng.integers(0, 256, shape, dtype=np.uint8)
            for mode in f.MODES:
                self.assert_roundtrip(source, mode)
        noncontiguous = torch.arange(17 * 33, dtype=torch.int64).to(torch.uint8).reshape(17, 33).T
        self.assertFalse(noncontiguous.is_contiguous())
        for mode in f.MODES:
            self.assert_roundtrip(noncontiguous, mode)

    def test_prediction_first_and_previous_are_independently_reversible(self):
        for mode in ("plane_first", "plane_prev"):
            source = _mixed_planes(mode, rows=259, channels=19)
            fields, meta = self.assert_roundtrip(source, mode)
            self.assertEqual(meta["storage"], "adaptive_bitplanes")
            self.assertEqual(meta["value_count"], (len(source) - 1) * source.shape[1])
            np.testing.assert_array_equal(fields["lower_first_row"].numpy(), source[0])
            oracle, _ = _oracle_fields(source, mode)
            self.assertEqual(sum(v.numel() for v in fields.values()), sum(v.size for v in oracle.values()))

    def test_raw_fallback_counts_every_byte_and_owns_payload_storage(self):
        rng = np.random.default_rng(294)
        source = rng.integers(0, 256, (37, 13), dtype=np.uint8)
        tensor = torch.from_numpy(source.copy())
        for mode in f.MODES:
            fields, meta = f.encode(tensor, mode)
            self.assertEqual(meta["storage"], "raw8")
            self.assertEqual(set(fields), {"lower_raw"})
            self.assertEqual(fields["lower_raw"].numel(), source.size)
            self.assertNotEqual(fields["lower_raw"].untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
            np.testing.assert_array_equal(f.decode(fields, meta).numpy(), source)

    def test_readonly_numpy_alias_is_never_mutated(self):
        source = _mixed_planes("plane_raw")
        expected = source.copy()
        source.setflags(write=False)
        # PyTorch emits its own warning when constructing a tensor alias of
        # readonly NumPy storage. The codec receives that tensor and must never
        # mutate it, even though PyTorch does not enforce the NumPy flag.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tensor = torch.from_numpy(source)
        for mode in f.MODES:
            self.assert_roundtrip(tensor, mode)
        np.testing.assert_array_equal(source, expected)
        self.assertFalse(source.flags.writeable)

    def test_codec_never_reads_payload_values_through_host_accessors(self):
        source = torch.from_numpy(_mixed_planes("plane_prev"))
        forbidden = AssertionError("Unexpected device payload read on host")
        with mock.patch.object(torch.Tensor, "cpu", side_effect=forbidden), \
             mock.patch.object(torch.Tensor, "item", side_effect=forbidden), \
             mock.patch.object(torch.Tensor, "numpy", side_effect=forbidden), \
             mock.patch.object(torch.Tensor, "tolist", side_effect=forbidden):
            fields, meta = f.encode(source, "plane_prev")
            restored = f.decode(fields, meta)
        self.assertTrue(torch.equal(source, restored))
        self.assertEqual(meta["storage"], "adaptive_bitplanes")

    def test_malformed_metadata_and_field_sizes_are_rejected(self):
        source = torch.from_numpy(_mixed_planes("plane_raw"))
        fields, meta = f.encode(source, "plane_raw")
        self.assertEqual(meta["storage"], "adaptive_bitplanes")
        for key, value in (("shape", [1]), ("group_count", meta["group_count"] + 1),
                ("group_bits", 128), ("minority_count_width_bits", 8), ("mode", "bad"),
                ("raw_bit_count", -1), ("minority_group_count", True)):
            changed = copy.deepcopy(meta)
            changed[key] = value
            with self.assertRaises(ValueError, msg=key):
                f.decode(fields, changed)
        broken = dict(fields)
        broken["lower_modes"] = broken["lower_modes"][:-1].contiguous()
        with self.assertRaisesRegex(ValueError, "field size"):
            f.decode(broken, meta)
        with self.assertRaisesRegex(ValueError, "torch.uint8"):
            f.encode(source.to(torch.int16))
        with self.assertRaisesRegex(ValueError, "mode"):
            f.encode(source, "bad")

    @unittest.skipUnless(torch.cuda.is_available(), "No local CUDA hardware; requires actual GPU validation")
    def test_cuda_all_symbols_tails_and_adaptive_wire_matches_cpu(self):
        words = torch.arange(65536, dtype=torch.int64, device="cuda").reshape(256, 256)
        lower = (((words >> 8) & 128) | (words & 127)).to(torch.uint8)
        for mode in f.MODES:
            self.assert_roundtrip(lower, mode)
        for mode in ("plane_raw", "plane_first", "plane_prev"):
            source = _mixed_planes(mode)
            gpu_fields, gpu_meta = self.assert_roundtrip(torch.from_numpy(source.copy()).to("cuda"), mode)
            cpu_fields, cpu_meta = f.encode(torch.from_numpy(source.copy()), mode)
            self.assertEqual(gpu_meta, cpu_meta)
            for name in cpu_fields:
                self.assertTrue(torch.equal(gpu_fields[name].cpu(), cpu_fields[name]))


if __name__ == "__main__":
    unittest.main()

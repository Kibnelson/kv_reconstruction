"""Independent wire-format and device-path tests for conditional Huffman coding."""
import copy
import unittest
from unittest import mock
import zlib

import numpy as np
import torch

from qwen_kv_bench import conditional_entropy as h


def _oracle_encode(values, book):
    text = "".join(format(int(book["code"][v]), f"0{int(book['length'][v])}b") for v in values)
    padding = (-len(text)) % 8
    padded = text + "0" * padding
    return bytes(int(padded[pos:pos + 8], 2) for pos in range(0, len(padded), 8)), len(text)


def _oracle_decode(payload, count, book):
    mapping = {format(int(code), f"0{int(width)}b"): symbol for symbol, (code, width)
        in enumerate(zip(book["code"], book["length"]))}
    bits = "".join(format(value, "08b") for value in payload)
    output, prefix = [], ""
    for bit in bits:
        prefix += bit
        if prefix in mapping:
            output.append(mapping[prefix])
            prefix = ""
            if len(output) == count:
                return output
    if count:
        raise AssertionError("oracle found a truncated stream")
    return []


class ConditionalEntropyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matches_independent_msb_prefix_oracle(self):
        book = h.fit_codebook(np.array([90, 7, 2, 0, 1]))
        state = h.prepare(book, "cpu")
        values = np.array([0, 0, 1, 4, 3, 0, 2, 1, 0, 4, 4], dtype=np.int64)
        fields, desc = h.encode(torch.tensor(values), state)
        expected, nbits = _oracle_encode(values, book)
        self.assertEqual(fields["payload"].numpy().tobytes(), expected)
        self.assertEqual(desc["nbits"], nbits)
        self.assertEqual(_oracle_decode(expected, len(values), book), values.tolist())
        self.assertTrue(torch.equal(h.decode(fields, desc, state), torch.tensor(values)))
        self.assertEqual(int.from_bytes(fields["checksum"].numpy().tobytes(), "little"), zlib.adler32(expected))
        self.assertEqual(state.physical_nbytes, sum(getattr(state, name).numel() * getattr(state, name).element_size()
            for name in ("code", "length", "lookup_symbol", "lookup_length")))

    def test_all_symbols_unseen_training_empty_singleton_and_tails(self):
        for alphabet in (1, 2, 3, 7, 256, 511, 512):
            counts = np.zeros(alphabet, dtype=np.int64)
            counts[0] = 100000
            book = h.fit_codebook(counts)
            self.assertTrue(np.all(book["length"] > 0))
            state = h.prepare(book, "cpu")
            for count in (0, 1, 7, 8, 9, 31, alphabet * 2 + 3):
                values = torch.arange(count, dtype=torch.int64) % alphabet
                fields, desc = h.encode(values, state)
                self.assertEqual(fields["payload"].numel(), (desc["nbits"] + 7) // 8)
                self.assertEqual(fields["checksum"].numel(), 4)
                self.assertTrue(torch.equal(h.decode(fields, desc, state), values))
        uniform = h.fit_codebook(np.zeros(512))
        self.assertTrue(np.all(uniform["length"] == 9))

    def test_bounded_lengths_smoothed_cost_and_canonical_identity(self):
        counts = np.power(1.55, np.arange(80, dtype=np.float64))
        book = h.fit_codebook(counts, max_bits=7)
        self.assertLessEqual(int(book["length"].max()), 7)
        self.assertGreater(book["smoothing_rounds"], 0)
        self.assertGreater(book["smoothing_pseudocount"], 0)
        self.assertAlmostEqual(book["training_mean_bits"], float(np.dot(counts / counts.sum(), book["length"])))
        self.assertEqual(book["codebook_id"], h.fit_codebook(counts, max_bits=7)["codebook_id"])
        values = torch.arange(80, dtype=torch.int64).repeat(3)
        fields, desc = h.encode(values, h.prepare(book, "cpu"))
        self.assertTrue(torch.equal(h.decode(fields, desc, h.prepare(book, "cpu")), values))
        for counts, bits in ((np.ones(9), 3), (np.array([-1, 2]), 3), (np.array([np.nan]), 3), (np.zeros(513), 15)):
            with self.assertRaises(ValueError):
                h.fit_codebook(counts, bits)

    def test_fifteen_bit_codes_cross_all_three_byte_windows(self):
        book = h.fit_codebook(np.power(4.0, np.arange(16)))
        self.assertEqual(int(book["length"].max()), 15)
        values = np.tile(np.arange(16, dtype=np.int64), 37)
        state = h.prepare(book, "cpu")
        fields, desc = h.encode(torch.tensor(values), state)
        expected, nbits = _oracle_encode(values, book)
        self.assertEqual(fields["payload"].numpy().tobytes(), expected)
        self.assertEqual(desc["nbits"], nbits)
        self.assertTrue(torch.equal(h.decode(fields, desc, state), torch.tensor(values)))

    def test_detects_corruption_truncation_padding_and_wrong_books(self):
        state = h.prepare(h.fit_codebook(np.array([10, 3, 1])), "cpu")
        fields, desc = h.encode(torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64), state)
        corrupted = {name: value.clone() for name, value in fields.items()}
        corrupted["payload"][0] ^= 128
        with self.assertRaisesRegex(ValueError, "checksum"):
            h.decode(corrupted, desc, state)
        truncated = dict(fields, payload=fields["payload"][:-1])
        with self.assertRaises(ValueError):
            h.decode(truncated, desc, state)
        changed = dict(desc, num_symbols=desc["num_symbols"] + 1)
        with self.assertRaises(ValueError):
            h.decode(fields, changed, state)
        other = h.prepare(h.fit_codebook(np.array([1, 3, 10])), "cpu")
        with self.assertRaisesRegex(ValueError, "identity"):
            h.decode(fields, desc, other)
        # A single-bit book has one invalid prefix.  Refreshing a checksum must
        # not make that malformed code path appear to be a valid stream.
        single = h.prepare(h.fit_codebook(np.array([0])), "cpu")
        sf, sd = h.encode(torch.tensor([0]), single)
        sf["payload"][0] = 128
        sf["checksum"] = h._checksum(sf["payload"])
        with self.assertRaisesRegex(ValueError, "boundary"):
            h.decode(sf, sd, single)
        sf, sd = h.encode(torch.tensor([0]), single)
        sf["payload"][0] = 1
        sf["checksum"] = h._checksum(sf["payload"])
        with self.assertRaisesRegex(ValueError, "padding"):
            h.decode(sf, sd, single)

    def test_payload_paths_do_not_call_host_conversion(self):
        state = h.prepare(h.fit_codebook(np.array([17, 2, 0, 1])), "cpu")
        values = torch.tensor([3, 0, 1, 0, 2, 0], dtype=torch.int64)
        with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("payload cpu copy")), \
                mock.patch.object(torch.Tensor, "numpy", side_effect=AssertionError("payload numpy copy")), \
                mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("payload list copy")):
            fields, desc = h.encode(values, state)
            result = h.decode(fields, desc, state)
            self.assertTrue(torch.equal(result, values))

    def test_invalid_tensor_and_codebook_inputs(self):
        book = h.fit_codebook(np.array([1, 2]))
        state = h.prepare(book, "cpu")
        for values in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([0.0]), torch.zeros((1, 1), dtype=torch.int64)):
            with self.assertRaises(ValueError):
                h.encode(values, state)
        changed = copy.deepcopy(book)
        changed["lookup_length"][0] += 1
        with self.assertRaises(ValueError):
            h.prepare(changed, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_cuda_same_wire_bytes_and_roundtrip(self):
        book = h.fit_codebook(np.arange(512, dtype=np.int64) ** 2)
        cpu, gpu = h.prepare(book, "cpu"), h.prepare(book, "cuda")
        values = torch.arange(4099, dtype=torch.int64) % 512
        cpu_fields, cpu_desc = h.encode(values, cpu)
        gpu_fields, gpu_desc = h.encode(values.cuda(), gpu)
        self.assertEqual(cpu_desc, gpu_desc)
        for name in cpu_fields:
            self.assertTrue(torch.equal(cpu_fields[name], gpu_fields[name].cpu()))
        self.assertTrue(torch.equal(h.decode(gpu_fields, gpu_desc, gpu).cpu(), values))


if __name__ == "__main__":
    unittest.main()

"""Full-domain exactness and deterministic binary-predictor arithmetic checks."""
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from kv_research.learned_codec import LearnedModel, _unpack
from kv_smoke.codec import bf16_words_to_float32
from kv_binary.binary_codec import (make_state, save_state, load_state, state_bytes,
    encode_matrix, decode_matrix, record_info, canonical_matmul, canonical_row_dot,
    reconstruct_prediction, direction_from_bits, _RECORD, _STATE)


def fixture_model():
    rng = np.random.default_rng(771)
    book = rng.integers(15200, 16500, size=(16, 8), dtype=np.uint16)
    book[0] = 0
    book[1] = np.arange(8, dtype=np.uint16) + 16000
    book.flags.writeable = False
    return LearnedModel({"test": book}, {"n_centroids": 16, "vector_dim": 8}, {})


def predictor_fields(data, model, state):
    info = record_info(data)
    total = int(np.prod(info["shape"]))
    padded = ((total + state.dimension - 1) // state.dimension) * state.dimension
    nids, nvectors = padded // 8, padded // state.dimension
    qbits = (model.n_centroids - 1).bit_length()
    cursor = _RECORD.size + len(info["dictionary_key"].encode())
    idbytes, signbytes = (nids * qbits + 7) // 8, (padded + 7) // 8
    ids = _unpack(data[cursor:cursor + idbytes], nids, qbits)
    cursor += idbytes + signbytes
    bits = _unpack(data[cursor:cursor + signbytes], padded, 1).reshape(-1, state.dimension)
    cursor += signbytes
    if info["amplitude_precision"] == "bf16":
        amplitudes = bf16_words_to_float32(np.frombuffer(data, dtype="<u2", count=nvectors, offset=cursor))
    else:
        amplitudes = np.frombuffer(data, dtype="<f4", count=nvectors, offset=cursor).copy()
    return ids, bits, amplitudes


class BinaryCodecTests(unittest.TestCase):
    def test_state_serialization_and_real_dense_rotation(self):
        model = fixture_model()
        with tempfile.TemporaryDirectory() as directory:
            for dimension in (8, 32, 64):
                for transform in ("identity", "dense", "hadamard"):
                    state = make_state(model, dimension, transform)
                    data = state_bytes(state)
                    expected = 52 + (dimension * dimension * 8 if transform == "dense" else 0)
                    self.assertEqual(len(data), expected)
                    self.assertEqual(state.runtime_rotation_array_bytes, dimension * dimension * 8)
                    restored = load_state(data)
                    self.assertEqual(state.state_id, restored.state_id)
                    self.assertTrue(np.array_equal(state.rotation, restored.rotation))
                    gram = canonical_matmul(state.rotation.T, state.rotation)
                    self.assertLess(float(np.max(np.abs(gram - np.eye(dimension)))), 1e-12)
                    if transform == "dense":
                        self.assertFalse(np.array_equal(state.rotation, np.eye(dimension)))
                    path = Path(directory) / f"{transform}_{dimension}.bin"
                    info = save_state(state, path)
                    self.assertEqual(info["serialized_state_bytes"], len(data))
                    self.assertEqual(load_state(path).state_id, state.state_id)

    def test_full_bf16_domain_and_actual_finite_mechanism(self):
        model = fixture_model()
        all_words = np.arange(65536, dtype=np.uint16)
        finite = all_words[(all_words & 0x7F80) != 0x7F80].reshape(255, 256)
        for dimension, transform in ((8, "identity"), (32, "hadamard"), (64, "dense")):
            state = make_state(model, dimension, transform)
            for precision in ("f32", "bf16"):
                encoded = encode_matrix(finite, model, state, "test", force=True, amplitude_precision=precision)
                self.assertTrue(record_info(encoded)["is_binary"])
                self.assertEqual(encoded.stats["mechanism_counts"]["binary_vectors"], finite.size // dimension)
                self.assertTrue(np.array_equal(decode_matrix(encoded, model, state), finite))
                self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
                encoded = encode_matrix(all_words.reshape(256, 256), model, state, "test", force=True, amplitude_precision=precision)
                self.assertEqual(encoded.stats["mechanism_counts"]["raw_nonfinite_records"], 1)
                self.assertTrue(np.array_equal(decode_matrix(encoded, model, state), all_words.reshape(256, 256)))

    def test_tails_nonfinites_signed_zero_and_assignment_controls(self):
        model = fixture_model()
        rng = np.random.default_rng(772)
        state = make_state(model, 64, "dense")
        cases = [rng.integers(14000, 17000, size=(1, n), dtype=np.uint16) for n in (1, 7, 15, 63, 65)]
        cases += [np.full((3, 31), 0x8000, dtype=np.uint16), np.empty((0, 7), dtype=np.uint16),
            np.asarray([[0, 0x8000, 1, 0x8001, 0x7F7F, 0xFF7F, 0x7F80, 0xFFC1]], dtype=np.uint16)]
        for words in cases:
            for assignment in ("radius", "l1"):
                for force in (True, False):
                    encoded = encode_matrix(words, model, state, "test", assignment=assignment, force=force)
                    self.assertTrue(np.array_equal(decode_matrix(encoded.data, model, state), words))
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))

    def test_zero_residual_tie_positive_and_stored_amplitude(self):
        model = fixture_model()
        state = make_state(model, 8, "dense")
        words = model.codebooks["test"][1].reshape(1, 8).copy()
        encoded = encode_matrix(words, model, state, "test", force=True)
        ids, bits, amplitude = predictor_fields(encoded.data, model, state)
        self.assertTrue(np.all(bits == 1))
        self.assertTrue(np.all(amplitude == 0))
        self.assertEqual(encoded.stats["mechanism_counts"]["binary_records"], 1)
        self.assertEqual(encoded.stats["mechanism_counts"]["nonzero_amplitudes"], 0)
        self.assertTrue(np.array_equal(decode_matrix(encoded, model, state), words))

    def test_round_to_even_combined_prediction_half_ties(self):
        model = fixture_model()
        state = make_state(model, 64, "identity")
        prototypes = np.full((1, 64), 10, dtype=np.uint16)
        prototypes[0, :6] = [1, 2, 1, 2, 32767, 0]
        bits = np.ones((1, 64), dtype=np.uint16)
        bits[0, [2, 3, 5]] = 0
        # sqrt64=8 exactly, so stored amplitude4 gives contributions+/-0.5.
        amplitude = np.asarray([4.0], dtype=np.float32)
        prediction = reconstruct_prediction(prototypes, bits, amplitude, state)
        self.assertEqual(prediction[0, :6].tolist(), [2, 2, 0, 2, 32767, 0])

    def test_bf16_amplitudes_above_float16_range_and_f32_compatibility(self):
        model = fixture_model()
        state = make_state(model, 64, "identity")
        words = np.full((16, 192), 0x7F7F, dtype=np.uint16)
        original = encode_matrix(words, model, state, "test", force=True)
        explicit = encode_matrix(words, model, state, "test", force=True, amplitude_precision="f32")
        self.assertEqual(original.data, explicit.data)
        encoded = encode_matrix(words, model, state, "test", force=True, amplitude_precision="bf16")
        info = record_info(encoded)
        self.assertEqual(info["amplitude_precision"], "bf16")
        self.assertEqual(info["amplitude_itemsize"], 2)
        _, bits, amplitudes = predictor_fields(encoded.data, model, state)
        self.assertTrue(np.all(np.isfinite(amplitudes)))
        self.assertGreater(float(amplitudes.max()), 65504)
        self.assertTrue(np.all((amplitudes.view(np.uint32) & 0xFFFF) == 0))
        self.assertEqual(encoded.stats["component_bytes"]["bf16_amplitudes"], 2 * len(amplitudes))
        self.assertTrue(np.array_equal(decode_matrix(encoded, model, state), words))

    def test_canonical_batch_scalar_and_d_aligned_segmentation(self):
        model = fixture_model()
        state = make_state(model, 64, "dense")
        rng = np.random.default_rng(773)
        left = rng.normal(size=(17, 64))
        whole = canonical_matmul(left, state.rotation)
        scalar = np.concatenate([canonical_matmul(row[None], state.rotation) for row in left])
        self.assertTrue(np.array_equal(whole.view(np.uint64), scalar.view(np.uint64)))
        dots = canonical_row_dot(left, whole)
        scalar_dots = np.concatenate([canonical_row_dot(a[None], b[None]) for a, b in zip(left, whole)])
        self.assertTrue(np.array_equal(dots.view(np.uint64), scalar_dots.view(np.uint64)))
        words = rng.integers(14000, 17500, size=(16, 192), dtype=np.uint16)
        for precision in ("f32", "bf16"):
            all_record = encode_matrix(words, model, state, "test", force=True, amplitude_precision=precision)
            all_fields = predictor_fields(all_record.data, model, state)
            split_fields = [predictor_fields(encode_matrix(part, model, state, "test", force=True,
                            amplitude_precision=precision).data, model, state) for part in (words[:8], words[8:])]
            for field, split in zip(all_fields, zip(*split_fields)):
                self.assertTrue(np.array_equal(field, np.concatenate(split)))
            ids, bits, amps = all_fields
            c = model.codebooks["test"][ids].reshape(-1, 64)
            batched = reconstruct_prediction(c, bits, amps, state)
            per_vector = np.concatenate([reconstruct_prediction(a[None], b[None], amp[None], state)
                                        for a, b, amp in zip(c, bits, amps)])
            self.assertTrue(np.array_equal(batched, per_vector))

    def test_model_state_identity_corruption_and_invalid_amplitudes(self):
        model = fixture_model()
        state = make_state(model, 32, "dense")
        words = np.full((2, 64), 0x3F80, dtype=np.uint16)
        encoded = encode_matrix(words, model, state, "test", force=True)
        wrong_state = make_state(model, 32, "dense", seed=104730)
        with self.assertRaises(ValueError):
            decode_matrix(encoded, model, wrong_state)
        wrong_model = fixture_model()
        wrong_model.config["changed"] = True
        with self.assertRaises(ValueError):
            decode_matrix(encoded, wrong_model, state)
        for payload in (encoded.data[:10], encoded.data[:-1], encoded.data + b"x"):
            with self.assertRaises(ValueError):
                decode_matrix(payload, model, state)
        for index in (0, _RECORD.size + 8, len(encoded.data) - 5):
            damaged = bytearray(encoded.data)
            damaged[index] ^= 1
            with self.assertRaises(ValueError):
                decode_matrix(damaged, model, state)
        damaged = bytearray(state_bytes(state)); damaged[_STATE.size] ^= 1
        with self.assertRaises(ValueError):
            load_state(damaged)
        # A caller who recomputes an outer CRC still cannot inject negative or
        # nonfinite stored amplitudes into the canonical predictor.
        for precision in ("f32", "bf16"):
            encoded = encode_matrix(words, model, state, "test", force=True, amplitude_precision=precision)
            info = record_info(encoded)
            n = 128; nids = n // 8; amp_offset = _RECORD.size + len(info["dictionary_key"]) + (nids * 4 + 7) // 8 + 2 * (n // 8)
            invalid = (struct.pack("<f", float("nan")), struct.pack("<f", -1.0)) if precision == "f32" else (struct.pack("<H", 0x7FC0), struct.pack("<H", 0xBF80))
            for amplitude_data in invalid:
                damaged = bytearray(encoded.data)
                damaged[amp_offset:amp_offset + len(amplitude_data)] = amplitude_data
                damaged[-4:] = struct.pack("<I", zlib.crc32(damaged[:-4]))
                with self.assertRaises(ValueError):
                    decode_matrix(damaged, model, state)


if __name__ == "__main__":
    unittest.main()

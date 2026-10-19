"""Independent format/accounting checks for matched predictor comparisons."""

import pathlib
import struct
import tempfile
import unittest
import zlib

import numpy as np

from kv_research import container, e8_codec, learned_codec
from kv_residual import learned, learned_e8
from kv_residual.run_eval import factories
from kv_stacked.run_eval import aggregate


def fixture_model(keys=("x",)):
    rng = np.random.default_rng(1277)
    book = rng.integers(13500, 17000, (16, 8), dtype=np.uint16)
    return learned_codec.LearnedModel(
        {key: book.copy() for key in keys},
        {"n_centroids": 16, "vector_dim": 8},
        {"synthetic_protocol_fixture": True},
    )


def learned_ids(record):
    fields = learned._HEADER.unpack_from(record)
    qbits, rows, columns, keylen = fields[4], fields[5], fields[6], fields[8]
    offset = learned._HEADER.size + keylen
    count = (rows * columns + 7) // 8
    length = (count * qbits + 7) // 8
    return learned_codec._unpack(record[offset:offset + length], count, qbits)


def residual_controls(record):
    if zlib.crc32(record[:-4]) != struct.unpack_from("<I", record, len(record) - 4)[0]:
        raise AssertionError("invalid residual matrix checksum")
    record = record[:-4]
    fields = learned_e8._HEADER.unpack_from(record)
    qbits, step, flags, group, rows, columns, keylen = (
        fields[2], fields[3], fields[4], fields[5], fields[6], fields[7], fields[9]
    )
    family = "scalar" if flags & 2 else "e8"
    offset = learned_e8._HEADER.size + keylen
    ids, points = [], []
    for start in range(0, rows * columns, group):
        mode, length = learned_e8._GROUP.unpack_from(record, offset)
        offset += learned_e8._GROUP.size
        if mode != 1:
            raise AssertionError("forced finite fixture used hidden raw fallback")
        payload = record[offset:offset + length]
        offset += length
        count = (min(group, rows * columns - start) + 7) // 8
        minimum, width = learned_e8._E8_METADATA.unpack_from(payload)
        cursor = learned_e8._E8_METADATA.size
        length_ids = (count * qbits + 7) // 8
        ids.append(learned_codec._unpack(payload[cursor:cursor + length_ids], count, qbits))
        cursor += length_ids
        if family == "e8":
            length_coords = e8_codec._coordinates_size(count, width)
            points.append(e8_codec._unpack_coordinates(
                payload[cursor:cursor + length_coords], count, minimum, width))
        else:
            length_coords = (count * 8 * width + 7) // 8
            coordinates = learned_codec._unpack(
                payload[cursor:cursor + length_coords], count * 8, width)
            points.append(coordinates.reshape(count, 8).astype(np.int32) + minimum)
    if offset != len(record):
        raise AssertionError("unexpected residual format tail")
    return np.concatenate(ids), np.concatenate(points), step


class ProtocolTests(unittest.TestCase):
    def test_serialized_radius_ids_and_residual_points_match_controls(self):
        rng = np.random.default_rng(811)
        words = rng.integers(12000, 19500, (3, 29), dtype=np.uint16)
        words |= rng.integers(0, 2, words.shape, dtype=np.uint16) << 15
        model = fixture_model()
        padded = np.pad(words.reshape(-1), (0, (-words.size) % 8)).reshape(-1, 8)
        magnitude = padded & 32767
        expected_ids, _, _ = learned_codec._assign(magnitude, model.codebooks["x"])
        residual = magnitude.astype(np.int32) - model.codebooks["x"][expected_ids].astype(np.int32)
        for correction in ("coset", "best_entropy"):
            encoded = learned.encode_matrix(words, model, "x", correction, "radius", force=True)
            np.testing.assert_array_equal(learned_ids(encoded.data), expected_ids)
        for family in ("e8", "scalar"):
            for correction in ("best", "coset"):
                encoded = learned_e8.encode_matrix(
                    words, model, "x", step_log2=7, correction=correction,
                    group_size=64, family=family, force=True)
                ids, points, step = residual_controls(encoded.data)
                np.testing.assert_array_equal(ids, expected_ids)
                normalized = residual.astype(np.float64) / (1 << step)
                expected_points = (
                    e8_codec.nearest_e8(normalized) if family == "e8"
                    else np.rint(normalized)
                )
                np.testing.assert_array_equal(points, expected_points)
                if family == "e8":
                    self.assertTrue(np.all(e8_codec.is_e8_point(points)))
                    self.assertGreater(np.count_nonzero(np.any(points != 0, axis=1)), 0)
                np.testing.assert_array_equal(learned_e8.decode_matrix(encoded, model), words)

    def test_factory_counts_full_state_and_actual_forced_branches(self):
        rng = np.random.default_rng(812)
        keys = ("layer_00_K", "layer_00_V")
        model = fixture_model(keys)
        matrices = [tuple(rng.integers(12000, 19500, (17, 16), dtype=np.uint16) for _ in keys)]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.bin"
            saved = learned_codec.save_model(model, path)
            config = {"model": str(path), "candidates": {
                "learned": {"kind": "learned", "options": {"correction": "best_entropy", "force": True}},
                "e8": {"kind": "learned_e8", "options": {"group_size": 3072, "correction": "best", "force": True}},
            }}
            encoders, decoders, states = factories(config)
            rows = {}
            for name in config["candidates"]:
                self.assertEqual(states[name], saved["serialized_bytes"])
                self.assertGreater(states[name], sum(book.nbytes for book in model.codebooks.values()))
                encoded = container.encode_cache(matrices, encoders[name], tile_tokens=16)
                self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
                diagnostics = encoded.stats["diagnostics"]
                if name == "learned":
                    self.assertEqual(diagnostics["record_modes"], {"entropy": 4})
                    self.assertEqual(diagnostics["mechanism_counts"]["stored_learned_vectors"], 68)
                else:
                    self.assertEqual(diagnostics["mode_counts"], {"1": 4})
                    self.assertEqual(diagnostics["mechanism_counts"]["e8_vectors"], 68)
                    self.assertEqual(diagnostics["mechanism_counts"]["learned_vectors"], 68)
                recovered = container.decode_cache(encoded.data, decoders[name])
                for expected_pair, actual_pair in zip(matrices, recovered):
                    for expected, actual in zip(expected_pair, actual_pair):
                        np.testing.assert_array_equal(actual, expected)
                rows[name] = {**encoded.stats, "roundtrip_exact": True, "encode_seconds": 0.0, "decode_seconds": 0.0}
            totals = aggregate([{"codecs": rows}], states, list(config["candidates"]))
            for name, row in rows.items():
                expected = 100 * (1 - (row["serialized_bytes"] + states[name]) / row["raw_payload_bytes"])
                self.assertAlmostEqual(totals[name]["savings_including_state_percent"]["1"], expected)

    def test_intentional_literal_fallback_still_charges_model(self):
        model = fixture_model(("layer_00_K", "layer_00_V"))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.bin"
            learned_codec.save_model(model, path)
            config = {"model": str(path), "candidates": {
                "literal": {"kind": "learned", "options": {"correction": "raw", "force": False}},
            }}
            encoders, decoders, states = factories(config)
            words = np.array([[15003]], dtype=np.uint16)
            encoded = encoders["literal"](words, 0, "K")
            self.assertEqual(encoded.stats["accepted_coding_mode"], "raw")
            self.assertEqual(encoded.stats["mechanism_counts"]["stored_learned_vectors"], 0)
            self.assertEqual(states["literal"], path.stat().st_size)
            np.testing.assert_array_equal(decoders["literal"](encoded.data, 0, "K"), words)


if __name__ == "__main__":
    unittest.main()

"""Actual page-byte exactness, matched controls, escapes and corruption checks."""
import copy
import json
import unittest
import zlib
from unittest import mock
import numpy as np
from qwen_kv_bench import syndrome_codec as codec


class Prepared:
    mode = "hidden"
    model_id = "a" * 64
    state_id = "b" * 64
    metadata = dict(geometry=dict(layers=2, channels=4, head_dim=4, rank=2))
    exponent_centers = np.full((2, 8), 127, dtype=np.uint8)

    def encode_features(self, record):
        return np.full((record["tokens"], 0 if self.mode == "none" else 2), 0x3f80, dtype=np.uint16)

    def predict_bits(self, layer, features, start, known_words, residue_bits):
        assert np.all((known_words & (127 ^ ((1 << residue_bits) - 1))) == 0), "omitted bits leaked"
        return np.zeros(known_words.shape + (7,), dtype=np.uint8)


def record(tokens=93, random=False):
    if random:
        words = np.random.default_rng(71).integers(0, 65536, (tokens, 8), dtype=np.uint16)
    else:
        words = np.full((tokens, 8), 0x3f81, np.uint16)
        for channel in range(8):
            for group in range(tokens // 31):
                words[group * 31 + (channel + group) % 31, channel] |= 1 << 6
        if tokens >= 31:
            # One full group needs a raw escape; other groups remain short.
            words[1, 0] |= 1 << 6
            words[2, 0] |= 1 << 6
    return dict(layer=1, start=17, tokens=tokens, head_dim=4,
                k=words[:, :4].copy(), v=words[:, 4:].copy(),
                hidden=np.full((tokens, 12), 0x3f80, np.uint16))


def target(row):
    return np.concatenate((row["k"], row["v"]), axis=1)


def rewrite(wire, **changes):
    metadata, payload = codec._parse(wire)
    metadata.update(changes)
    header = codec._canonical(metadata)
    body = header + payload
    return codec._HEADER.pack(codec._MAGIC, len(header), len(payload), zlib.crc32(body)) + body


class SyndromeCodecTests(unittest.TestCase):
    def test_every_single_error_syndrome_and_multiple_error_escapes(self):
        predicted = np.zeros((31, 128), np.uint8)
        actual = predicted.copy()
        for column in range(128):
            if column % 32:
                actual[column % 32 - 1, column] = 1
        wire, desc, stats = codec._plane_encode(actual, predicted, "syndrome", 4)
        self.assertEqual(desc["mode"], "syndrome")
        self.assertEqual(stats["retained_short_groups"], 128)
        np.testing.assert_array_equal(codec._plane_decode(wire, desc, predicted, 31, 128), actual)
        actual[0, :] = 1
        actual[1, :] = 1
        wire, desc, stats = codec._plane_encode(actual, predicted, "syndrome", 4)
        self.assertGreater(stats["multi_error_groups"], 0)
        np.testing.assert_array_equal(codec._plane_decode(wire, desc, predicted, 31, 128), actual)

    def test_actual_compressed_pages_restore_with_only_wire_and_frozen_state(self):
        prepared = Prepared()
        row = record()
        expected = target(row)
        wire, stats = codec.encode_page(row, prepared, residue_bits=2)
        self.assertTrue(stats["roundtrip_exact"])
        self.assertTrue(stats["prediction_used_for_retained_short_groups"])
        self.assertGreater(stats["planes_used_syndrome"], 0)
        self.assertGreater(stats["retained_short_groups"], 0)
        self.assertEqual(sum(stats["component_bytes"].values()), len(wire))
        self.assertEqual(stats["feature_bytes"], 93 * 2 * 2)
        self.assertEqual(codec.page_metadata(wire), dict(layer=1, start=17, tokens=93, channels=4))
        row.clear()
        other = Prepared()
        other.encode_features = lambda _: (_ for _ in ()).throw(AssertionError("decoder consulted encoder"))
        actual, timing = codec.decode_page(wire, other, return_stats=True)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(timing["raw_prerequisite_pages"], 0)
        self.assertEqual(timing["dependency_depth"], 0)
        for plane in stats["bitplanes"]:
            self.assertEqual(sum(plane["errors_per_group"]), plane["groups"])
            self.assertEqual(sum(s["bit_errors"] for s in plane["by_kind"].values()), plane["bit_errors"])
            self.assertEqual(sum(s["bit_errors"] for s in plane["by_head"].values()), plane["bit_errors"])
            self.assertEqual(sum(s["original_ones"] for s in plane["by_kind"].values()), plane["original_ones"])
            alternative = plane["alternative_layout_diagnostic_not_encoded"]
            self.assertEqual(sum(alternative["errors_per_group"]), alternative["groups"])
            self.assertIn("Diagnostic only", alternative["scope"])
            for alternative in plane["alternative_group_diagnostics_not_encoded"].values():
                self.assertEqual(sum(alternative["errors_per_group"]), alternative["groups"])

    def test_decoder_uses_transmitted_features_as_the_actual_side_information(self):
        class Conditional(Prepared):
            def predict_bits(self, layer, features, start, known_words, residue_bits):
                observed = ((features[:, 0] >> 7) & 1).astype(np.uint8)
                return np.broadcast_to(observed[:, None, None], known_words.shape + (7,)).copy()
        prepared = Conditional()
        features = np.full((93, 2), 0x3f80, np.uint16)
        features[::2] = 0x3f00
        bits = ((features[:, 0] >> 7) & 1).astype(np.uint16)
        words = np.full((93, 8), 0x3f80, np.uint16) | (bits[:, None] * 127)
        row = dict(layer=1, start=0, tokens=93, head_dim=4, k=words[:, :4], v=words[:, 4:])
        wire, stats = codec.encode_page(row, prepared, residue_bits=2, feature_words=features)
        self.assertTrue(stats["prediction_used_for_retained_short_groups"])
        np.testing.assert_array_equal(codec.decode_page(wire, Conditional()), words)
        meta, payload = codec._parse(wire)
        changed = features.copy()
        changed[:, 0] ^= 0x80
        changed_bytes = changed.astype("<u2").tobytes()
        payload = changed_bytes + payload[meta["feature_bytes"]:]
        meta["features_sha256"] = codec._sha(changed_bytes)
        header = codec._canonical(meta)
        body = header + payload
        damaged = codec._HEADER.pack(codec._MAGIC, len(header), len(payload), zlib.crc32(body)) + body
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            codec.decode_page(damaged, Conditional())

    def test_positions_and_syndromes_have_equal_actual_complete_sizes(self):
        # Includes raw-plane decisions near descriptor/byte boundaries.
        for tokens in (1, 7, 30, 31, 32, 48, 62, 63, 64, 93, 124):
            for bits in (0, 2, 4, 6, 7):
                with self.subTest(tokens=tokens, residue_bits=bits):
                    row = record(tokens)
                    a, sa = codec.encode_page(row, Prepared(), residue_bits=bits, correction="syndrome")
                    b, sb = codec.encode_page(row, Prepared(), residue_bits=bits, correction="positions")
                    self.assertEqual(len(a), len(b))
                    self.assertEqual(sa["component_bytes"], sb["component_bytes"])
                    self.assertEqual([p["retained_mode"] == "raw" for p in sa["bitplanes"]],
                                     [p["retained_mode"] == "raw" for p in sb["bitplanes"]])
                    np.testing.assert_array_equal(codec.decode_page(a, Prepared()), target(row))
                    np.testing.assert_array_equal(codec.decode_page(b, Prepared()), target(row))

    def test_all_65536_original_bf16_patterns_and_nonfinite_payloads(self):
        words = np.arange(65536, dtype=np.uint16).reshape(8192, 8)
        row = dict(layer=0, start=0, tokens=8192, head_dim=4, k=words[:, :4], v=words[:, 4:])
        # Includes all NaN payloads, infinities, signed zero and subnormals.
        for bits in (0, 4, 7):
            wire, _ = codec.encode_page(row, Prepared(), residue_bits=bits)
            np.testing.assert_array_equal(codec.decode_page(wire, Prepared()), words)

    def test_none_baseline_transmits_no_features_and_b7_never_predicts(self):
        prepared = Prepared(); prepared.mode = "none"
        prepared.predict_bits = lambda *args: (_ for _ in ()).throw(AssertionError("b7 predicted"))
        wire, stats = codec.encode_page(record(17, random=True), prepared, residue_bits=7)
        self.assertEqual(stats["feature_bytes"], 0)
        self.assertEqual(stats["bitplanes"], [])
        self.assertFalse(stats["prediction_used_for_retained_short_groups"])
        np.testing.assert_array_equal(codec.decode_page(wire, prepared), target(record(17, random=True)))

    def test_inline_features_are_transmitted_copied_and_validated(self):
        prepared = Prepared()
        prepared.encode_features = lambda _: (_ for _ in ()).throw(AssertionError("inline code not reused"))
        supplied = np.full((93, 2), 0x4000, np.uint16)
        wire, stats = codec.encode_page(record(), prepared, feature_words=supplied)
        self.assertTrue(stats["feature_words_supplied"])
        self.assertEqual(stats["feature_encode_seconds"], 0.)
        meta, payload = codec._parse(wire)
        self.assertEqual(payload[:meta["feature_bytes"]], supplied.astype("<u2").tobytes())
        supplied[:] = 0
        np.testing.assert_array_equal(codec.decode_page(wire, Prepared()), target(record()))
        for invalid in (np.zeros((93, 1), np.uint16), np.zeros((93, 2), np.float32), np.full((93, 2), 0x7fc1, np.uint16)):
            with self.assertRaises(ValueError):
                codec.encode_page(record(), prepared, feature_words=invalid)

    def test_corruption_wrong_state_geometry_and_bad_streams_rejected(self):
        wire, _ = codec.encode_page(record(), Prepared())
        for bad in (wire[:-1], wire[:12], wire[:-1] + bytes([wire[-1] ^ 1])):
            with self.assertRaises(ValueError):
                codec.decode_page(bad, Prepared())
        for key, value in (("model_id", "c" * 64), ("state_id", "d" * 64), ("mode", "kv"),
                           ("layer", 9), ("tokens", 92), ("feature_bytes", 0),
                           ("features_sha256", "0" * 64), ("words_sha256", "0" * 64),
                           ("residue_bits", True), ("planes", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                codec.decode_page(rewrite(wire, **{key: value}), Prepared())
        with self.assertRaisesRegex(ValueError, "padding"):
            codec._unpack(bytes([0x80]), 1, 1)
        with self.assertRaises(ValueError):
            codec._inflate(zlib.compress(b"x" * 1000), "zlib", 3)
        with self.assertRaises(ValueError):
            codec._inflate(zlib.compress(b"x") + b"trailing", "zlib", 1)

    def test_predictor_replay_mismatch_is_detected(self):
        wire, _ = codec.encode_page(record(), Prepared())
        bad = Prepared()
        bad.predict_bits = lambda layer,features,start,known,b: np.ones(known.shape + (7,), np.uint8)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            codec.decode_page(wire, bad)

    def test_empty_malformed_and_omitted_bit_leakage_contract(self):
        for bits in (-1, 8, True):
            with self.assertRaises(ValueError):
                codec.encode_page(record(), Prepared(), residue_bits=bits)
        for row in ({}, dict(record(), tokens=0), dict(record(), k=np.zeros((93, 4), np.float32))):
            with self.assertRaises(ValueError):
                codec.encode_page(row, Prepared())
        prepared = Prepared()
        prepared.predict_bits = lambda *args: np.full((93, 8, 7), 2, np.uint8)
        with self.assertRaisesRegex(ValueError, "binary"):
            codec.encode_page(record(), prepared)


if __name__ == "__main__":
    unittest.main()

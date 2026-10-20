"""Bit-exact compact pages, exercised entropy streams and paid side metadata."""
import copy
import unittest
import zlib

import numpy as np

from qwen_kv_bench import compact_codec as codec


class Prepared:
    model_id = "a" * 64
    state_id = "b" * 64
    metadata = dict(geometry=dict(layers=2, channels=4, head_dim=4))

    def __init__(self, **changes):
        self.spec = dict(name="test", source="hidden", side_bits=2, page_shared=False,
                         pooled=True, key_first=False, layout="lsb", base_bits=2)
        self.spec.update(changes)
        self.calls = []

    def encode_hints(self, record):
        n = 0 if not self.spec["side_bits"] else 1 if self.spec["page_shared"] else record["tokens"]
        return (np.arange(n) % (1 << self.spec["side_bits"])).astype(np.uint8)

    def predict_counts(self, layer, start, known_words, hint_ids, stage_index):
        shift, width = codec.stage_plan(self.spec)[stage_index]
        mask = codec.initial_mask(self.spec)
        for s, w in codec.stage_plan(self.spec)[:stage_index]:
            mask |= np.uint16(((1 << w) - 1) << s)
        v = known_words[:, 4:] if self.spec["key_first"] else known_words
        if np.any(v & np.uint16(0xffff ^ int(mask))):
            raise AssertionError("predictor consulted future bits")
        self.calls.append((stage_index, known_words.copy(), hint_ids.copy()))
        alphabet = 1 << width
        counts = np.full(known_words.shape + (alphabet,), codec.TOTAL // alphabet, np.int32)
        return counts


class HintPredictor(Prepared):
    def predict_counts(self, layer, start, known_words, hint_ids, stage_index):
        super().predict_counts(layer, start, known_words, hint_ids, stage_index)
        _, width = codec.stage_plan(self.spec)[stage_index]
        alphabet = 1 << width
        counts = np.ones(known_words.shape + (alphabet,), np.int32)
        hints = np.full(len(known_words), hint_ids[0], np.uint8) if self.spec["page_shared"] else hint_ids
        for t, hint in enumerate(hints):
            counts[t, :, int(hint) % alphabet] = codec.TOTAL - alphabet + 1
        return counts


def record(words=None, tokens=13):
    if words is None:
        words = np.random.default_rng(71).integers(0, 65536, (tokens, 8), dtype=np.uint16)
    return dict(layer=1, start=17, tokens=len(words), head_dim=4,
        k=words[:, :4].copy(), v=words[:, 4:].copy(), hidden=np.zeros((len(words), 12), np.uint16))


def target(row):
    return np.concatenate((row["k"], row["v"]), axis=1)


def rewrite(wire, updates=None, payload_update=None):
    meta, payload = codec._parse(wire)
    if updates:
        meta.update(updates)
    if payload_update:
        payload = payload_update(meta, payload)
    header = codec._canonical(meta)
    body = header + payload
    return codec._HEADER.pack(codec._MAGIC, len(header), len(payload), zlib.crc32(body)) + body


class CompactCodecTests(unittest.TestCase):
    def test_every_bf16_word_and_both_layouts_are_bit_exact(self):
        words = np.arange(65536, dtype=np.uint16).reshape(-1, 8)
        for layout in ("lsb", "msb"):
            with self.subTest(layout=layout):
                p = Prepared(layout=layout, source="none", side_bits=0)
                row = record(words)
                wire, stats = codec.encode_page(row, p)
                np.testing.assert_array_equal(codec.decode_page(wire, Prepared(layout=layout, source="none", side_bits=0)), words)
                self.assertEqual(sum(stats["component_bytes"].values()), len(wire))

    def test_all_hint_widths_page_sharing_and_key_first(self):
        row = record()
        for layout in ("lsb", "msb"):
            for bits in (1, 2, 4):
                for shared in (False, True):
                    for key_first in (False, True):
                        p = Prepared(layout=layout, side_bits=bits, page_shared=shared, key_first=key_first)
                        wire, stats = codec.encode_page(row, p)
                        restored, decoded = codec.decode_page(wire, p, return_stats=True)
                        np.testing.assert_array_equal(restored, target(row))
                        n = 1 if shared else row["tokens"]
                        self.assertEqual(stats["hint_bytes"], (n * bits + 7) // 8)
                        self.assertEqual(sum(stats["component_bytes"].values()), len(wire))
                        self.assertFalse(decoded["source_arrays_used"])
                        self.assertEqual(decoded["raw_prerequisite_pages"], 0)
                        self.assertEqual(codec.page_metadata(wire), dict(layer=1, start=17, tokens=13, channels=4))

    def test_actual_arithmetic_is_exercised_and_uses_only_transmitted_hints(self):
        n = 4096
        hints = np.random.default_rng(73).integers(0, 4, n, dtype=np.uint8)
        words = np.full((n, 8), 0x3f80, np.uint16)
        words |= hints[:, None].astype(np.uint16) << 2
        words |= hints[:, None].astype(np.uint16) << 4
        p = HintPredictor()
        p.encode_hints = lambda _: (_ for _ in ()).throw(AssertionError("must use supplied hints"))
        wire, stats = codec.encode_page(record(words), p, hint_ids=hints)
        self.assertTrue(stats["hint_ids_supplied"])
        self.assertTrue(any(x["retained_codec"] == "arithmetic" for x in stats["stages"]))
        other = HintPredictor()
        other.encode_hints = lambda _: (_ for _ in ()).throw(AssertionError("decoder consulted encoder"))
        restored, d = codec.decode_page(wire, other, return_stats=True)
        self.assertGreater(d["arithmetic_stages"], 0)
        np.testing.assert_array_equal(restored, words)
        with self.assertRaisesRegex(ValueError, "counts"):
            codec.decode_page(wire, Prepared())

    def test_stage_raw_fallback_skips_predictor_and_preserves_signed_zero_nan_payloads(self):
        words = np.array([0, 0x8000, 1, 0x807f, 0x3f80, 0xbf80, 0x7f80, 0xff80,
            0x7fc1, 0xffc2, 0x7f7f, 0xff7f, 0x0080, 0x8080, 0x1234, 0xabcd], np.uint16).reshape(2, 8)
        p = Prepared()
        wire, stats = codec.encode_page(record(words), p)
        self.assertTrue(all(x["retained_codec"] == "packed" for x in stats["stages"]))
        other = Prepared()
        other.predict_counts = lambda *a: (_ for _ in ()).throw(AssertionError("raw fallback invoked neural predictor"))
        np.testing.assert_array_equal(codec.decode_page(wire, other), words)

    def test_packing_padding_and_roundtrip_all_widths(self):
        for width in (1, 2, 3, 4, 7, 11, 16):
            values = (np.arange(13, dtype=np.uint32) % (1 << width)).astype(np.uint16)
            data = codec._pack(values, width)
            np.testing.assert_array_equal(codec._unpack(data, len(values), width), values)
        with self.assertRaisesRegex(ValueError, "padding"):
            codec._unpack(b"\xff", 1, 2)
        with self.assertRaises(ValueError):
            codec._pack(np.array([-1]), 2)

    def test_invalid_predictions_hints_and_shapes_are_rejected(self):
        row = record()
        for bad in (np.zeros(13, np.uint16), np.zeros(12, np.uint8), np.full(13, 4, np.uint8)):
            with self.assertRaises(ValueError):
                codec.encode_page(row, Prepared(), hint_ids=bad)
        p = Prepared()
        p.predict_counts = lambda *a: np.zeros((13, 8, 4), np.int32)
        with self.assertRaisesRegex(ValueError, "counts"):
            codec.encode_page(row, p)
        with self.assertRaises(ValueError):
            codec.encode_page(dict(row, k=np.zeros((13, 4), np.float32)), Prepared())

    def test_integrity_and_schedule_checks(self):
        p = Prepared()
        wire, _ = codec.encode_page(record(), p)
        for damaged in (wire[:-1], wire + b"x", wire[:30] + bytes([wire[30] ^ 1]) + wire[31:]):
            with self.assertRaises(ValueError):
                codec.decode_page(damaged, p)
        for changes in (dict(model_id="f" * 64), dict(state_id="f" * 64), dict(tokens=0),
                        dict(hint_count=99), dict(hint_bytes=99), dict(words_sha256="f" * 64)):
            with self.assertRaises(ValueError):
                codec.decode_page(rewrite(wire, changes), p)
        meta = codec.inspect_page(wire)
        stages = copy.deepcopy(meta["stages"]); stages[0]["shift"] += 1
        with self.assertRaises(ValueError):
            codec.decode_page(rewrite(wire, dict(stages=stages)), p)

    def test_zlib_bombs_trailing_streams_and_length_mismatch_rejected(self):
        data = zlib.compress(b"x" * 10000)
        desc = dict(storage="zlib", raw_bytes=1, payload_bytes=len(data))
        with self.assertRaises(ValueError):
            codec._open_stored(data, desc, 1)
        data = zlib.compress(b"x") + b"extra"
        desc = dict(storage="zlib", raw_bytes=1, payload_bytes=len(data))
        with self.assertRaises(ValueError):
            codec._open_stored(data, desc, 1)


if __name__ == "__main__":
    unittest.main()

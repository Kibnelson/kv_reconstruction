"""Independent page restoration, complete framing costs and integrity checks."""
import copy
import json
import struct
import unittest
import zlib

import numpy as np

from qwen_kv_bench import side_codec as codec


class Prepared:
    """A deterministic stand-in; these tests exercise the actual entropy codec."""
    mode = "hidden"
    state_id = "a" * 64
    model_id = "b" * 64
    metadata = dict(geometry=dict(layers=3, channels=4, head_dim=4, rank=2, hidden_size=8, width=16))

    def encode_features(self, record):
        rank = self.metadata["geometry"]["rank"]
        return np.full((record["tokens"], rank), 0x3f80, dtype=np.uint16)

    def predict(self, layer, features, start=0):
        shape = (len(features), self.metadata["geometry"]["channels"] * 2)
        # Distinct scales avoid the separate shared-CDF optimization and test
        # actual per-value probability reproduction from the stored features.
        mu = np.full(shape, layer * .01 + start * .001, dtype=np.float32)
        if features.shape[1]:
            values = (features[:, 0].astype(np.uint32) << 16).view(np.float32)
            mu += values[:, None] * .1
        logs = np.broadcast_to(np.arange(shape[1], dtype=np.float32)[None] * .01, shape).copy()
        return mu, logs


def record(layer=0, start=0):
    special = np.array([0, 0x8000, 1, 0x807f, 0x3f80, 0xbf80, 0x7f80, 0xff80,
        0x7fc1, 0xffc2, 0x7f7f, 0xff7f], dtype=np.uint16).reshape(3, 4)
    return dict(layer=layer, start=start, tokens=3, head_dim=4,
        k=special.copy(), v=special[:, ::-1].copy(), hidden=np.zeros((3, 8), dtype=np.uint16))


def rewrite(wire, **changes):
    meta, payload = codec._parse(wire)
    meta.update(changes)
    header = codec._canonical(meta)
    body = header + payload
    return codec._HEADER.pack(codec._MAGIC, len(header), len(payload), zlib.crc32(body)) + body


class SideCodecTests(unittest.TestCase):
    def test_canonical_kv_mode_has_the_same_independent_packet_contract(self):
        prepared = Prepared()
        prepared.mode = "kv_canonical"
        wire, _ = codec.encode_page(record(2, 5), prepared)
        self.assertEqual(codec.inspect_page(wire)["mode"], "kv_canonical")
        np.testing.assert_array_equal(codec.decode_page(wire, prepared),
            np.concatenate((record(2, 5)["k"], record(2, 5)["v"]), axis=1))
        with self.assertRaises(ValueError):
            codec.decode_page(wire, Prepared())

    def test_inside_forward_feature_override_is_transmitted_and_used(self):
        prepared = Prepared()
        prepared.encode_features = lambda *args: (_ for _ in ()).throw(AssertionError("must use callback code"))
        supplied = np.array([[0x3f00, 0x4000], [0x3f80, 0x4040], [0x4000, 0x4080]], dtype=np.uint16)
        wire, stats = codec.encode_page(record(), prepared, feature_words=supplied)
        meta, payload = codec._parse(wire)
        self.assertEqual(payload[:meta["feature_bytes"]], supplied.astype("<u2").tobytes())
        self.assertTrue(stats["feature_words_supplied"])
        self.assertEqual(stats["feature_encode_seconds"], 0.)
        supplied[:] = 0
        np.testing.assert_array_equal(codec.decode_page(wire, Prepared()),
            np.concatenate((record()["k"], record()["v"]), axis=1))
        for invalid in (np.zeros((3, 1), np.uint16), np.zeros((3, 2), np.float32),
                        np.full((3, 2), 0x7f80, np.uint16)):
            with self.assertRaises(ValueError):
                codec.encode_page(record(), prepared, feature_words=invalid)

    def test_any_order_restores_all_original_words_without_sources(self):
        prepared = Prepared()
        sources = [record(0, 0), record(2, 17), record(1, 6)]
        expected = [np.concatenate((r["k"], r["v"]), axis=1) for r in sources]
        pages = [codec.encode_page(r, prepared)[0] for r in sources]
        sources.clear()
        for i in (2, 0, 1):
            restored = codec.decode_page(pages[i], Prepared())
            self.assertEqual(restored.dtype, np.uint16)
            np.testing.assert_array_equal(restored, expected[i])
        self.assertEqual(codec.page_metadata(pages[1]), dict(layer=2, start=17, tokens=3, channels=4))

    def test_none_retains_no_code_and_all_transmitted_bytes_are_charged(self):
        prepared = Prepared()
        prepared.mode = "none"
        prepared.metadata = copy.deepcopy(prepared.metadata)
        prepared.metadata["geometry"]["rank"] = 0
        wire, stats = codec.encode_page(record(), prepared)
        self.assertEqual(stats["feature_bytes"], 0)
        self.assertEqual(stats["raw_bytes"], 3 * 8 * 2)
        self.assertEqual(stats["packet_bytes"], len(wire))
        self.assertEqual(stats["packet_bytes"], stats["feature_bytes"] + stats["entropy_payload_bytes"] + stats["framing_bytes"])
        self.assertGreater(stats["ideal_bits"], 0)
        np.testing.assert_array_equal(codec.decode_page(wire, prepared), np.concatenate((record()["k"], record()["v"]), axis=1))

    def test_wrong_model_runtime_geometry_and_corruption_rejected(self):
        wire, _ = codec.encode_page(record(), Prepared())
        for key, value in (("model_id", "c" * 64), ("state_id", "d" * 64), ("mode", "kv"),
                           ("layer", 3), ("tokens", 4), ("feature_bytes", 0),
                           ("features_sha256", "0" * 64), ("words_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                codec.decode_page(rewrite(wire, **{key: value}), Prepared())
        changed = bytearray(wire)
        changed[-1] ^= 1
        for bad in (changed, wire[:-1], b"", wire + b"extra"):
            with self.assertRaises(ValueError):
                codec.decode_page(bad, Prepared())
        prepared = Prepared()
        prepared.state_id = "f" * 64
        with self.assertRaises(ValueError):
            codec.decode_page(wire, prepared)

    def test_invalid_input_and_predictor_shapes_fail(self):
        for change in (dict(layer=True), dict(start=-1), dict(head_dim=2), dict(tokens=0),
                       dict(k=np.zeros((3, 4), dtype=np.float32)), dict(v=np.zeros((2, 4), dtype=np.uint16))):
            bad = record()
            bad.update(change)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                codec.encode_page(bad, Prepared())
        prepared = Prepared()
        prepared.predict = lambda *args: (np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32))
        with self.assertRaises(ValueError):
            codec.encode_page(record(), prepared)
        prepared = Prepared()
        prepared.encode_features = lambda *args: np.full((3, 2), 0x7f80, dtype=np.uint16)
        with self.assertRaises(ValueError):
            codec.encode_page(record(), prepared)


if __name__ == "__main__":
    unittest.main()

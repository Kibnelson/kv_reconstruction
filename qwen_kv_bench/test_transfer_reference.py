"""Correctness and source-equivalence checks for the supplied transfer codec."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qwen_kv_bench import transfer_reference as reference


def _bits(value):
    value = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((value + ((value >> 16) & 1) + 0x7fff) >> 16).astype(np.uint16)


def capture(seed=1, tokens=64):
    rng = np.random.default_rng(seed)
    # Three layers, K/V width eight each, two heads and complete RoPE pairs.
    previous = rng.normal(size=(tokens, 16))
    rows = []
    for layer in range(3):
        current = .75 * previous + .1 * rng.normal(size=previous.shape)
        rows.append(_bits(current)); previous = current
    value = np.stack(rows)
    return dict(K_bits=value[:, :, :8], V_bits=value[:, :, 8:], Hk=2, hd=4, base=100000.0)


@unittest.skipUnless(importlib.util.find_spec("constriction"), "constriction is optional")
class TransferReferenceTests(unittest.TestCase):
    def setUp(self):
        self.train = [capture(1), capture(2)]
        self.scale = capture(3)
        self.test = capture(4, 16)

    def test_four_predictor_modes_round_trip(self):
        for mode in ("mean", "previous", "aqua", "block"):
            with self.subTest(mode=mode):
                model = reference.fit(self.train, self.scale, mode=mode, block_size=4, max_positions=128)
                packet, stats = reference.encode(model, reference.words(self.test), start=16)
                actual, decoded_stats = reference.decode(model, packet)
                np.testing.assert_array_equal(actual, reference.words(self.test))
                self.assertTrue(decoded_stats["exact_checksum_verified"])
                self.assertEqual(stats["packet_bytes"], stats["framing_bytes"] + stats["payload_bytes"])
                self.assertEqual(stats["metadata_bytes"], 0)
                self.assertIsNone(stats["fallback"])

    def test_metadata_is_paid_and_decoded_from_packet(self):
        rng = np.random.default_rng(7)
        train_meta = [rng.integers(-127, 128, size=(3, 64, 2), dtype=np.int8) for _ in self.train]
        scale_meta = rng.integers(-127, 128, size=(3, 64, 2), dtype=np.int8)
        model = reference.fit(self.train, self.scale, block_size=4, metadata_train=train_meta, metadata_scale=scale_meta)
        metadata = rng.integers(-127, 128, size=(3, 16, 2), dtype=np.int8)
        packet, stats = reference.encode(model, reference.words(self.test), metadata=metadata)
        self.assertEqual(stats["metadata_bytes"], metadata.nbytes)
        metadata[:] = 0  # Packet is the decoder's only metadata input.
        decoded, _ = reference.decode(model, packet)
        np.testing.assert_array_equal(decoded, reference.words(self.test))
        with self.assertRaisesRegex(ValueError, "metadata"):
            reference.encode(model, reference.words(self.test))

    def test_frozen_state_round_trip_and_mismatch(self):
        model = reference.fit(self.train, self.scale, block_size=4)
        packet, _ = reference.encode(model, reference.words(self.test))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.npz"
            state_stats = reference.save_model(model, path)
            loaded = reference.load_model(path)
            self.assertEqual(state_stats["serialized_bytes"], path.stat().st_size)
            self.assertEqual(loaded.state_sha256, model.state_sha256)
            actual, _ = reference.decode(loaded, packet)
            np.testing.assert_array_equal(actual, reference.words(self.test))
        other = reference.fit([capture(19)], self.scale, block_size=4)
        with self.assertRaisesRegex(ValueError, "different frozen state"):
            reference.decode(other, packet)

    def test_packet_tamper_and_unsupported_words(self):
        model = reference.fit(self.train, self.scale, block_size=4)
        source = reference.words(self.test)
        packet, _ = reference.encode(model, source)
        changed = bytearray(packet); changed[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            reference.decode(model, bytes(changed))
        source[0, 0, :4] = [0x7f80, 0x7fc1, 0xff80, 0x8000]
        packet, stats = reference.encode(model, source)
        self.assertIsNotNone(stats["fallback"])
        self.assertGreater(stats["raw_fallback_bytes"], 0)
        actual, _ = reference.decode(model, packet)
        np.testing.assert_array_equal(actual, source)

    def test_original_block_predictor_and_stream_equivalence(self):
        original = Path(__file__).resolve().parents[2] / "team_transfer_original" / "combined.py"
        if not original.is_file():
            original = Path(__file__).resolve().parents[1] / "experiments" / "team_transfer_original" / "combined.py"
        if not original.is_file():
            self.skipTest("unmodified supplied combined.py unavailable")
        import constriction
        from qwen_kv_bench import transfer_coset
        model = reference.fit(self.train, self.scale, block_size=4)
        tree = ast.parse(original.read_text())
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
        ns = dict(np=np, constriction=constriction, FIT=self.train, TAB=self.scale,
                  L=3, N=64, C=8, NV=16, Hk=2, hd=4, base=100000.0, BSZ=4, NBLK=4,
                  ang=np.arange(64, dtype=np.float64)[:, None] * 100000.0 ** (-np.arange(0, 4, 2, dtype=np.float64) / 4)[None, :],
                  COSI=model.cosine, SINI=model.sine, H14=1 << 14,
                  BD=transfer_coset.BD, GB=transfer_coset.BD[::128], TT=32.0, TR=256, NTAB=16385,
                  F_TAB=model.f_tab, ONE16=1 << 16, CAT=constriction.stream.model.Categorical(perfect=False))
        exec(compile(functions, str(original), "exec"), ns)
        for layer in range(3):
            expected = ns["fit_layer"](layer)
            for index, block in enumerate(model.predictors[layer]):
                selection = slice(index * 4, (index + 1) * 4)
                np.testing.assert_array_equal(block["A"], expected["A"][selection, :len(block["columns"])])
                np.testing.assert_array_equal(block["b"], expected["b"][selection])
                np.testing.assert_array_equal(block["inv_s"], expected["inv_s"][selection])
        # Same predictions and original direct probability coding emit identical bytes.
        mean = np.array([0.0, .5, -1.0, 2.0])
        inverse = np.array([1.0, 2.0, .5, 4.0])
        target = _bits(np.array([.1, .4, -.9, 1.8]))
        left = constriction.stream.queue.RangeEncoder(); right = constriction.stream.queue.RangeEncoder()
        ns["code_block"](left, mean, inverse, target)
        transfer_coset.code_block(right, mean, inverse, target, f_tab=model.f_tab)
        np.testing.assert_array_equal(left.get_compressed(), right.get_compressed())


if __name__ == "__main__":
    unittest.main()

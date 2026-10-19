"""Exactness/provenance tests; CPU execution does not establish GPU speed."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import gpu_ops as g
from qwen_kv_bench import residual_dictionary as r


class ResidualDictionaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def caches(self):
        rng = np.random.default_rng(43)
        return [{f"layer_00_{kind}": rng.integers(15000, 17000, size=(17, 16), dtype=np.uint16)
                 for kind in ("K", "V")} for _ in range(3)]

    def fit(self, output, layout="channel", **options):
        return r.train(self.caches(), output, tile_tokens=8, layout=layout, n_centroids=4,
            max_samples_per_key=64, iterations=2,
            training_metadata=[dict(split="train", name=f"fixture_{i}") for i in range(3)],
            model_identity=dict(model="tiny", num_hidden_layers=1, head_dim=8, num_key_value_heads=2), **options)

    def state(self, layout="channel", device="cpu", count=4):
        # Public make_state convention uses shifted signed residual books.
        book = np.zeros((count, 8), dtype=np.int16)
        book[1:, :] = np.arange(1, count)[:, None] * np.arange(-4, 4)[None, :]
        return g.make_state({"layer_00_K": (book.astype(np.int32) + 16384).astype(np.uint16)},
            device=device, state_id="test_residual_model")

    def assert_exact(self, words, state, **options):
        packet = r.encode(words, state, key="layer_00_K", **options)
        restored = r.decode(packet, state)
        self.assertTrue(torch.equal(words.to(torch.int32), restored.to(torch.int32)))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))
        return packet

    def test_training_freeze_deterministic_reload_and_complete_state_charge(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = self.fit(a), self.fit(b)
            self.assertEqual(first.model_id, second.model_id)
            self.assertEqual(Path(first.path).read_bytes(), Path(second.path).read_bytes())
            self.assertEqual(first.state_bytes, Path(first.path).stat().st_size)
            self.assertGreater(first.state_bytes, sum(book.nbytes for book in first.codebooks.values()))
            self.assertTrue(all(book.dtype == np.int16 and not book.flags.writeable for book in first.codebooks.values()))
            self.assertEqual(first.metadata["configuration"]["fit_objective"],
                "circular zigzag group-width assignment; coordinate median L1 updates; fixed zero prototype")
            loaded = r.load(first.path, expected_model_id=first.model_id,
                expected_model_identity=first.metadata["source_model_identity"], expected_training_hashes=first.metadata["training_hashes"])
            self.assertEqual(first.model_id, loaded.model_id)
            state = r.make_state(loaded)
            self.assertEqual(state.state_id, loaded.model_id)
            self.assertEqual(state.physical_nbytes, sum(book.size * 4 for book in first.codebooks.values()))
            packet = self.assert_exact(torch.from_numpy(self.caches()[0]["layer_00_K"][:8].copy()), state)
            self.assertEqual(packet.metadata["dictionary_training_config"]["tile_tokens"], 8)
            with self.assertRaises(ValueError):
                r.load(first.path, expected_model_id="wrong")
            with self.assertRaises(ValueError):
                r.load(first.path, expected_model_identity="wrong")
            with self.assertRaises(ValueError):
                r.load(first.path, expected_training_hashes=["wrong"] * 3)

    def test_reject_dev_test_missing_training_architecture_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            caches = self.caches()
            for split in ("dev", "test"):
                with self.assertRaises(ValueError):
                    r.train(caches, directory, tile_tokens=8, n_centroids=4,
                        training_metadata=[dict(split="train"), dict(split=split), dict(split="train")], model_identity="tiny")
                wrappers = [{"cache": cache, "split": split} for cache in caches]
                with self.assertRaises(ValueError):
                    r.train(wrappers, directory, tile_tokens=8, n_centroids=4, model_identity="tiny")
            for model in (None, {"num_hidden_layers": 2}, {"head_dim": 8, "num_key_value_heads": 3}):
                with self.assertRaises(ValueError):
                    r.train(caches, directory, tile_tokens=8, n_centroids=4, model_identity=model)
            with self.assertRaises(ValueError):
                r.train(caches[:2], directory, tile_tokens=8, model_identity="tiny")
            with self.assertRaises(ValueError):
                r.train(caches, directory, tile_tokens=0, model_identity="tiny")
            with self.assertRaises(ValueError):
                r.train(caches, directory, tile_tokens=8, n_centroids=7, model_identity="tiny")

    def test_crc_metadata_mutation_and_before_after_training_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.fit(directory)
            data = bytearray(Path(model.path).read_bytes())
            data[-1] ^= 1
            damaged = Path(directory) / "damaged.bin"
            damaged.write_bytes(data)
            with self.assertRaises(ValueError):
                r.load(damaged)
            modified = r.ResidualModel(model.codebooks, copy.deepcopy(model.metadata))
            modified.metadata["configuration"]["layout"] = "time"
            self.assertNotEqual(modified.model_id, model.model_id)
            modified.metadata["training_cases"][0]["split"] = "test"
            with self.assertRaises(ValueError):
                r.model_bytes(modified)
            caches = self.caches()
            with self.assertRaises(ValueError):
                r.train(caches, directory, tile_tokens=8, n_centroids=4, model_identity="tiny",
                    training_metadata=[dict(raw_sha256="0" * 64)] * 3)
            original_fit = r._fit
            # Two keys call the fitter twice; mutate a different position on
            # each call so the before/after digest cannot accidentally cancel.
            mutated_calls = [0]
            def mutate_once(*args, **kwargs):
                result = original_fit(*args, **kwargs)
                if mutated_calls[0] == 0:
                    caches[0]["layer_00_K"][0, 0] ^= np.uint16(1)
                mutated_calls[0] += 1
                return result
            with mock.patch.object(r, "_fit", side_effect=mutate_once):
                with self.assertRaises(AssertionError):
                    r.train(caches, directory, tile_tokens=8, n_centroids=4,
                        iterations=1, max_samples_per_key=32, model_identity="tiny")

    def test_numpy_affine_residual_and_actual_bitcost_assignment_oracle(self):
        rng = np.random.default_rng(91)
        for layout in ("channel", "time"):
            source = rng.integers(0, 65536, size=(5, 9), dtype=np.uint16)
            magnitude, prediction = r._np_affine(source)
            np.testing.assert_array_equal(prediction[0], magnitude[0])
            np.testing.assert_array_equal(prediction[-1], magnitude[-1])
            flat = r._arrange(r._circular(magnitude - prediction), layout)
            vectors = np.pad(flat, (0, (-len(flat)) % 8)).reshape(-1, 8)
            state = self.state()
            book = state.codebooks["layer_00_K"].numpy().astype(np.int64) - 16384
            packet = self.assert_exact(torch.from_numpy(source), state, layout=layout)
            ids = g._unpack_fixed(packet.fields["residual_ids"], len(vectors), 2).numpy()
            np.testing.assert_array_equal(ids, r._assign_np(vectors, book))
            error = r._circular(vectors - book[ids]).reshape(-1)[:source.size]
            encoded = g._restore(packet.fields, packet.metadata["descriptions"], "prefill_correction").numpy()
            np.testing.assert_array_equal(encoded, error)
            widths = g._restore_widths(packet.fields, packet.metadata["descriptions"], "prefill_correction").numpy()
            zz = np.where(error >= 0, 2 * error, -2 * error - 1)
            expected = np.pad(zz, (0, (-len(zz)) % 8)).reshape(-1, 8).max(axis=1)
            np.testing.assert_array_equal(widths, np.array([int(x).bit_length() for x in expected]))

    def test_all_65536_patterns_and_layout_tails_empty_and_single_centroid(self):
        state = self.state()
        full_words = torch.arange(65536, dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for layout in ("channel", "time"):
            self.assert_exact(full_words, state, layout=layout)
            for shape in ((1, 9), (9, 1), (3, 3), (0, 7), (3, 0), (0, 0)):
                tail_words = (torch.arange(np.prod(shape), dtype=torch.int64) * 7919).to(torch.uint16).reshape(shape)
                self.assert_exact(tail_words, state, layout=layout)
                self.assert_exact(tail_words, self.state(count=1), layout=layout)

    def test_state_model_binding_and_wire_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.fit(directory, layout="time")
            state = r.make_state(model)
            words = torch.from_numpy(self.caches()[0]["layer_00_K"][:8].copy())
            packet = self.assert_exact(words, state, layout="time")
            wire = g.packet_to_cpu(packet)
            restored = g.packet_from_cpu_parts(wire.header.clone(), {name: value.clone() for name, value in wire.fields.items()})
            self.assertTrue(torch.equal(r.decode(restored, state).to(torch.int32), words.to(torch.int32)))
            with self.assertRaises(ValueError):
                r.decode(packet, self.state())
            with self.assertRaises(ValueError):
                r.encode(words, state, key="layer_00_K", layout="channel")
            with self.assertRaises(ValueError):
                r.encode(torch.cat((words, words[:1])), state, key="layer_00_K", layout="time")
            packet.metadata["model_id"] = "altered"
            with self.assertRaises(ValueError):
                r.decode(packet, state)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; no GPU performance claim")
    def test_cuda_exact_layout_and_wire_transfer(self):
        words = torch.arange(65536, device="cuda", dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        for layout in ("channel", "time"):
            state = self.state(device="cuda")
            packet = self.assert_exact(words, state, layout=layout)
            wire = g.packet_to_cpu(packet)
            transferred = g.packet_to_device(g.packet_from_cpu_parts(wire.header, wire.fields), "cuda")
            self.assertTrue(torch.equal(r.decode(transferred, state).to(torch.int32), words.to(torch.int32)))


if __name__ == "__main__":
    unittest.main()

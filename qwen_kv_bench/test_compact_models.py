"""TRAIN isolation, paid discrete hints and unknown-field boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from qwen_kv_bench import compact_models as models, producer_models as producer


def words(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def captures(seed=17, tokens=24):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        rows = []
        for layer in range(2):
            h = rng.standard_normal((tokens, 12)).astype(np.float32)
            k = words(h[:, :8] * .7)
            v = words(h[:, 2:10] + .2)
            rows.append(dict(layer=layer, start=0, tokens=tokens, head_dim=4,
                k=k, v=v, hidden=words(h), split="train"))
        out.append(rows)
    return out


def fit(path, data=None, specs=None, steps=8):
    data = captures() if data is None else data
    return models.fit(data, path, training_metadata=[dict(split="train", name=str(i)) for i in range(len(data))],
        model_identity=dict(model="compact-synthetic"), head_dim=4, steps=steps, batch_tokens=12,
        channels_per_step=8, width=12, device="cpu", seed=17, specs=specs)


class CompactModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.data = captures()
        cls.fitted = fit(Path(cls.directory.name) / "models", cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup(); torch.set_num_threads(cls.threads)

    def test_complete_artifact_reload_and_actual_state_bytes(self):
        for name, m in self.fitted.items():
            self.assertEqual(m.state_bytes, sum(Path(p).stat().st_size for p in m.artifact_paths))
            self.assertNotIn("losses_bits", m.metadata["training"])
            self.assertEqual(m.model_id, hashlib.sha256(Path(m.path).read_bytes()).hexdigest())
            self.assertEqual(m.metadata["training_hashes"], [producer._source_hash(r) for r in self.data])
            q = models.load(m.path, expected_model_identity=m.metadata["model_identity"],
                expected_model_id=m.model_id, expected_training_hashes=m.metadata["training_hashes"])
            p, r = m.prepare(), q.prepare()
            record = self.data[0][0]
            hint = p.encode_hints(record)
            target = np.concatenate((record["k"], record["v"]), axis=1)
            for stage in (0, 1):
                np.testing.assert_array_equal(p.predict_counts(0, 0, target, hint, stage),
                    r.predict_counts(0, 0, target, hint, stage))
            self.assertGreaterEqual(p.state_bytes, p.physical_nbytes)
            self.assertTrue(all(not a.flags.writeable for a in q.params.values()))

    def test_packed_hint_budget_shapes_and_diagnostics(self):
        self.assertEqual(len({m.metadata["training"]["data_sampling_sha256"] for m in self.fitted.values()}), 1)
        for name, m in self.fitted.items():
            p = m.prepare(); hint = p.encode_hints(self.data[0][0])
            expected = 0 if m.spec["source"] == "none" else (1 if m.spec["page_shared"] else 24)
            self.assertEqual(hint.dtype, np.uint8); self.assertEqual(hint.shape, (expected,))
            self.assertEqual(sum(p.last_hint_diagnostics["histogram"]), expected)
            if hint.size:
                self.assertLess(int(hint.max()), 1 << m.spec["side_bits"])
            self.assertNotIn("source_mu", m.params); self.assertNotIn("source_sd", m.params)
            if m.spec["source"] != "none":
                self.assertEqual(m.params["projection_mu"].shape, (2, 1))
                self.assertEqual(m.params["projection_sd"].shape, (2, 1))
            self.assertEqual(m.metadata["training"]["eligible_channel_range"],
                [8 if m.spec["key_first"] else 0, 16])

    def test_unknown_words_are_remasked_at_each_stage(self):
        rng = np.random.default_rng(4)
        record = self.data[0][0]
        target = np.concatenate((record["k"], record["v"]), axis=1)
        for name in ("lsb_pool_hidden2", "lsb_kfirst_hidden2", "msb_pool_hidden2"):
            p = self.fitted[name].prepare(); hint = p.encode_hints(record)
            for stage in (0, 1):
                mask = models.stage_mask(p.spec, stage)
                unknown = np.uint16(0xffff ^ mask)
                changed = (target & np.uint16(mask)) | (rng.integers(0, 65536, target.shape, dtype=np.uint16) & unknown)
                if p.spec["key_first"]:
                    changed[:, :8] = target[:, :8]
                a = p.predict_counts(0, 0, target, hint, stage)
                b = p.predict_counts(0, 0, changed, hint, stage)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.dtype, np.int32)
                self.assertEqual(a.shape, (24, 16, 1 << models.stage_plan(p.spec)[stage][1]))
                self.assertTrue(np.all(a > 0)); self.assertTrue(np.all(a.sum(axis=-1) == 65536))
        self.assertNotEqual(models.initial_mask(self.fitted["lsb_pool_none"].spec),
            models.initial_mask(self.fitted["msb_pool_none"].spec))

    def test_no_source_or_original_values_needed_for_prediction(self):
        p = self.fitted["lsb_pool_hidden2"].prepare()
        record = copy.deepcopy(self.data[0][0]); hint = p.encode_hints(record)
        known = np.concatenate((record["k"], record["v"]), axis=1)
        expected = p.predict_counts(0, 0, known, hint, 0)
        record.clear()
        for key in ("encoder_W", "encoder_b", "projection_mu", "projection_sd", "innovation_W"):
            p.params.pop(key)
        np.testing.assert_array_equal(expected, p.predict_counts(0, 0, known, hint, 0))

    def test_pooled_known_key_context_reaches_value_predictions(self):
        target = np.concatenate((self.data[0][0]["k"], self.data[0][0]["v"]), axis=1)
        changed = target.copy(); changed[:, :8] ^= np.uint16(0x8000)
        hints = np.empty(0, np.uint8)
        local = self.fitted["lsb_local_none"].prepare()
        a = local.predict_counts(0, 0, target, hints, 0)
        b = local.predict_counts(0, 0, changed, hints, 0)
        np.testing.assert_array_equal(a[:, 8:], b[:, 8:])
        pooled = self.fitted["lsb_pool_none"].prepare()
        a = pooled.predict_counts(0, 0, target, hints, 0)
        b = pooled.predict_counts(0, 0, changed, hints, 0)
        self.assertTrue(np.any(a[:, 8:] != b[:, 8:]))

    def test_nonfinite_integer_fields_and_bad_hint_validation(self):
        p = self.fitted["msb_pool_none"].prepare()
        patterns = np.array([0, 0x8000, 1, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffff], np.uint16)
        target = np.tile(patterns, (2, 2))
        counts = p.predict_counts(0, 0, target, np.empty(0, np.uint8), 0)
        self.assertTrue(np.all(counts > 0))
        with self.assertRaisesRegex(ValueError, "hint"):
            p.predict_counts(0, 0, target, np.array([1], np.uint8), 0)
        h = self.fitted["lsb_pool_hidden1"].prepare()
        with self.assertRaisesRegex(ValueError, "hint"):
            h.predict_counts(0, 0, target, np.full(2, 2, np.uint8), 0)

    def test_train_provenance_and_corrupt_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = copy.deepcopy(self.data); bad[0][0]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                fit(Path(directory) / "bad", bad, specs=[models.default_specs()[0]], steps=1)
            bad = copy.deepcopy(self.data); bad[0].append(copy.deepcopy(bad[0][0]))
            with self.assertRaisesRegex(ValueError, "overlap"):
                fit(Path(directory) / "overlap", bad, specs=[models.default_specs()[0]], steps=1)
            m = self.fitted["lsb_pool_hidden2"]
            manifest = Path(directory) / Path(m.path).name
            manifest.write_bytes(Path(m.path).read_bytes())
            shard = Path(directory) / Path(m.artifact_paths[1]).name
            wire = Path(m.artifact_paths[1]).read_bytes(); shard.write_bytes(wire[:-1] + bytes([wire[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA"):
                models.load(manifest)
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                models.load(m.path, expected_training_hashes=["wrong"])

    def test_real_training_learns_predictable_joint_symbols(self):
        data = captures()
        for rows in data:
            for r in rows:
                r["k"].fill(127 << 7); r["v"].fill(127 << 7)
        with tempfile.TemporaryDirectory() as directory:
            m = fit(directory, data, [models.default_specs()[1]], steps=120)["lsb_pool_none"]
            diagnostic = m.metadata["training"]["diagnostics"]
            log_wire = (Path(m.path).parent / diagnostic["file"]).read_bytes()
            self.assertEqual(len(log_wire), diagnostic["bytes"])
            self.assertEqual(hashlib.sha256(log_wire).hexdigest(), diagnostic["sha256"])
            training = json.loads(log_wire)["training"]
            losses = training["losses_bits"]
            self.assertLess(np.mean(losses[-8:]), .15 * np.mean(losses[:8]))
            self.assertEqual(m.metadata["training"]["updates"], 120)
            for layer in training["exposure_by_layer_stage"].values():
                self.assertTrue(all(row["channels_without_examples"] == 0 for row in layer.values()))


if __name__ == "__main__":
    unittest.main()

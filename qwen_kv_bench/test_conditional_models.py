"""Provenance, complete state and actual producer-feature predictor contracts."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import conditional_models as c
from qwen_kv_bench import producer_models as p


def sources():
    rng = np.random.default_rng(271)
    result = []
    for capture in range(3):
        records = []
        for layer in (0, 1):
            hidden = rng.standard_normal((13, 12)).astype(np.float32)
            hidden_words = c._bf16_words(hidden)
            actual_hidden = c._float(hidden_words)
            weights = np.arange(12 * 12, dtype=np.float32).reshape(12, 12) / 97 - .7
            joint = actual_hidden @ weights + np.float32(layer + .25)
            k_pre_rope = c._bf16_words(joint[:, :6])
            v = c._bf16_words(joint[:, 6:])
            records.append(dict(layer=layer, start=0, tokens=13, head_dim=2,
                hidden=hidden_words, k_pre_rope=k_pre_rope, v=v,
                k=c._bf16_words(-c._float(k_pre_rope)),
                k_proj=c._bf16_words(c._float(k_pre_rope) * np.float32(1.75)),
                unused=np.array([capture, layer], dtype=np.int32)))
        result.append(records)
    return result


def fit(path, records=None):
    return c.fit(sources() if records is None else records, path, rank=2,
        training_metadata=[dict(split="train", name=f"train_{i}") for i in range(3)],
        model_identity=dict(model="conditional-tiny", head_dim=2, hidden_size=12), seed=71)


class ConditionalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.records = sources()
        cls.before = [p._source_hash(rows) for rows in cls.records]
        cls.models = fit(cls.directory.name, cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_complete_canonical_state_readonly_and_provenance(self):
        expected = {"conditional_entropy", "conditional_mean", "conditional_anchor_r8",
                    "conditional_anchor_r32", "conditional_latent_r8", "conditional_latent_r32"}
        self.assertTrue(expected.issubset(self.models))
        self.assertEqual(self.before, [p._source_hash(rows) for rows in self.records])
        for name in expected:
            model = self.models[name]
            raw = Path(model.path).read_bytes()
            self.assertEqual(model.state_bytes, len(raw))
            self.assertEqual(model.model_id, hashlib.sha256(raw).hexdigest())
            self.assertEqual(model.metadata["training_hashes"], self.before)
            self.assertGreater(len(raw), sum(a.nbytes for values in model.params.values() for a in values.values()))
            self.assertTrue(all(not a.flags.writeable for values in model.params.values() for a in values.values()))
            reloaded = c.load(model.path, expected_model_identity=model.metadata["model_identity"],
                expected_training_hashes=self.before, expected_model_id=model.model_id, expected_kind=model.kind)
            self.assertEqual(c.model_bytes(reloaded), raw)
        self.assertEqual(self.models["conditional_anchor_r32"].metadata["layer_schema"]["layer_00"]["actual_rank"], 12)

    def test_entropy_books_use_actual_pre_rope_sources_and_exact_sign_order(self):
        model = self.models["conditional_entropy"]
        expected = {key: np.zeros(512 if key.startswith("exp") else 256, dtype=np.int64) for key in c._BOOKS}
        for group in self.records:
            for record in group:
                base = self.models["cost_pre_rope"].layer_params(record["layer"])
                for side, source in (("K", "k_pre_rope"), ("V", "v")):
                    words = record[source]
                    error = ((words.astype(np.int64) >> 7) & 255) - base["base_" + side]
                    zigzag = np.where(error >= 0, 2 * error, -2 * error - 1)
                    expected["exp_" + side] += np.bincount(zigzag.ravel(), minlength=512)
                    signs = (words >> 15).astype(np.uint8)
                    stream = np.packbits((signs[1:] ^ signs[:-1]).T.ravel(), bitorder="little")
                    expected["sign_" + side] += np.bincount(stream, minlength=256)
        for prefix, histogram in expected.items():
            stats = model.metadata["training_stats"][prefix]
            np.testing.assert_array_equal(stats["training_symbol_histogram"], histogram)
            self.assertEqual(sum(stats["training_symbol_histogram"]), stats["training_symbols"])
            self.assertEqual(stats["count_sha256"], hashlib.sha256(histogram.astype("<i8").tobytes()).hexdigest())
            book = model.book(prefix)
            self.assertEqual(stats["training_code_bits"], int(histogram @ book["length"].astype(np.int64)))
            self.assertTrue(np.all(book["length"] > 0))  # Unseen symbols remain representable.

    def test_actual_anchor_features_and_packet_only_joint_prediction(self):
        for name in ("conditional_anchor_r8", "conditional_anchor_r32", "conditional_latent_r8", "conditional_latent_r32"):
            model = self.models[name]
            record = copy.deepcopy(self.records[1][0])
            params = {key: torch.tensor(value.copy()) for key, value in model.layer_params(0).items()}
            fields = {key: torch.from_numpy(record[key].copy()) for key in ("hidden", "k_pre_rope", "v")}
            features = c.torch_features_params(params, model.kind, fields)
            self.assertEqual(features.dtype, torch.uint16)
            self.assertEqual(features.shape[1], params["coeff"].shape[0])
            if model.kind == "conditional_anchor":
                expected = np.concatenate((record["k_pre_rope"], record["v"]), axis=1)[:, model.layer_params(0)["anchor_indices"]]
                np.testing.assert_array_equal(features.numpy(), expected)
            fields.clear()
            record.clear()
            prediction = c.torch_predict_params(params, model.kind, features)
            self.assertEqual(tuple(prediction.shape), (13, 12))
            self.assertTrue(bool(torch.isfinite(prediction.view(torch.bfloat16)).all()))
            if model.kind == "conditional_anchor":
                # Compare through signed views for backends lacking uint16 indexing.
                self.assertTrue(torch.equal(prediction.view(torch.int16)[:, params["anchor_indices"].long()], features.view(torch.int16)))
            proxy = model.metadata["training_stats"]["layer_00"]
            self.assertTrue(proxy["training_only"])
            self.assertIn("TRAIN NumPy BF16 whole-layer shrinkage", proxy["fit_objective"])
            self.assertEqual(len(proxy["shrinkage_candidates"]), 4)
            self.assertEqual(proxy["selection_proxy_total_bits"],
                min(row["total_proxy_bits"] for row in proxy["shrinkage_candidates"]))

    def test_mean_is_feature_free_actual_train_joint_mean(self):
        model = self.models["conditional_mean"]
        for layer in (0, 1):
            words = np.concatenate([c._joint(record) for group in self.records for record in group if record["layer"] == layer])
            expected = c._float(words).mean(axis=0, dtype=np.float64).astype(np.float32)
            params = model.layer_params(layer)
            self.assertEqual(set(params), {"intercept"})
            np.testing.assert_array_equal(params["intercept"], expected)
            self.assertEqual(model.metadata["training_stats"][p._key(layer)]["feature_bytes"], 0)

    def test_anchor_retention_copies_signed_zero_and_subnormal_bits(self):
        words = torch.tensor([[0x8000, 0x0001, 0x8001, 0x007f]], dtype=torch.uint16)
        params = {"anchor_indices": torch.arange(4, dtype=torch.int32)}
        retained = c.torch_features_params(params, "conditional_anchor", {"k_pre_rope": words, "v": words.clone()})
        self.assertTrue(torch.equal(retained.view(torch.int16), words.view(torch.int16)))

    def test_rate_proxy_matches_actual_five_bit_widths_and_unsent_tail_padding(self):
        from qwen_kv_bench import producer_codec
        rng = np.random.default_rng(19)
        actual = rng.integers(0, 65536, (5, 14), dtype=np.uint16)
        predicted = rng.integers(0, 65536, (5, 14), dtype=np.uint16)
        feature = np.zeros((5, 3), dtype=np.uint16)
        stats = c._proxy(actual, predicted, feature, [2, 3])
        expected_payload = expected_widths = expected_groups = expected_bits = offset = 0
        for count in (2, 3):
            for side in (slice(0, 7), slice(7, 14)):
                fields = {}
                desc = producer_codec._encode_full_correction(torch.from_numpy(actual[offset:offset + count, side].copy()),
                    torch.from_numpy(predicted[offset:offset + count, side].copy()), fields, "check", "residual")
                expected_payload += fields["check__correction"].numel()
                expected_widths += fields["check__correction_widths"].numel()
                expected_groups += desc["groups"]
                expected_bits += desc["nbits"]
            offset += count
        self.assertEqual(stats["correction_payload_bytes"], expected_payload)
        self.assertEqual(stats["correction_width_bytes"], expected_widths)
        self.assertEqual(stats["correction_groups"], expected_groups)
        self.assertEqual(stats["correction_payload_data_bits"], expected_bits)
        self.assertEqual(stats["selection_proxy_total_bits"], (expected_payload + expected_widths + feature.nbytes) * 8)
        all_anchors = np.arange(14, dtype=np.int32)
        anchored = c._proxy(actual, actual, actual, [2, 3], all_anchors)
        self.assertEqual(anchored["correction_payload_bytes"], 0)
        self.assertEqual(anchored["correction_width_bytes"], 0)
        self.assertEqual(anchored["selection_proxy_total_bits"], actual.nbytes * 8)

    def test_crc_sha_and_semantic_mutations_are_rejected(self):
        model = self.models["conditional_entropy"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            raw = bytearray(Path(model.path).read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "CRC"):
                c.load(path)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                c.load(model.path, expected_model_id="0" * 64)
            with self.assertRaisesRegex(ValueError, "TRAIN source"):
                c.load(model.path, expected_training_hashes=["x"] * 3)
            altered = c.Model(model.kind, {key: {k: v.copy() for k, v in values.items()} for key, values in model.params.items()}, copy.deepcopy(model.metadata))
            altered.params["global"]["exp_K_lookup_symbol"][0] ^= 1
            with self.assertRaisesRegex(ValueError, "lookup differs"):
                c.model_bytes(altered)
        predictor = copy.deepcopy(self.models["conditional_latent_r8"])
        predictor.params["layer_00"]["coeff"] = np.full_like(predictor.params["layer_00"]["coeff"], np.nan)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            c.model_bytes(predictor)

    def test_deterministic_fit_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            second = fit(directory)
            for name in self.models:
                self.assertEqual(self.models[name].model_id, second[name].model_id)

    def test_train_only_missing_source_and_mutation_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = sources()
            bad[0][0]["split"] = "dev"
            with self.assertRaises(ValueError):
                fit(directory, bad)
            bad = sources()
            del bad[0][0]["k_pre_rope"]
            with self.assertRaises(ValueError):
                fit(directory, bad)
            bad = sources()
            original = c._fit_entropy
            def mutate(*args, **kwargs):
                result = original(*args, **kwargs)
                bad[0][0]["unused"][0] += 1
                return result
            with mock.patch.object(c, "_fit_entropy", side_effect=mutate):
                with self.assertRaisesRegex(AssertionError, "changed during fitting"):
                    fit(directory, bad)


if __name__ == "__main__":
    unittest.main()

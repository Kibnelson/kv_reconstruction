"""Exact producer-pair tests; CPU checks are not GPU throughput evidence."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from qwen_kv_bench import gpu_ops as g
from qwen_kv_bench import producer_codec as p


def _bf16(values):
    return torch.as_tensor(values, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy().copy()


def _fixtures(tokens=7, channels=16, head_dim=8):
    rng = np.random.default_rng(32891 + tokens + channels)
    record = dict(layer=0, start=127, tokens=tokens, head_dim=head_dim,
        k=rng.integers(0, 65536, (tokens, channels), dtype=np.uint16),
        v=rng.integers(0, 65536, (tokens, channels), dtype=np.uint16),
        hidden=_bf16(rng.normal(size=(tokens, 12))),
        k_proj=rng.integers(0, 65536, (tokens, channels), dtype=np.uint16),
        k_pre_rope=rng.integers(0, 65536, (tokens, channels), dtype=np.uint16))
    context = dict(rope_inv_freq=(10000.0 ** (-np.arange(head_dim // 2) * 2 / head_dim)).astype(np.float32),
        rope_attention_scaling=1.0, rope_stable=True,
        norm_weights={0: _bf16(np.linspace(0.75, 1.25, head_dim))}, norm_eps={0: 1e-6})
    models = {}
    for kind in ("frozen", "hidden", "hidden_abs"):
        params = dict(base_K=rng.integers(110, 135, channels, dtype=np.uint8),
                      base_V=rng.integers(110, 135, channels, dtype=np.uint8))
        if kind != "frozen":
            params.update(indices=np.array([0, 3, 7], dtype=np.int32),
                scales=np.array([0.1, 0.2, 0.05], dtype=np.float32),
                coeff_K=rng.integers(-1000, 1001, (3, channels), dtype=np.int16),
                coeff_V=rng.integers(-1000, 1001, (3, channels), dtype=np.int16))
        config = dict(rank=0 if kind == "frozen" else 3,
                      feature_transform={"frozen": "none", "hidden": "signed", "hidden_abs": "abs"}[kind])
        models[kind] = SimpleNamespace(kind=kind, params={"layer_00": params}, config=config,
            metadata=dict(config=config, split="train", source="synthetic unit fixture"),
            model_id="synthetic-" + kind, path=None)
    return record, models, context


class ProducerCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check(self, record, spec, models, context, device="cpu"):
        state = p.make_state(spec, models, context, device=device)
        packet = p.encode(record, spec, state)
        expected = tuple(torch.from_numpy(record[key].copy()).to(device) for key in ("k", "v"))
        got = p.decode(packet, state)
        for restored, original in zip(got, expected):
            self.assertTrue(torch.equal(restored.to(torch.int32), original.to(torch.int32)))
        self.assertEqual(packet.stats["raw_payload_bytes"], sum(value.numel() * 2 for value in expected))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))
        host = g.packet_to_cpu(packet)
        wire = g.packet_from_cpu_parts(host.header.clone(), {name: value.clone() for name, value in host.fields.items()})
        if device != "cpu":
            wire = g.packet_to_device(wire, device)
        for restored, original in zip(g.decode(wire, state), expected):
            self.assertTrue(torch.equal(restored.to(torch.int32), original.to(torch.int32)))
        return packet, state

    def test_every_BF16_pattern_all_nine_recipes_and_nan_sources(self):
        record, models, context = _fixtures(256, 256)
        values = np.arange(65536, dtype=np.uint32)
        record["k"] = values.astype(np.uint16).reshape(256, 256)
        record["v"] = values[::-1].astype(np.uint16).reshape(256, 256)
        record["k_proj"] = ((values * 29) & 65535).astype(np.uint16).reshape(256, 256)
        record["k_pre_rope"] = ((values * 71) & 65535).astype(np.uint16).reshape(256, 256)
        self.assertEqual(len(p.recipes()), 9)
        self.assertEqual(len({spec["name"] for spec in p.recipes()}), 9)
        for spec in p.recipes():
            self.check(record, spec, models, context)

    def test_tails_empty_and_single_token_inputs(self):
        for tokens, channels, head_dim in ((0, 16, 8), (1, 18, 2), (5, 18, 2)):
            record, models, context = _fixtures(tokens, channels, head_dim)
            for spec in p.recipes():
                self.check(record, spec, models, context)

    def test_shared_features_numpy_quantization_and_integer_prediction(self):
        record, models, context = _fixtures()
        for kind in ("hidden", "hidden_abs"):
            spec = next(spec for spec in p.recipes() if spec["kind"] == kind)
            packet, state = self.check(record, spec, models, context)
            params = models[kind].params["layer_00"]
            hidden = (record["hidden"].astype(np.uint32) << 16).view(np.float32)
            features = np.clip(np.rint(hidden[:, params["indices"]] / params["scales"]), -127, 127).astype(np.int8)
            stored = packet.fields["shared_features"].view(torch.int8).reshape(features.shape).numpy()
            np.testing.assert_array_equal(stored, features)
            self.assertEqual(packet.fields["shared_features"].numel(), len(features) * features.shape[1])
            design = np.abs(features.astype(np.int64)) if kind == "hidden_abs" else features.astype(np.int64)
            for kv in ("K", "V"):
                expected = np.clip(params["base_" + kv].astype(np.int64)[None, :] +
                    (design @ params["coeff_" + kv].astype(np.int64) + 128) // 256, 0, 255)
                actual = p._predict(state, 0, kv, len(features), torch.from_numpy(features.copy())).numpy()
                np.testing.assert_array_equal(actual, expected)
            self.assertFalse(any("hidden" in name or "k_proj" in name for name in packet.fields))
            original = (record["k"].copy(), record["v"].copy())
            record_without_sources = {key: value for key, value in record.items() if key in ("layer", "start")}
            self.assertEqual(set(record_without_sources), {"layer", "start"})
            for actual, expected in zip(p.decode(packet, state), original):
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_full_word_coset_residual_widths_are_matched(self):
        record, models, context = _fixtures(19, 18, 2)
        for source in ("pre_rope", "kproj_norm_rope"):
            packets = [self.check(record, spec, models, context)[0] for spec in p.recipes() if spec["kind"] == source]
            self.assertEqual(len(packets), 2)
            self.assertEqual(packets[0].metadata["full_word_correction"]["nbits"], packets[1].metadata["full_word_correction"]["nbits"])
            self.assertTrue(torch.equal(packets[0].fields["K__correction_widths"], packets[1].fields["K__correction_widths"]))
            for packet in packets:
                description = packet.metadata["full_word_correction"]
                self.assertEqual(description["width_storage_bits"], 5)
                widths = g._unpack_fixed(packet.fields["K__correction_widths"], description["groups"], 5)
                self.assertTrue(bool(torch.all(widths <= 16)))

    def test_minimum_producer_staging_and_complete_state_charge(self):
        record, models, context = _fixtures()
        with tempfile.TemporaryDirectory() as directory:
            for spec in p.recipes():
                staged = p.prepare_record(record, "cpu", spec)
                source = {"hidden": "hidden", "hidden_abs": "hidden", "pre_rope": "k_pre_rope", "kproj_norm_rope": "k_proj"}.get(spec["kind"])
                self.assertEqual(set(staged), {"layer", "start", "tokens", "head_dim", "k", "v"} | ({source} if source else set()))
                packet, state = self.check(record, spec, models, context)
                path = state.dump(Path(directory) / (spec["name"] + ".bin"))
                self.assertEqual(path.read_bytes(), p.state_bytes(state))
                self.assertEqual(state.serialized_bytes, path.stat().st_size)
                self.assertEqual(state.state_bytes, max(state.serialized_bytes, state.physical_nbytes))
                if state.serialized_bytes:
                    self.assertEqual(state.state_id, hashlib.sha256(path.read_bytes()).hexdigest())
                if spec["kind"] in ("pre_rope", "kproj_norm_rope"):
                    self.assertFalse(any("cos" in name or "sin" in name for name in state.fields))
                    self.assertFalse(any("hidden" in name for name in state.fields))
                    self.assertEqual(state.fields["rope_inv_freq"].numel(), record["head_dim"] // 2)

    def test_wrong_state_header_source_and_geometry_rejected(self):
        record, models, context = _fixtures()
        spec = next(spec for spec in p.recipes() if spec["kind"] == "frozen")
        packet, state = self.check(record, spec, models, context)
        bad_models = copy.deepcopy(models)
        bad_models["frozen"].params["layer_00"]["base_K"][0] += np.uint8(1)
        with self.assertRaises(ValueError):
            p.decode(packet, p.make_state(spec, bad_models, context))
        changed = g.packet_to_cpu(packet)
        changed.metadata["layer"] = 1
        with self.assertRaises(ValueError):
            p.decode(changed, state)
        for kind in ("hidden", "pre_rope", "kproj_norm_rope"):
            recipe = next(spec for spec in p.recipes() if spec["kind"] == kind)
            state = p.make_state(recipe, models, context)
            source = {"hidden": "hidden", "pre_rope": "k_pre_rope", "kproj_norm_rope": "k_proj"}[kind]
            bad = dict(record)
            del bad[source]
            with self.assertRaises(ValueError):
                p.encode(bad, recipe, state)
        hidden = next(spec for spec in p.recipes() if spec["kind"] == "hidden")
        state = p.make_state(hidden, models, context)
        for change in ({"hidden": record["hidden"][:, :2]}, {"tokens": 10}, {"start": -1}):
            with self.assertRaises(ValueError):
                p.encode(dict(record, **change), hidden, state)
        replay = next(spec for spec in p.recipes() if spec["kind"] == "pre_rope")
        replay_packet, replay_state = self.check(record, replay, models, context)
        changed_context = copy.deepcopy(context)
        changed_context["rope_inv_freq"][0] += np.float32(0.01)
        with self.assertRaises(ValueError):
            p.decode(replay_packet, p.make_state(replay, models, changed_context))
        for change in ({"head_dim": 6}, {"k_pre_rope": record["k_pre_rope"][:, :2]}):
            with self.assertRaises(ValueError):
                p.encode(dict(record, **change), replay, replay_state)
        with self.assertRaises(ValueError):
            p.make_state(dict(replay, correction="unknown"), models, context)
        with self.assertRaises(ValueError):
            p.make_state(replay, models, dict(context, rope_stable=False))
        with self.assertRaises(ValueError):
            p.make_state(replay, models, dict(context, rope_inv_freq=np.array([np.nan], np.float32)))
        bad_models = copy.deepcopy(models)
        bad_models["hidden"].params["layer_00"]["scales"][0] = 0
        with self.assertRaises(ValueError):
            p.make_state(hidden, bad_models, context)
        bad_models = copy.deepcopy(models)
        bad_models["hidden"].config["feature_transform"] = "abs"
        with self.assertRaises(ValueError):
            p.make_state(hidden, bad_models, context)

    def test_actual_TRAIN_model_file_and_numpy_oracle(self):
        from qwen_kv_bench import producer_models as models_module
        records = [[dict(_fixtures(8, 16)[0], split="train")] for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            models = models_module.fit(records, directory, rank=3,
                training_metadata=[dict(split="train", name=f"train{i}") for i in range(3)],
                model_identity=dict(model="synthetic", head_dim=8, num_key_value_heads=2, num_hidden_layers=1))
            record, _, context = _fixtures(5, 16)
            for spec in p.recipes():
                if spec["kind"] not in models:
                    continue
                packet, state = self.check(record, spec, models, context)
                model = models[spec["kind"]]
                self.assertEqual(state.wire_bytes, Path(model.path).read_bytes())
                self.assertEqual(state.serialized_bytes, model.state_bytes)
                features = model.features(record) if spec["kind"] != "frozen" else None
                for kv in ("K", "V"):
                    actual = p._predict(state, 0, kv, len(record["k"]),
                        torch.from_numpy(features.copy()) if features is not None else None).numpy()
                    np.testing.assert_array_equal(actual, model.predict_exponents(0, kv, len(record["k"]), features))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; CPU tests provide no GPU performance evidence")
    def test_cuda_pair_roundtrip_all_recipes(self):
        record, models, context = _fixtures(17, 16)
        for spec in p.recipes():
            self.check(record, spec, models, context, device="cuda")


if __name__ == "__main__":
    unittest.main()

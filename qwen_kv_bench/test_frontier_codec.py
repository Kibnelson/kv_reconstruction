"""Exactness and actual-byte frontier controls, not pretrained savings claims."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import frontier_codec as f
from qwen_kv_bench import gpu_ops as g
from qwen_kv_bench.test_producer_codec import _bf16, _fixtures


def fixture(tokens=7, channels=16, head_dim=8):
    record, models, context = _fixtures(tokens, channels, head_dim)
    for source, key in (("final", "k"), ("pre_rope", "k_pre_rope"), ("kproj", "k_proj")):
        model = copy.deepcopy(models["frozen"])
        model.config.update(source_k=key, source_v="v", exponent_base="actual_2_3_9_tier_cost")
        model.model_id = "synthetic-cost-" + source
        models["cost_" + source] = model
    return record, models, context


class FrontierCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check(self, record, spec, models, context, device="cpu"):
        state = f.make_state(spec, models, context, device)
        packet = f.encode(record, spec, state)
        expected = tuple(torch.from_numpy(record[key].copy()).to(device) for key in ("k", "v"))
        for actual, original in zip(f.decode(packet, state), expected):
            self.assertTrue(torch.equal(actual.to(torch.int32), original.to(torch.int32)))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))
        self.assertEqual(packet.stats["raw_payload_bytes"], sum(item.numel() * 2 for item in expected))
        host = g.packet_to_cpu(packet)
        wire = g.packet_from_cpu_parts(host.header.clone(), {name: item.clone() for name, item in host.fields.items()})
        wire = g.packet_to_device(wire, device)
        for actual, original in zip(g.decode(wire, state), expected):
            self.assertTrue(torch.equal(actual.to(torch.int32), original.to(torch.int32)))
        return packet, state

    def test_all_BF16_patterns_all_sixteen_candidates(self):
        record, models, context = fixture(256, 256)
        values = np.arange(65536, dtype=np.uint32)
        record["k"] = values.astype(np.uint16).reshape(256, 256)
        record["v"] = values[::-1].astype(np.uint16).reshape(256, 256)
        record["k_proj"] = ((values * 29) & 65535).astype(np.uint16).reshape(256, 256)
        record["k_pre_rope"] = ((values * 71) & 65535).astype(np.uint16).reshape(256, 256)
        self.assertEqual(len(f.recipes()), 16)
        self.assertEqual(len({spec["name"] for spec in f.recipes()}), 16)
        self.assertTrue(set(f.generation_representatives()) <= {spec["name"] for spec in f.recipes()})
        for spec in f.recipes():
            self.check(record, spec, models, context)

    def test_empty_tail_singleton_and_complete_state_files(self):
        for tokens, channels, head in ((0, 16, 8), (1, 18, 2), (5, 18, 2)):
            record, models, context = fixture(tokens, channels, head)
            with tempfile.TemporaryDirectory() as directory:
                for spec in f.recipes():
                    packet, state = self.check(record, spec, models, context)
                    path = state.dump(Path(directory) / (spec["name"] + ".bin"))
                    self.assertEqual(path.read_bytes(), f.state_bytes(state))
                    self.assertEqual(state.serialized_bytes, path.stat().st_size)
                    self.assertEqual(state.state_bytes, max(state.serialized_bytes, state.physical_nbytes))
                    if state.serialized_bytes:
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), state.state_id)

    def test_compact_start_terminal_65536_and_uint32_fallback(self):
        spec = next(row for row in f.recipes() if row["name"] == "frontier_cost_final_raw8")
        for tokens, channels, expected_bits in ((64, 1024, 16), (65, 1024, 32)):
            record, models, context = fixture(tokens, channels, 8)
            for key in ("k", "v"):
                record[key].fill(0)
            for kv in ("K", "V"):
                models["cost_final"].params["layer_00"]["base_" + kv].fill(127)
            packet, state = self.check(record, spec, models, context)
            for label in ("K", "V"):
                description = packet.metadata["encoded_sides"][label]
                self.assertEqual(description["offset_storage_bits"], expected_bits)
                self.assertTrue(description["offset_starts_only"])
                self.assertEqual(description["secondary_count"], tokens * channels)
                self.assertEqual(description["rare_count"], tokens * channels)
                segments = (tokens * channels + 255) // 256
                for suffix in ("secondary_offsets", "rare_offsets"):
                    tensor = packet.fields[label + "__" + suffix]
                    self.assertEqual(tensor.numel(), segments * expected_bits // 8)
                    self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel())

    def test_head_shift_actual_byte_gain_and_zero_fallback(self):
        record, models, context = fixture(64, 64, 8)
        for key in ("k", "v"):
            record[key] = np.full((64, 64), 127 << 7, dtype=np.uint16)
        for kv in ("K", "V"):
            models["cost_final"].params["layer_00"]["base_" + kv].fill(120)
        shifted = next(row for row in f.recipes() if row["name"] == "frontier_shift_final_raw8")
        control = next(row for row in f.recipes() if row["name"] == "frontier_cost_final_raw8")
        base, _ = self.check(record, control, models, context)
        packet, state = self.check(record, shifted, models, context)
        self.assertLess(packet.physical_nbytes, base.physical_nbytes)
        for label in ("K", "V"):
            self.assertTrue(packet.metadata["encoded_sides"][label]["head_shift_applied"])
            # Error +1 and error 0 both occupy the same 2-bit primary tier;
            # the smallest-magnitude tied shift is therefore +6, not +7.
            self.assertEqual(packet.fields[label + "__head_shift"].view(torch.int8).tolist(), [6] * 8)
        for kv in ("K", "V"):
            models["cost_final"].params["layer_00"]["base_" + kv].fill(127)
        packet, state = self.check(record, shifted, models, context)
        self.assertFalse(any(name.endswith("__head_shift") for name in packet.fields))
        self.assertTrue(all(not value["head_shift_applied"] for value in packet.metadata["encoded_sides"].values()))

    def test_primary_auto_planes_positive_and_fixed_two_bit_fallback(self):
        record, models, context = fixture(64, 64, 8)
        spec = next(row for row in f.recipes() if row["name"] == "frontier_cost_final_raw8")
        self.assertEqual(spec["primary_mode"], "auto_plane")
        for kv in ("K", "V"):
            models["cost_final"].params["layer_00"]["base_" + kv].fill(127)
        for key in ("k", "v"):
            record[key] = np.full((64, 64), 127 << 7, dtype=np.uint16)
        packet, state = self.check(record, spec, models, context)
        self.assertTrue(all(part["primary_storage"] == "bitplanes" for part in packet.metadata["encoded_sides"].values()))
        self.assertFalse(any(name in packet.fields for name in ("K__primary", "V__primary")))
        # Uniform random primary symbols keep both meaningful bitplanes dense;
        # six zero high planes cannot repay their metadata over fixed 2-bit data.
        rng = np.random.default_rng(175)
        errors = np.array([0, -1, 1, -2], dtype=np.int64)
        for key in ("k", "v"):
            exp = 127 + errors[rng.integers(0, 4, (64, 64))]
            record[key] = ((exp << 7) | rng.integers(0, 128, (64, 64))).astype(np.uint16)
        packet, state = self.check(record, spec, models, context)
        self.assertTrue(all(part["primary_storage"] == "fixed2" for part in packet.metadata["encoded_sides"].values()))
        self.assertTrue({"K__primary", "V__primary"} <= set(packet.fields))

    def test_lower_full_contribution_fallback_can_override_child_choice(self):
        from qwen_kv_bench import frontier_lower
        record, models, context = fixture(16, 48, 8)
        generator = torch.Generator().manual_seed(991)
        lower = torch.randint(0, 2, (16, 48), generator=generator, dtype=torch.int64).to(torch.uint8)
        child_fields, child_meta = frontier_lower.encode(lower, mode="plane_raw")
        raw_fields, raw_meta = frontier_lower.encode(lower, mode="raw8")
        self.assertEqual(child_meta["storage"], "adaptive_bitplanes")
        child = sum(value.numel() for value in child_fields.values()) + len(f._json(child_meta))
        raw = sum(value.numel() for value in raw_fields.values()) + len(f._json(raw_meta))
        self.assertLess(child, raw)
        for kv in ("K", "V"):
            models["cost_final"].params["layer_00"]["base_" + kv].fill(127)
        for key in ("k", "v"):
            record[key] = ((np.full((16, 48), 127, np.uint16) << 7) | lower.numpy().astype(np.uint16))
        spec = next(row for row in f.recipes() if row["name"] == "frontier_shift_final_plane_raw")
        packet, state = self.check(record, spec, models, context)
        for description in packet.metadata["encoded_sides"].values():
            self.assertEqual(description["lower"]["storage"], "raw8")
            self.assertTrue(description["lower_complete_fallback_to_raw"])

    def test_replay_runtime_contract_is_serialized_hashed_and_guarded(self):
        record, models, context = fixture()
        spec = next(row for row in f.recipes() if row["name"] == "frontier_shift_pre_rope_raw8")
        packet, state = self.check(record, spec, models, context)
        contract = f.state_info(state)["replay_execution_contract"]
        self.assertEqual(contract["device_type"], "cpu")
        self.assertIn("not universal canonical trigonometry", contract["numerical_scope"])
        self.assertEqual(contract["torch_version"], str(torch.__version__))
        self.assertIn(b"replay_execution_contract", state.wire_bytes)
        changed = dict(contract, torch_version="deliberately different runtime")
        with mock.patch.object(f, "_execution_contract", return_value=changed):
            with self.assertRaises(ValueError):
                f.decode(packet, state)
            with self.assertRaises(ValueError):
                f.encode(record, spec, state)
            changed_state = f.make_state(spec, models, context)
            self.assertNotEqual(changed_state.state_id, state.state_id)
            self.assertNotEqual(changed_state.wire_bytes, state.wire_bytes)

    def test_replay_verified_zero_and_full_word_nonzero_fallback(self):
        record, models, context = fixture(1, 16, 8)
        record["start"] = 0
        for key in ("k", "v", "k_pre_rope", "k_proj"):
            record[key] = _bf16(np.ones((1, 16), dtype=np.float32))
        context["norm_weights"][0] = _bf16(np.ones(8, dtype=np.float32))
        for source in ("pre_rope", "kproj"):
            spec = next(row for row in f.recipes() if row["name"] == f"frontier_shift_{source}_plane_first")
            packet, state = self.check(record, spec, models, context)
            self.assertEqual(packet.metadata["replay_correction"]["mode"], "verified_zero")
            self.assertFalse(any(name.startswith("K_replay__") for name in packet.fields))
            changed = dict(record, k=record["k"].copy())
            changed["k"][0, 0] ^= np.uint16(0x8001)
            packet, state = self.check(changed, spec, models, context)
            self.assertEqual(packet.metadata["replay_correction"]["mode"], "coset_fallback")
            self.assertGreater(packet.metadata["replay_correction"]["nbits"], 0)
            self.assertIn("K_replay__correction", packet.fields)

    def test_synthetic_constant_plane_positive_control_can_exceed_40_percent(self):
        # A deliberately structured synthetic input tests codec capability, not
        # an empirical savings claim about Qwen or any pretrained model.
        record, models, context = fixture(64, 1024, 128)
        for key in ("k", "v"):
            record[key] = _bf16(np.ones((64, 1024), dtype=np.float32))
        for kv in ("K", "V"):
            models["cost_final"].params["layer_00"]["base_" + kv].fill(127)
        spec = next(row for row in f.recipes() if row["name"] == "frontier_shift_final_plane_first")
        packet, state = self.check(record, spec, models, context)
        raw = record["k"].nbytes + record["v"].nbytes
        self.assertGreater(100 * (1 - packet.physical_nbytes / raw), 40)

    def test_wrong_stage_missing_source_state_and_header_guards(self):
        record, models, context = fixture()
        spec = next(row for row in f.recipes() if row["name"] == "frontier_shift_pre_rope_plane_prev")
        for config in ({"source_k": "k"}, {"source_k": None}, {"source_v": "k"}):
            bad = copy.deepcopy(models)
            bad["cost_pre_rope"].config.update(config)
            with self.assertRaises(ValueError):
                f.make_state(spec, bad, context)
        bad = copy.deepcopy(models)
        del bad["cost_pre_rope"].config["source_k"]
        with self.assertRaises(ValueError):
            f.make_state(spec, bad, context)
        packet, state = self.check(record, spec, models, context)
        bad_context = copy.deepcopy(context)
        bad_context["rope_inv_freq"][0] += np.float32(0.01)
        with self.assertRaises(ValueError):
            f.decode(packet, f.make_state(spec, models, bad_context))
        bad = dict(record)
        del bad["k_pre_rope"]
        with self.assertRaises(ValueError):
            f.encode(bad, spec, state)
        wire = g.packet_to_cpu(packet)
        wire.metadata["source"] = "final"
        with self.assertRaises(ValueError):
            f.decode(wire, state)

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA hardware; functional CPU checks make no GPU performance claim")
    def test_cuda_all_frontier_candidates(self):
        record, models, context = fixture(17, 16, 8)
        for spec in f.recipes():
            self.check(record, spec, models, context, "cuda")


if __name__ == "__main__":
    unittest.main()

"""Pair codec exactness, packet-only restoration and complete byte contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import conditional_codec as c
from qwen_kv_bench import conditional_models as models_module
from qwen_kv_bench import gpu_ops as g
from qwen_kv_bench import producer_codec as producer
from qwen_kv_bench.test_conditional_models import fit, sources


class ConditionalCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.models = fit(cls.directory.name)
        cls.record = sources()[0][0]
        cls.context = dict(rope_stable=True, rope_inv_freq=np.array([1.], dtype=np.float32), rope_attention_scaling=1.)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def spec(self, name):
        return next(spec for spec in c.recipes() if spec["name"] == name)

    def state(self, name, device="cpu"):
        spec = self.spec(name)
        return spec, c.make_state(spec, self.models, self.context, device)

    def assert_exact(self, packet, state, record):
        recovered = c.decode(packet, state)
        for actual, field in zip(recovered, ("k", "v")):
            expected = record[field] if isinstance(record[field], torch.Tensor) else torch.from_numpy(record[field].copy())
            self.assertTrue(torch.equal(actual.to(torch.int32).cpu(), expected.to(torch.int32).cpu()))
        self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))
        return recovered

    def test_thirteen_frozen_recipes_and_every_family_wire_roundtrip(self):
        self.assertEqual(len(c.recipes()), 13)
        self.assertEqual({s["family"] for s in c.recipes()}, {"control", "exponent", "sign", "anchor", "latent"})
        self.assertEqual(set(c.generation_representatives()), {s["name"] for s in c.recipes()})
        for spec in c.recipes():
            state = c.make_state(spec, self.models, self.context)
            record = copy.deepcopy(self.record)
            packet = c.encode(record, spec, state)
            self.assert_exact(packet, state, record)
            description = c.describe(packet)
            self.assertEqual(description["recipe"], spec["name"])
            raw = c.state_bytes(state)
            if raw:
                self.assertEqual(hashlib.sha256(raw).hexdigest(), state.state_id)
            self.assertGreaterEqual(c.state_info(state)["charged_bytes"], len(raw))
            if packet.metadata["method"] == "conditional_pair":
                self.assertNotIn("predictor_quality", packet.metadata)
                self.assertNotIn("candidate_complete_bytes", packet.metadata)
            host = g.packet_to_cpu(packet)
            restored = g.packet_from_cpu_parts(host.header.clone(), {k: v.clone() for k, v in host.fields.items()})
            # The launch diagnostic sidecar and original producer record are not
            # supplied to decoding the independently restored actual wire.
            self.assertFalse(hasattr(restored, "analysis"))
            self.assert_exact(restored, state, record)

    def test_true_raw_pair_has_no_model_state_and_materializes_fresh_outputs(self):
        spec, state = self.state("conditional_raw_pair")
        record = {key: value for key, value in self.record.items() if key != "k_pre_rope"}
        packet = c.encode(record, spec, state)
        self.assertEqual(state.state_bytes, 0)
        self.assertEqual(set(packet.fields), {"final_K", "final_V"})
        self.assertEqual(sum(v.numel() for v in packet.fields.values()), self.record["k"].nbytes + self.record["v"].nbytes)
        recovered = self.assert_exact(packet, state, record)
        self.assertNotEqual(recovered[0].untyped_storage().data_ptr(), packet.fields["final_K"].untyped_storage().data_ptr())
        self.assertEqual(c.describe(packet)["replay_correction_mode"], "none")

    def test_sign_seeds_singletons_odd_tails_and_nonfinite_exact_symbol_fallback(self):
        for name in ("conditional_huffman_exponent", "conditional_huffman_exponent_sign", "conditional_tier_sign"):
            spec, state = self.state(name)
            for tokens in (1, 3, 9):
                record = copy.deepcopy(self.record)
                record["tokens"] = tokens
                for field in ("k", "v", "k_pre_rope", "hidden"):
                    record[field] = record[field][:tokens].copy()
                # Arbitrary target/source words including NaNs and signed zero
                # remain exact; entropy coding operates on integer symbols.
                record["k_pre_rope"][0, :4] = [0, 32768, 0x7f80, 0xffc1]
                packet = c.encode(record, spec, state)
                self.assert_exact(packet, state, record)
                if "sign" in name:
                    self.assertEqual(packet.fields["K__sign_seed"].numel(), 1)  # six source channels
                    self.assertEqual(packet.metadata["sides"]["K"]["sign"]["transition_bits"], (tokens - 1) * 6)

    def test_entropy_count_sidecars_match_actual_code_bits_without_wire_inflation(self):
        for name in ("conditional_huffman_exponent", "conditional_huffman_exponent_sign", "conditional_tier_sign"):
            spec, state = self.state(name)
            packet = c.encode(self.record, spec, state)
            for stream, stats in packet.analysis["stream_statistics"].items():
                self.assertEqual(sum(stats["symbol_histogram"].values()), stats["symbols"])
                self.assertEqual(sum(stats["code_length_histogram"].values()), stats["symbols"])
                self.assertEqual(sum(int(width)*count for width,count in stats["code_length_histogram"].items()), stats["sum_code_bits"])
                self.assertEqual(len(stats["symbol_histogram"]), 512 if stream.endswith("exponent") else 256)
                if stream in packet.analysis["entropy"]:
                    self.assertEqual(stats["sum_code_bits"], packet.analysis["entropy"][stream]["encoded_bits"])
            self.assertNotIn("stream_statistics", json.loads(bytes(packet.header.tolist())))
            self.assert_exact(packet, state, self.record)

    def _perfect_record(self, state, tokens=256):
        record = c.prepare_record(copy.deepcopy(self.record), state.device, state.spec)
        for field in ("k", "v", "k_pre_rope", "hidden"):
            if field in record:
                record[field] = record[field].repeat((tokens + len(record[field]) - 1) // len(record[field]), 1)[:tokens].contiguous()
        record["tokens"] = tokens
        features, prediction = c._features_and_prediction(record, state)
        record["k_pre_rope"] = prediction[:, :6].contiguous().clone()
        record["v"] = prediction[:, 6:].contiguous().clone()
        record["k"] = producer._rope(record["k_pre_rope"], state.rope_state, record["start"], record["head_dim"], record["layer"])
        return record

    def test_positive_anchor_and_latent_prediction_and_complete_byte_fallback(self):
        modes = set()
        for name in ("conditional_anchor_r8", "conditional_latent_r8"):
            spec, state = self.state(name)
            record = self._perfect_record(state)
            packet = c.encode(record, spec, state)
            self.assert_exact(packet, state, record)
            # Carefully constructed mathematical fixtures make the predictor
            # exact; this tests the actual branch, not empirical model savings.
            self.assertEqual(packet.metadata["mode"], "predicted")
            self.assertEqual(packet.metadata["replay"]["mode"], "exact")
            self.assertEqual(packet.analysis["predictor_quality"]["word_matches"], 2 * 256 * 6)
            sizes = packet.analysis["candidate_complete_bytes"]
            self.assertEqual(packet.physical_nbytes, min(sizes.values()))
            modes.add(packet.metadata["mode"])
            if spec["family"] == "anchor":
                kept = sum(description["count"] for description in packet.metadata["corrections"].values())
                rank = packet.metadata["feature_rank"]
                self.assertEqual(kept, record["tokens"] * (12 - rank))
                self.assertEqual(packet.fields["shared_features"].numel(), record["tokens"] * rank * 2)
            fallback = c.encode(self.record, spec, state)
            self.assert_exact(fallback, state, self.record)
            self.assertEqual(fallback.physical_nbytes, min(fallback.analysis["candidate_complete_bytes"].values()))
            modes.add(fallback.metadata["mode"])
        self.assertIn("raw", modes)
        self.assertIn("predicted", modes)

    def test_all_anchor_coordinates_no_duplicate_corrections_and_empty_complements(self):
        spec, state = self.state("conditional_anchor_r32")
        rank = len(state.predictor_params["layer_00"]["anchor_indices"])
        self.assertEqual(rank, 12)
        original = c._baseline_packet
        def expensive_fallback(*args, **kwargs):
            packet = original(*args, **kwargs)
            fields = dict(packet.fields, test_only_padding=torch.zeros(8192, dtype=torch.uint8))
            return g._finish(fields, dict(packet.metadata), packet.header.device)
        with mock.patch.object(c, "_baseline_packet", side_effect=expensive_fallback):
            packet = c.encode(self.record, spec, state)
        self.assertEqual(packet.metadata["mode"], "predicted")
        self.assertTrue(all(description["count"] == 0 for description in packet.metadata["corrections"].values()))
        self.assertTrue(all(description["groups"] == 0 for description in packet.metadata["corrections"].values()))
        self.assert_exact(packet, state, self.record)
        self.assertEqual(packet.fields["K__correction"].numel() + packet.fields["V__correction"].numel(), 0)

    def test_all_65536_bf16_patterns_fullword_control_and_nonzero_replay(self):
        spec, state = self.state("conditional_raw_fullword")
        full = torch.arange(65536, dtype=torch.int64).to(torch.uint16).reshape(256, 256)
        record = dict(layer=0, start=0, tokens=256, head_dim=2,
            k=full, v=full.view(torch.int16).flip(1).contiguous().view(torch.uint16), k_pre_rope=full.clone())
        packet = c.encode(record, spec, state)
        self.assert_exact(packet, state, record)
        self.assertEqual(packet.metadata["replay"]["mode"], "coset")
        self.assertEqual(sum(packet.analysis["predictor_width_histogram"].values()), 2 * 65536 // 64)
        quality = c._quality(full, full)
        self.assertEqual(quality["matching_bits"], 65536 * 16)
        self.assertEqual(quality["word_matches"], 65536)

    def test_wrong_model_sources_rank_entropy_base_and_runtime_contract_rejected(self):
        wrong = dict(self.models)
        wrong["conditional_anchor_r8"] = self.models["conditional_latent_r8"]
        with self.assertRaises(ValueError):
            c.make_state(self.spec("conditional_anchor_r8"), wrong, self.context)
        entropy = copy.deepcopy(self.models["conditional_entropy"])
        entropy.metadata["config"]["exponent_base_model_id"] = "wrong"
        wrong = dict(self.models, conditional_entropy=entropy)
        with self.assertRaises(ValueError):
            c.make_state(self.spec("conditional_huffman_exponent"), wrong, self.context)
        spec, state = self.state("conditional_latent_r8")
        packet = c.encode(self.record, spec, state)
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium" if previous != "medium" else "highest")
            with self.assertRaises(ValueError):
                c.decode(packet, state)
        finally:
            torch.set_float32_matmul_precision(previous)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaises(ValueError):
                c.decode(packet, state)
        restored = g.packet_to_cpu(packet)
        restored.metadata["state_id"] = "tampered"
        with self.assertRaises(ValueError):
            c.decode(restored, state)

    def test_complete_state_roles_and_sidecar_is_not_wire(self):
        spec, state = self.state("conditional_latent_r8")
        info = c.state_info(state)
        self.assertGreater(info["encoder_only_predictor_tensor_bytes"], 0)
        self.assertGreater(info["decoder_predictor_tensor_bytes"], 0)
        self.assertEqual(info["required_decoder_numerical_bytes_excluding_encoder_only_predictor"] +
            info["encoder_only_predictor_tensor_bytes"], info["physical_tensor_bytes"])
        self.assertGreaterEqual(info["charged_bytes"], info["physical_tensor_bytes"])
        self.assertGreaterEqual(info["charged_bytes"], len(c.state_bytes(state)))
        packet = c.encode(self.record, spec, state)
        header_before = packet.header.clone()
        packet.analysis["test_diagnostic"] = [1, 2, 3]
        self.assertTrue(torch.equal(header_before, packet.header))
        self.assertNotIn("test_diagnostic", json.loads(bytes(packet.header.tolist())))
        self.assert_exact(packet, state, self.record)

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA hardware; CPU tensor exactness is not GPU performance")
    def test_actual_cuda_all_recipes_wire_transfer_and_exact_bf16(self):
        for spec in c.recipes():
            state = c.make_state(spec, self.models, self.context, "cuda")
            packet = c.encode(self.record, spec, state)
            self.assert_exact(packet, state, self.record)
            cpu = g.packet_to_cpu(packet)
            wire = g.packet_from_cpu_parts(cpu.header, cpu.fields)
            gpu = g.packet_to_device(wire, "cuda")
            self.assert_exact(gpu, state, self.record)


if __name__ == "__main__":
    unittest.main()

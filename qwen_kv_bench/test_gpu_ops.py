"""CPU functional tests for the identical tensor-native GPU code paths."""
import unittest

import torch

from qwen_kv_bench import gpu_ops as g


def _method_options(method):
    """Use each family's declared representation contract in shared tests.

    Existing families keep their defaults.  Residual dictionaries are 8D and
    optimize actual bit width; exponent recipes use the prefill median rather
    than the magnitude-codec default midpoint predictor.
    """
    if method == "prefill_dictionary":
        return dict(group_size=8, assignment="bitcost")
    if method == "prefill_exponent":
        return dict(predictor="median", layout="time", segment_size=256)
    return {}


class NativePacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def state(self, device="cpu"):
        book = torch.arange(32 * 8, dtype=torch.int32).reshape(32, 8) + 15700
        return g.make_state({"test": book}, device=device)

    def test_all_65536_patterns_all_native_families(self):
        report = g.self_test("cpu")
        self.assertEqual(set(report["methods"]), set(g.NATIVE_METHODS))
        self.assertTrue(all(row["exact"] for row in report["methods"].values()))

    def test_incremental_sweep_names_and_residual_model_identities(self):
        from qwen_kv_bench.recipe_sweep import insight_options
        specs = insight_options()
        self.assertEqual(len(specs), 33)
        self.assertEqual(len({spec["name"] for spec in specs}), len(specs))
        exponents = [spec for spec in specs if spec["method"] == "prefill_exponent"]
        self.assertEqual({(spec["options"]["predictor"], spec["options"]["layout"]) for spec in exponents},
            {("zero", "channel"), ("fixed127", "channel"), ("fixed127", "time"),
             ("median", "channel"), ("median", "time"), ("affine", "channel"), ("affine", "time")})
        dictionaries = [spec for spec in specs if spec["method"] == "prefill_dictionary"]
        self.assertEqual({spec["recipe_model"] for spec in dictionaries},
            {f"recipe_residual{count}_{layout}" for count in (16, 64) for layout in ("channel", "time")})
        for spec in dictionaries:
            self.assertEqual(spec["recipe_model"],
                f"recipe_residual{spec['dictionary_size']}_{spec['options']['layout']}")
            self.assertEqual(spec["options"]["group_size"], 8)
            self.assertEqual(spec["options"]["assignment"], "bitcost")

    def test_group_bitpacking_full_signed_domain_and_fixed_tails(self):
        values = torch.arange(-32767, 32768, dtype=torch.int64)
        fields, descriptions = {}, {}
        g._adaptive(values, fields, descriptions, "values", 64, True, 16)
        self.assertTrue(torch.equal(g._restore(fields, descriptions, "values"), values))
        for width in range(18):
            values = torch.arange(37, dtype=torch.int64) % max(1, 1 << width)
            packed, _ = g._pack_fixed(values, width)
            self.assertTrue(torch.equal(g._unpack_fixed(packed, len(values), width), values))

    def test_tails_special_words_precision_and_state_accounting(self):
        for count in (1, 7, 65, 193):
            words = ((torch.arange(count, dtype=torch.int64) * 997) & 65535).to(torch.uint16).reshape(1, -1)
            for method in g.NATIVE_METHODS:
                state = self.state()
                for precision in (("bf16", "f32") if method == "binary" else ("bf16",)):
                    packet = g.encode(words, state, method=method, key="test", dimension=64,
                        transform="hadamard", amplitude_precision=precision, **_method_options(method))
                    self.assertTrue(torch.equal(g.decode(packet, state).to(torch.int32), words.to(torch.int32)))
                    self.assertEqual(packet.physical_nbytes, sum(packet.stats["component_bytes"].values()))
                    self.assertGreater(packet.header.numel(), 0)
                    if method == "prefill_exponent":
                        self.assertEqual(packet.metadata["state_id"], "no_learned_state")
                        self.assertTrue({"sign_mantissa", "exponent_primary", "exponent_secondary",
                            "exponent_rare", "secondary_offsets", "rare_offsets"} <= set(packet.fields))
                        segments = (words.numel() + 255) // 256
                        self.assertEqual(packet.fields["secondary_offsets"].numel(), 4 * (segments + 1))
                        self.assertEqual(packet.fields["rare_offsets"].numel(), 4 * (segments + 1))
                        self.assertEqual(packet.stats["prefill_recipe"]["external_state_bytes"], 0)
                    if method == "prefill_dictionary":
                        self.assertIn("residual_ids", packet.fields)
                        self.assertEqual(packet.metadata["group_size"], 8)
                        self.assertEqual(packet.metadata["assignment"], "bitcost")
                        self.assertEqual(packet.metadata["model_id"], state.state_id)
                if method == "binary":
                    self.assertEqual(state.physical_nbytes, 32 * 8 * 4 + 64 * 64 * 8)
        empty = torch.empty((0, 5), dtype=torch.uint16)
        for method in g.NATIVE_METHODS:
            state = self.state()
            self.assertTrue(torch.equal(g.decode(g.encode(empty, state, method, key="test",
                **_method_options(method)), state), empty))

    def test_e8_membership_idempotence_and_coordinate_parity_roundtrip(self):
        generator = torch.Generator().manual_seed(319)
        values = torch.randn((113, 8), dtype=torch.float64, generator=generator) * 5
        points2 = g.nearest_e8_twice(values)
        self.assertTrue(bool(torch.all((points2 & 1) == (points2[:, :1] & 1))))
        self.assertTrue(bool(torch.all((points2.sum(dim=1) & 3) == 0)))
        self.assertTrue(torch.equal(g.nearest_e8_twice(points2.to(torch.float64) / 2), points2))
        fields, descriptions = {}, {}
        g._store_e8(points2, fields, descriptions, 64)
        self.assertTrue(torch.equal(g._restore_e8(fields, descriptions, len(points2)), points2))

    def test_binary_canonical_batch_and_combined_half_ties(self):
        state = self.state()
        rotation = state.rotation(64, "dense", device="cpu")
        x = torch.arange(7 * 64, dtype=torch.float64).reshape(7, 64) / 3
        all_values = g.canonical_matmul(x, rotation)
        separate = torch.cat([g.canonical_matmul(row[None], rotation) for row in x])
        self.assertTrue(torch.equal(all_values.view(torch.int64), separate.view(torch.int64)))
        c = torch.full((1, 64), 10, dtype=torch.int64)
        c[0, :6] = torch.tensor([1, 2, 1, 2, 32767, 0])
        bits = torch.ones((1, 64), dtype=torch.int64)
        bits[0, [2, 3, 5]] = 0
        prediction = g._binary_prediction(c, bits, torch.tensor([4.0]), torch.eye(64, dtype=torch.float64))
        self.assertEqual(prediction[0, :6].tolist(), [2, 2, 0, 2, 32767, 0])

    def test_bulk_packet_copy_and_state_identity(self):
        state = self.state()
        words = torch.arange(16 * 1024, dtype=torch.int64).to(torch.uint16).reshape(16, 1024)
        packet = g.encode(words, state, "learned", key="test")
        copied = g.packet_to_cpu(packet)
        self.assertTrue(torch.equal(g.decode(copied, state), words))
        self.assertEqual(copied.physical_nbytes, packet.physical_nbytes)
        self.assertNotEqual(packet.header.data_ptr(), copied.header.data_ptr())
        wrong = g.make_state({"test": torch.zeros((32, 8), dtype=torch.int32)})
        with self.assertRaises(ValueError):
            g.decode(packet, wrong)
        with self.assertRaises(NotImplementedError):
            g.encode(words, state, "deflate", key="test")
        with self.assertRaises(ValueError):
            g.words_from_bf16(torch.ones((2, 3), dtype=torch.float16))
        raw_bf16 = words.view(torch.bfloat16)
        self.assertTrue(torch.equal(g.words_from_bf16(raw_bf16), words))
        self.assertTrue(torch.equal(g.bf16_from_words(words).view(torch.uint16), words))

    def test_authoritative_header_restore_and_prefetched_descriptor_guards(self):
        state = self.state()
        words = torch.arange(512, dtype=torch.int64).to(torch.uint16).reshape(4, 128)
        encoded = g.encode(words, state, "learned", key="test")
        restored = g.packet_from_cpu_parts(encoded.header.clone(),
            {key: value.clone() for key, value in encoded.fields.items()})
        self.assertTrue(torch.equal(g.decode(restored, state), words))
        g.validate_packet(restored, check_header=True)
        # The isolated device decoder rejects edits to its prefetched host
        # descriptor without introducing a D2H header read on every launch.
        edited = g.packet_to_cpu(encoded)
        edited.metadata["shape"] = [2, 256]
        with self.assertRaises(ValueError):
            g.decode(edited, state)
        # Actual wire-header edits are detected at explicit transfer/restore.
        edited = g.packet_to_cpu(encoded)
        edited.header[0] ^= 1
        with self.assertRaises(ValueError):
            g.packet_to_device(edited, "cpu")
        with self.assertRaises(ValueError):
            g.packet_from_cpu_parts(edited.header, edited.fields)
        edited = g.packet_to_cpu(encoded)
        edited.fields["exact_residual"] = edited.fields["exact_residual"][:-1]
        with self.assertRaises(ValueError):
            g.validate_packet(edited, check_header=False)

    @unittest.skipUnless(torch.cuda.is_available(), "No local CUDA hardware; native CUDA path remains unvalidated here")
    def test_cuda_device_roundtrip(self):
        report = g.self_test("cuda")
        self.assertTrue(all(row["exact"] for row in report["methods"].values()))


if __name__ == "__main__":
    unittest.main()

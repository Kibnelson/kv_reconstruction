"""Direct compact address translation, exact native replay and ownership."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare.team_compact import TeamCompactCodec
from kv_compare.team_packet import TeamCodec, TeamPacket, save_state
from tests.test_team_packet import fixed_state, packet_fixture


class CompactTests(unittest.TestCase):
    def test_every_alignment_gap_translates_states_and_stream(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            codec = TeamCompactCodec(td, "cpu", "dense_uniform")
            for gap in range(8):
                packet, arena, native_starts = packet_fixture(codec.header((2, 64, 256)), gap=gap)
                wire = packet.export()
                codec.stage(packet)
                self.assertFalse(hasattr(codec, "arena"))
                self.assertTrue(torch.equal(codec.packed[:packet.payload.numel()], packet.payload))
                for seg, start in enumerate(codec.starts.tolist()):
                    compact_base = ((start - 256) // 8) * 8
                    native_base = ((int(native_starts[seg]) - 256) // 8) * 8
                    length = int(packet.offsets[seg + 1] - packet.offsets[seg])
                    self.assertEqual(compact_base, int(packet.offsets[seg]))
                    self.assertEqual(start - compact_base, 256 + gap)
                    self.assertTrue(torch.equal(codec.packed[compact_base:compact_base + 256],
                                                arena[native_base:native_base + 256]))
                    self.assertTrue(torch.equal(codec.packed[start:compact_base + length],
                                                arena[int(native_starts[seg]):native_base + length]))
                self.assertEqual(packet.export(), wire)
                acc = codec.accounting(packet)
                self.assertEqual(acc["arena_bytes"], 0)
                self.assertEqual(acc["packet_bytes"], len(wire))
                self.assertEqual(acc["compact_staging_bytes"], codec.packed.numel() * 2)

    def test_reservation_growth_invalidates_capture_but_warm_staging_does_not(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            codec = TeamCompactCodec(td, "cpu", "dense_uniform")
            with self.assertRaises(RuntimeError):
                codec.reserve_payload_words(1)
            packet, _, _ = packet_fixture(codec.header((2, 64, 256)))
            codec.prepare(packet)
            old_pointer = codec.packed.data_ptr()
            sentinel = object(); codec.graph = sentinel
            codec.reserve_payload_words(packet.payload.numel())
            codec.stage(packet)
            self.assertIs(codec.graph, sentinel)
            self.assertEqual(codec.packed.data_ptr(), old_pointer)
            codec.reserve_payload_words(packet.payload.numel() + 8)
            self.assertIsNone(codec.graph)
            self.assertNotEqual(codec.packed.data_ptr(), old_pointer)
            S, cap = codec.native.layout(64)
            for bad in (0, -1, True, 1.5, S * cap + 1, 1 << 31):
                with self.assertRaises(ValueError):
                    codec.reserve_payload_words(bad)
            with self.assertRaises(RuntimeError):
                codec.graph_prepare(packet)

    def test_mutated_directory_and_wrong_state_rejected_before_staging(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            codec = TeamCompactCodec(td, "cpu", "dense_uniform")
            packet, _, _ = packet_fixture(codec.header((2, 64, 256)))
            codec.stage(packet)
            initial = codec.starts.clone()
            packet.relative_starts[0] += 1
            codec.stage(packet)
            self.assertEqual(int(codec.starts[0]), int(initial[0]) + 1)
            packet.offsets[1] += 1
            with self.assertRaises(ValueError):
                codec.stage(packet)
            packet.offsets[1] -= 1
            packet.relative_starts = packet.relative_starts.float()
            with self.assertRaises(ValueError):
                codec.stage(packet)
            packet.relative_starts = packet.relative_starts.to(torch.int32)
            packet.header["state_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                codec.stage(packet)

    def test_graph_decode_captures_after_valid_growth(self):
        # CPU control-flow regression: staging may invalidate an existing graph;
        # capture must happen afterwards, never replay a stale payload pointer.
        codec = object.__new__(TeamCompactCodec)
        codec.graph = object()
        calls = []
        def stage(packet):
            calls.append("stage"); codec.graph = None
        def prepare(packet):
            calls.append("capture"); codec.graph = object()
        with mock.patch.object(codec, "stage", side_effect=stage), \
             mock.patch.object(codec, "graph_prepare", side_effect=prepare), \
             mock.patch.object(codec, "graph_replay_native", side_effect=lambda: calls.append("replay") or torch.zeros(2, dtype=torch.int16)):
            codec.graph_decode(object())
        self.assertEqual(calls, ["stage", "capture", "replay"])


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU Triton interpreter required")
class InterpreterTests(unittest.TestCase):
    def test_native_exact_and_uniform_direct_compact_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            consts = fixed_state(layers=2, tokens=64)
            gen = torch.Generator().manual_seed(347)
            consts["Aq"][1] = torch.randint(-2, 3, (256, 256), generator=gen, dtype=torch.int8)
            consts["sa"].fill_(0.01)
            save_state(td, consts)
            words = torch.randn(2, 64, 256, generator=gen).bfloat16().view(torch.int16)
            for arm in ("dense_uniform", "dense_exact"):
                with self.subTest(arm=arm):
                    reference = TeamCodec(td, "cpu", arm)
                    direct = TeamCompactCodec(td, "cpu", arm)
                    packet = reference.encode(words)
                    held = direct.decode(packet)
                    self.assertTrue(torch.equal(held, reference.decode(packet)))
                    self.assertTrue(torch.equal(held, words))
                    if arm == "dense_uniform":
                        # Replaying into the shared native output must not alias
                        # a previously returned result, even on the CPU path.
                        direct.output.fill_(0)
                        self.assertTrue(torch.equal(held, words))
                        self.assertTrue(torch.equal(direct.decode(packet), words))
                        self.assertTrue(torch.equal(held, words))
                    self.assertFalse(hasattr(direct, "arena"))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1", "actual CUDA required")
class CudaTests(unittest.TestCase):
    def test_actual_cuda_direct_modes_alternating_owned_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            consts = fixed_state(layers=2, tokens=256)
            gen = torch.Generator().manual_seed(772)
            rank = dict(rank=256,
                Bq=torch.randint(-2, 3, (2, 256, 256), generator=gen, dtype=torch.int8),
                Uq=torch.randint(-2, 3, (2, 256, 256), generator=gen, dtype=torch.int8),
                m1=torch.full((2, 256), 0.01), m0=torch.full((2, 256), 0.5),
                sa=torch.full((2, 256), 0.01), b=torch.zeros(2, 256),
                sp=consts["sp"].clone(), inv=consts["inv"].clone())
            save_state(td, consts, rank)
            words = [torch.randn(2, 256, 256, generator=gen).bfloat16().view(torch.int16).to("cuda:0"),
                     torch.full((2, 256, 256), 5., dtype=torch.bfloat16, device="cuda:0").view(torch.int16)]
            for arm in ("dense_uniform", "dense_exact", "rank256_uniform"):
                with self.subTest(arm=arm):
                    reference = TeamCodec(td, "cuda:0", arm)
                    direct = TeamCompactCodec(td, "cuda:0", arm)
                    packets = [TeamPacket.import_bytes(reference.encode(w).export(), "cuda:0") for w in words]
                    direct.prepare(packets[0])
                    direct.reserve_payload_words(max(p.payload.numel() for p in packets))
                    direct.graph_prepare(packets[0])
                    held = direct.graph_decode(packets[0])
                    for i in (1, 0, 1):
                        self.assertTrue(torch.equal(direct.graph_decode(packets[i]), words[i]))
                        self.assertTrue(torch.equal(held, words[0]))
                    self.assertEqual(direct.graph_capture_count, 1)
                    self.assertEqual(direct.accounting(packets[0])["arena_bytes"], 0)
                    self.assertFalse(hasattr(direct, "arena"))
                    if arm == "dense_uniform":
                        small, large = sorted(range(2), key=lambda i: packets[i].payload.numel())
                        self.assertLess(packets[small].payload.numel(), packets[large].payload.numel())
                        growing = TeamCompactCodec(td, "cuda:0", arm)
                        growing.graph_prepare(packets[small])
                        old = growing.graph_decode(packets[small])
                        self.assertTrue(torch.equal(growing.graph_decode(packets[large]), words[large]))
                        self.assertTrue(torch.equal(old, words[small]))
                        self.assertEqual(growing.graph_capture_count, 2)

    def test_actual_cuda_all_bf16_patterns_direct(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state(layers=1, tokens=256))
            reference = TeamCodec(td, "cuda:0", "dense_uniform")
            direct = TeamCompactCodec(td, "cuda:0", "dense_uniform")
            words = torch.arange(65536, dtype=torch.int32, device="cuda:0").to(torch.int16).view(1, 256, 256)
            packet = reference.encode(words)
            direct.graph_prepare(packet)
            self.assertTrue(torch.equal(direct.graph_decode(packet), words))


if __name__ == "__main__":
    unittest.main()

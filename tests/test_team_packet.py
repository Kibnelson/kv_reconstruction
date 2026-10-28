"""Compact transport checks; CUDA tests exercise the unmodified native coder."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from kv_compare.team_packet import (ARMS, FORMAT, PREFIX, TeamCodec, TeamPacket,
    canonical, compact_segments, geometry, load_state, save_state, source_identity)


def fixed_state(layers=2, tokens=256):
    """Small deterministic predictor; channels keep the source's 128-wide heads."""
    L, W = layers, 256
    return dict(Aq=torch.zeros(L, W, W, dtype=torch.int8),
        sa=torch.ones(L, W), b=torch.zeros(L, W), sp=torch.ones(L, W),
        inv=torch.full((L, W), 4096 / (2 * np.log(2))),
        sx=torch.ones(L, W), isx=torch.ones(L, W),
        cos=torch.ones(tokens, 64), sin=torch.zeros(tokens, 64))


def header(tokens=64, layers=2):
    hd, W, TB, chunks = 128, 256, 32, 2
    return dict(format=FORMAT, arm="dense_uniform", shape=[layers, tokens, W], start=0,
        state_sha256="a" * 64, source_manifest_sha256="b" * 64,
        head_dim=hd, key_columns=W // 2, tokens_per_program=TB, chunks=chunks,
        segments=(tokens // TB) * (W // hd),
        segment_capacity_words=hd * (2 * layers * TB + 2) + 8)


def packet_fixture(h=None, gap=0, seed=1):
    h = header() if h is None else h
    _L, _T, _W, hd, S, cap = geometry(h)
    lengths = torch.arange(S, dtype=torch.int64) * 8 + 2 * hd + 32
    starts = torch.arange(1, S + 1, dtype=torch.int64) * cap - lengths + 2 * hd + gap
    arena = torch.arange(S * cap, dtype=torch.int64).to(torch.int16)
    p, o, r = compact_segments(arena, starts, segments=S, capacity_words=cap, head_dim=hd)
    return TeamPacket(h, p, o, r), arena, starts


class PacketTests(unittest.TestCase):
    def test_snapshot_source_identity(self):
        identity = source_identity()
        self.assertEqual(identity["commit"], "a7778073902bc2bb8bb0433ff7cd2e10bd79f5c3")
        self.assertEqual(len(identity["manifest_sha256"]), 64)

    def test_all_bf16_patterns_framed_without_changes(self):
        h = header(tokens=256, layers=1)
        S, cap = h["segments"], h["segment_capacity_words"]
        length = 65536 // S
        original = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        arena = torch.full((S * cap,), -999, dtype=torch.int16)
        starts = []
        for seg in range(S):
            base = (seg + 1) * cap - length
            arena[base:(seg + 1) * cap] = original[seg * length:(seg + 1) * length]
            starts.append(base + 256)
        payload, offsets, relative = compact_segments(arena, torch.tensor(starts),
            segments=S, capacity_words=cap, head_dim=128)
        packet = TeamPacket(h, payload, offsets, relative)
        restored = TeamPacket.import_bytes(packet.export())
        self.assertTrue(torch.equal(restored.payload, original))
        self.assertEqual(packet.export(), restored.export())
        self.assertEqual(len(packet.export()), packet.packet_bytes)
        self.assertEqual(packet.packet_bytes, packet.payload_bytes + packet.directory_bytes + PREFIX.size + len(canonical(h)) + 32)

    def test_alignment_padding_is_canonical_not_allocator_bytes(self):
        packet, arena, starts = packet_fixture(gap=3)
        hd, cap = 128, packet.header["segment_capacity_words"]
        arena2 = arena.clone()
        for i, st in enumerate(starts):
            base = ((int(st) - 2 * hd) // 8) * 8
            arena2[base + 2 * hd:int(st)] = 12345
        p, o, r = compact_segments(arena2, starts, segments=packet.header["segments"], capacity_words=cap, head_dim=hd)
        other = TeamPacket(packet.header, p, o, r)
        self.assertEqual(packet.export(), other.export())
        for i in range(packet.header["segments"]):
            begin = int(packet.offsets[i])
            self.assertTrue(torch.equal(packet.payload[begin + 256:begin + 259], torch.zeros(3, dtype=torch.int16)))

    def test_corrupt_truncated_and_extra_bytes_rejected(self):
        packet, _, _ = packet_fixture()
        wire = packet.export()
        for bad in (wire[:-1], wire + b"x", wire[:12], bytes([wire[0] ^ 1]) + wire[1:],
                    wire[:-33] + bytes([wire[-33] ^ 1]) + wire[-32:]):
            with self.assertRaises(ValueError):
                TeamPacket.import_bytes(bad)

    def test_directory_and_geometry_validation(self):
        p, _, _ = packet_fixture()
        bad_offsets = [p.offsets.clone() for _ in range(4)]
        bad_offsets[0][0] = -1
        bad_offsets[1][1] = 0
        bad_offsets[2][-1] += 8
        bad_offsets[3][1] += 1
        for o in bad_offsets:
            with self.assertRaises(ValueError):
                TeamPacket(p.header, p.payload, o, p.relative_starts)
        for rel in (255, 264, 1 << 32):
            r = p.relative_starts.to(torch.int64)
            r[0] = rel
            with self.assertRaises(ValueError):
                TeamPacket(p.header, p.payload, p.offsets, r)
        for key, value in (("start", 1), ("head_dim", 64), ("segments", 99), ("shape", [2, 63, 256]),
                           ("arm", "unknown")):
            with self.assertRaises(ValueError):
                TeamPacket(dict(p.header, **{key: value}), p.payload, p.offsets, p.relative_starts)
        with self.assertRaises(ValueError):
            TeamPacket(p.header, p.payload, p.offsets.float(), p.relative_starts)

    def test_state_roundtrip_and_identity_checks(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixed"
            consts = fixed_state()
            manifest = save_state(path, consts, metadata={"purpose": "software test"})
            loaded, rank, again = load_state(path)
            self.assertIsNone(rank)
            self.assertEqual(manifest, again)
            self.assertTrue(all(torch.equal(v, loaded[k]) for k, v in consts.items()))
            with self.assertRaises(FileExistsError):
                save_state(path, consts)
            archive = path / "tensors.npz"
            archive.write_bytes(archive.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                load_state(path)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_state(td)

    def test_interrupted_state_write_does_not_block_retry(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixed"
            with mock.patch("kv_compare.team_packet.np.savez", side_effect=OSError("simulated interrupted archive")):
                with self.assertRaises(OSError):
                    save_state(path, fixed_state())
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(td).iterdir()), [])
            save_state(path, fixed_state())
            self.assertEqual(load_state(path)[2]["metadata"]["format"], "team_onestep_state_v1")

    def test_cpu_stage_restores_only_used_suffixes(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            codec = TeamCodec(td, "cpu", "dense_uniform")
            packet, original, _ = packet_fixture(codec.header((2, 64, 256)), gap=2)
            codec.prepare(packet)
            codec.arena.fill_(-30000)
            codec.stage(packet)
            S, cap = codec.native.layout(64)
            for seg in range(S):
                a, b = int(packet.offsets[seg]), int(packet.offsets[seg + 1])
                base = (seg + 1) * cap - (b - a)
                self.assertTrue(torch.equal(codec.arena[base:(seg + 1) * cap], packet.payload[a:b]))
                self.assertTrue(bool((codec.arena[seg * cap:base] == -30000).all()))
                self.assertEqual(int(codec.starts[seg]), base + int(packet.relative_starts[seg]))
            self.assertEqual(codec.accounting(packet)["packet_bytes"], len(packet.export()))
            bad = TeamPacket(dict(packet.header, state_sha256="0" * 64), packet.payload, packet.offsets, packet.relative_starts)
            with self.assertRaises(ValueError):
                codec.stage(bad)
            with self.assertRaises(ValueError):
                codec.encode(torch.zeros(2, 64, 256, dtype=torch.int16), start=1)
            with self.assertRaises(ValueError):
                codec.encode(torch.zeros(2, 64, 256))
            with self.assertRaises(ValueError):
                codec.encode(torch.zeros(2, 63, 256, dtype=torch.int16))
            with self.assertRaises(RuntimeError):
                codec.graph_prepare(packet)


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU Triton interpreter required")
class InterpreterTests(unittest.TestCase):
    def test_native_exact_and_uniform_real_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            consts = fixed_state(layers=2, tokens=64)
            gen = torch.Generator().manual_seed(119)
            consts["Aq"][1] = torch.randint(-2, 3, (256, 256), generator=gen, dtype=torch.int8)
            consts["sa"].fill_(0.01)
            save_state(td, consts)
            words = torch.randn(2, 64, 256, generator=gen).bfloat16().view(torch.int16)
            for arm in ("dense_uniform", "dense_exact"):
                with self.subTest(arm=arm):
                    codec = TeamCodec(td, "cpu", arm)
                    packet = codec.encode(words)
                    restored = TeamPacket.import_bytes(packet.export())
                    self.assertTrue(torch.equal(codec.decode(restored), words))
                    S, cap = codec.native.layout(64)
                    lengths = packet.offsets[1:] - packet.offsets[:-1]
                    starts = (torch.arange(S, dtype=torch.int64) + 1) * cap - lengths + packet.relative_starts
                    self.assertEqual(codec.native.bits(starts, 64) // 8,
                                     codec.accounting(packet)["native_used_stream_bytes"])


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1", "actual CUDA required")
class CudaTests(unittest.TestCase):
    def test_actual_cuda_all_bf16_patterns(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state(layers=1))
            codec = TeamCodec(td, "cuda:0", "dense_uniform")
            words = torch.arange(65536, dtype=torch.int32, device="cuda:0").to(torch.int16).view(1, 256, 256)
            packet = codec.encode(words)
            restored = TeamPacket.import_bytes(packet.export(), "cuda:0")
            self.assertTrue(torch.equal(codec.decode(restored), words))

    def test_actual_cuda_modes_graph_alternating_owned_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            gen = torch.Generator().manual_seed(901)
            first = torch.randn(2, 64, 256, generator=gen).bfloat16().view(torch.int16).to("cuda:0")
            second = torch.full((2, 64, 256), 5., dtype=torch.bfloat16, device="cuda:0").view(torch.int16)
            for arm in ("dense_exact", "dense_uniform"):
                codec = TeamCodec(td, "cuda:0", arm)
                packets = [TeamPacket.import_bytes(codec.encode(w).export(), "cuda:0") for w in (first, second)]
                self.assertNotEqual(packets[0].payload.numel(), packets[1].payload.numel())
                codec.graph_prepare(packets[0])
                held = codec.graph_decode(packets[0])
                for i in (1, 0, 1):
                    self.assertTrue(torch.equal(codec.graph_decode(packets[i]), (first, second)[i]))
                    self.assertTrue(torch.equal(held, first))
                self.assertEqual(codec.graph_capture_count, 1)

    def test_actual_cuda_rank256_epilogue(self):
        with tempfile.TemporaryDirectory() as td:
            consts = fixed_state(layers=2, tokens=256)
            gen = torch.Generator().manual_seed(221)
            rank = dict(rank=256,
                Bq=torch.randint(-2, 3, (2, 256, 256), generator=gen, dtype=torch.int8),
                Uq=torch.randint(-2, 3, (2, 256, 256), generator=gen, dtype=torch.int8),
                m1=torch.full((2, 256), 0.01), m0=torch.full((2, 256), 0.5),
                sa=torch.full((2, 256), 0.01), b=torch.zeros(2, 256),
                sp=consts["sp"].clone(), inv=consts["inv"].clone())
            save_state(td, consts, rank)
            codec = TeamCodec(td, "cuda:0", "rank256_uniform")
            words = torch.randn(2, 256, 256, generator=gen).bfloat16().view(torch.int16).to("cuda:0")
            packet = TeamPacket.import_bytes(codec.encode(words).export(), "cuda:0")
            codec.graph_prepare(packet)
            self.assertTrue(torch.equal(codec.graph_decode(packet), words))


if __name__ == "__main__":
    unittest.main()

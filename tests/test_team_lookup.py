"""CDF inverse equivalence and owned-output, borrowed-input decode contracts."""
import gc as python_gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from kv_compare.team_lookup import TeamLookupCodec, cdf_inverse_proof
from kv_compare.team_packet import TeamCodec, TeamPacket, save_state
from tests.test_team_packet import fixed_state, packet_fixture


VARIANTS = (("graph", False), ("graph", True), ("direct", False), ("direct", True))


class LookupTests(unittest.TestCase):
    def test_exhaustive_all_cdf_slots_and_independent_proof(self):
        proof = cdf_inverse_proof()
        self.assertTrue(proof["verified"])
        self.assertEqual((proof["classes"], proof["slots_per_class"], proof["checked_slots"]),
                         (10, 65536, 655360))
        self.assertEqual(proof["reference_sha256"], proof["lookup_sha256"])
        self.assertEqual(set(proof["table_sha256"]), {"cum", "off", "K", "cumf", "ko", "lut", "kc", "tail"})
        proof["table_sha256"]["lut"]["sha256"] = "changed"
        self.assertNotEqual(cdf_inverse_proof()["table_sha256"]["lut"]["sha256"], "changed")

    def test_options_preserve_original_wire_metadata_and_fixed_tables(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            original = TeamCodec(td, "cpu", "dense_uniform")
            for mode, lookup in VARIANTS:
                codec = TeamLookupCodec(td, "cpu", "dense_uniform", decoder_mode=mode, lookup=lookup)
                self.assertEqual(codec.header((2, 64, 256)), original.header((2, 64, 256)))
                self.assertTrue(codec.header((2, 64, 256))["search"])
                self.assertEqual(codec.native.search, not lookup)
                self.assertTrue(codec.inverse_proof["native_tables_verified"])
                for name, value in original.native.tables.items():
                    self.assertTrue(torch.equal(codec.native.tables[name], value))
            for kw in ({"decoder_mode": "other"}, {"lookup": 1}):
                with self.assertRaises(ValueError):
                    TeamLookupCodec(td, "cpu", "dense_uniform", **kw)

    def test_direct_stages_only_current_directory_and_rejects_graph_operations(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            codec = TeamLookupCodec(td, "cpu", "dense_uniform", decoder_mode="direct", lookup=True)
            for gap in range(8):
                packet, _, _ = packet_fixture(codec.header((2, 64, 256)), gap=gap)
                codec.stage(packet)
                self.assertTrue(torch.equal(codec.starts.long(), packet.offsets[:-1] + packet.relative_starts))
                self.assertFalse(hasattr(codec, "packed"))
                self.assertFalse(hasattr(codec, "output"))
                self.assertIsNone(codec.graph)
                self.assertEqual(codec.graph_capture_count, 0)
                acc = codec.accounting(packet)
                for key in ("arena_bytes", "compact_staging_bytes", "persistent_output_bytes", "output_clone_bytes_per_call"):
                    self.assertEqual(acc[key], 0)
                self.assertTrue(acc["borrowed_payload"])
            for fn in (lambda:codec.reserve_payload_words(1024), lambda:codec.graph_prepare(packet),
                       lambda:codec.graph_decode(packet), codec.graph_replay_native, codec.native_decode):
                with self.assertRaises(RuntimeError):
                    fn()

    def test_direct_returns_new_owned_output_and_borrows_exact_input_tensor(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            codec = TeamLookupCodec(td, "cpu", "dense_uniform", decoder_mode="direct")
            packet, _, _ = packet_fixture(codec.header((2, 64, 256)))
            saved = packet.payload.clone()
            def native(buf, starts, tokens, *, out, bufs):
                self.assertIs(buf, packet.payload)
                out.fill_(int(buf[0]))
                return out
            with mock.patch.object(codec.native, "decode", side_effect=native):
                held = codec.decode(packet)
                again = codec.decode(packet)
            self.assertNotEqual(held.data_ptr(), again.data_ptr())
            self.assertNotEqual(held.data_ptr(), packet.payload.data_ptr())
            again.zero_()
            self.assertTrue(bool((held == int(saved[0])).all()))
            self.assertTrue(torch.equal(packet.payload, saved))
            packet.offsets[1] += 1
            with mock.patch.object(codec.native, "decode") as native:
                with self.assertRaises(ValueError):
                    codec.decode(packet)
                native.assert_not_called()

    def test_changed_caller_stream_waits_before_workspace_reuse(self):
        # Exercise the ordering rule without requiring a local CUDA device.
        shape = (2, 64, 256)
        for mode in ("graph", "direct"):
            codec = object.__new__(TeamLookupCodec)
            codec.decoder_mode, codec.device, codec.shape = mode, torch.device("cuda:0"), shape
            previous = mock.Mock(cuda_stream=7)
            current = mock.Mock(cuda_stream=8)
            codec._workspace_last_stream = previous
            with mock.patch("kv_compare.team_lookup.torch.cuda.current_stream", return_value=current), \
                 mock.patch.object(codec, "validate", return_value=shape), \
                 mock.patch("kv_compare.team_lookup.TeamCompactCodec.prepare", return_value=codec) as graph_prepare:
                codec.prepare(object())
                current.wait_stream.assert_called_once_with(previous)
                self.assertIs(codec._workspace_last_stream, current)
                codec.prepare(object())
                current.wait_stream.assert_called_once_with(previous)
                self.assertEqual(graph_prepare.call_count, 2 if mode == "graph" else 0)

    def test_unaligned_borrowed_view_is_rejected_without_hidden_copy(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            direct = TeamLookupCodec(td, "cpu", "dense_uniform", decoder_mode="direct")
            packet, _, _ = packet_fixture(direct.header((2, 64, 256)))
            allocation = torch.empty(packet.payload.numel() + 1, dtype=torch.int16)
            payload = allocation[1:]
            payload.copy_(packet.payload)
            view = TeamPacket(packet.header, payload, packet.offsets, packet.relative_starts)
            self.assertNotEqual(view.payload.data_ptr() % 16, 0)
            with mock.patch.object(direct.native, "decode") as native:
                with self.assertRaises(ValueError):
                    direct.decode(view)
                native.assert_not_called()
            staged = TeamLookupCodec(td, "cpu", "dense_uniform", decoder_mode="graph")
            staged.stage(view)
            self.assertTrue(torch.equal(staged.packed[:payload.numel()], packet.payload))


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU Triton interpreter required")
class InterpreterTests(unittest.TestCase):
    def test_real_native_lookup_search_parity_uniform_and_exact(self):
        with tempfile.TemporaryDirectory() as td:
            consts = fixed_state(layers=2, tokens=64)
            generator = torch.Generator().manual_seed(347)
            consts["Aq"][1] = torch.randint(-2, 3, (256, 256), generator=generator, dtype=torch.int8)
            consts["sa"].fill_(0.01)
            save_state(td, consts)
            words = torch.randn(2, 64, 256, generator=generator).bfloat16().view(torch.int16)
            for arm in ("dense_uniform", "dense_exact"):
                with self.subTest(arm=arm):
                    source = TeamCodec(td, "cpu", arm)
                    packet = source.encode(words)
                    wire = packet.export()
                    search = TeamLookupCodec(td, "cpu", arm, decoder_mode="direct", lookup=False)
                    lookup = TeamLookupCodec(td, "cpu", arm, decoder_mode="direct", lookup=True)
                    expected = search.decode(packet)
                    actual = lookup.decode(packet)
                    self.assertTrue(torch.equal(expected, words))
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(packet.export(), wire)
                    self.assertFalse(hasattr(lookup, "packed"))
                    self.assertFalse(hasattr(lookup, "output"))
                    # A subsequent actual decoder call must not overwrite an
                    # earlier return, even while its workspace is reused.
                    if arm == "dense_uniform":
                        actual.zero_()
                        self.assertTrue(torch.equal(expected, words))
                        graph_kernel = TeamLookupCodec(td, "cpu", arm, decoder_mode="graph", lookup=True)
                        self.assertTrue(torch.equal(graph_kernel.kernel_decode(packet), words))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1", "actual CUDA required")
class CudaTests(unittest.TestCase):
    def test_actual_cuda_variants_switch_caller_streams(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state(layers=2, tokens=256))
            generator = torch.Generator().manual_seed(531)
            words = [torch.randn(2, 256, 256, generator=generator).bfloat16().view(torch.int16).to("cuda:0"),
                     torch.full((2, 256, 256), 5., dtype=torch.bfloat16, device="cuda:0").view(torch.int16)]
            source = TeamCodec(td, "cuda:0", "dense_uniform")
            packets = [source.encode(w) for w in words]
            first, second = torch.cuda.Stream(), torch.cuda.Stream()
            for mode, lookup in VARIANTS:
                codec = TeamLookupCodec(td, "cuda:0", "dense_uniform", decoder_mode=mode, lookup=lookup)
                largest = max(packets, key=lambda p:p.payload.numel())
                codec.prepare(largest)  # Workspace initialization on default.
                if mode == "graph":
                    codec.graph_prepare(largest)
                with torch.cuda.stream(first):
                    held = codec.decode(packets[0])
                with torch.cuda.stream(second):
                    other = codec.decode(packets[1])
                with torch.cuda.stream(first):
                    again = codec.decode(packets[0])
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(held, words[0]))
                self.assertTrue(torch.equal(other, words[1]))
                self.assertTrue(torch.equal(again, words[0]))
                self.assertEqual(codec.graph_capture_count, int(mode == "graph"))

    def test_actual_cuda_all_modes_variants_alternating_owned_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            consts = fixed_state(layers=2, tokens=256)
            generator = torch.Generator().manual_seed(772)
            rank = dict(rank=256,
                Bq=torch.randint(-2, 3, (2, 256, 256), generator=generator, dtype=torch.int8),
                Uq=torch.randint(-2, 3, (2, 256, 256), generator=generator, dtype=torch.int8),
                m1=torch.full((2, 256), 0.01), m0=torch.full((2, 256), 0.5),
                sa=torch.full((2, 256), 0.01), b=torch.zeros(2, 256),
                sp=consts["sp"].clone(), inv=consts["inv"].clone())
            save_state(td, consts, rank)
            words = [torch.randn(2, 256, 256, generator=generator).bfloat16().view(torch.int16).to("cuda:0"),
                     torch.full((2, 256, 256), 5., dtype=torch.bfloat16, device="cuda:0").view(torch.int16)]
            for arm in ("dense_uniform", "dense_exact", "rank256_uniform"):
                source = TeamCodec(td, "cuda:0", arm)
                packets = [source.encode(w) for w in words]
                initial = [p.payload.clone() for p in packets]
                for mode, lookup in VARIANTS:
                    with self.subTest(arm=arm, mode=mode, lookup=lookup):
                        codec = TeamLookupCodec(td, "cuda:0", arm, decoder_mode=mode, lookup=lookup)
                        largest = max(packets, key=lambda p:p.payload.numel())
                        codec.prepare(largest)
                        if mode == "graph":
                            codec.reserve_payload_words(largest.payload.numel())
                            codec.graph_prepare(largest)
                        held = codec.decode(packets[0])
                        for i in (1, 0):
                            self.assertTrue(torch.equal(codec.decode(packets[i]), words[i]))
                            self.assertTrue(torch.equal(held, words[0]))
                            self.assertTrue(torch.equal(packets[i].payload, initial[i]))
                        self.assertEqual(codec.graph_capture_count, int(mode == "graph"))
                        if mode == "direct":
                            self.assertFalse(hasattr(codec, "output"))
                            self.assertFalse(hasattr(codec, "packed"))

    def test_actual_cuda_all_bf16_patterns_and_borrow_lifetime(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state(layers=1, tokens=256))
            source = TeamCodec(td, "cuda:0", "dense_uniform")
            words = torch.arange(65536, dtype=torch.int32, device="cuda:0").to(torch.int16).view(1, 256, 256)
            wire = source.encode(words).export()
            for mode, lookup in VARIANTS:
                codec = TeamLookupCodec(td, "cuda:0", "dense_uniform", decoder_mode=mode, lookup=lookup)
                caller = torch.cuda.Stream()
                with torch.cuda.stream(caller):
                    packet = TeamPacket.import_bytes(wire, "cuda:0")
                    count = packet.payload.numel()
                    result = codec.decode(packet)
                    del packet
                python_gc.collect()
                # Exercise unrelated allocation pressure before waiting on the
                # caller. Direct decode must keep borrowed input reads alive.
                trash = [torch.empty(count, dtype=torch.int16, device="cuda:0").fill_(-1) for _ in range(4)]
                torch.cuda.current_stream().wait_stream(caller)
                self.assertTrue(torch.equal(result, words))
                del trash


if __name__ == "__main__":
    unittest.main()

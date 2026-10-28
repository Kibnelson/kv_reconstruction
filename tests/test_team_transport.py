"""Wire parity, validation and ownership for the new transport-only functions."""
import hashlib
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch

from kv_compare.team_packet import TeamCodec, TeamPacket, MAGIC, PREFIX, canonical, save_state
from kv_compare.team_transport import fast_export, fast_import, reference_export, reference_import
from tests.test_team_packet import fixed_state, header, packet_fixture


def all_words_packet(*, odd=False):
    h = header(tokens=256, layers=1)
    if odd:
        h["transport_test_tag"] = "x"
        if len(canonical(h)) % 2 == 0:
            h["transport_test_tag"] += "x"
    S = h["segments"]
    payload = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    offsets = torch.arange(S + 1, dtype=torch.int64) * (65536 // S)
    starts = torch.full((S,), 256, dtype=torch.int32)
    return TeamPacket(h, payload, offsets, starts)


def reframe(blob, *, header_bytes=None, directory=None):
    magic, nh, nd, npayload = PREFIX.unpack_from(blob)
    h = blob[PREFIX.size:PREFIX.size + nh] if header_bytes is None else header_bytes
    d0 = PREFIX.size + nh
    d = blob[d0:d0 + nd] if directory is None else directory
    payload = blob[d0 + nd:d0 + nd + npayload]
    data = PREFIX.pack(magic, len(h), len(d), len(payload)) + h + d + payload
    return data + hashlib.sha256(data).digest()


class TransportTests(unittest.TestCase):
    def test_all_bf16_wire_parity_and_profile_modes(self):
        packet = all_words_packet()
        expected = packet.export()
        for export in (reference_export, fast_export):
            for profile in (False, True):
                with self.subTest(export=export.__name__, profile=profile):
                    wire, record = export(packet, profile=profile)
                    self.assertIsInstance(wire, bytes)
                    self.assertEqual(wire, expected)
                    self.assertIs(record["profiled"], profile)
                    if profile:
                        self.assertEqual(record["wire_bytes"], len(wire))
                        self.assertIn("checksum", record["phases"])
                        self.assertGreaterEqual(record["total_wall_ms"], 0.)
        for restore in (reference_import, fast_import):
            for profile in (False, True):
                with self.subTest(restore=restore.__name__, profile=profile):
                    packet2, record = restore(expected, profile=profile)
                    self.assertTrue(torch.equal(packet2.payload, packet.payload))
                    self.assertEqual(packet2.export(), expected)
                    self.assertIs(record["profiled"], profile)

    def test_odd_byte_offset_payload_is_preserved_and_owned(self):
        packet = all_words_packet(odd=True)
        wire, _ = fast_export(packet)
        _, nh, nd, npayload = PREFIX.unpack_from(wire)
        offset = PREFIX.size + nh + nd
        self.assertEqual(offset % 2, 1)
        source = np.frombuffer(wire, dtype="<i2", offset=offset, count=npayload // 2)
        restored, _ = fast_import(wire)
        self.assertTrue(torch.equal(restored.payload, packet.payload))
        self.assertNotEqual(restored.payload.data_ptr(), source.ctypes.data)
        self.assertIsNone(restored.payload._base)
        restored.payload.fill_(123)
        self.assertEqual(wire, packet.export())

    def test_mutable_and_readonly_mutable_views_are_snapshotted(self):
        packet, _, _ = packet_fixture(gap=3)
        expected = packet.export()
        for readonly_view in (False, True):
            backing = bytearray(expected)
            argument = memoryview(backing).toreadonly() if readonly_view else backing
            restored, record = fast_import(argument, profile=True)
            backing[:] = b"\xff" * len(backing)
            self.assertTrue(record["mutable_input_snapshot"])
            self.assertEqual(restored.export(), expected)
            self.assertTrue(torch.equal(restored.offsets, packet.offsets))
            self.assertTrue(torch.equal(restored.relative_starts, packet.relative_starts))

    def test_export_returns_owned_immutable_bytes(self):
        packet, _, _ = packet_fixture()
        wire, _ = fast_export(packet)
        expected = bytes(bytearray(wire))
        packet.payload.zero_()
        packet.offsets.zero_()
        self.assertEqual(wire, expected)

    def test_input_lifetime_and_scoped_warning_filter(self):
        packet, _, _ = packet_fixture()
        expected = packet.payload.clone()
        filters = list(warnings.filters)
        # Both immutable temporary bytes and a temporary read-only view may die
        # immediately after this call; the returned CPU tensor must own storage.
        restored, _ = fast_import(memoryview(bytearray(packet.export())).toreadonly())
        gc.collect()
        self.assertTrue(torch.equal(restored.payload, expected))
        self.assertEqual(warnings.filters, filters)
        restored.payload.fill_(-123)
        self.assertTrue(torch.equal(packet.payload, expected))

    def test_corruption_and_validation_match_frozen_import(self):
        packet, _, _ = packet_fixture()
        wire = packet.export()
        nh, nd = PREFIX.unpack_from(wire)[1:3]
        directory = bytearray(wire[PREFIX.size + nh:PREFIX.size + nh + nd])
        bad_directory = []
        for offset, value in ((0, -1), (8, 0), (8, 257)):
            changed = directory.copy()
            changed[offset:offset + 8] = int(value).to_bytes(8, "little", signed=True)
            bad_directory.append(reframe(wire, directory=changed))
        invalid_header = dict(packet.header, start=1)
        bad = [wire[:-1], wire + b"x", wire[:17], bytes([wire[0] ^ 1]) + wire[1:],
               wire[:-33] + bytes([wire[-33] ^ 1]) + wire[-32:],
               reframe(wire, header_bytes=json.dumps(packet.header).encode()),
               reframe(wire, header_bytes=canonical(invalid_header)), *bad_directory]
        for blob in bad:
            for restore in (reference_import, fast_import):
                for profile in (False, True):
                    with self.subTest(restore=restore.__name__, profile=profile, length=len(blob)):
                        with self.assertRaises(ValueError):
                            restore(blob, profile=profile)

    def test_invalid_directory_is_rejected_before_gpu_allocation(self):
        packet, _, _ = packet_fixture()
        wire = packet.export()
        _, nh, nd, _ = PREFIX.unpack_from(wire)
        directory = bytearray(wire[PREFIX.size + nh:PREFIX.size + nh + nd])
        directory[:8] = (-1).to_bytes(8, "little", signed=True)
        bad = reframe(wire, directory=directory)
        # Even on this CPU-only worker, ValueError must precede any CUDA setup.
        with self.assertRaisesRegex(ValueError, "invalid compact offsets"):
            fast_import(bad, device="cuda:0")

    def test_uninstrumented_path_has_no_timers_or_cuda_events(self):
        packet, _, _ = packet_fixture()
        wire = packet.export()
        with mock.patch("kv_compare.team_transport.time.perf_counter", side_effect=AssertionError("timer in ordinary path")), \
             mock.patch("torch.cuda.Event", side_effect=AssertionError("event in ordinary path")):
            for export in (reference_export, fast_export):
                self.assertEqual(export(packet, cuda_events=True)[0], wire)
            for restore in (reference_import, fast_import):
                self.assertTrue(torch.equal(restore(wire, cuda_events=True)[0].payload, packet.payload))


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU Triton interpreter required")
class InterpreterTests(unittest.TestCase):
    def test_low_copy_packet_restores_through_real_native_decoder(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state(layers=2, tokens=64))
            codec = TeamCodec(td, "cpu", "dense_uniform")
            words = torch.randn(2, 64, 256, generator=torch.Generator().manual_seed(992)).bfloat16().view(torch.int16)
            packet = codec.encode(words)
            wire, _ = fast_export(packet)
            self.assertEqual(wire, packet.export())
            receiver, _ = fast_import(wire)
            self.assertTrue(torch.equal(codec.decode(receiver), words))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1", "actual CUDA required")
class CudaTests(unittest.TestCase):
    def test_actual_cuda_upload_export_parity_and_ownership(self):
        cpu = all_words_packet(odd=True)
        wire = cpu.export()
        gpu = cpu.to("cuda:0")
        for profile in (False, True):
            exported, record = fast_export(gpu, profile=profile, cuda_events=True)
            self.assertEqual(exported, wire)
            source = bytearray(wire)
            restored, receipt = fast_import(source, "cuda:0", profile=profile, cuda_events=True)
            source[:] = b"\x00" * len(source)
            self.assertEqual(restored.payload.device.type, "cuda")
            self.assertTrue(torch.equal(restored.payload, gpu.payload))
            if profile:
                self.assertIn("cuda_event_ms", record["phases"]["D2H"])
                self.assertIn("cuda_event_ms", receipt["phases"]["H2D"])
            restored.payload.zero_()
            self.assertTrue(torch.equal(gpu.payload.cpu(), cpu.payload))


if __name__ == "__main__":
    unittest.main()

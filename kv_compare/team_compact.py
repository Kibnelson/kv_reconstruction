"""Decode unchanged one-step packets directly from compact GPU storage.

The native decoder addresses a segment only through its initial read pointer.
For an eight-word-aligned compact offset ``o`` and relative pointer ``r`` in
256..263, its state address ``floor((o+r-256)/8)*8`` is exactly ``o``.
Translating each initial pointer therefore preserves every native read while
removing the expanded fixed-capacity arena. No predictor or entropy arithmetic,
packet bytes, or codec state changes here.

The adapter owns stable staging storage and returns independently owned BF16
words. Reserve the largest packet before graph preparation when alternating
packets. Growing that reservation or changing shape invalidates the old graph;
the subsequent public decode captures a new graph outside the native replay.
"""
from __future__ import annotations

import torch

from .team_packet import TeamCodec, _tensor_bytes, geometry, triton, tl


if triton is not None:
    @triton.jit
    def _compact_starts_kernel(offsets, relative, starts, S: tl.constexpr,
                               BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        o = tl.load(offsets + i, mask=i < S, other=0)
        r = tl.load(relative + i, mask=i < S, other=0)
        tl.store(starts + i, (o + r).to(tl.int32), mask=i < S)


class TeamCompactCodec(TeamCodec):
    """Same wire and native decoder, with no expanded receiver arena.

    The public API requires sequential calls like ``TeamCodec``; the unchanged
    native decoder keeps its internal chunk streams. The reservation is explicit
    reusable scratch, not compressed resident KV.
    """

    def validate(self, packet):
        shape = super().validate(packet)
        if (packet.offsets.device.type != "cpu" or packet.offsets.dtype != torch.int64 or
                packet.relative_starts.device.type != "cpu" or packet.relative_starts.dtype != torch.int32 or
                not packet.offsets.is_contiguous() or not packet.relative_starts.is_contiguous()):
            raise ValueError("compact packet directory storage differs from checked frame")
        return shape

    def _invalidate_graph(self):
        if self.graph is not None and self.device.type == "cuda":
            # A cold shape/capacity transition may release addresses referenced
            # by a preceding asynchronous replay. Warm calls never enter here.
            torch.cuda.synchronize(self.device)
        self.graph = None
        self.graph_cold_wall_ms = None

    def prepare(self, packet):
        shape = self.validate(packet)
        if shape != self.shape:
            self._invalidate_graph()
            self.shape = shape
            _L, T, _W = shape
            S, _cap = self.native.layout(T)
            self.starts = torch.empty(S, dtype=torch.int32, device=self.device)
            self.output = torch.empty(shape, dtype=torch.int16, device=self.device)
            self.buffers = self.native.buffers(T)
            self.offsets_device = torch.empty(S + 1, dtype=torch.int64, device=self.device)
            self.relative_device = torch.empty(S, dtype=torch.int32, device=self.device)
            # Do not retain a previous geometry's oversized payload allocation.
            self.packed = torch.empty(0, dtype=torch.int16, device=self.device)
        self.reserve_payload_words(packet.payload.numel())
        return self

    def reserve_payload_words(self, words):
        """Reserve stable scratch after ``prepare`` and before graph capture.

        A larger valid reservation invalidates capture instead of replaying a
        graph containing the old payload address. Smaller requests never shrink
        storage. Bounds follow the original packet's signed-int32 geometry.
        """
        if self.shape is None:
            raise RuntimeError("prepare a checked packet before reserving compact storage")
        S, cap = self.native.layout(self.shape[1])
        if type(words) is not int or words <= 0 or words > S * cap or words >= 1 << 31:
            raise ValueError("compact reservation exceeds checked indexed geometry")
        if words > self.packed.numel():
            self._invalidate_graph()
            self.packed = torch.empty(words, dtype=torch.int16, device=self.device)
        return self

    def stage(self, packet):
        self.prepare(packet)
        _L, _T, _W, _hd, S, _cap = geometry(packet.header)
        # Packet validation runs on each call: neither mutable directories nor
        # a different incoming packet may reuse previously calculated pointers.
        self.packed[:packet.payload.numel()].copy_(packet.payload)
        self.offsets_device.copy_(packet.offsets)
        self.relative_device.copy_(packet.relative_starts)
        if self.device.type == "cuda":
            _compact_starts_kernel[(triton.cdiv(S, 256),)](
                self.offsets_device, self.relative_device, self.starts,
                S=S, BLOCK=256, num_warps=4)
        else:
            self.starts.copy_((self.offsets_device[:-1] + self.relative_device).to(torch.int32))
        return self

    def native_decode(self):
        return self.native.decode(self.packed, self.starts, self.shape[1],
                                  out=self.output, bufs=self.buffers)

    def graph_decode(self, packet):
        # Stage first: valid incoming growth can invalidate a previously live
        # graph even when shape is unchanged. Capture only stable addresses.
        self.stage(packet)
        if self.graph is None:
            self.graph_prepare(packet)
        return self.graph_replay_native().clone()

    def accounting(self, packet):
        packet.validate()
        _L, _T, _W, _hd, S, cap = geometry(packet.header)
        return dict(native_used_stream_bytes=packet.payload_bytes + 4 * S,
            compact_payload_bytes=packet.payload_bytes,
            directory_bytes=packet.directory_bytes,
            header_checksum_bytes=packet.packet_bytes - packet.payload_bytes - packet.directory_bytes,
            packet_bytes=packet.packet_bytes, arena_bytes=0,
            avoided_expanded_arena_bytes=S * cap * 2,
            output_bytes=packet.header["shape"][0] * packet.header["shape"][1] * packet.header["shape"][2] * 2,
            shared_predictor_bytes=self.shared_bytes,
            coder_table_bytes=self.coder_table_bytes,
            decoder_workspace_tensor_bytes=_tensor_bytes(getattr(self, "buffers", {})),
            compact_staging_bytes=_tensor_bytes(getattr(self, "packed", None)),
            directory_staging_bytes=_tensor_bytes(getattr(self, "offsets_device", None)) +
                _tensor_bytes(getattr(self, "relative_device", None)),
            native_starts_bytes=_tensor_bytes(getattr(self, "starts", None)),
            decode_storage="direct compact payload; unchanged native arithmetic")

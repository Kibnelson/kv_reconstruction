"""Byte-compatible low-copy transport for the frozen TOS1 packet format.

The native codec, entropy streams and ``TeamPacket`` are unchanged. Ordinary
calls do not create timers or CUDA events. ``profile=True`` is a separate,
instrumented diagnostic; its phase sums are not ordinary receiver latency.

Immutable input bytes are borrowed only while parsing or during a blocking GPU
upload. Returned CPU payloads have an owned copy; returned GPU payloads own GPU
storage. No writable Torch tensor that aliases immutable bytes is returned.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings

import numpy as np
import torch

from .team_packet import MAGIC, PREFIX, TeamPacket, canonical, geometry


class _Phases:
    def __init__(self, implementation, direction, enabled, cuda_events):
        self.enabled = bool(enabled)
        self.cuda_events = bool(cuda_events) and self.enabled
        self.record = dict(implementation=implementation, direction=direction,
                           profiled=self.enabled)
        if self.enabled:
            self.record.update(phases={}, cuda_events_requested=self.cuda_events,
                scope="separate instrumented diagnostic; includes Python dispatch/allocation; do not add to ordinary timings")
            self.begin = time.perf_counter()

    def run(self, name, fn, cuda_device=None):
        if not self.enabled:
            return fn()
        events = None
        if self.cuda_events and cuda_device is not None:
            cuda_device = torch.device(cuda_device)
            if cuda_device.type == "cuda":
                torch.cuda.synchronize(cuda_device)
                with torch.cuda.device(cuda_device):
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record(torch.cuda.current_stream(cuda_device))
        begin = time.perf_counter()
        value = fn()
        if events is not None:
            events[1].record(torch.cuda.current_stream(cuda_device))
            events[1].synchronize()
        phase = dict(wall_ms=(time.perf_counter() - begin) * 1000)
        if events is not None:
            phase.update(cuda_event_ms=events[0].elapsed_time(events[1]),
                         cuda_interval_scope="copy call including host dispatch/allocation gaps")
        self.record["phases"][name] = phase
        return value

    def finish(self, wire_bytes, payload_bytes, **details):
        if self.enabled:
            self.record.update(total_wall_ms=(time.perf_counter() - self.begin) * 1000,
                               wire_bytes=wire_bytes, payload_bytes=payload_bytes, **details)
        return self.record


def _snapshot(blob):
    # bytes(existing_bytes) is itself, but spelling this out documents the
    # normal no-copy path and the one required mutable-input ownership copy.
    return blob if isinstance(blob, bytes) else bytes(blob)


def _little_view(array, dtype):
    """Contiguous byte view; only an actual endian mismatch requires a copy."""
    expected = np.dtype(dtype)
    if array.dtype != expected or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=expected)
    return memoryview(array).cast("B")


def _checksum_parts(parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def reference_export(packet, *, profile=False, cuda_events=False):
    """Frozen export, optionally split into diagnostic phases without changing bytes."""
    if not profile:
        return packet.export(), dict(implementation="reference", direction="export", profiled=False)
    p = _Phases("reference", "export", True, cuda_events)
    p.run("packet_validation", packet.validate)
    head = p.run("header_encode", lambda: canonical(packet.header))
    directory = p.run("directory_encode", lambda:
        packet.offsets.numpy().astype("<i8").tobytes() + packet.relative_starts.numpy().astype("<i4").tobytes())
    cpu = p.run("D2H", lambda: packet.payload.detach().contiguous().cpu(),
                packet.payload.device if packet.payload.is_cuda else None)
    array = p.run("dtype_layout_copy", lambda: cpu.numpy().astype("<i2"))
    payload = p.run("payload_tobytes", array.tobytes)
    data = p.run("framing_assembly", lambda:
        PREFIX.pack(MAGIC, len(head), len(directory), len(payload)) + head + directory + payload)
    checksum = p.run("checksum", lambda: hashlib.sha256(data).digest())
    wire = p.run("checksum_append", lambda: data + checksum)
    return wire, p.finish(len(wire), packet.payload_bytes,
        ownership="immutable owned bytes", mutable_input_snapshot=False,
        payload_sized_host_operations=["D2H if CUDA", "astype copy", "tobytes copy", "frame assembly copy", "checksum append copy"],
        allocation_scope="D2H includes host tensor allocation; astype/tobytes/frame assembly each allocate; phases are not additive ordinary timings")


def fast_export(packet, *, profile=False, cuda_events=False):
    """Stream SHA over views and assemble the exact immutable TOS1 frame once."""
    p = _Phases("low_copy", "export", profile, cuda_events)
    p.run("packet_validation", packet.validate)
    head = p.run("header_encode", lambda: canonical(packet.header))
    cpu = p.run("D2H", lambda: packet.payload.detach().contiguous().cpu(),
                packet.payload.device if packet.payload.is_cuda else None)
    payload = p.run("dtype_layout_copy", lambda: _little_view(cpu.numpy(), "<i2"))
    directories = p.run("directory_views", lambda: (
        _little_view(packet.offsets.numpy(), "<i8"), _little_view(packet.relative_starts.numpy(), "<i4")))
    prefix = PREFIX.pack(MAGIC, len(head), sum(v.nbytes for v in directories), payload.nbytes)
    parts = (prefix, head, *directories, payload)
    checksum = p.run("checksum", lambda: _checksum_parts(parts))
    wire = p.run("framing_assembly", lambda: b"".join((*parts, checksum)))
    return wire, p.finish(len(wire), packet.payload_bytes,
        ownership="immutable owned bytes", mutable_input_snapshot=False,
        payload_sized_host_operations=["D2H if CUDA", "one final frame assembly copy"],
        dtype_layout_copy_performed=(cpu.numpy().dtype != np.dtype("<i2") or not cpu.numpy().flags.c_contiguous),
        allocation_scope="D2H includes host tensor allocation; final join allocates one owned immutable frame; no pinned workspace")


def _frame_lengths(blob):
    if len(blob) < PREFIX.size + 32:
        raise ValueError("truncated one-step packet")
    magic, nh, nd, npayload = PREFIX.unpack_from(blob)
    if magic != MAGIC or nh > (1 << 20) or npayload % 2 or PREFIX.size + nh + nd + npayload + 32 != len(blob):
        raise ValueError("invalid one-step frame lengths")
    return nh, nd, npayload


def _header(header_bytes, nd):
    header = json.loads(header_bytes)
    if canonical(header) != header_bytes:
        raise ValueError("noncanonical one-step header")
    _L, _T, _W, _hd, S, _cap = geometry(header)
    if nd != 8 * (S + 1) + 4 * S:
        raise ValueError("one-step directory byte count differs")
    return header, S


def reference_import(blob, device="cpu", *, profile=False, cuda_events=False):
    """Frozen import; diagnostic copies and validation follow its original order."""
    if not profile:
        return TeamPacket.import_bytes(blob, device), dict(implementation="reference", direction="import", profiled=False)
    p = _Phases("reference", "import", True, cuda_events)
    mutable = not isinstance(blob, bytes)
    blob = p.run("snapshot", lambda: bytes(blob))
    nh, nd, npayload = p.run("frame_parse", lambda: _frame_lengths(blob))
    checksum_input = p.run("checksum_input_copy", lambda: blob[:-32])
    checksum = p.run("checksum", lambda: hashlib.sha256(checksum_input).digest())
    if checksum != blob[-32:]:
        raise ValueError("one-step packet checksum differs")
    pos = PREFIX.size
    header, S = p.run("header_parse", lambda: _header(blob[pos:pos + nh], nd))
    pos += nh

    def directories():
        offsets = torch.from_numpy(np.frombuffer(blob[pos:pos + 8 * (S + 1)], dtype="<i8").copy())
        start = pos + 8 * (S + 1)
        relative = torch.from_numpy(np.frombuffer(blob[start:start + 4 * S], dtype="<i4").copy())
        return offsets, relative

    offsets, starts = p.run("directory_parse", directories)
    pos += nd
    payload_bytes = p.run("payload_bytes_slice", lambda: blob[pos:pos + npayload])
    payload = p.run("payload_cpu_copy", lambda: torch.from_numpy(np.frombuffer(payload_bytes, dtype="<i2").copy()))
    d = torch.device(device)
    payload = p.run("H2D", lambda: payload.to(d), d if d.type == "cuda" else None)
    packet = p.run("packet_validation", lambda: TeamPacket(header, payload, offsets, starts))
    return packet, p.finish(len(blob), npayload,
        ownership="owned CPU payload copy or owned GPU upload; owned directory tensors",
        mutable_input_snapshot=mutable,
        payload_sized_host_operations=["mutable input snapshot if needed", "checksum input slice", "payload bytes slice", "NumPy owned copy", "H2D if CUDA"],
        allocation_scope="payload_cpu_copy allocates owned host storage; H2D includes GPU allocation")


def _readonly_tensor(array):
    # Torch does not enforce NumPy's read-only flag. This borrowed tensor is
    # private, used solely as a copy source, and never escapes this module.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*", category=UserWarning)
        return torch.from_numpy(array)


def fast_import(blob, device="cpu", *, profile=False, cuda_events=False):
    """Validate views and copy the payload only into its returned owned storage."""
    p = _Phases("low_copy", "import", profile, cuda_events)
    mutable = not isinstance(blob, bytes)
    blob = p.run("snapshot", lambda: _snapshot(blob))
    view = memoryview(blob)
    nh, nd, npayload = p.run("frame_parse", lambda: _frame_lengths(view))
    checksum = p.run("checksum", lambda: hashlib.sha256(view[:-32]).digest())
    if checksum != view[-32:]:
        raise ValueError("one-step packet checksum differs")
    pos = PREFIX.size
    # JSON/header copies remain small; the payload stays a view throughout.
    header, S = p.run("header_parse", lambda: _header(bytes(view[pos:pos + nh]), nd))
    pos += nh

    def directories():
        offsets = torch.from_numpy(np.frombuffer(view, dtype="<i8", count=S + 1, offset=pos).copy())
        starts = torch.from_numpy(np.frombuffer(view, dtype="<i4", count=S, offset=pos + 8 * (S + 1)).copy())
        return offsets, starts

    offsets, starts = p.run("directory_parse", directories)
    pos += nd
    borrowed = p.run("payload_view", lambda:
        _readonly_tensor(np.frombuffer(view, dtype="<i2", count=npayload // 2, offset=pos)))
    # Validate the directory with the frozen validator BEFORE any GPU upload.
    # This temporary packet's private borrowed payload is replaced with owned
    # storage below; no read-only alias is returned to a caller.
    packet = p.run("packet_validation", lambda: TeamPacket(header, borrowed, offsets, starts))
    d = torch.device(device)
    if d.type == "cpu":
        owned = p.run("payload_cpu_copy", borrowed.clone)
    else:
        # Blocking copy is intentional: the immutable host owner may be released
        # after this function returns. No asynchronous host-lifetime promise.
        owned = p.run("H2D", lambda: borrowed.to(d, non_blocking=False), d if d.type == "cuda" else None)
    packet.payload = owned
    return packet, p.finish(len(blob), npayload,
        ownership="owned CPU clone or owned blocking GPU upload; owned directory tensors",
        mutable_input_snapshot=mutable, host_payload_owner_retained_until_upload_complete=True,
        payload_sized_host_operations=["mutable input snapshot if needed", "one owned CPU clone if CPU destination", "blocking H2D if CUDA"],
        allocation_scope="CPU clone or H2D allocates returned owned payload; no pinned workspace")

"""Wire formats of the PD protocol.

A transfer is one header (JSON, a few kB) and one blob: the codec's tensors laid end to end, each 16-byte aligned,
as raw bytes on the sender's GPU. The header names each tensor (dtype, shape, offset) and carries the codec's
scalars, so the receiver rebuilds views of the same tensors on its own device with no copy.

Our codec writes each stream segment backwards from the end of a fixed-capacity slot; only its tail (states,
alignment and data: exactly what LayerwiseOneStep.bits counts) is sent. compact_segments packs those tails into a
dense array (one kernel on the GPU; elsewhere a few hundred slots at a time so the boolean mask stays small). The
receiver decodes the dense array as it is, with compact_starts; expand_segments (its inverse) is kept for tests.
"""
from __future__ import annotations

import json

import torch

try:                                                 # the GPU packing kernel; without Triton, the torch loop
    import triton
    import triton.language as tl
except Exception:                                    # pragma: no cover
    triton = None

ALIGN = 16
DTYPES = {str(d): d for d in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint16,
                               torch.bfloat16, torch.float16, torch.float32)}


def _nbytes(t):
    return t.numel() * t.element_size()


def bundle(parts, meta=None):
    """parts {name: tensor} (all on one device) -> (header dict, blob uint8 tensor on that device)."""
    names = list(parts)
    if not names:
        raise ValueError("an empty bundle")
    dev = parts[names[0]].device
    entries, off = [], 0
    for n in names:
        t = parts[n]
        if t.device != dev:
            raise ValueError(f"part {n} is on {t.device}, not {dev}")
        if str(t.dtype) not in DTYPES:
            raise ValueError(f"part {n}: dtype {t.dtype} not supported on the wire")
        entries.append({"name": n, "dtype": str(t.dtype), "shape": list(t.shape), "offset": off,
                        "nbytes": _nbytes(t)})
        off += -(-_nbytes(t) // ALIGN) * ALIGN
    blob = torch.zeros(max(off, ALIGN), dtype=torch.uint8, device=dev)
    for e, n in zip(entries, names):
        if e["nbytes"]:
            blob[e["offset"]:e["offset"] + e["nbytes"]] = parts[n].contiguous().reshape(-1).view(torch.uint8)
    return {"parts": entries, "meta": dict(meta or {}), "blob_bytes": int(blob.numel())}, blob


def unbundle(header, blob):
    """(header, blob) -> ({name: tensor view into blob}, meta)."""
    out = {}
    for e in header["parts"]:
        dt = DTYPES[e["dtype"]]
        raw = blob[e["offset"]:e["offset"] + e["nbytes"]]
        t = raw.view(dt) if e["nbytes"] else torch.empty(0, dtype=dt, device=blob.device)
        out[e["name"]] = t.reshape(e["shape"])
    return out, header.get("meta", {})


def payload_bytes(header):
    """The bytes a bundle's tensors hold (without alignment padding or the header)."""
    return sum(e["nbytes"] for e in header["parts"])


def header_to_bytes(header):
    return json.dumps(header, separators=(",", ":")).encode()


def header_from_bytes(b):
    return json.loads(bytes(b).decode())


# ------------------------------------------------------------------------------------------- our stream segments
def segment_bases(starts, cap, hd):
    """First used word of every segment (LayerwiseOneStep.bits: the states sit 2 x hd words before the start,
    rounded down to 8 words); segment s ends at (s + 1) x cap."""
    s = starts.to(torch.int64)
    return torch.div(s - 2 * hd, 8, rounding_mode="floor") * 8


def segment_lengths(starts, cap, hd):
    S = starts.numel()
    ends = (torch.arange(S, device=starts.device, dtype=torch.int64) + 1) * cap
    return ends - segment_bases(starts, cap, hd)


if triton is not None:
    @triton.jit
    def _tail_copy(src_ptr, dst_ptr, base_ptr, off_ptr, len_ptr, BLOCK: tl.constexpr):
        """Segment s's used words src[base_s : base_s + len_s] to dst[off_s : off_s + len_s] (one program each)."""
        s = tl.program_id(0)
        b = tl.load(base_ptr + s)
        o = tl.load(off_ptr + s)
        n = tl.load(len_ptr + s)
        for i in range(0, n, BLOCK):
            j = i + tl.arange(0, BLOCK)
            m = j < n
            tl.store(dst_ptr + o + j, tl.load(src_ptr + b + j, mask=m), mask=m)


def compact_starts(starts, cap, hd):
    """(starts in compact_segments' dense array, its length): the decoder reads a segment's states 2 x hd words
    below its start (rounded down to 8 words: the segment's first used word) and its words upward from the start,
    reading exactly the words the encoder wrote; the dense array keeps each tail in one piece 8-aligned, so the
    same distances hold there and the receiver decodes it without the S x cap slot buffer (twice the request's KV)
    or the copy into it."""
    bases = segment_bases(starts, cap, hd)
    lens = (torch.arange(starts.numel(), device=starts.device, dtype=torch.int64) + 1) * cap - bases
    offs = torch.cumsum(lens, 0) - lens
    return (offs + starts.to(torch.int64) - bases).to(torch.int32), int(lens.sum())


def compact_segments(buf, starts, cap, hd, rows=256, kernel=None):
    """The used tail of every segment, in segment order: buf int16 [S x cap] -> int16 [sum of lengths].
    kernel: one Triton launch for all segments (default on the GPU) instead of the torch loop."""
    S = starts.numel()
    lens = segment_lengths(starts, cap, hd)
    lo, hi, total = (int(x) for x in torch.stack([lens.min(), lens.max(), lens.sum()]).cpu().tolist())
    if lo < 0 or hi > cap:
        raise ValueError("segment starts outside their slots")
    out = torch.empty(total, dtype=buf.dtype, device=buf.device)
    if kernel is None:
        kernel = buf.is_cuda and triton is not None
    if kernel:
        offs = torch.cumsum(lens, 0) - lens
        _tail_copy[(S,)](buf, out, segment_bases(starts, cap, hd), offs, lens, BLOCK=1024)
        return out
    b2 = buf.view(S, cap)
    col = torch.arange(cap, device=buf.device)
    o = 0
    for a in range(0, S, rows):
        b = min(a + rows, S)
        mask = col[None, :] >= (cap - lens[a:b])[:, None]
        piece = b2[a:b][mask]
        out[o:o + piece.numel()] = piece
        o += piece.numel()
    return out


def expand_segments(packed, starts, cap, hd, out=None, rows=256):
    """Inverse of compact_segments: the tails back at the end of their slots (the rest left as it was)."""
    S = starts.numel()
    lens = segment_lengths(starts, cap, hd)
    if int(lens.sum()) != packed.numel():
        raise ValueError(f"{packed.numel()} words for segments that use {int(lens.sum())}")
    if out is None:
        out = torch.zeros(S * cap, dtype=packed.dtype, device=packed.device)
    b2 = out.view(S, cap)
    col = torch.arange(cap, device=packed.device)
    o = 0
    for a in range(0, S, rows):
        b = min(a + rows, S)
        mask = col[None, :] >= (cap - lens[a:b])[:, None]
        n = int(lens[a:b].sum())
        sub = b2[a:b]
        sub[mask] = packed[o:o + n]
        o += n
    return out


# --------------------------------------------------------------------------------------------------- checksums
def word_checksum(words):
    """Three 64-bit sums of int16 words [L, T, W] (any device): plain, channel-weighted and token-weighted, per
    layer and in all. Cheap on the GPU; sender and receiver compare them to prove the transfer bit-exact."""
    L, T, W = words.shape
    wc = torch.arange(1, W + 1, device=words.device, dtype=torch.int64)
    wt = torch.arange(1, T + 1, device=words.device, dtype=torch.int64)
    acc = []
    for l in range(L):
        x = words[l].to(torch.int64)
        acc.append(torch.stack([x.sum(), (x * wc[None, :]).sum(), (x.sum(1) * wt).sum()]))
    return torch.stack(acc).cpu().tolist()


class Checksum:
    """word_checksum accumulated layer by layer (the receiver decodes into a two-layer ring)."""

    def __init__(self):
        self.rows = []

    def add(self, words_l):
        T, W = words_l.shape
        wc = torch.arange(1, W + 1, device=words_l.device, dtype=torch.int64)
        wt = torch.arange(1, T + 1, device=words_l.device, dtype=torch.int64)
        x = words_l.to(torch.int64)
        self.rows.append(torch.stack([x.sum(), (x * wc[None, :]).sum(), (x.sum(1) * wt).sum()]))

    def value(self):
        return torch.stack(self.rows).cpu().tolist() if self.rows else []

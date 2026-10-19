"""Exact, field-aware BF16 coding with independent exponent entropy coding.

Sign and seven fraction bits remain one verbatim byte per original value.
Only the eight exponent bits receive group packing and/or Deflate. These are
reversible bit transforms, including every NaN payload and signed zero.
The caller makes independently addressable temporal tiles; this matrix format
has no dependency on another tile and requires no trained dictionary.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

import numpy as np


_HEADER = struct.Struct("<4sBBHIIII")
_MAGIC, _VERSION = b"KVEF", 1
_RAW, _EXP_RAW, _EXP_PACKED, _EXP_DEFLATE, _PACKED_DEFLATE = range(5)
MODE_NAMES = {0: "raw_words", 1: "raw_exponents", 2: "packed_exponents",
              3: "deflate_exponents", 4: "deflate_packed_exponents"}
STRATEGIES = ("raw", "packed", "deflate", "packed_deflate", "best")


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def _pack(values, width):
    if width == 0:
        return b""
    a = np.asarray(values, dtype=np.uint16).reshape(-1)
    bits = ((a[:, None] >> np.arange(width, dtype=np.uint16)) & 1)
    return np.packbits(bits.reshape(-1).astype(np.uint8), bitorder="little").tobytes()


def _unpack(payload, count, width):
    if not 0 <= width <= 8 or len(payload) != (count * width + 7) // 8:
        raise ValueError("invalid exponent bit stream")
    if not width:
        return np.zeros(count, dtype=np.uint16)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    bits = bits[:count * width].reshape(count, width).astype(np.uint16)
    return np.sum(bits << np.arange(width, dtype=np.uint16), axis=1).astype(np.uint16)


def _pack_exponents(exponents, group_size):
    payload = bytearray()
    exponents = np.asarray(exponents, dtype=np.uint16)
    for start in range(0, len(exponents), group_size):
        group = exponents[start:start + group_size]
        minimum = int(group.min())
        width = (int(group.max()) - minimum).bit_length()
        payload.extend((minimum, width))
        payload.extend(_pack(group - minimum, width))
    return bytes(payload)


def _unpack_exponents(payload, count, group_size):
    out = np.empty(count, dtype=np.uint16)
    offset = 0
    for start in range(0, count, group_size):
        if offset + 2 > len(payload):
            raise ValueError("truncated exponent group header")
        minimum, width = payload[offset:offset + 2]
        size = min(group_size, count - start)
        nbytes = (size * width + 7) // 8
        offset += 2
        if width > 8 or offset + nbytes > len(payload):
            raise ValueError("invalid exponent group")
        values = _unpack(payload[offset:offset + nbytes], size, width) + minimum
        if np.any(values > 255):
            raise ValueError("exponent outside byte range")
        out[start:start + size] = values
        offset += nbytes
    if offset != len(payload):
        raise ValueError("trailing exponent data")
    return out


def _inflate(payload, maximum):
    dec = zlib.decompressobj()
    result = dec.decompress(payload, maximum + 1)
    if len(result) > maximum or not dec.eof or dec.unused_data or dec.unconsumed_tail:
        raise ValueError("invalid or oversized Deflate stream")
    return result


def encode_matrix(words, *, strategy="best", level=6, group_size=64):
    """Return actual complete bytes and accounting; raw fallback is mandatory.

    `packed_deflate` literally stacks group exponent packing then Deflate,
    while `deflate` entropy-codes original exponent bytes. `best` permits both
    and records only the smallest candidate, with deterministic tie breaking.
    Shape and groups are local to this record. All payloads are byte-aligned.
    """
    words = np.asarray(words)
    if words.ndim != 2 or words.dtype != np.uint16:
        raise ValueError("expected uint16 matrix")
    if strategy not in STRATEGIES or not 0 <= level <= 9 or not 1 <= group_size <= 65535:
        raise ValueError("invalid codec configuration")
    rows, cols = words.shape
    if max(rows, cols, words.size) > 0xFFFFFFFF:
        raise ValueError("matrix too large")
    flat = np.ascontiguousarray(words).reshape(-1)
    n = flat.size
    raw = flat.astype("<u2").tobytes()
    exponent = ((flat >> 7) & 255).astype(np.uint8).tobytes()
    lower = (((flat >> 8) & 128) | (flat & 127)).astype(np.uint8).tobytes()
    candidates = [(_RAW, raw)]
    packed = None
    if strategy in ("packed", "packed_deflate", "best"):
        packed = _pack_exponents(np.frombuffer(exponent, dtype=np.uint8), group_size)
    if strategy == "raw":
        pass
    elif strategy == "packed":
        candidates.append((_EXP_PACKED, lower + packed))
    elif strategy == "deflate":
        candidates.extend([(_EXP_RAW, lower + exponent),
                           (_EXP_DEFLATE, lower + zlib.compress(exponent, level))])
    elif strategy == "packed_deflate":
        candidates.extend([(_EXP_PACKED, lower + packed),
                           (_PACKED_DEFLATE, lower + zlib.compress(packed, level))])
    else:
        candidates.extend([(_EXP_RAW, lower + exponent), (_EXP_PACKED, lower + packed),
                           (_EXP_DEFLATE, lower + zlib.compress(exponent, level)),
                           (_PACKED_DEFLATE, lower + zlib.compress(packed, level))])
    mode, payload = min(candidates, key=lambda x: (len(x[1]), x[0]))
    exp_len = len(payload) - n if mode else 0
    header = _HEADER.pack(_MAGIC, _VERSION, mode, group_size, rows, cols, exp_len,
                          zlib.crc32(raw))
    data = header + payload
    components = {"header": len(header), "raw_words": len(payload) if mode == _RAW else 0,
                  "sign_fraction": n if mode != _RAW else 0,
                  "exponent_payload": exp_len}
    assert sum(components.values()) == len(data)
    return EncodedMatrix(data, {"serialized_bytes": len(data), "raw_bytes": len(raw),
                               "mode": MODE_NAMES[mode], "mode_id": mode,
                               "strategy": strategy, "level": level, "group_size": group_size,
                               "component_bytes": components,
                               "candidate_payload_bytes": {MODE_NAMES[m]: len(p) for m, p in candidates}})


def decode_matrix(data):
    if isinstance(data, EncodedMatrix):
        data = data.data
    data = bytes(data)
    if len(data) < _HEADER.size:
        raise ValueError("truncated field record")
    magic, version, mode, group_size, rows, cols, exp_len, checksum = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION or mode not in MODE_NAMES or not group_size:
        raise ValueError("invalid field record header")
    n = rows * cols
    payload = data[_HEADER.size:]
    if mode == _RAW:
        if exp_len or len(payload) != 2 * n:
            raise ValueError("invalid raw payload")
        out = np.frombuffer(payload, dtype="<u2").copy()
    else:
        if len(payload) != n + exp_len:
            raise ValueError("invalid field payload length")
        lower = np.frombuffer(payload[:n], dtype=np.uint8).astype(np.uint16)
        exp = payload[n:]
        if mode == _EXP_DEFLATE:
            exp = _inflate(exp, n)
        elif mode == _PACKED_DEFLATE:
            exp = _inflate(exp, n + 2 * ((n + group_size - 1) // group_size))
        if mode in (_EXP_RAW, _EXP_DEFLATE):
            if len(exp) != n:
                raise ValueError("incorrect exponent count")
            exp = np.frombuffer(exp, dtype=np.uint8).astype(np.uint16)
        else:
            exp = _unpack_exponents(exp, n, group_size)
        out = ((lower & 128) << 8) | (lower & 127) | (exp << 7)
    if zlib.crc32(out.astype("<u2").tobytes()) != checksum:
        raise ValueError("field record checksum mismatch")
    return out.reshape(rows, cols)

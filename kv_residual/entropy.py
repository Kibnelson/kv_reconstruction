"""Self-contained exact signed-integer correction records.

Only NumPy and the standard library are used. Every record contains its shape,
codec mode/parameter, payload length and checksum. Strategies expose genuinely
different lossless residual coders; `best` keeps only the smallest byte stream.
All strategies permit an explicit signed-byte/word/int32 raw fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

import numpy as np


_HEADER = struct.Struct("<4sBBBBIIII")
_MAGIC, _VERSION = b"KRIN", 1
_RAW32, _RAW16, _RAW8, _RICE, _ZDEFLATE, _CDEFLATE, _CPACK, _ZPACK = range(8)
MODES = {0: "raw_int32", 1: "raw_int16", 2: "raw_int8", 3: "zigzag_rice",
         4: "zigzag_byteplanes_deflate", 5: "centered_packed_deflate",
         6: "centered_packed", 7: "zigzag_packed"}
STRATEGIES = ("raw", "rice", "zigzag_deflate", "centered_deflate", "centered_packed",
              "zigzag_packed", "best")


@dataclass(frozen=True)
class Encoded:
    data: bytes
    stats: dict


def _pack(values, width):
    if not 0 <= width <= 32:
        raise ValueError("invalid integer width")
    if not width:
        return b""
    a = np.asarray(values, dtype=np.uint64).reshape(-1)
    bits = ((a[:, None] >> np.arange(width - 1, -1, -1, dtype=np.uint64)) & 1)
    return np.packbits(bits.reshape(-1).astype(np.uint8), bitorder="big").tobytes()


def _unpack(payload, count, width):
    if not 0 <= width <= 32 or len(payload) != (count * width + 7) // 8:
        raise ValueError("invalid packed integer stream")
    if not width:
        return np.zeros(count, dtype=np.uint64)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if np.any(bits[count * width:]):
        raise ValueError("nonzero packed padding")
    bits = bits[:count * width].reshape(count, width).astype(np.uint64)
    return np.sum(bits << np.arange(width - 1, -1, -1, dtype=np.uint64), axis=1)


def _zigzag(values):
    x = np.asarray(values, dtype=np.int64)
    return ((x << 1) ^ (x >> 31)).astype(np.uint64)


def _unzigzag(values):
    z = np.asarray(values, dtype=np.uint64)
    return ((z >> 1).astype(np.int64) ^ -(z & 1).astype(np.int64))


def _rice_encode(z):
    n = len(z)
    if not n:
        return 0, 0, b""
    costs = [n * (k + 1) + int((z >> np.uint64(k)).sum(dtype=np.uint64)) for k in range(32)]
    k = min(range(32), key=lambda i: (costs[i], i))
    q = (z >> np.uint64(k)).astype(np.int64)
    lengths = q + 1 + k
    total = int(lengths.sum())
    starts = np.cumsum(np.r_[np.int64(0), lengths[:-1]])
    bits = np.zeros(total, dtype=np.uint8)
    bits[starts + q] = 1
    if k:
        positions = starts[:, None] + q[:, None] + 1 + np.arange(k)[None, :]
        bits[positions] = ((z[:, None] >> np.arange(k - 1, -1, -1, dtype=np.uint64)) & 1)
    return k, total, np.packbits(bits, bitorder="big").tobytes()


def _rice_decode(payload, count, k, total):
    if not 0 <= k <= 31 or total < count * (k + 1) or len(payload) != (total + 7) // 8:
        raise ValueError("invalid Rice payload")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if np.any(bits[total:]):
        raise ValueError("nonzero Rice padding")
    out = np.empty(count, dtype=np.uint64)
    offset = 0
    for i in range(count):
        q = 0
        while offset < total and bits[offset] == 0:
            offset += 1
            q += 1
        if offset >= total or offset + 1 + k > total:
            raise ValueError("truncated Rice value")
        offset += 1
        low = 0
        for bit in bits[offset:offset + k]:
            low = (low << 1) | int(bit)
        offset += k
        value = (q << k) | low
        if value > 0xFFFFFFFF:
            raise ValueError("Rice value outside int32 zigzag range")
        out[i] = value
    if offset != total:
        raise ValueError("trailing Rice bits")
    return out


def _inflate(payload, expected):
    try:
        decoder = zlib.decompressobj()
        out = decoder.decompress(payload, expected + 1)
    except zlib.error as exc:
        raise ValueError("corrupt Deflate payload") from exc
    if len(out) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("incorrect Deflate payload size")
    return out


def encode(values, strategy="best", *, level=6):
    """Encode an int32 array exactly; all selectable methods keep raw fallback."""
    values = np.asarray(values)
    if values.dtype != np.int32 or values.ndim > 255:
        raise ValueError("expected int32 array with at most255dimensions")
    if strategy not in STRATEGIES or not 0 <= level <= 9:
        raise ValueError("invalid correction codec configuration")
    if values.size > 0xFFFFFFFF or any(s > 0xFFFFFFFF for s in values.shape):
        raise ValueError("correction array too large")
    flat = np.ascontiguousarray(values).reshape(-1)
    n = flat.size
    raw = flat.astype("<i4").tobytes()
    minimum = int(flat.min()) if n else 0
    maximum = int(flat.max()) if n else 0
    candidates = [(_RAW32, 0, 0, raw, 0)]
    if minimum >= -32768 and maximum <= 32767:
        candidates.append((_RAW16, 0, 0, flat.astype("<i2").tobytes(), 0))
    if minimum >= -128 and maximum <= 127:
        candidates.append((_RAW8, 0, 0, flat.astype("i1").tobytes(), 0))
    zigzag = _zigzag(flat)
    if strategy in ("rice", "best"):
        k, total, body = _rice_encode(zigzag)
        candidates.append((_RICE, k, total, body, 0))
    if strategy in ("zigzag_packed", "best"):
        width = int(zigzag.max()).bit_length() if n else 0
        candidates.append((_ZPACK, width, 0, _pack(zigzag, width), 0))
    if strategy in ("zigzag_deflate", "best"):
        nbytes = max(1, ((int(zigzag.max()).bit_length() if n else 0) + 7) // 8)
        planes = ((zigzag[None, :] >> (8 * np.arange(nbytes, dtype=np.uint64))[:, None]) & 255).astype(np.uint8)
        candidates.append((_ZDEFLATE, nbytes, 0, zlib.compress(planes.tobytes(), level), 0))
    if strategy in ("centered_packed", "centered_deflate", "best"):
        centered = (flat.astype(np.int64) - minimum).astype(np.uint64)
        width = (maximum - minimum).bit_length()
        packed = _pack(centered, width)
        prefix = struct.pack("<i", minimum)
        if strategy in ("centered_packed", "best"):
            candidates.append((_CPACK, width, 0, prefix + packed, 4))
        if strategy in ("centered_deflate", "best"):
            candidates.append((_CDEFLATE, width, 0, prefix + zlib.compress(packed, level), 4))
    mode, parameter, auxiliary, payload, nparams = min(candidates, key=lambda row: (len(row[3]), row[0]))
    shape = struct.pack("<" + "I" * values.ndim, *values.shape)
    header = _HEADER.pack(_MAGIC, _VERSION, mode, parameter, values.ndim,
                          n, len(payload), auxiliary, zlib.crc32(raw))
    data = header + shape + payload
    components = {"header": len(header), "shape": len(shape), "parameters": nparams,
                  "correction_payload": len(payload) - nparams}
    assert sum(components.values()) == len(data)
    return Encoded(data, {"strategy": strategy, "mode": MODES[mode], "parameter": parameter,
                          "serialized_bytes": len(data), "raw_int32_bytes": len(raw),
                          "value_count": n, "component_bytes": components,
                          "candidate_payload_bytes": {MODES[m]: len(p) for m, _, _, p, _ in candidates}})


def decode(data):
    if isinstance(data, Encoded):
        data = data.data
    data = bytes(data)
    if len(data) < _HEADER.size:
        raise ValueError("truncated correction record")
    magic, version, mode, parameter, ndim, n, length, auxiliary, checksum = _HEADER.unpack_from(data)
    head = _HEADER.size + 4 * ndim
    if magic != _MAGIC or version != _VERSION or mode not in MODES or len(data) != head + length:
        raise ValueError("invalid correction record header")
    shape = struct.unpack_from("<" + "I" * ndim, data, _HEADER.size)
    product = 1
    for d in shape:
        product *= d
    if product != n:
        raise ValueError("shape/count mismatch")
    payload = data[head:]
    if mode in (_RAW32, _RAW16, _RAW8):
        dtype = {_RAW32: "<i4", _RAW16: "<i2", _RAW8: "i1"}[mode]
        if parameter or auxiliary or len(payload) != n * np.dtype(dtype).itemsize:
            raise ValueError("invalid raw correction payload")
        out = np.frombuffer(payload, dtype=dtype).astype(np.int64)
    elif mode == _RICE:
        out = _unzigzag(_rice_decode(payload, n, parameter, auxiliary))
    elif mode == _ZPACK:
        if auxiliary:
            raise ValueError("unexpected packed auxiliary bits")
        out = _unzigzag(_unpack(payload, n, parameter))
    elif mode == _ZDEFLATE:
        if not 1 <= parameter <= 4 or auxiliary:
            raise ValueError("invalid byteplane width")
        planes = np.frombuffer(_inflate(payload, n * parameter), dtype=np.uint8).reshape(parameter, n)
        z = np.sum(planes.astype(np.uint64) << (8 * np.arange(parameter, dtype=np.uint64))[:, None], axis=0)
        out = _unzigzag(z)
    else:
        if len(payload) < 4 or auxiliary or not 0 <= parameter <= 32:
            raise ValueError("invalid centered correction payload")
        minimum = struct.unpack_from("<i", payload)[0]
        packed = payload[4:]
        if mode == _CDEFLATE:
            packed = _inflate(packed, (n * parameter + 7) // 8)
        out = _unpack(packed, n, parameter).astype(np.int64) + minimum
    if np.any(out < -2147483648) or np.any(out > 2147483647):
        raise ValueError("decoded correction outside int32 range")
    out = out.astype(np.int32)
    if zlib.crc32(out.astype("<i4").tobytes()) != checksum:
        raise ValueError("correction record checksum mismatch")
    return out.reshape(shape)

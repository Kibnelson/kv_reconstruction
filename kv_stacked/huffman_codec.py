"""Exact BF16 field codec with a local canonical Huffman exponent table.

This control preserves sign/fraction bits verbatim and entropy-codes exponent
bytes only. Every record includes its own sparse (symbol, code-length) table,
shape, payload length and checksum. There are no external trained tables.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import struct
import zlib

import numpy as np


_HEADER = struct.Struct("<4sBBHIIII")
_MAGIC, _VERSION = b"KVHU", 1


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def _lengths(symbols):
    frequencies = np.bincount(symbols, minlength=256)
    heap = [(int(n), int(s), int(s)) for s, n in enumerate(frequencies) if n]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][1]: 0}
    while len(heap) > 1:
        na, sa, a = heapq.heappop(heap)
        nb, sb, b = heapq.heappop(heap)
        heapq.heappush(heap, (na + nb, min(sa, sb), (a, b)))
    result = {}
    if not heap:
        return result
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            result[node] = depth
        else:
            stack.extend([(node[0], depth + 1), (node[1], depth + 1)])
    return result


def _codes(lengths):
    if len(lengths) == 1 and next(iter(lengths.values())) == 0:
        return {next(iter(lengths)): (0, 0)}
    if not lengths or any(not 1 <= width <= 255 for width in lengths.values()):
        raise ValueError("invalid Huffman code lengths")
    result = {}
    code = previous = 0
    for symbol, width in sorted(lengths.items(), key=lambda x: (x[1], x[0])):
        code <<= width - previous
        if code >= 1 << width:
            raise ValueError("oversubscribed Huffman table")
        result[symbol] = (code, width)
        code += 1
        previous = width
    return result


def _encode_symbols(symbols, codes):
    output = bytearray()
    accumulator = nbits = 0
    for symbol in symbols:
        code, width = codes[int(symbol)]
        accumulator = (accumulator << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            output.append((accumulator >> nbits) & 255)
            accumulator &= (1 << nbits) - 1
    if nbits:
        output.append(accumulator << (8 - nbits))
    return bytes(output)


def _decode_symbols(payload, count, codes):
    if len(codes) == 1 and next(iter(codes.values()))[1] == 0:
        if payload:
            raise ValueError("payload for constant symbol")
        return np.full(count, next(iter(codes)), dtype=np.uint16)
    # A local prefix tree avoids any dependence on symbols from another tile.
    root = {}
    for symbol, (code, width) in codes.items():
        node = root
        for shift in range(width - 1, -1, -1):
            if "symbol" in node:
                raise ValueError("invalid prefix table")
            node = node.setdefault((code >> shift) & 1, {})
        if node:
            raise ValueError("invalid prefix table")
        node["symbol"] = symbol
    out = np.empty(count, dtype=np.uint16)
    bitindex = 0
    for i in range(count):
        node = root
        while "symbol" not in node:
            if bitindex >= len(payload) * 8:
                raise ValueError("truncated Huffman bit stream")
            bit = (payload[bitindex // 8] >> (7 - bitindex % 8)) & 1
            bitindex += 1
            if bit not in node:
                raise ValueError("invalid Huffman code")
            node = node[bit]
        out[i] = node["symbol"]
    if len(payload) != (bitindex + 7) // 8:
        raise ValueError("trailing Huffman bytes")
    padding = -bitindex % 8
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ValueError("nonzero Huffman padding")
    return out


def encode_matrix(words):
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError("expected uint16 matrix")
    rows, cols = words.shape
    if max(rows, cols, words.size) > 0xFFFFFFFF:
        raise ValueError("matrix too large")
    flat = np.ascontiguousarray(words).reshape(-1)
    raw = flat.astype("<u2").tobytes()
    lower = (((flat >> 8) & 128) | (flat & 127)).astype(np.uint8).tobytes()
    exp = ((flat >> 7) & 255).astype(np.uint8)
    lengths = _lengths(exp)
    table = bytes(v for symbol in sorted(lengths) for v in (symbol, lengths[symbol]))
    coded = _encode_symbols(exp, _codes(lengths)) if lengths else b""
    candidate = lower + table + coded
    mode = int(bool(lengths) and len(candidate) < len(raw))
    payload = candidate if mode else raw
    header = _HEADER.pack(_MAGIC, _VERSION, mode, len(lengths) if mode else 0,
                          rows, cols, len(coded) if mode else 0, zlib.crc32(raw))
    data = header + payload
    parts = {"header": len(header), "raw_words": 0 if mode else len(raw),
             "sign_fraction": len(lower) if mode else 0,
             "exponent_table": len(table) if mode else 0,
             "exponent_payload": len(coded) if mode else 0}
    assert sum(parts.values()) == len(data)
    return EncodedMatrix(data, {"raw_bytes": len(raw), "serialized_bytes": len(data),
                               "mode": "huffman_exponents" if mode else "raw_words",
                               "component_bytes": parts,
                               "active_exponent_symbols": len(lengths),
                               "max_code_length": max(lengths.values(), default=0),
                               "candidate_payload_bytes": len(candidate)})


def decode_matrix(data):
    if isinstance(data, EncodedMatrix):
        data = data.data
    data = bytes(data)
    if len(data) < _HEADER.size:
        raise ValueError("truncated Huffman record")
    magic, version, mode, nsym, rows, cols, exp_len, checksum = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION or mode not in (0, 1) or nsym > 256:
        raise ValueError("invalid Huffman header")
    n = rows * cols
    payload = data[_HEADER.size:]
    if mode == 0:
        if nsym or exp_len or len(payload) != 2 * n:
            raise ValueError("invalid raw payload")
        out = np.frombuffer(payload, dtype="<u2").copy()
    else:
        if not nsym or len(payload) != n + 2 * nsym + exp_len:
            raise ValueError("invalid Huffman payload length")
        lower = np.frombuffer(payload[:n], dtype=np.uint8).astype(np.uint16)
        table = payload[n:n + 2 * nsym]
        lengths = {table[i]: table[i + 1] for i in range(0, len(table), 2)}
        if len(lengths) != nsym:
            raise ValueError("duplicate Huffman symbol")
        exponents = _decode_symbols(payload[n + 2 * nsym:], n, _codes(lengths))
        out = ((lower & 128) << 8) | (lower & 127) | (exponents << 7)
    if zlib.crc32(out.astype("<u2").tobytes()) != checksum:
        raise ValueError("Huffman record checksum mismatch")
    return out.reshape(rows, cols)

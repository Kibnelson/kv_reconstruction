"""An indexed, exact KV-cache container for the CPU mechanism smoke test.

Each temporal tile is a complete codec.py matrix record. The outer container has
a fixed 32-byte header and an explicit 64-bit absolute-offset table. Its record
order is layer, K/V, then time. A tile can be decoded without decoding earlier
tiles or allocating the complete expanded cache. This is a CPU format prototype,
not a GPU kernel or a claim about serving performance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct
from typing import Any

import numpy as np

try:
    from .codec import decode_matrix, encode_matrix
except ImportError:  # Direct execution/import from the kv_smoke directory.
    from codec import decode_matrix, encode_matrix


_MAGIC = b"KVCC"
_VERSION = 1
# magic, version, flags, layers, tokens, channels, tile_tokens, record_count
_HEADER = struct.Struct("<4sHHIIIIQ")
_OFFSET = struct.Struct("<Q")
_MAX_U32 = (1 << 32) - 1


@dataclass(frozen=True)
class EncodedCache:
    data: bytes
    stats: dict[str, Any]


def _as_bytes(blob: EncodedCache | bytes) -> bytes:
    return blob.data if isinstance(blob, EncodedCache) else bytes(blob)


def _parse(blob: EncodedCache | bytes) -> tuple[bytes, dict[str, int]]:
    data = _as_bytes(blob)
    if len(data) < _HEADER.size:
        raise ValueError("truncated cache header")
    magic, version, flags, nlayers, tokens, channels, tile_tokens, nrecords = (
        _HEADER.unpack_from(data)
    )
    if magic != _MAGIC or version != _VERSION or flags:
        raise ValueError("unsupported cache header")
    if min(nlayers, tokens, channels, tile_tokens) <= 0:
        raise ValueError("cache dimensions and tile_tokens must be positive")
    ntiles = (tokens + tile_tokens - 1) // tile_tokens
    if nrecords != nlayers * 2 * ntiles:
        raise ValueError("record count does not match cache dimensions")
    offset_table_bytes = (nrecords + 1) * _OFFSET.size
    payload_start = _HEADER.size + offset_table_bytes
    if payload_start > len(data):
        raise ValueError("truncated cache offset table")
    first = _OFFSET.unpack_from(data, _HEADER.size)[0]
    last = _OFFSET.unpack_from(data, _HEADER.size + nrecords * _OFFSET.size)[0]
    if first != payload_start or last != len(data):
        raise ValueError("invalid first or final cache offset")
    # Other offsets are checked when their records are accessed. Avoid allocating
    # an O(number_of_records) validation array for every bounded tile read.
    shape = {
        "nlayers": nlayers,
        "tokens": tokens,
        "channels": channels,
        "tile_tokens": tile_tokens,
        "ntiles": ntiles,
        "nrecords": nrecords,
        "serialized_bytes": len(data),
        "raw_payload_bytes": nlayers * 2 * tokens * channels * 2,
        "header_bytes": _HEADER.size,
        "offset_table_bytes": offset_table_bytes,
        "container_overhead_bytes": payload_start,
    }
    return data, shape


def _record_bytes(data: bytes, shape: dict[str, int], index: int) -> bytes:
    if not 0 <= index < shape["nrecords"]:
        raise IndexError("record index outside cache")
    table_position = _HEADER.size + index * _OFFSET.size
    start, end = struct.unpack_from("<QQ", data, table_position)
    if not shape["container_overhead_bytes"] <= start < end <= len(data):
        raise ValueError("invalid record offsets")
    return data[start:end]


def cache_shape(blob: EncodedCache | bytes) -> dict[str, int]:
    """Read dimensions and outer-format byte counts without decoding payloads."""
    _, shape = _parse(blob)
    return shape


def encode_cache(
    matrices: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    group_size: int = 64,
    tile_tokens: int = 16,
) -> EncodedCache:
    """Encode aligned layers of (K,V) uint16 matrices, each [tokens,channels].

    Every K tile and V tile carries the complete matrix codec metadata. Shared
    outer metadata and its actual offset table are charged separately. No raw
    tensor, external state, or earlier temporal record is needed to decode a tile.
    """
    layers = list(matrices)
    if not layers:
        raise ValueError("at least one layer is required")
    if not isinstance(tile_tokens, int) or not 1 <= tile_tokens <= _MAX_U32:
        raise ValueError("invalid tile_tokens")
    if len(layers) > _MAX_U32:
        raise ValueError("too many layers for cache format")

    first_pair = layers[0]
    if len(first_pair) != 2:
        raise ValueError("each layer must contain exactly K and V")
    first = np.asarray(first_pair[0])
    if first.dtype != np.uint16 or first.ndim != 2:
        raise ValueError("expected two-dimensional uint16 BF16 word matrices")
    tokens, channels = first.shape
    if min(tokens, channels) <= 0 or max(tokens, channels) > _MAX_U32:
        raise ValueError("invalid cache matrix dimensions")
    for pair in layers:
        if len(pair) != 2:
            raise ValueError("each layer must contain exactly K and V")
        for matrix in pair:
            array = np.asarray(matrix)
            if array.dtype != np.uint16 or array.shape != (tokens, channels):
                raise ValueError("all K/V matrices must share uint16 dtype and shape")

    nlayers = len(layers)
    ntiles = (tokens + tile_tokens - 1) // tile_tokens
    nrecords = nlayers * 2 * ntiles
    offset_table_bytes = (nrecords + 1) * _OFFSET.size
    outer_bytes = _HEADER.size + offset_table_bytes
    payloads: list[bytes] = []
    sizes_by_kind: dict[str, int] = {"K": 0, "V": 0}
    modes: Counter = Counter()
    candidate_widths: Counter = Counter()
    accepted_widths: Counter = Counter()
    exponent_widths: Counter = Counter()
    accepted_exponent_widths: Counter = Counter()
    accepted_payload_by_mode: Counter = Counter()
    matrix_components: Counter = Counter()
    fallbacks: Counter = Counter()
    groups = 0
    matrix_metadata_bytes = 0
    matrix_payload_bytes = 0
    accepted_widths_available = True

    for pair in layers:
        for kind, matrix in zip(("K", "V"), pair):
            matrix = np.asarray(matrix)
            for start in range(0, tokens, tile_tokens):
                record = encode_matrix(
                    matrix[start:start + tile_tokens], method, group_size=group_size
                )
                payloads.append(record.data)
                sizes_by_kind[kind] += len(record.data)
                stats = record.stats
                modes.update(stats.get("mode_counts", {}))
                candidate_widths.update(stats.get("candidate_correction_width_histogram", {}))
                exponent_widths.update(stats.get("candidate_exponent_width_histogram", {}))
                accepted_exponent_widths.update(stats.get("accepted_exponent_width_histogram", {}))
                accepted_payload_by_mode.update(stats.get("accepted_payload_bytes_by_mode", {}))
                matrix_components.update(stats.get("component_bytes", {}))
                fallbacks.update(stats.get("fallback_counts", {}))
                if "accepted_correction_width_histogram" in stats:
                    accepted_widths.update(stats["accepted_correction_width_histogram"])
                else:
                    accepted_widths_available = False
                groups += stats["groups"]
                matrix_metadata_bytes += stats["container_overhead_bytes"]
                matrix_payload_bytes += stats["payload_bytes"]

    offsets = [outer_bytes]
    for payload in payloads:
        offsets.append(offsets[-1] + len(payload))
    header = _HEADER.pack(
        _MAGIC, _VERSION, 0, nlayers, tokens, channels, tile_tokens, nrecords
    )
    offset_table = b"".join(_OFFSET.pack(offset) for offset in offsets)
    data = header + offset_table + b"".join(payloads)
    raw_payload_bytes = nlayers * 2 * tokens * channels * 2
    if len(data) != outer_bytes + matrix_metadata_bytes + matrix_payload_bytes:
        raise AssertionError("cache byte accounting does not balance")
    if len(data) != outer_bytes + sizes_by_kind["K"] + sizes_by_kind["V"]:
        raise AssertionError("K/V byte accounting does not balance")

    stats: dict[str, Any] = {
        "method": method,
        "nlayers": nlayers,
        "tokens": tokens,
        "channels": channels,
        "group_size": group_size,
        "tile_tokens": tile_tokens,
        "ntiles": ntiles,
        "records": nrecords,
        "groups": groups,
        "raw_payload_bytes": raw_payload_bytes,
        "serialized_bytes": len(data),
        "ratio_vs_raw_payload": raw_payload_bytes / len(data),
        "bits_per_value_including_container": len(data) * 16 / raw_payload_bytes,
        "header_bytes": _HEADER.size,
        "offset_table_bytes": offset_table_bytes,
        "container_overhead_bytes": outer_bytes,
        "matrix_metadata_bytes": matrix_metadata_bytes,
        "matrix_payload_bytes": matrix_payload_bytes,
        "framing_bytes": outer_bytes + matrix_metadata_bytes,
        "K_serialized_record_bytes": sizes_by_kind["K"],
        "V_serialized_record_bytes": sizes_by_kind["V"],
        "mode_counts": dict(modes),
        # Candidate statistics may include predictions ultimately replaced by
        # raw records. Preserve that distinction in the aggregate field names.
        "candidate_correction_width_histogram": dict(candidate_widths),
        "candidate_exponent_width_histogram": dict(exponent_widths),
        "accepted_exponent_width_histogram": dict(accepted_exponent_widths),
        "accepted_payload_bytes_by_mode": dict(accepted_payload_by_mode),
        "fallback_counts": dict(fallbacks),
        "byte_accounting_note": (
            "matrix_metadata_bytes means matrix/group framing; matrix_payload_bytes "
            "also includes predictor scales and other codec metadata. component_bytes "
            "provides the detailed accounting when exposed by the matrix codec."
        ),
    }
    if accepted_widths_available:
        stats["accepted_correction_width_histogram"] = dict(accepted_widths)
    if matrix_components:
        components = {"cache_header": _HEADER.size, "record_offset_table": offset_table_bytes}
        components.update({f"matrix_{key}": value for key, value in matrix_components.items()})
        if sum(components.values()) != len(data):
            raise AssertionError("detailed component byte accounting does not balance")
        stats["component_bytes"] = components
    return EncodedCache(data=data, stats=stats)


def _decode_tile(
    data: bytes, shape: dict[str, int], layer: int, tile_idx: int
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0 <= layer < shape["nlayers"]:
        raise IndexError("layer outside cache")
    if not 0 <= tile_idx < shape["ntiles"]:
        raise IndexError("tile index outside cache")
    start_row = tile_idx * shape["tile_tokens"]
    tile_rows = min(shape["tile_tokens"], shape["tokens"] - start_row)
    base = layer * 2 * shape["ntiles"]
    k = decode_matrix(_record_bytes(data, shape, base + tile_idx))
    v = decode_matrix(_record_bytes(data, shape, base + shape["ntiles"] + tile_idx))
    expected_shape = (tile_rows, shape["channels"])
    if k.shape != expected_shape or v.shape != expected_shape:
        raise ValueError("tile matrix shape does not match cache header")
    return k, v, start_row


def decode_cache_tile(
    blob: EncodedCache | bytes, layer: int, tile_idx: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (K words, V words, start_row), decoding only the requested tile."""
    data, shape = _parse(blob)
    return _decode_tile(data, shape, layer, tile_idx)


def decode_cache(blob: EncodedCache | bytes) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expand a complete cache for reference comparison, not bounded execution."""
    data, shape = _parse(blob)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in range(shape["nlayers"]):
        k = np.empty((shape["tokens"], shape["channels"]), dtype=np.uint16)
        v = np.empty_like(k)
        for tile_idx in range(shape["ntiles"]):
            ktile, vtile, start = _decode_tile(data, shape, layer, tile_idx)
            stop = start + ktile.shape[0]
            k[start:stop] = ktile
            v[start:stop] = vtile
        result.append((k, v))
    return result

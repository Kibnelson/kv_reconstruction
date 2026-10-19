"""Frozen learned magnitude predictor followed by a genuine E8 residual.

This independent prototype applies E8 to signed residuals in reversible BF16
magnitude-symbol space, NOT floating-point KV space.  It does not reproduce
any published quantizer.  A record stores learned IDs, E8 coordinates/coset
flags, exact signs and lossless remaining corrections.  Only the explicitly
identified frozen dictionary is external; all request-specific state is in
the bytes.  The learned-only control must use the same correction coder.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct
import zlib

import numpy as np

from kv_research.e8_codec import (nearest_e8, _pack_coordinates,
    _unpack_coordinates, _coordinates_size)
from kv_research.learned_codec import (_assign, _pack, _unpack,
    _unpack_corrections, model_bytes)
from kv_residual.learned import _cosets


_MAGIC = b"LKE8"
_VERSION = 1
_HEADER = struct.Struct("<4sBBBBIII32sH")
_GROUP = struct.Struct("<BI")
_E8_METADATA = struct.Struct("<hB")


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def _correction_encode(original, prediction, method):
    # The common entropy codec is authored alongside this file.  Import at
    # call-time so projection-only tests do not depend on optional extensions.
    if method == "coset":
        # Width15 retains the full magnitude symbol and is exact even at or
        # beyond the half-modulus boundary.  Smaller widths still use the
        # strict nearest-coset radius guarantee.
        blob, _, widths = _cosets(original.reshape(-1, 8), prediction.reshape(-1, 8))
        widthbytes = (len(widths) * 4 + 7) // 8
        return b"\x00" + blob, dict(mode="coset", component_bytes={
            "method_header": 1, "width_metadata": widthbytes,
            "correction_payload": len(blob) - widthbytes})
    from kv_residual.entropy import encode
    residual = original.astype(np.int32) - prediction.astype(np.int32)
    result = encode(residual, strategy="best")
    parts = Counter(result.stats["component_bytes"])
    parts["method_header"] += 1
    return b"\x01" + result.data, dict(result.stats, component_bytes=dict(parts))


def _correction_decode(data, prediction):
    if not data:
        raise ValueError("truncated correction method")
    if data[0] == 0:
        nvec = prediction.size // 8
        widthbytes = (nvec * 4 + 7) // 8
        if len(data) < 1 + widthbytes:
            raise ValueError("truncated correction widths")
        widths = _unpack(data[1:1 + widthbytes], nvec, 4)
        residues = _unpack_corrections(data[1 + widthbytes:], widths).astype(np.int32)
        modulus = 1 << widths.astype(np.int32)
        half = np.where(widths == 0, 0, modulus // 2)
        c = prediction.reshape(-1, 8).astype(np.int32)
        recovered = (c + ((residues - c + half[:, None]) & (modulus[:, None] - 1)) - half[:, None]) & 0x7FFF
        return recovered.reshape(-1).astype(np.uint16)
    if data[0] != 1:
        raise ValueError("unknown correction method")
    from kv_residual.entropy import decode
    residual = decode(data[1:])
    if residual.shape != prediction.shape:
        raise ValueError("remaining residual shape mismatch")
    magnitude = prediction.astype(np.int32) + residual
    if np.any((magnitude < 0) | (magnitude > 32767)):
        raise ValueError("recovered magnitude outside BF16 domain")
    return magnitude.astype(np.uint16)


def encode_matrix(words, model, key="global", step_log2=6, correction="best",
                  assignment="radius", force=False, group_size=64, family="e8"):
    """Learned ID -> E8 residual -> exact remaining correction.

    ``force=True`` retains real E8 records even when they are larger than raw
    words.  Nonfinite source groups still use raw; this preserves every BF16
    pattern.  ``force=False`` chooses raw when the complete candidate payload
    is no smaller.  Final vector tails are padded with known zero words and
    trimmed after decode.  All groups are independent within the matrix.
    """
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError("expected uint16 BF16 [tokens,channels]")
    if group_size < 8 or group_size % 8 or not 1 <= step_log2 <= 14:
        raise ValueError("invalid group size or E8 step")
    if correction not in ("best", "coset") or assignment != "radius":
        raise ValueError("unsupported correction or assignment")
    if family not in ("e8", "scalar"):
        raise ValueError("unknown residual predictor family")
    if key not in model.codebooks:
        raise ValueError("unknown dictionary key")
    qbits = (model.n_centroids - 1).bit_length()
    key_bytes = key.encode("utf8")
    if len(key_bytes) > 65535:
        raise ValueError("dictionary key too long")
    identity = bytes.fromhex(model.model_id)
    rows, columns = words.shape
    flags = int(bool(force)) | ((family == "scalar") << 1)
    header = _HEADER.pack(_MAGIC, _VERSION, qbits, step_log2, flags,
                          group_size, rows, columns, identity, len(key_bytes)) + key_bytes
    flat = np.ascontiguousarray(words).reshape(-1)
    padded = np.pad(flat, (0, (-len(flat)) % 8)).reshape(-1, 8)
    magnitudes = padded & 0x7FFF
    indexes, _, _ = _assign(magnitudes, model.codebooks[key])
    learned_prediction = model.codebooks[key][indexes].astype(np.int32)
    residual = magnitudes.astype(np.int32) - learned_prediction
    if family == "e8":
        points2 = (2 * nearest_e8(residual.astype(np.float64) / (1 << step_log2))).astype(np.int32)
    else:
        points2 = 2 * np.rint(residual.astype(np.float64) / (1 << step_log2)).astype(np.int32)
    # Every half-coordinate times2**step is integral for step>=1.
    prediction = np.clip(learned_prediction + points2 * (1 << (step_log2 - 1)),
                         0, 32767).astype(np.uint16)
    output = bytearray(header)
    components = Counter(matrix_header_and_model_identity=len(header))
    modes, reasons, coordinate_widths, correction_modes = Counter(), Counter(), Counter(), Counter()
    candidate_bytes, candidate_groups = 0, 0
    accepted_corrections = Counter()
    candidate_error_histogram = Counter()
    nonzero_vectors = half_coset_vectors = encoded_vectors = 0
    mechanism_counts = Counter(e8_groups=0, scalar_groups=0, learned_vectors=0,
        e8_vectors=0, nonzero_e8_vectors=0, half_integer_e8_vectors=0)
    for start in range(0, len(flat), group_size):
        original = flat[start:start + group_size]
        first, last = start // 8, (start + len(original) + 7) // 8
        nvec = last - first
        raw = original.astype("<u2").tobytes()
        finite = not np.any((original & 0x7F80) == 0x7F80)
        payload = None
        candidate_parts = None
        if finite:
            ids = _pack(indexes[first:last], qbits)
            q2 = points2[first:last]
            if family == "e8":
                minimum, coordinate_width, coords = _pack_coordinates(q2.astype(np.float64) * .5)
            else:
                integers = q2 // 2
                minimum = int(integers.min())
                coordinate_width = int(int(integers.max()) - minimum).bit_length()
                coords = _pack(integers - minimum, coordinate_width)
            meta = _E8_METADATA.pack(minimum, coordinate_width)
            signs = _pack(padded[first:last].reshape(-1) >> 15, 1)
            correction_blob, correction_stats = _correction_encode(
                magnitudes[first:last].reshape(-1), prediction[first:last].reshape(-1), correction)
            payload = meta + ids + coords + signs + correction_blob
            candidate_parts = Counter(residual_predictor_metadata=len(meta), learned_indices=len(ids),
                                      exact_signs=len(signs))
            candidate_parts["e8_coordinates_and_coset_flags" if family == "e8" else "scalar_coordinates"] = len(coords)
            correction_components = correction_stats.get("component_bytes")
            if correction_components is None:
                candidate_parts["correction_stream"] = len(correction_blob)
            else:
                if sum(correction_components.values()) != len(correction_blob):
                    raise AssertionError("correction byte accounting")
                candidate_parts.update({"correction_" + k: v for k, v in correction_components.items()})
            assert sum(candidate_parts.values()) == len(payload)
            candidate_groups += 1
            encoded_vectors += nvec
            nonzero_vectors += int(np.any(q2 != 0, axis=1).sum())
            half_coset_vectors += int(((q2[:, 0] & 1) != 0).sum())
            candidate_bytes += len(payload)
            coordinate_widths[str(coordinate_width)] += 1
            correction_modes[str(correction_stats.get("mode", "unknown"))] += 1
            max_error = np.max(np.abs(magnitudes[first:last].astype(np.int32) -
                                      prediction[first:last].astype(np.int32)))
            candidate_error_histogram[str(int(max_error))] += 1
        if payload is None or (not force and len(payload) >= len(raw)):
            reason = "nonfinite_input" if not finite else "candidate_not_smaller"
            reasons[reason] += 1
            mode, payload = 0, raw
            components["raw_words"] += len(payload)
        else:
            mode = 1
            components.update(candidate_parts)
            accepted_corrections[str(correction_stats.get("mode", "unknown"))] += 1
            mechanism_counts[family + "_groups"] += 1
            mechanism_counts["learned_vectors"] += nvec
            if family == "e8":
                mechanism_counts["e8_vectors"] += nvec
                mechanism_counts["nonzero_e8_vectors"] += int(np.any(q2 != 0, axis=1).sum())
                mechanism_counts["half_integer_e8_vectors"] += int(((q2[:, 0] & 1) != 0).sum())
        output.extend(_GROUP.pack(mode, len(payload)))
        output.extend(payload)
        components["group_headers"] += _GROUP.size
        modes[str(mode)] += 1
    checksum = struct.pack("<I", zlib.crc32(output))
    output.extend(checksum)
    components["matrix_checksum"] += len(checksum)
    data = bytes(output)
    assert sum(components.values()) == len(data)
    stats = dict(method="learned_" + family + "_residual", raw_payload_bytes=int(words.nbytes),
        serialized_bytes=len(data), shape=[rows, columns], group_size=group_size,
        model_id=model.model_id, dictionary_key=key, n_centroids=model.n_centroids,
        external_model_bytes=len(model_bytes(model)), step_log2=step_log2,
        correction_requested=correction, assignment=assignment, force=bool(force), family=family,
        component_bytes=dict(components), mode_counts=dict(modes), fallback_counts=dict(reasons),
        candidate_groups=candidate_groups, candidate_payload_bytes=candidate_bytes,
        candidate_coordinate_width_histogram=dict(coordinate_widths),
        candidate_correction_mode_histogram=dict(correction_modes),
        accepted_correction_mode_histogram=dict(accepted_corrections),
        mechanism_counts=dict(mechanism_counts),
        candidate_maximum_error_histogram=dict(candidate_error_histogram),
        branch_diagnostics=dict(candidate_vectors=encoded_vectors,
            nonzero_residual_vectors=nonzero_vectors,
            half_coset_vectors=half_coset_vectors))
    return EncodedMatrix(data, stats)


def decode_matrix(encoded, model):
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _HEADER.size + 4:
        raise ValueError("truncated learned E8 matrix")
    expected = struct.unpack_from("<I", data, len(data) - 4)[0]
    if zlib.crc32(data[:-4]) != expected:
        raise ValueError("learned E8 matrix checksum mismatch")
    data = data[:-4]
    magic, version, qbits, step_log2, flags, group_size, rows, columns, identity, keylen = _HEADER.unpack_from(data)
    if (magic != _MAGIC or version != _VERSION or flags not in (0, 1, 2, 3) or
            group_size < 8 or group_size % 8 or not 1 <= step_log2 <= 14):
        raise ValueError("invalid learned E8 matrix header")
    if identity.hex() != model.model_id or qbits != (model.n_centroids - 1).bit_length():
        raise ValueError("learned model identity or index width mismatch")
    family = "scalar" if flags & 2 else "e8"
    offset = _HEADER.size
    if offset + keylen > len(data):
        raise ValueError("truncated dictionary key")
    try:
        key = data[offset:offset + keylen].decode("utf8")
    except UnicodeDecodeError as error:
        raise ValueError("invalid dictionary key encoding") from error
    offset += keylen
    if key not in model.codebooks:
        raise ValueError("unknown learned dictionary key")
    total = rows * columns
    flat = np.empty(total, dtype=np.uint16)
    for start in range(0, total, group_size):
        if offset + _GROUP.size > len(data):
            raise ValueError("truncated learned E8 group header")
        mode, length = _GROUP.unpack_from(data, offset)
        offset += _GROUP.size
        if offset + length > len(data):
            raise ValueError("truncated learned E8 group payload")
        payload = data[offset:offset + length]
        offset += length
        count = min(group_size, total - start)
        nvec = (count + 7) // 8
        if mode == 0:
            if length != 2 * count:
                raise ValueError("invalid learned E8 raw payload length")
            decoded = np.frombuffer(payload, dtype="<u2")
        elif mode == 1:
            if length < _E8_METADATA.size:
                raise ValueError("truncated E8 predictor metadata")
            minimum, width = _E8_METADATA.unpack_from(payload)
            idbytes = (nvec * qbits + 7) // 8
            if not 0 <= width <= 16:
                raise ValueError("invalid residual coordinate width")
            coordbytes = _coordinates_size(nvec, width) if family == "e8" else (nvec * 8 * width + 7) // 8
            signbytes = nvec
            controls = _E8_METADATA.size + idbytes + coordbytes + signbytes
            if length < controls:
                raise ValueError("truncated learned E8 controls")
            p = _E8_METADATA.size
            ids = _unpack(payload[p:p + idbytes], nvec, qbits)
            p += idbytes
            if np.any(ids >= model.n_centroids):
                raise ValueError("invalid dictionary ID")
            if family == "e8":
                points2 = (2 * _unpack_coordinates(payload[p:p + coordbytes], nvec, minimum, width)).astype(np.int32)
            else:
                integers = _unpack(payload[p:p + coordbytes], nvec * 8, width).astype(np.int32).reshape(nvec, 8) + minimum
                points2 = 2 * integers
            p += coordbytes
            signs = _unpack(payload[p:p + signbytes], nvec * 8, 1)
            p += signbytes
            learned = model.codebooks[key][ids].astype(np.int32)
            prediction = np.clip(learned + points2 * (1 << (step_log2 - 1)),
                                 0, 32767).astype(np.uint16).reshape(-1)
            magnitude = np.asarray(_correction_decode(payload[p:], prediction), dtype=np.uint16).reshape(-1)
            if magnitude.shape != prediction.shape or np.any(magnitude > 32767):
                raise ValueError("invalid decoded magnitude")
            decoded = (magnitude | (signs << 15))[:count]
        else:
            raise ValueError("unknown learned E8 group mode")
        flat[start:start + count] = decoded
    if offset != len(data):
        raise ValueError("unexpected learned E8 trailing bytes")
    return flat.reshape(rows, columns)

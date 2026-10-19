"""Small, exact BF16 record codecs for a CPU mechanism smoke test.

These deliberately simple predictors are baselines, not implementations of E8,
SplitZip, or a learned KV codec. All sizes are sizes of actual serialized bytes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct

import numpy as np


_HEADER = struct.Struct("<4sBBHIII")
_GROUP = struct.Struct("<BI")
_MAGIC = b"KVRC"
_VERSION = 1
_METHODS = {"raw": 0, "exponent": 1, "zero_coset": 2,
            "lattice4_coset": 3, "lattice2_coset": 4,
            "lattice2_sign_coset": 5}
_RAW, _EXPONENT, _ZERO, _LATTICE4, _LATTICE2, _LATTICE2_SIGN = range(6)


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def float32_to_bf16_words(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16, ties to even; retain nonfinite classification."""
    x = np.asarray(values, dtype=np.float32)
    u = x.view(np.uint32)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nonfinite = (u & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    if np.any(nonfinite):
        # Direct truncation preserves infinities. Ensure a NaN payload cannot
        # disappear when all its significant payload bits were below bit 16.
        special = (u >> 16).astype(np.uint16)
        nan = nonfinite & ((u & np.uint32(0x007FFFFF)) != 0)
        special = np.where(nan & ((special & 0x7F) == 0), special | 1, special)
        rounded = np.where(nonfinite, special, rounded).astype(np.uint16)
    return rounded


def bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def ordered_u16(words: np.ndarray) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint16)
    return np.where((w & 0x8000) != 0, (~w) & 0xFFFF, w ^ 0x8000).astype(np.uint16)


def inverse_ordered_u16(symbols: np.ndarray) -> np.ndarray:
    x = np.asarray(symbols, dtype=np.uint16)
    return np.where((x & 0x8000) != 0, x ^ 0x8000, (~x) & 0xFFFF).astype(np.uint16)


def _nonfinite(words: np.ndarray) -> bool:
    return bool(np.any((words & 0x7F80) == 0x7F80))


def _pack_bits(values: np.ndarray, width: int) -> bytes:
    if width == 0:
        return b""
    output = bytearray()
    accumulator = 0
    nbits = 0
    limit = 1 << width
    for value in np.asarray(values).reshape(-1):
        value = int(value)
        if not 0 <= value < limit:
            raise ValueError("bit-packing input outside requested width")
        accumulator |= value << nbits
        nbits += width
        while nbits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            nbits -= 8
    if nbits:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _unpack_bits(payload: bytes, count: int, width: int) -> np.ndarray:
    expected = (count * width + 7) // 8
    if len(payload) != expected:
        raise ValueError("incorrect packed payload length")
    result = np.zeros(count, dtype=np.uint16)
    if width == 0:
        return result
    accumulator = 0
    nbits = 0
    offset = 0
    mask = (1 << width) - 1
    for i in range(count):
        while nbits < width:
            accumulator |= payload[offset] << nbits
            offset += 1
            nbits += 8
        result[i] = accumulator & mask
        accumulator >>= width
        nbits -= width
    return result


def _coset_payload(original: np.ndarray, prediction: np.ndarray):
    x = ordered_u16(original).astype(np.int32)
    c = ordered_u16(prediction).astype(np.int32)
    radius = int(np.max(np.abs(x - c)))
    if radius == 0:
        return 0, b"", radius
    width = next((b for b in range(1, 16) if radius < (1 << (b - 1))), None)
    if width is None:
        return None, None, radius
    return width, _pack_bits(x & ((1 << width) - 1), width), radius


def _coset_decode(prediction: np.ndarray, width: int, payload: bytes) -> np.ndarray:
    if not 0 <= width < 16:
        raise ValueError("invalid correction width")
    count = prediction.size
    c = ordered_u16(prediction).astype(np.int32)
    if width == 0:
        if payload:
            raise ValueError("unexpected exact-prediction correction bytes")
        x = c
    else:
        half = 1 << (width - 1)
        mask = (1 << width) - 1
        s = _unpack_bits(payload, count, width).astype(np.int32)
        x = c + ((s - c + half) & mask) - half
    if np.any((x < 0) | (x > 65535)):
        raise ValueError("decoded symbol outside BF16 word range")
    return inverse_ordered_u16(x)


def _candidate(original: np.ndarray, method: str):
    """Return mode, payload, diagnostics; predictor state is in payload."""
    if method == "raw":
        return _RAW, original.astype("<u2").tobytes(), {"reason": "requested_raw"}
    if _nonfinite(original):
        return None, None, {"reason": "nonfinite_original"}
    if method == "exponent":
        exponent = ((original >> 7) & 0xFF).astype(np.uint16)
        minimum = int(exponent.min())
        width = int(int(exponent.max()) - minimum).bit_length()
        sign_mantissa = (((original >> 8) & 0x80) | (original & 0x7F)).astype(np.uint8)
        payload = bytes((minimum, width)) + sign_mantissa.tobytes()
        payload += _pack_bits(exponent - minimum, width)
        return _EXPONENT, payload, {"exponent_width": width}
    if method == "zero_coset":
        prediction = np.zeros_like(original)
        width, corrections, radius = _coset_payload(original, prediction)
        if width is None:
            return None, None, {"reason": "no_useful_width", "radius": radius}
        return _ZERO, bytes((width,)) + corrections, {"correction_width": width,
                                                    "radius": radius}
    if method == "lattice2_sign_coset":
        # A positive offset scalar grid {0.5,1.5,2.5,3.5}*scale. Exact sign
        # bits are retained separately. Unlike the symmetric grid, a nonzero
        # small value is not predicted as zero. Corrections act on reversible
        # 15-bit magnitude symbols, not arithmetic floating-point residuals.
        values = np.abs(bf16_words_to_float32(original))
        maximum = np.max(values)
        scale = np.float32(maximum / np.float32(3.5)) if maximum != 0 else np.float32(1)
        scale_word = int(float32_to_bf16_words(np.asarray([scale]))[0])
        stored_scale = bf16_words_to_float32(np.asarray([scale_word], dtype=np.uint16))[0]
        if not np.isfinite(stored_scale) or stored_scale <= 0:
            return None, None, {"reason": "invalid_scale"}
        q = np.clip(np.rint(values / stored_scale - np.float32(0.5)), 0, 3).astype(np.int16)
        prediction = float32_to_bf16_words((q.astype(np.float32) + np.float32(0.5)) * stored_scale)
        if _nonfinite(prediction):
            return None, None, {"reason": "nonfinite_prediction"}
        x = (original & 0x7FFF).astype(np.int32)
        c = prediction.astype(np.int32)
        radius = int(np.max(np.abs(x - c)))
        width = 0 if radius == 0 else next((b for b in range(1, 16)
                                            if radius < (1 << (b - 1))), None)
        if width is None:
            return None, None, {"reason": "no_useful_width", "radius": radius}
        corrections = b"" if width == 0 else _pack_bits(x & ((1 << width) - 1), width)
        payload = struct.pack("<HB", scale_word, width)
        payload += _pack_bits(q, 2) + _pack_bits(original >> 15, 1) + corrections
        return _LATTICE2_SIGN, payload, {"correction_width": width, "radius": radius,
                                        "predictor_bits_per_value": 3}
    if method in ("lattice4_coset", "lattice2_coset"):
        # A finite scalar Z^d grid, not E8. Coordinates and the stored BF16
        # scale constitute the complete predictor state; no source values or
        # external activation buffer are needed by the decoder.
        qbits = 4 if method == "lattice4_coset" else 2
        qmax = 7 if qbits == 4 else 1
        values = bf16_words_to_float32(original)
        maximum = np.max(np.abs(values))
        scale = np.float32(maximum / np.float32(qmax)) if maximum != 0 else np.float32(1)
        scale_word = int(float32_to_bf16_words(np.asarray([scale]))[0])
        stored_scale = bf16_words_to_float32(np.asarray([scale_word], dtype=np.uint16))[0]
        if not np.isfinite(stored_scale) or stored_scale <= 0:
            return None, None, {"reason": "invalid_scale"}
        with np.errstate(over="ignore", invalid="ignore"):
            q = np.clip(np.rint(values / stored_scale), -qmax, qmax).astype(np.int16)
            prediction = float32_to_bf16_words(q.astype(np.float32) * stored_scale)
        if _nonfinite(prediction):
            return None, None, {"reason": "nonfinite_prediction"}
        width, corrections, radius = _coset_payload(original, prediction)
        if width is None:
            return None, None, {"reason": "no_useful_width", "radius": radius}
        payload = struct.pack("<HB", scale_word, width)
        payload += _pack_bits(q + qmax, qbits) + corrections
        mode = _LATTICE4 if qbits == 4 else _LATTICE2
        return mode, payload, {"correction_width": width, "radius": radius,
                               "predictor_bits_per_value": qbits}
    raise ValueError(f"unknown codec method: {method}")


def encode_matrix(words: np.ndarray, method: str = "lattice4_coset",
                  group_size: int = 64) -> EncodedMatrix:
    """Encode a [tokens,channels] uint16 matrix; flat groups are independent."""
    if method not in _METHODS:
        raise ValueError(f"unknown codec method: {method}")
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError("expected a two-dimensional uint16 BF16 word matrix")
    if not 1 <= group_size <= 2**32 - 1:
        raise ValueError("invalid group_size")
    rows, columns = words.shape
    if rows > 2**32 - 1 or columns > 2**32 - 1:
        raise ValueError("matrix too large for format")
    output = bytearray(_HEADER.pack(_MAGIC, _VERSION, _METHODS[method], 0,
                                    group_size, rows, columns))
    modes, widths, exponent_widths, fallbacks = Counter(), Counter(), Counter(), Counter()
    accepted_widths, accepted_exponent_widths, payload_by_mode = Counter(), Counter(), Counter()
    components = Counter({"container_header": _HEADER.size})
    raw_payload_bytes = words.size * 2
    flat = np.ascontiguousarray(words).reshape(-1)
    groups = 0
    payload_bytes = 0
    for offset in range(0, flat.size, group_size):
        original = flat[offset:offset + group_size]
        raw_payload = original.astype("<u2").tobytes()
        mode, payload, diagnostic = _candidate(original, method)
        # Group headers have identical size in both modes. Predictor state and
        # padding are already part of the candidate payload at this point.
        if mode is None:
            fallbacks[diagnostic["reason"]] += 1
            mode, payload = _RAW, raw_payload
        elif mode != _RAW and len(payload) >= len(raw_payload):
            fallbacks["candidate_not_smaller"] += 1
            mode, payload = _RAW, raw_payload
        modes[str(mode)] += 1
        payload_by_mode[str(mode)] += len(payload)
        components["group_headers"] += _GROUP.size
        if "correction_width" in diagnostic:
            widths[str(diagnostic["correction_width"])] += 1
        if "exponent_width" in diagnostic:
            exponent_widths[str(diagnostic["exponent_width"])] += 1
        if mode == _RAW:
            components["raw_words"] += len(payload)
        elif mode == _EXPONENT:
            accepted_exponent_widths[str(diagnostic["exponent_width"])] += 1
            components["codec_metadata"] += 2
            components["exponent_sign_mantissa"] += original.size
            components["exponent_deltas"] += len(payload) - 2 - original.size
        elif mode == _ZERO:
            accepted_widths[str(diagnostic["correction_width"])] += 1
            components["codec_metadata"] += 1
            components["exact_corrections"] += len(payload) - 1
        else:
            accepted_widths[str(diagnostic["correction_width"])] += 1
            components["predictor_scales"] += 2
            components["codec_metadata"] += 1
            qbits = 4 if mode == _LATTICE4 else 2
            qbytes = (original.size * qbits + 7) // 8
            components["predictor_coordinates"] += qbytes
            signbytes = (original.size + 7) // 8 if mode == _LATTICE2_SIGN else 0
            components["explicit_signs"] += signbytes
            components["exact_corrections"] += len(payload) - 3 - qbytes - signbytes
        output.extend(_GROUP.pack(mode, len(payload)))
        output.extend(payload)
        payload_bytes += len(payload)
        groups += 1
    data = bytes(output)
    stats = {
        "method": method, "shape": [rows, columns], "group_size": group_size,
        "groups": groups, "raw_payload_bytes": raw_payload_bytes,
        "serialized_bytes": len(data), "payload_bytes": payload_bytes,
        "container_overhead_bytes": _HEADER.size + groups * _GROUP.size,
        "ratio_vs_raw_payload": raw_payload_bytes / len(data) if len(data) else 0,
        "bits_per_value_including_container": len(data) * 8 / words.size if words.size else None,
        "mode_counts": dict(modes), "candidate_correction_width_histogram": dict(widths),
        "candidate_exponent_width_histogram": dict(exponent_widths),
        "accepted_correction_width_histogram": dict(accepted_widths),
        "accepted_exponent_width_histogram": dict(accepted_exponent_widths),
        "accepted_payload_bytes_by_mode": dict(payload_by_mode),
        "component_bytes": dict(components),
        "fallback_counts": dict(fallbacks),
    }
    assert sum(components.values()) == len(data)
    return EncodedMatrix(data, stats)


def decode_matrix(encoded: EncodedMatrix | bytes) -> np.ndarray:
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _HEADER.size:
        raise ValueError("truncated container header")
    magic, version, method_id, reserved, group_size, rows, columns = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION or method_id not in _METHODS.values() or reserved:
        raise ValueError("unsupported container header")
    if group_size == 0:
        raise ValueError("invalid zero group size")
    total = rows * columns
    result = np.empty(total, dtype=np.uint16)
    cursor = _HEADER.size
    for offset in range(0, total, group_size):
        count = min(group_size, total - offset)
        if cursor + _GROUP.size > len(data):
            raise ValueError("truncated group header")
        mode, length = _GROUP.unpack_from(data, cursor)
        cursor += _GROUP.size
        if cursor + length > len(data):
            raise ValueError("truncated group payload")
        payload = data[cursor:cursor + length]
        cursor += length
        if mode == _RAW:
            if length != 2 * count:
                raise ValueError("incorrect raw payload length")
            decoded = np.frombuffer(payload, dtype="<u2").astype(np.uint16)
        elif mode == _EXPONENT:
            if length < count + 2:
                raise ValueError("truncated exponent payload")
            minimum, width = payload[0], payload[1]
            if width > 8:
                raise ValueError("invalid exponent width")
            sm = np.frombuffer(payload[2:2 + count], dtype=np.uint8).astype(np.uint16)
            exponent = _unpack_bits(payload[2 + count:], count, width).astype(np.uint32) + minimum
            if np.any(exponent > 255):
                raise ValueError("invalid decoded exponent")
            decoded = (((sm & 0x80) << 8) | (sm & 0x7F) | (exponent << 7)).astype(np.uint16)
        elif mode == _ZERO:
            if length < 1:
                raise ValueError("truncated zero predictor payload")
            decoded = _coset_decode(np.zeros(count, dtype=np.uint16), payload[0], payload[1:])
        elif mode in (_LATTICE4, _LATTICE2):
            if length < 3:
                raise ValueError("truncated lattice payload")
            scale_word, width = struct.unpack_from("<HB", payload)
            qbits, qmax = (4, 7) if mode == _LATTICE4 else (2, 1)
            qbytes = (count * qbits + 7) // 8
            if length < 3 + qbytes:
                raise ValueError("truncated predictor coordinates")
            q = _unpack_bits(payload[3:3 + qbytes], count, qbits).astype(np.int16) - qmax
            if np.any(np.abs(q) > qmax):
                raise ValueError("invalid predictor coordinate")
            scale = bf16_words_to_float32(np.asarray([scale_word], dtype=np.uint16))[0]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid predictor scale")
            prediction = float32_to_bf16_words(q.astype(np.float32) * scale)
            if _nonfinite(prediction):
                raise ValueError("nonfinite predictor")
            decoded = _coset_decode(prediction, width, payload[3 + qbytes:])
        elif mode == _LATTICE2_SIGN:
            if length < 3:
                raise ValueError("truncated signed-grid payload")
            scale_word, width = struct.unpack_from("<HB", payload)
            if not 0 <= width < 16:
                raise ValueError("invalid magnitude correction width")
            qbytes = (count * 2 + 7) // 8
            signbytes = (count + 7) // 8
            if length < 3 + qbytes + signbytes:
                raise ValueError("truncated signed-grid coordinates")
            q = _unpack_bits(payload[3:3 + qbytes], count, 2).astype(np.float32)
            signs = _unpack_bits(payload[3 + qbytes:3 + qbytes + signbytes], count, 1)
            scale = bf16_words_to_float32(np.asarray([scale_word], dtype=np.uint16))[0]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid signed-grid scale")
            prediction = float32_to_bf16_words((q + np.float32(0.5)) * scale)
            if _nonfinite(prediction):
                raise ValueError("nonfinite signed-grid predictor")
            c = prediction.astype(np.int32)
            correction_payload = payload[3 + qbytes + signbytes:]
            if width == 0:
                if correction_payload:
                    raise ValueError("unexpected signed-grid correction bytes")
                x = c
            else:
                half, mask = 1 << (width - 1), (1 << width) - 1
                s = _unpack_bits(correction_payload, count, width).astype(np.int32)
                x = c + ((s - c + half) & mask) - half
            if np.any((x < 0) | (x > 32767)):
                raise ValueError("invalid decoded magnitude symbol")
            decoded = x.astype(np.uint16) | (signs << 15)
        else:
            raise ValueError("unknown group mode")
        result[offset:offset + count] = decoded
    if cursor != len(data):
        raise ValueError("unexpected trailing bytes")
    return result.reshape(rows, columns)

"""Exact BF16 prototype using genuine nearest-E8 predictors.

This is not a reproduction of any published KV quantizer.  E8 nearest-point
projection is the union-of-D8-cosets construction in Conway & Sloane (1982),
https://neilsloane.com/doc/Me83.pdf, Section VI.  Every independent record
contains all predictor state, exact signs and modular corrections.  E8 float
quantization minimizes ordinary Euclidean error; ``e8_symbol`` instead applies
the same lattice to the monotone BF16 magnitude-word coordinate, a companding
domain chosen for exact reconstruction rather than Euclidean distortion.

No learned weights, external codebook or source activation is required.  The
integer-coordinate parity bit omitted from each E8 point is reconstructed by
the decoder; the saved bit pays for its integer/half-integer coset flag.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import struct

import numpy as np

from kv_smoke.codec import (_pack_bits, _unpack_bits,
                           bf16_words_to_float32, float32_to_bf16_words)


_MAGIC = b"KVE8"
_VERSION = 1
_HEADER = struct.Struct("<4sBBHIII")
_GROUP = struct.Struct("<BI")
_PRED = struct.Struct("<HhBB")  # BF16scale or magnitude base, qminimum, step_log2, qwidth
_METHODS = {"e8_float": 1, "e8_symbol": 2}
_RAW = 0
_E8 = 1


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def _nearest_d8(x: np.ndarray) -> np.ndarray:
    """Nearest even-sum integer vector.  Ties are deterministic, not unique."""
    y = np.rint(x).astype(np.int64)
    bad = (y.sum(axis=-1) & 1).astype(bool)
    if np.any(bad):
        error = x - y
        k = np.argmax(np.abs(error), axis=-1)
        row = np.arange(len(y))
        direction = np.where(error[row, k] >= 0, 1, -1)
        y[row, k] += direction * bad
    return y


def nearest_e8(x: np.ndarray) -> np.ndarray:
    """Nearest point in E8 = D8 union (D8 + (1/2,...,1/2)).

    Input shape is [...,8].  Computation uses float64.  At equal distances the
    integer coset wins; within a D8 coset the first largest rounding error wins.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.shape[-1:] != (8,) or not np.all(np.isfinite(a)):
        raise ValueError("expected finite vectors with eight coordinates")
    shape = a.shape
    a = a.reshape(-1, 8)
    integer = _nearest_d8(a).astype(np.float64)
    half = _nearest_d8(a - .5).astype(np.float64) + .5
    use_half = np.sum((a - half) ** 2, axis=1) < np.sum((a - integer) ** 2, axis=1)
    return np.where(use_half[:, None], half, integer).reshape(shape)


def is_e8_point(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape[-1:] != (8,):
        raise ValueError("expected eight coordinates")
    twice = 2 * a
    integral = np.all(twice == np.rint(twice), axis=-1)
    parity = np.mod(np.rint(twice).astype(np.int64), 2)
    same_coset = np.all(parity == parity[..., :1], axis=-1)
    even_sum = np.mod(np.sum(a, axis=-1), 2) == 0
    return integral & same_coset & even_sum


def _pack_coordinates(points: np.ndarray):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 8)
    assert np.all(is_e8_point(p))
    integers = np.floor(p).astype(np.int32)
    flags = ((p[:, 0] - integers[:, 0]) * 2).astype(np.uint16)
    minimum = int(integers.min())
    if not -32768 <= minimum <= 32767:
        raise ValueError("E8 coordinates exceed prototype signed metadata")
    shifted = integers - minimum
    width = int(int(shifted.max())).bit_length()
    if width > 16:
        raise ValueError("E8 coordinate range too large")
    # Subtracting a common minimum leaves the sum even (dimension is eight).
    assert np.all((shifted.sum(axis=1) & 1) == 0)
    flags_blob = _pack_bits(flags, 1)
    if width == 0:
        return minimum, width, flags_blob
    first_blob = _pack_bits(shifted[:, :7], width)
    last_blob = _pack_bits(shifted[:, 7] >> 1, width - 1)
    return minimum, width, flags_blob + first_blob + last_blob


def _coordinates_size(nvectors: int, width: int) -> int:
    if not 0 <= width <= 16:
        raise ValueError("invalid coordinate width")
    return ((nvectors + 7) // 8 +
            ((nvectors * 7 * width + 7) // 8 if width else 0) +
            ((nvectors * (width - 1) + 7) // 8 if width else 0))


def _unpack_coordinates(payload: bytes, nvectors: int, minimum: int, width: int):
    if len(payload) != _coordinates_size(nvectors, width):
        raise ValueError("incorrect coordinate payload length")
    fbytes = (nvectors + 7) // 8
    flags = _unpack_bits(payload[:fbytes], nvectors, 1)
    if width:
        firstbytes = (nvectors * 7 * width + 7) // 8
        first = _unpack_bits(payload[fbytes:fbytes + firstbytes], nvectors * 7, width)
        first = first.reshape(nvectors, 7).astype(np.int32)
        high = _unpack_bits(payload[fbytes + firstbytes:], nvectors, width - 1)
        last = (high.astype(np.int32) << 1) | (first.sum(axis=1) & 1)
        integers = np.column_stack((first, last)) + minimum
    else:
        integers = np.full((nvectors, 8), minimum, dtype=np.int32)
    points = integers.astype(np.float64) + flags[:, None] * .5
    if not np.all(is_e8_point(points)):
        raise ValueError("invalid E8 coordinate parity")
    return points


def _correction_payload(magnitudes: np.ndarray, prediction: np.ndarray):
    """Exact 15-bit symbol cosets, with a width for each eight-value vector."""
    x = magnitudes.reshape(-1, 8).astype(np.int32)
    c = prediction.reshape(-1, 8).astype(np.int32)
    radii = np.max(np.abs(x - c), axis=1)
    widths = np.zeros(len(radii), dtype=np.uint16)
    for i, radius in enumerate(radii):
        if radius:
            # Strict radius<half-modulus prevents a boundary ambiguity.
            widths[i] = int(int(radius).bit_length() + 1)
    if np.any(widths > 15):
        return None
    chunks = [_pack_bits(widths, 4)]
    for row, width in zip(x, widths):
        width = int(width)
        chunks.append(_pack_bits(row & ((1 << width) - 1), width))
    return b"".join(chunks), widths, radii


def _decode_corrections(payload: bytes, prediction: np.ndarray):
    c = prediction.reshape(-1, 8).astype(np.int32)
    widthbytes = (len(c) * 4 + 7) // 8
    if len(payload) < widthbytes:
        raise ValueError("truncated correction widths")
    widths = _unpack_bits(payload[:widthbytes], len(c), 4)
    cursor = widthbytes
    result = np.empty_like(c)
    for i, width in enumerate(widths):
        width = int(width)
        nbytes = width  # eight values, width bits each
        if cursor + nbytes > len(payload):
            raise ValueError("truncated corrections")
        s = _unpack_bits(payload[cursor:cursor + nbytes], 8, width).astype(np.int32)
        cursor += nbytes
        if width:
            half = 1 << (width - 1)
            result[i] = c[i] + ((s - c[i] + half) & ((1 << width) - 1)) - half
        else:
            result[i] = c[i]
    if cursor != len(payload):
        raise ValueError("unexpected correction bytes")
    if np.any((result < 0) | (result > 32767)):
        raise ValueError("invalid decoded BF16 magnitude")
    return result.reshape(-1).astype(np.uint16)


def _predict(points: np.ndarray, parameter: int, step_log2: int, method: str):
    if method == "e8_symbol":
        prediction = np.rint(parameter + points * (1 << step_log2))
        return np.clip(prediction, 0, 32767).astype(np.uint16).reshape(-1)
    scale = bf16_words_to_float32(np.asarray([parameter], dtype=np.uint16))[0]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid E8 float scale")
    # E8 coordinates may be negative even for nonnegative inputs near zero.
    # Clipping only changes the predictor, never the exact target symbols.
    prediction = np.maximum(points.astype(np.float32) * scale, np.float32(0))
    return float32_to_bf16_words(prediction).reshape(-1)


def _one_candidate(words, method: str, parameter: int, step_log2: int, normalized):
    points = nearest_e8(normalized.reshape(-1, 8))
    minimum, qwidth, coordinates = _pack_coordinates(points)
    prediction = _predict(points, parameter, step_log2, method)
    if np.any((prediction & 0x7F80) == 0x7F80):
        return None
    corrections = _correction_payload(words & 0x7FFF, prediction)
    if corrections is None:
        return None
    correction_blob, widths, radii = corrections
    signs = _pack_bits(words >> 15, 1)
    payload = _PRED.pack(parameter, minimum, step_log2, qwidth)
    payload += coordinates + signs + correction_blob
    components = {"predictor_metadata": _PRED.size,
                  "e8_coordinates_and_coset_flags": len(coordinates),
                  "exact_signs": len(signs),
                  "correction_widths": (len(widths) * 4 + 7) // 8,
                  "exact_corrections": len(correction_blob) - (len(widths) * 4 + 7) // 8}
    return payload, {"coordinate_width": qwidth,
                     "correction_widths": widths.tolist(),
                     "maximum_magnitude_symbol_error": int(radii.max()),
                     "components": components, "step_log2": step_log2}


def _candidate(words, method: str, symbol_step_log2: int, float_scale_multiplier: float):
    if len(words) % 8:
        return None, {"reason": "tail_not_multiple_of_eight"}
    if np.any((words & 0x7F80) == 0x7F80):
        return None, {"reason": "nonfinite_input"}
    if method == "e8_symbol":
        # Source-derived base and fixed configuration are fully serialized.
        magnitude = (words & 0x7FFF).astype(np.int32)
        parameter = int(np.min(magnitude))
        normalized = (magnitude - parameter).astype(np.float64) / (1 << symbol_step_log2)
        result = _one_candidate(words, method, parameter, symbol_step_log2, normalized)
    else:
        values = np.abs(bf16_words_to_float32(words))
        rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        scale = np.float32(rms * float_scale_multiplier if rms > 0 else 1.0)
        parameter = int(float32_to_bf16_words(np.asarray([scale]))[0])
        stored_scale = float(bf16_words_to_float32(np.asarray([parameter], dtype=np.uint16))[0])
        if not math.isfinite(stored_scale) or stored_scale <= 0:
            return None, {"reason": "invalid_scale"}
        normalized = values.astype(np.float64) / stored_scale
        result = _one_candidate(words, method, parameter, 0, normalized)
    if result is None:
        return None, {"reason": "correction_not_useful"}
    return result


def encode_matrix(words: np.ndarray, method: str = "e8_symbol", group_size: int = 64,
                  symbol_step_log2: int = 6,
                  float_scale_multiplier: float = 0.015625) -> EncodedMatrix:
    """Encode [tokens,channels] BF16 words with independent flat groups.

    Configuration is encoder-side only; all decoder-needed choices are in the
    output bytes.  Symbol step is 2**symbol_step_log2 magnitude-word units.
    The fixed group option is deliberately not tuned on the evaluation cache.
    """
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError("expected a uint16 [tokens,channels] matrix")
    if method not in _METHODS or not 8 <= group_size <= 2**32 - 1 or group_size % 8:
        raise ValueError("invalid method or group_size (must be a multiple of eight)")
    if not 1 <= symbol_step_log2 <= 14 or not math.isfinite(float_scale_multiplier) or float_scale_multiplier <= 0:
        raise ValueError("invalid predictor configuration")
    rows, columns = words.shape
    output = bytearray(_HEADER.pack(_MAGIC, _VERSION, _METHODS[method], 0,
                                    group_size, rows, columns))
    components = Counter({"matrix_header": _HEADER.size})
    modes, widths, qwidths, rejected, fallbacks = Counter(), Counter(), Counter(), Counter(), Counter()
    payloads_by_mode = Counter()
    flat = np.ascontiguousarray(words).reshape(-1)
    for start in range(0, len(flat), group_size):
        group = flat[start:start + group_size]
        raw = group.astype("<u2").tobytes()
        candidate, diagnostics = _candidate(group, method, symbol_step_log2, float_scale_multiplier)
        if candidate is not None:
            widths.update(map(str, diagnostics["correction_widths"]))
            qwidths[str(diagnostics["coordinate_width"])] += 1
        if candidate is None or len(candidate) >= len(raw):
            reason = diagnostics.get("reason", "candidate_not_smaller")
            fallbacks[reason] += 1
            if candidate is not None:
                rejected.update(map(str, diagnostics["correction_widths"]))
            mode, payload = _RAW, raw
            components["raw_words"] += len(raw)
        else:
            mode, payload = _E8, candidate
            components.update(diagnostics["components"])
        modes[str(mode)] += 1
        payloads_by_mode[str(mode)] += len(payload)
        components["group_headers"] += _GROUP.size
        output.extend(_GROUP.pack(mode, len(payload)))
        output.extend(payload)
    data = bytes(output)
    assert sum(components.values()) == len(data)
    accepted = widths - rejected
    return EncodedMatrix(data, {
        "method": method, "shape": [rows, columns], "group_size": group_size,
        "groups": sum(modes.values()),
        "raw_payload_bytes": words.size * 2, "serialized_bytes": len(data),
        "payload_bytes": sum(payloads_by_mode.values()),
        "container_overhead_bytes": _HEADER.size + sum(modes.values()) * _GROUP.size,
        "ratio_vs_raw_payload": words.size * 2 / len(data),
        "bits_per_value_including_container": len(data) * 8 / words.size if words.size else None,
        "encoder_config": {"symbol_step_log2": symbol_step_log2,
                           "float_scale_multiplier": float_scale_multiplier},
        "mode_counts": dict(modes), "fallback_counts": dict(fallbacks),
        "candidate_correction_width_histogram": dict(widths),
        "accepted_correction_width_histogram": dict(accepted),
        "candidate_coordinate_width_histogram": dict(qwidths),
        "accepted_payload_bytes_by_mode": dict(payloads_by_mode),
        "component_bytes": dict(components),
        "learned_parameter_bytes": 0, "external_codebook_bytes": 0,
    })


def decode_matrix(encoded: EncodedMatrix | bytes) -> np.ndarray:
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _HEADER.size:
        raise ValueError("truncated E8 matrix header")
    magic, version, method_id, reserved, group_size, rows, columns = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION or method_id not in _METHODS.values() or reserved:
        raise ValueError("unsupported E8 matrix header")
    if group_size < 8 or group_size % 8:
        raise ValueError("invalid E8 group size")
    method = next(k for k, v in _METHODS.items() if v == method_id)
    result = np.empty(rows * columns, dtype=np.uint16)
    cursor = _HEADER.size
    for start in range(0, len(result), group_size):
        count = min(group_size, len(result) - start)
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
                raise ValueError("invalid raw payload length")
            decoded = np.frombuffer(payload, dtype="<u2").astype(np.uint16)
        elif mode == _E8:
            if count % 8 or length < _PRED.size:
                raise ValueError("invalid E8 record shape")
            parameter, minimum, step_log2, qwidth = _PRED.unpack_from(payload)
            if method == "e8_symbol" and not 1 <= step_log2 <= 14:
                raise ValueError("invalid symbol step")
            if method == "e8_float" and step_log2:
                raise ValueError("invalid float step")
            nvectors = count // 8
            qbytes = _coordinates_size(nvectors, qwidth)
            signbytes = (count + 7) // 8
            if length < _PRED.size + qbytes + signbytes:
                raise ValueError("truncated predictor payload")
            qstart = _PRED.size
            points = _unpack_coordinates(payload[qstart:qstart + qbytes], nvectors, minimum, qwidth)
            prediction = _predict(points, parameter, step_log2, method)
            if np.any((prediction & 0x7F80) == 0x7F80):
                raise ValueError("nonfinite predictor")
            sstart = qstart + qbytes
            signs = _unpack_bits(payload[sstart:sstart + signbytes], count, 1)
            decoded = _decode_corrections(payload[sstart + signbytes:], prediction) | (signs << 15)
        else:
            raise ValueError("unknown E8 record mode")
        result[start:start + count] = decoded
    if cursor != len(data):
        raise ValueError("unexpected E8 matrix trailing bytes")
    return result.reshape(rows, columns)

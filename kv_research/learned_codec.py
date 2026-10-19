"""Trained magnitude-symbol vector predictor with exact BF16 corrections.

This is a small learned control, not AQUA-KV or a neural autoencoder. Lloyd
training learns 8-dimensional prototype vectors in reversible BF16 magnitude
symbol space. Encoding picks a prototype by the *actual* exact-correction
width, retains signs, and stores low magnitude bits. Every matrix identifies
the frozen model by SHA256. Models and records are independently serialized.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


_MODEL_HEADER = struct.Struct("<4sBBHII")
_RECORD_HEADER = struct.Struct("<4sBBHIII32sH")
_GROUP_HEADER = struct.Struct("<BI")
_MODEL_MAGIC, _RECORD_MAGIC, _VERSION = b"LKVW", b"LKVR", 1
_DIM = 8


def _pack(values, width):
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    if not 0 <= width <= 16:
        raise ValueError("invalid bit width")
    if width == 0:
        if np.any(values):
            raise ValueError("nonzero zero-width code")
        return b""
    if np.any(values.astype(np.uint32) >= (1 << width)):
        raise ValueError("out-of-range packed code")
    bits = ((values[:, None] >> np.arange(width, dtype=np.uint16)) & 1)
    return np.packbits(bits.reshape(-1).astype(np.uint8), bitorder="little").tobytes()


def _unpack(data, count, width):
    if not 0 <= width <= 16 or len(data) != (count * width + 7) // 8:
        raise ValueError("invalid packed code length")
    if width == 0:
        return np.zeros(count, dtype=np.uint16)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    bits = bits[:count * width].reshape(count, width).astype(np.uint32)
    return np.sum(bits << np.arange(width, dtype=np.uint32), axis=1).astype(np.uint16)


def _pack_corrections(values, widths):
    """Each 8-value vector contributes exactly `width` bytes, without padding."""
    values = np.asarray(values, dtype=np.uint16).reshape(-1, _DIM)
    widths = np.asarray(widths, dtype=np.uint16)
    shifts = np.arange(15, dtype=np.uint16)
    bits = ((values[:, :, None] >> shifts) & 1).astype(np.uint8)
    mask = np.broadcast_to(shifts[None, None, :] < widths[:, None, None], bits.shape)
    return np.packbits(bits[mask], bitorder="little").tobytes()


def _unpack_corrections(data, widths):
    widths = np.asarray(widths, dtype=np.uint16)
    if len(data) != int(widths.sum()):
        raise ValueError("incorrect correction length")
    if not len(data):
        return np.zeros((len(widths), _DIM), dtype=np.uint16)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    starts = np.cumsum(np.r_[0, widths[:-1].astype(np.int64)]) * _DIM
    shifts = np.arange(15, dtype=np.int64)
    indexes = starts[:, None, None] + np.arange(_DIM)[None, :, None] * widths[:, None, None] + shifts
    mask = np.broadcast_to(shifts[None, None, :] < widths[:, None, None], indexes.shape)
    selected = np.zeros_like(indexes, dtype=np.uint16)
    selected[mask] = bits[indexes[mask]]
    return np.sum(selected.astype(np.uint32) << shifts, axis=2).astype(np.uint16)


@dataclass
class LearnedModel:
    codebooks: dict[str, np.ndarray]
    config: dict
    training_stats: dict

    @property
    def model_id(self):
        return hashlib.sha256(model_bytes(self)).hexdigest()

    @property
    def n_centroids(self):
        return int(self.config["n_centroids"])


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def _lloyd(vectors, n_centroids, iterations, rng):
    x = vectors.astype(np.float32)
    if len(x) == 0:
        raise ValueError("no finite training vectors")
    # k-means++ initialization; deterministic for a fixed seed and NumPy build.
    centers = np.empty((n_centroids, _DIM), dtype=np.float32)
    centers[0] = x[rng.integers(len(x))]
    distance = np.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, n_centroids):
        total = float(distance.sum(dtype=np.float64))
        index = int(rng.choice(len(x), p=distance.astype(np.float64) / total)) if total else j % len(x)
        centers[j] = x[index]
        distance = np.minimum(distance, np.sum((x - centers[j]) ** 2, axis=1))
    for _ in range(iterations):
        # No BLAS dependency or nondeterministic parallel reductions.
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignment = distances.argmin(axis=1)
        minimum = distances[np.arange(len(x)), assignment]
        for j in range(n_centroids):
            members = x[assignment == j]
            centers[j] = members.mean(axis=0) if len(members) else x[int(minimum.argmax())]
    quantized = np.clip(np.rint(centers), 0, 0x7F7F).astype(np.uint16)
    distances = np.sum((x[:, None, :] - quantized[None, :, :].astype(np.float32)) ** 2, axis=2)
    return quantized, float(np.min(distances, axis=1).mean())


def train_model(training, config=None):
    """Train frozen dictionaries from {key: uint16[T,C] or list of arrays}.

    Config keys: n_centroids (16, 64, 256), iterations (default 10), seed (19),
    max_samples_per_key (4096). Training uses only magnitude symbols; signs are
    retained verbatim per evaluation record. No evaluation input is used here.
    """
    cfg = {"n_centroids": 16, "iterations": 10, "seed": 19,
           "max_samples_per_key": 4096, "vector_dim": _DIM}
    cfg.update(config or {})
    if cfg["n_centroids"] not in (16, 64, 256) or cfg["vector_dim"] != _DIM:
        raise ValueError("supported codebooks have 16/64/256 8D prototypes")
    if cfg["iterations"] < 1 or cfg["max_samples_per_key"] < cfg["n_centroids"]:
        raise ValueError("invalid training budget")
    if isinstance(training, np.ndarray):
        training = {"global": training}
    rng = np.random.default_rng(cfg["seed"])
    codebooks, statistics = {}, {}
    for key in sorted(training):
        arrays = training[key] if isinstance(training[key], (list, tuple)) else [training[key]]
        vectors = []
        total_vectors, rejected_vectors = 0, 0
        for array in arrays:
            array = np.asarray(array)
            if array.dtype != np.uint16 or array.ndim != 2:
                raise ValueError("training arrays must be uint16 matrices")
            words = np.ascontiguousarray(array).reshape(-1)
            words = words[:words.size // _DIM * _DIM].reshape(-1, _DIM)
            finite = ~np.any((words & 0x7F80) == 0x7F80, axis=1)
            total_vectors += len(words)
            rejected_vectors += int((~finite).sum())
            vectors.append(words[finite] & 0x7FFF)
        vectors = np.concatenate(vectors)
        if len(vectors) > cfg["max_samples_per_key"]:
            vectors = vectors[rng.choice(len(vectors), cfg["max_samples_per_key"], replace=False)]
        prototypes, mse = _lloyd(vectors, cfg["n_centroids"], cfg["iterations"], rng)
        prototypes.flags.writeable = False
        codebooks[str(key)] = prototypes
        statistics[str(key)] = {"available_vectors": total_vectors, "nonfinite_vectors_rejected": rejected_vectors,
                                "sampled_vectors": len(vectors), "training_symbol_mse": mse}
    if not codebooks:
        raise ValueError("empty training collection")
    return LearnedModel(codebooks, cfg, statistics)


def model_bytes(model):
    metadata = json.dumps({"config": model.config, "training_stats": model.training_stats},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = bytearray(_MODEL_HEADER.pack(_MODEL_MAGIC, _VERSION, _DIM, len(model.codebooks),
                                         model.n_centroids, len(metadata)))
    result.extend(metadata)
    for key in sorted(model.codebooks):
        name = key.encode("utf8")
        book = np.asarray(model.codebooks[key])
        if len(name) > 65535 or book.shape != (model.n_centroids, _DIM) or book.dtype != np.uint16:
            raise ValueError("invalid model dictionary")
        if np.any(book > 0x7F7F):
            raise ValueError("nonfinite model prototype")
        result.extend(struct.pack("<H", len(name)))
        result.extend(name)
        result.extend(book.astype("<u2").tobytes())
    return bytes(result)


def save_model(model, path):
    data = model_bytes(model)
    Path(path).write_bytes(data)
    prototype_bytes = sum(book.nbytes for book in model.codebooks.values())
    return {"model_id": hashlib.sha256(data).hexdigest(), "serialized_bytes": len(data),
            "prototype_bytes": prototype_bytes, "metadata_and_header_bytes": len(data) - prototype_bytes,
            "keys": len(model.codebooks), "n_centroids": model.n_centroids}


def load_model(path):
    data = Path(path).read_bytes() if isinstance(path, (str, Path)) else bytes(path)
    if len(data) < _MODEL_HEADER.size:
        raise ValueError("truncated learned model")
    magic, version, dim, nkeys, ncentroids, metalen = _MODEL_HEADER.unpack_from(data)
    if magic != _MODEL_MAGIC or version != _VERSION or dim != _DIM or ncentroids not in (16, 64, 256):
        raise ValueError("invalid learned model header")
    offset = _MODEL_HEADER.size
    if offset + metalen > len(data):
        raise ValueError("truncated learned model metadata")
    try:
        metadata = json.loads(data[offset:offset + metalen])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid learned model metadata") from error
    offset += metalen
    codebooks = {}
    for _ in range(nkeys):
        if offset + 2 > len(data):
            raise ValueError("truncated dictionary key")
        length = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        if offset + length + ncentroids * _DIM * 2 > len(data):
            raise ValueError("truncated dictionary payload")
        key = data[offset:offset + length].decode("utf8")
        offset += length
        book = np.frombuffer(data, dtype="<u2", count=ncentroids * _DIM, offset=offset).reshape(ncentroids, _DIM).copy()
        if key in codebooks or np.any(book > 0x7F7F):
            raise ValueError("invalid learned dictionary")
        book.flags.writeable = False
        codebooks[key] = book
        offset += book.nbytes
    if offset != len(data) or metadata["config"]["n_centroids"] != ncentroids:
        raise ValueError("inconsistent learned model")
    return LearnedModel(codebooks, metadata["config"], metadata["training_stats"])


def _assign(magnitudes, codebook):
    chosen, widths, radii = [], [], []
    # Bound the temporary distance array even for a full un-tiled cache.
    for start in range(0, len(magnitudes), 1024):
        block = magnitudes[start:start + 1024].astype(np.int32)
        radius = np.max(np.abs(block[:, None, :] - codebook[None, :, :].astype(np.int32)), axis=2)
        # Within a group we use the minimal width that guarantees a unique
        # nearest coset representative: radius < 2**(width - 1).
        index = radius.argmin(axis=1)
        best = radius[np.arange(len(block)), index]
        width = np.where(best == 0, 0, np.floor(np.log2(np.maximum(best, 1))).astype(np.int32) + 2)
        chosen.append(index.astype(np.uint16))
        widths.append(width.astype(np.uint16))
        radii.append(best)
    if not chosen:
        return np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.int32)
    return np.concatenate(chosen), np.concatenate(widths), np.concatenate(radii)


def encode_matrix(words, model, key="global", group_size=64):
    """Encode [T,C] BF16 words; no original activations needed for recovery."""
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2 or group_size < _DIM or group_size % _DIM:
        raise ValueError("expected uint16 matrix and group_size multiple of 8")
    if key not in model.codebooks:
        raise ValueError("unknown learned dictionary key")
    ncentroids = model.n_centroids
    qbits = (ncentroids - 1).bit_length()
    name = key.encode("utf8")
    identity = bytes.fromhex(model.model_id)
    rows, columns = words.shape
    output = bytearray(_RECORD_HEADER.pack(_RECORD_MAGIC, _VERSION, qbits, 0, group_size,
                                          rows, columns, identity, len(name)))
    output.extend(name)
    flat = np.ascontiguousarray(words).reshape(-1)
    padded = np.pad(flat, (0, (-len(flat)) % _DIM)).reshape(-1, _DIM)
    indexes, widths, radii = _assign(padded & 0x7FFF, model.codebooks[key])
    components = Counter({"matrix_header_and_model_identity": _RECORD_HEADER.size + len(name)})
    modes, accepted_widths, candidate_widths, reasons = Counter(), Counter(), Counter(), Counter()
    for start in range(0, flat.size, group_size):
        original = flat[start:start + group_size]
        first, last = start // _DIM, (start + len(original) + _DIM - 1) // _DIM
        ids, ws = indexes[first:last], widths[first:last]
        count = last - first
        candidate_widths.update(map(int, ws))
        raw = original.astype("<u2").tobytes()
        finite = not np.any((original & 0x7F80) == 0x7F80)
        payload = None
        if finite and np.all(ws < 16):
            code_data = _pack(ids, qbits)
            width_data = _pack(ws, 4)
            sign_data = _pack(padded[first:last].reshape(-1) >> 15, 1)
            correction_data = _pack_corrections(padded[first:last] & 0x7FFF, ws)
            payload = code_data + width_data + sign_data + correction_data
        if payload is None or len(payload) >= len(raw):
            mode, payload = 0, raw
            reasons["nonfinite" if not finite else "width_16" if np.any(ws >= 16) else "candidate_not_smaller"] += 1
            components["raw_payload"] += len(payload)
        else:
            mode = 1
            accepted_widths.update(map(int, ws))
            components["code_indices"] += len(code_data)
            components["width_metadata"] += len(width_data)
            components["exact_signs"] += len(sign_data)
            components["exact_corrections"] += len(correction_data)
        output.extend(_GROUP_HEADER.pack(mode, len(payload)))
        output.extend(payload)
        components["group_headers"] += _GROUP_HEADER.size
        modes[str(mode)] += 1
    statistics = {"raw_payload_bytes": int(words.nbytes), "serialized_bytes": len(output),
                  "model_id": model.model_id, "dictionary_key": key, "n_centroids": ncentroids,
                  "component_bytes": dict(components), "mode_counts": dict(modes), "fallback_counts": dict(reasons),
                  "accepted_correction_width_histogram": {str(k): v for k, v in sorted(accepted_widths.items())},
                  "candidate_correction_width_histogram": {str(k): v for k, v in sorted(candidate_widths.items())}}
    assert sum(components.values()) == len(output)
    return EncodedMatrix(bytes(output), statistics)


def decode_matrix(encoded, model):
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _RECORD_HEADER.size:
        raise ValueError("truncated learned matrix")
    magic, version, qbits, flags, group_size, rows, columns, identity, keylen = _RECORD_HEADER.unpack_from(data)
    if magic != _RECORD_MAGIC or version != _VERSION or flags or group_size < _DIM or group_size % _DIM:
        raise ValueError("invalid learned matrix header")
    if identity.hex() != model.model_id:
        raise ValueError("learned model SHA256 mismatch")
    if qbits != (model.n_centroids - 1).bit_length():
        raise ValueError("inconsistent learned index width")
    offset = _RECORD_HEADER.size
    if offset + keylen > len(data):
        raise ValueError("truncated dictionary name")
    key = data[offset:offset + keylen].decode("utf8")
    offset += keylen
    if key not in model.codebooks:
        raise ValueError("unknown learned dictionary key")
    total = rows * columns
    flat = np.empty(total, dtype=np.uint16)
    for start in range(0, total, group_size):
        if offset + _GROUP_HEADER.size > len(data):
            raise ValueError("truncated learned group")
        mode, length = _GROUP_HEADER.unpack_from(data, offset)
        offset += _GROUP_HEADER.size
        if offset + length > len(data):
            raise ValueError("truncated learned group payload")
        payload = data[offset:offset + length]
        offset += length
        count = min(group_size, total - start)
        nvec = (count + _DIM - 1) // _DIM
        if mode == 0:
            if length != count * 2:
                raise ValueError("incorrect raw group length")
            result = np.frombuffer(payload, dtype="<u2")
        elif mode == 1:
            qlength, wlength, slength = (nvec * qbits + 7) // 8, (nvec * 4 + 7) // 8, nvec
            if qlength + wlength + slength > length:
                raise ValueError("truncated learned control codes")
            indexes = _unpack(payload[:qlength], nvec, qbits)
            widths = _unpack(payload[qlength:qlength + wlength], nvec, 4)
            signs = _unpack(payload[qlength + wlength:qlength + wlength + slength], nvec * _DIM, 1).reshape(nvec, _DIM)
            residues = _unpack_corrections(payload[qlength + wlength + slength:], widths).astype(np.int32)
            prediction = model.codebooks[key][indexes].astype(np.int32)
            modulus = 1 << widths.astype(np.int32)
            half = np.where(widths == 0, 0, modulus // 2)
            recovered = prediction + ((residues - prediction + half[:, None]) & (modulus[:, None] - 1)) - half[:, None]
            if np.any((recovered < 0) | (recovered > 0x7FFF)):
                raise ValueError("decoded BF16 magnitude outside range")
            result = (recovered.astype(np.uint16) | (signs << 15)).reshape(-1)[:count]
        else:
            raise ValueError("unknown learned group mode")
        flat[start:start + count] = result
    if offset != len(data):
        raise ValueError("unexpected trailing learned bytes")
    return flat.reshape(rows, columns)

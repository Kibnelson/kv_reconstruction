"""Learned circular VQ plus exact modular cosets for BF16 sign/fraction bytes.

The exponent stream is decoded by the enclosing codec. This candidate uses
those known exponents only for an optional frozen mean-fraction lookup, then
looks up an 8D learned fraction prototype. Exact modular residues recover all
seven fraction bits; signs remain verbatim. Actual bytes and frozen model
state count even when the surrounding format elects to use its baseline.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from kv_research.learned_codec import _pack, _unpack, _pack_corrections, _unpack_corrections

_MODEL = struct.Struct("<4sBBHII")
_RECORD = struct.Struct("<4sBBHII32sH")
_DIM = 8


@dataclass
class LowerModel:
    codebooks: dict
    exponent_means: dict
    config: dict
    training_stats: dict

    @property
    def model_id(self):
        return hashlib.sha256(model_bytes(self)).hexdigest()

    @property
    def n_centroids(self):
        return int(self.config["n_centroids"])


@dataclass(frozen=True)
class EncodedLower:
    data: bytes
    stats: dict


def _distances(x, centers):
    delta = np.abs(x[:, None, :].astype(np.int16) - centers[None, :, :].astype(np.int16))
    return np.minimum(delta, 128 - delta)


def _train_book(x, count, iterations, rng):
    centers = np.empty((count, _DIM), dtype=np.uint8)
    centers[0] = x[rng.integers(len(x))]
    best = np.sum(_distances(x, centers[:1]).astype(np.float32) ** 2, axis=2)[:, 0]
    for j in range(1, count):
        total = best.sum(dtype=np.float64)
        index = int(rng.choice(len(x), p=best.astype(np.float64) / total)) if total else j % len(x)
        centers[j] = x[index]
        best = np.minimum(best, np.sum(_distances(x, centers[j:j + 1]).astype(np.float32) ** 2, axis=2)[:, 0])
    for _ in range(iterations):
        distances = np.sum(_distances(x, centers).astype(np.float32) ** 2, axis=2)
        assignment = distances.argmin(axis=1)
        for j in range(count):
            members = x[assignment == j]
            if not len(members):
                centers[j] = x[int(distances.min(axis=1).argmax())]
                continue
            angle = members.astype(np.float64) * (2 * np.pi / 128)
            mean = np.arctan2(np.sin(angle).sum(axis=0), np.cos(angle).sum(axis=0))
            centers[j] = (np.rint(mean * (128 / (2 * np.pi))).astype(np.int32) & 127).astype(np.uint8)
    error = np.min(np.sum(_distances(x, centers).astype(np.float32) ** 2, axis=2), axis=1).mean()
    centers.flags.writeable = False
    return centers, float(error)


def train_model(training, config=None):
    """Fit dictionaries from {key: list of uint16 BF16 matrices}; no dev data."""
    cfg = {"n_centroids": 16, "conditional": False, "iterations": 8,
           "max_samples_per_key": 4096, "seed": 71, "mean_prior_count": 16}
    cfg.update(config or {})
    if cfg["n_centroids"] not in (16, 64) or cfg["iterations"] < 1:
        raise ValueError("unsupported lower dictionary configuration")
    rng = np.random.default_rng(cfg["seed"])
    books, means, statistics = {}, {}, {}
    for key in sorted(training):
        arrays = training[key] if isinstance(training[key], (list, tuple)) else [training[key]]
        words = []
        for array in arrays:
            array = np.asarray(array)
            if array.dtype != np.uint16 or array.ndim != 2:
                raise ValueError("training expects BF16 uint16 matrices")
            words.append(array.reshape(-1))
        flat = np.concatenate(words)
        fraction, exponent = flat & 127, (flat >> 7) & 255
        if cfg["conditional"]:
            counts = np.bincount(exponent, minlength=256)
            total = np.bincount(exponent, weights=fraction, minlength=256)
            prior = cfg["mean_prior_count"]
            mean = np.rint((total + prior * 63.5) / (counts + prior)).astype(np.uint8)
            base = mean[exponent]
            means[key] = mean
        else:
            base = np.full(len(flat), 64, np.uint8)
        vectors = ((fraction.astype(np.int16) - base) & 127).astype(np.uint8)
        vectors = vectors[:len(vectors) // _DIM * _DIM].reshape(-1, _DIM)
        available = len(vectors)
        if len(vectors) > cfg["max_samples_per_key"]:
            vectors = vectors[rng.choice(len(vectors), cfg["max_samples_per_key"], replace=False)]
        books[key], error = _train_book(vectors, cfg["n_centroids"], cfg["iterations"], rng)
        statistics[key] = {"available_vectors": available, "sampled_vectors": len(vectors),
                           "training_circular_symbol_squared_error": error}
    return LowerModel(books, means, cfg, statistics)


def model_bytes(model):
    metadata = json.dumps({"config": model.config, "training_stats": model.training_stats},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    data = bytearray(_MODEL.pack(b"LFLW", 1, int(model.config["conditional"]),
                                 len(model.codebooks), model.n_centroids, len(metadata)))
    data.extend(metadata)
    for key in sorted(model.codebooks):
        name = key.encode()
        book = np.asarray(model.codebooks[key])
        if book.dtype != np.uint8 or book.shape != (model.n_centroids, _DIM) or np.any(book > 127):
            raise ValueError("invalid fraction codebook")
        data.extend(struct.pack("<H", len(name))); data.extend(name); data.extend(book.tobytes())
        if model.config["conditional"]:
            mean = np.asarray(model.exponent_means[key])
            if mean.dtype != np.uint8 or mean.shape != (256,) or np.any(mean > 127):
                raise ValueError("invalid exponent-conditional table")
            data.extend(mean.tobytes())
    return bytes(data)


def save_model(model, path):
    data = model_bytes(model); Path(path).write_bytes(data)
    arrays = sum(book.nbytes for book in model.codebooks.values()) + sum(mean.nbytes for mean in model.exponent_means.values())
    return {"serialized_bytes": len(data), "model_id": hashlib.sha256(data).hexdigest(),
            "parameter_array_bytes": arrays, "metadata_and_header_bytes": len(data) - arrays,
            "n_centroids": model.n_centroids, "conditional": bool(model.config["conditional"])}


def load_model(path):
    data = Path(path).read_bytes() if isinstance(path, (str, Path)) else bytes(path)
    if len(data) < _MODEL.size:
        raise ValueError("truncated lower model")
    magic, version, conditional, nkeys, ncentroids, metalen = _MODEL.unpack_from(data)
    if magic != b"LFLW" or version != 1 or conditional not in (0, 1) or ncentroids not in (16, 64):
        raise ValueError("invalid lower model")
    offset = _MODEL.size
    if offset + metalen > len(data):
        raise ValueError("truncated lower model metadata")
    metadata = json.loads(data[offset:offset + metalen]); offset += metalen
    books, means = {}, {}
    for _ in range(nkeys):
        if offset + 2 > len(data):
            raise ValueError("truncated lower model key")
        keylen = struct.unpack_from("<H", data, offset)[0]; offset += 2
        length = keylen + ncentroids * _DIM + (256 if conditional else 0)
        if offset + length > len(data):
            raise ValueError("truncated lower model parameters")
        key = data[offset:offset + keylen].decode(); offset += keylen
        if key in books:
            raise ValueError("duplicate lower model dictionary")
        books[key] = np.frombuffer(data, np.uint8, ncentroids * _DIM, offset).reshape(ncentroids, _DIM).copy()
        offset += ncentroids * _DIM
        if conditional:
            means[key] = np.frombuffer(data, np.uint8, 256, offset).copy(); offset += 256
    if offset != len(data) or bool(metadata["config"]["conditional"]) != bool(conditional):
        raise ValueError("inconsistent lower model")
    result = LowerModel(books, means, metadata["config"], metadata["training_stats"])
    if model_bytes(result) != data:
        raise ValueError("noncanonical lower model")
    return result


def _base(model, key, exponents):
    return model.exponent_means[key][exponents] if model.config["conditional"] else np.full(exponents.shape, 64, np.uint8)


def encode_lower(lower, exponents, model, key, force_cosets=False):
    """Encode lower fields; force_cosets is an expansion-allowed diagnostic.

    The default preserves raw fallback. Forcing stores a prototype and modular
    residue for every vector, including full seven-bit residues when necessary;
    it does not imply compression and is never used during model training.
    """
    lower, exponents = np.asarray(lower), np.asarray(exponents)
    if lower.dtype != np.uint8 or lower.ndim != 2 or exponents.dtype != np.uint8 or exponents.shape != lower.shape:
        raise ValueError("expected matching uint8 lower/exponent matrices")
    if key not in model.codebooks:
        raise ValueError("unknown lower model dictionary")
    name = key.encode(); qbits = (model.n_centroids - 1).bit_length()
    flat = lower.reshape(-1); exp = exponents.reshape(-1)
    padding = (-len(flat)) % _DIM
    vectors = np.pad(flat, (0, padding)).reshape(-1, _DIM)
    bases = np.pad(_base(model, key, exp), (0, padding), constant_values=64).reshape(-1, _DIM)
    ids, widths = [], []
    for start in range(0, len(vectors), 1024):
        prediction = (model.codebooks[key][None, :, :].astype(np.uint16) + bases[start:start + 1024, None, :]) & 127
        distance = np.abs((vectors[start:start + 1024, None, :] & 127).astype(np.int16) - prediction.astype(np.int16))
        radius = np.minimum(distance, 128 - distance).max(axis=2)
        chosen = radius.argmin(axis=1); best = radius[np.arange(len(chosen)), chosen]
        width = np.where(best == 0, 0, np.minimum(7, np.floor(np.log2(np.maximum(best, 1))).astype(np.int32) + 2))
        ids.append(chosen.astype(np.uint16)); widths.append(width.astype(np.uint16))
    ids = np.concatenate(ids) if ids else np.empty(0, np.uint16)
    widths = np.concatenate(widths) if widths else np.empty(0, np.uint16)
    predicted = (np.ones(len(vectors), dtype=bool) if force_cosets else
                 qbits + 3 + 8 + 8 * widths.astype(np.int32) < 64)
    # A bitmap says which vectors use the predictor; all other vectors remain
    # literal. Both alternatives count this bitmap and all packed padding.
    components = {"lower_header_and_model_identity": _RECORD.size + len(name)}
    bitmap = _pack(predicted.astype(np.uint16), 1)
    code_data = _pack(ids[predicted], qbits)
    width_data = _pack(widths[predicted], 3)
    signs = _pack(vectors[predicted].reshape(-1) >> 7, 1)
    corrections = _pack_corrections(vectors[predicted] & 127, widths[predicted])
    literals = vectors[~predicted].tobytes()
    candidate = bitmap + code_data + width_data + signs + corrections + literals
    mode = int(force_cosets or len(candidate) < len(flat))
    if mode:
        payload = candidate
        components.update(vector_bitmap=len(bitmap), code_indices=len(code_data), width_metadata=len(width_data),
                          exact_signs=len(signs), modular_corrections=len(corrections), raw_lower_vectors=len(literals))
    else:
        payload = flat.tobytes(); components["raw_lower_payload"] = len(payload)
    header = _RECORD.pack(b"LFLR", 1, mode, 0, *lower.shape, bytes.fromhex(model.model_id), len(name)) + name
    data = header + payload
    stats = {"serialized_bytes": len(data), "raw_lower_bytes": lower.nbytes, "mode": "predicted" if mode else "raw_fallback",
             "forced_coset_diagnostic": bool(force_cosets),
             "model_id": model.model_id, "dictionary_key": key, "candidate_payload_bytes": len(candidate),
             "candidate_predicted_vectors": int(predicted.sum()), "candidate_raw_vectors": int((~predicted).sum()),
             "candidate_correction_width_histogram": {str(k): v for k, v in sorted(Counter(map(int, widths)).items())},
             "component_bytes": components}
    assert sum(components.values()) == len(data)
    return EncodedLower(data, stats)


def decode_lower(encoded, exponents, model, key):
    data = encoded.data if isinstance(encoded, EncodedLower) else bytes(encoded)
    exponents = np.asarray(exponents)
    if len(data) < _RECORD.size:
        raise ValueError("truncated lower record")
    magic, version, mode, reserved, rows, columns, identity, keylen = _RECORD.unpack_from(data)
    if magic != b"LFLR" or version != 1 or mode not in (0, 1) or reserved:
        raise ValueError("invalid lower record")
    if identity.hex() != model.model_id:
        raise ValueError("lower model SHA256 mismatch")
    if exponents.dtype != np.uint8 or exponents.shape != (rows, columns):
        raise ValueError("inconsistent decoded exponents")
    offset = _RECORD.size
    if offset + keylen > len(data) or data[offset:offset + keylen].decode() != key or key not in model.codebooks:
        raise ValueError("lower model dictionary key mismatch")
    offset += keylen
    total, nvec = rows * columns, (rows * columns + 7) // 8
    if mode == 0:
        if len(data) - offset != total:
            raise ValueError("incorrect literal lower size")
        return np.frombuffer(data, np.uint8, total, offset).copy().reshape(rows, columns)
    qbits = (model.n_centroids - 1).bit_length()
    def read(length):
        nonlocal offset
        if offset + length > len(data):
            raise ValueError("truncated modular lower payload")
        result = data[offset:offset + length]; offset += length
        return result
    predicted = _unpack(read((nvec + 7) // 8), nvec, 1).astype(bool)
    count = int(predicted.sum())
    ids = _unpack(read((count * qbits + 7) // 8), count, qbits)
    widths = _unpack(read((count * 3 + 7) // 8), count, 3)
    signs = _unpack(read(count), count * _DIM, 1).reshape(-1, _DIM)
    residues = _unpack_corrections(read(int(widths.sum())), widths).astype(np.int32)
    literal = np.frombuffer(read(int((~predicted).sum()) * _DIM), np.uint8).reshape(-1, _DIM)
    if offset != len(data):
        raise ValueError("unexpected lower trailing bytes")
    base = np.pad(_base(model, key, exponents.reshape(-1)), (0, nvec * _DIM - total), constant_values=64).reshape(-1, _DIM)
    prediction = ((model.codebooks[key][ids].astype(np.int32) + base[predicted]) & 127)
    modulus = 1 << widths.astype(np.int32); half = np.where(widths == 0, 0, modulus // 2)
    fraction = (prediction + ((residues - prediction + half[:, None]) & (modulus[:, None] - 1)) - half[:, None]) & 127
    result = np.empty((nvec, _DIM), np.uint8)
    result[predicted] = fraction.astype(np.uint8) | (signs.astype(np.uint8) << 7)
    result[~predicted] = literal
    return result.reshape(-1)[:total].reshape(rows, columns)

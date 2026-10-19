"""Frozen full-magnitude VQ predictor with matched exact correction controls.

Every learned record stores prototype IDs, exact signs, and a complete exact
correction stream. Corrections are either per-8D modular cosets or a signed
integer entropy stream. No exponent stream, other KV, or activations are inputs
to reconstruction. The frozen model is separately serialized and charged by
the surrounding benchmark. CRC32 covers every record byte except its own field.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct
import zlib

import numpy as np

from kv_research.learned_codec import _pack, _unpack, _pack_corrections, _unpack_corrections

_HEADER = struct.Struct("<4sBBBBII32sHI")
_CRC = struct.Struct("<I")
_DIM = 8
_ENTROPY_STRATEGIES = {"raw", "rice", "zigzag_deflate", "centered_deflate",
                       "centered_packed", "zigzag_packed", "best_entropy"}


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def _assign(magnitudes, book, assignment):
    if assignment not in ("radius", "l1"):
        raise ValueError("assignment must be radius or l1")
    chosen = []
    for start in range(0, len(magnitudes), 1024):
        error = np.abs(magnitudes[start:start + 1024, None, :].astype(np.int32)
                       - book[None, :, :].astype(np.int32))
        score = error.max(axis=2) if assignment == "radius" else error.sum(axis=2)
        chosen.append(score.argmin(axis=1).astype(np.uint16))
    return np.concatenate(chosen) if chosen else np.empty(0, np.uint16)


def _cosets(magnitudes, prediction):
    radius = np.abs(magnitudes.astype(np.int32) - prediction.astype(np.int32)).max(axis=1)
    widths = np.where(radius == 0, 0, np.minimum(15,
                        np.floor(np.log2(np.maximum(radius, 1))).astype(np.int32) + 2)).astype(np.uint16)
    width_data = _pack(widths, 4)
    corrections = _pack_corrections(magnitudes, widths)
    return width_data + corrections, {"correction_width_metadata": len(width_data),
                                      "coset_residues": len(corrections)}, widths


def _entropy(residual, strategy):
    from kv_residual import entropy
    return entropy.encode(residual.reshape(-1).astype(np.int32),
                          strategy="best" if strategy == "best_entropy" else strategy)


def encode_matrix(words, model, key, correction="best", assignment="radius", force=True):
    """Encode all BF16 word patterns exactly using a frozen 8D dictionary.

    force=True is the mechanism diagnostic: predictor plus corrections remains
    stored even if it loses to raw. force=False allows a whole-matrix literal
    alternative, with the model state still charged by the benchmark. best
    compares complete coset and entropy-best payloads under identical IDs.
    """
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError("expected a uint16 BF16 matrix")
    if key not in model.codebooks:
        raise ValueError("unknown learned dictionary key")
    if correction not in _ENTROPY_STRATEGIES | {"coset", "best"}:
        raise ValueError("unknown correction strategy")
    book = np.asarray(model.codebooks[key])
    if book.dtype != np.uint16 or book.shape != (model.n_centroids, _DIM) or np.any(book > 0x7FFF):
        raise ValueError("invalid frozen magnitude dictionary")
    flat = np.ascontiguousarray(words).reshape(-1)
    vectors = np.pad(flat, (0, (-len(flat)) % _DIM)).reshape(-1, _DIM)
    magnitudes = vectors & 0x7FFF
    indexes = _assign(magnitudes, book, assignment)
    prediction = book[indexes]
    residual = magnitudes.astype(np.int32) - prediction.astype(np.int32)
    qbits = (model.n_centroids - 1).bit_length()
    code_data = _pack(indexes, qbits)
    signs = _pack(vectors.reshape(-1) >> 15, 1)
    coset_data, coset_components, widths = _cosets(magnitudes, prediction)
    candidates = {"coset": len(coset_data)}
    entropy_stats = None
    if correction in ("coset", "best"):
        mode, correction_data, correction_components = 1, coset_data, coset_components
    if correction != "coset":
        result = _entropy(residual, "best_entropy" if correction == "best" else correction)
        candidates["entropy"] = len(result.data)
        entropy_stats = result.stats
        if correction != "best" or len(result.data) < len(coset_data):
            mode, correction_data = 2, result.data
            # Preserve the entropy stream's exact component accounting where
            # available, including all inner headers, tables and checksums.
            inner = result.stats.get("component_bytes", {})
            if inner and sum(inner.values()) == len(result.data):
                correction_components = {"correction_" + name: size for name, size in inner.items()}
            else:
                correction_components = {"entropy_correction_stream": len(result.data)}
    name = key.encode()
    components = {"record_header_and_model_identity": _HEADER.size,
                  "dictionary_key": len(name), "record_checksum": _CRC.size,
                  "prototype_indices": len(code_data), "exact_signs": len(signs), **correction_components}
    learned_payload = code_data + signs + correction_data
    candidate_mode = mode
    if not force and len(learned_payload) >= len(flat) * 2:
        mode, payload, correction_length = 0, flat.astype("<u2").tobytes(), 0
        components = {"record_header_and_model_identity": _HEADER.size,
                      "dictionary_key": len(name), "record_checksum": _CRC.size,
                      "raw_payload": len(payload)}
    else:
        payload, correction_length = learned_payload, len(correction_data)
    header = _HEADER.pack(b"KVRL", 1, mode, int(assignment == "l1"), qbits,
                          *words.shape, bytes.fromhex(model.model_id), len(name), correction_length)
    body = header + name + payload
    data = body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)
    hist = {str(k): v for k, v in sorted(Counter(map(int, widths)).items())}
    stats = {
        "raw_payload_bytes": words.nbytes, "serialized_bytes": len(data),
        "component_bytes": components, "model_id": model.model_id, "dictionary_key": key,
        "assignment": assignment, "requested_correction": correction, "force_learned": bool(force),
        "accepted_coding_mode": {0: "raw", 1: "coset", 2: "entropy"}[mode],
        "record_modes": {{0: "raw", 1: "coset", 2: "entropy"}[mode]: 1},
        "mechanism_counts": {"stored_learned_vectors": len(indexes) if mode else 0,
                             "stored_entropy_records": int(mode == 2),
                             "stored_coset_records": int(mode == 1)},
        "candidate_coding_mode": {1: "coset", 2: "entropy"}[candidate_mode],
        "candidate_correction_payload_bytes": candidates,
        "candidate_coset_width_histogram": hist,
        "accepted_coset_width_histogram": hist if mode == 1 else {},
        "residual_min": int(residual.min()) if residual.size else 0,
        "residual_max": int(residual.max()) if residual.size else 0,
        "correction_entropy_stats": entropy_stats if candidate_mode == 2 else None,
        "correction_backend_histogram": {str(entropy_stats['mode']): 1} if mode == 2 else {},
        "stored_predictor_vectors": len(indexes) if mode else 0,
    }
    assert sum(components.values()) == len(data)
    return EncodedMatrix(data, stats)


def decode_matrix(encoded, model):
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _HEADER.size + _CRC.size:
        raise ValueError("truncated learned residual record")
    body, checksum = data[:-_CRC.size], _CRC.unpack_from(data, len(data) - _CRC.size)[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise ValueError("learned residual record checksum mismatch")
    magic, version, mode, assignment, qbits, rows, columns, identity, keylen, correction_length = _HEADER.unpack_from(body)
    if magic != b"KVRL" or version != 1 or mode not in (0, 1, 2) or assignment not in (0, 1):
        raise ValueError("invalid learned residual header")
    if identity.hex() != model.model_id:
        raise ValueError("learned residual model SHA256 mismatch")
    if qbits != (model.n_centroids - 1).bit_length():
        raise ValueError("learned residual index width mismatch")
    offset = _HEADER.size
    if offset + keylen > len(body):
        raise ValueError("truncated learned residual dictionary name")
    key = body[offset:offset + keylen].decode(); offset += keylen
    if key not in model.codebooks:
        raise ValueError("unknown learned residual dictionary")
    total = rows * columns
    if mode == 0:
        if correction_length or len(body) - offset != total * 2:
            raise ValueError("invalid learned residual literal payload")
        return np.frombuffer(body, dtype="<u2", count=total, offset=offset).copy().reshape(rows, columns)
    nvec = (total + _DIM - 1) // _DIM
    code_length, sign_length = (nvec * qbits + 7) // 8, nvec
    if len(body) - offset != code_length + sign_length + correction_length:
        raise ValueError("invalid learned residual payload lengths")
    indexes = _unpack(body[offset:offset + code_length], nvec, qbits); offset += code_length
    signs = _unpack(body[offset:offset + sign_length], nvec * _DIM, 1).reshape(nvec, _DIM); offset += sign_length
    prediction = model.codebooks[key][indexes].astype(np.int32)
    payload = body[offset:]
    if mode == 1:
        width_length = (nvec * 4 + 7) // 8
        if len(payload) < width_length:
            raise ValueError("truncated coset widths")
        widths = _unpack(payload[:width_length], nvec, 4)
        residues = _unpack_corrections(payload[width_length:], widths).astype(np.int32)
        modulus = 1 << widths.astype(np.int32)
        half = np.where(widths == 0, 0, modulus // 2)
        magnitude = (prediction + ((residues - prediction + half[:, None]) & (modulus[:, None] - 1)) - half[:, None]) & 0x7FFF
    else:
        from kv_residual import entropy
        residual = entropy.decode(payload)
        if residual.dtype != np.int32 or residual.shape != (nvec * _DIM,):
            raise ValueError("inconsistent signed correction shape")
        magnitude = prediction + residual.reshape(nvec, _DIM)
        if np.any((magnitude < 0) | (magnitude > 0x7FFF)):
            raise ValueError("decoded BF16 magnitude outside range")
    words = (magnitude.astype(np.uint16) | (signs << 15)).reshape(-1)[:total]
    return words.reshape(rows, columns)

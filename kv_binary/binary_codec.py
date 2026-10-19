"""RaBitQ-inspired binary direction predictor with exact KV corrections.

This is an independently implemented lossless record codec, not a reproduction
of RaBitQ's ANN distance estimator.  Coordinates are BF16 magnitude-word
integers, not floating-point KV values.  A frozen learned 8D dictionary is the
first predictor; the remaining signed residual is approximated by a binary
direction after an identity, dense orthogonal, or Walsh-Hadamard transform.
Actual stored float32 or BF16 amplitudes and fixed-order float64 arithmetic
define the second predictor.  A self-contained exact signed residual stream restores all
original bits.  Records and transform state have complete CRC32 checksums.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import zlib

import numpy as np

from kv_research.learned_codec import _pack, _unpack
from kv_residual.learned import _assign
from kv_residual import entropy
from kv_smoke.codec import float32_to_bf16_words, bf16_words_to_float32


_STATE = struct.Struct("<4sBBBBI32sI")
_RECORD = struct.Struct("<4sBBBBII32s32sH")
_VERSION = 1
_TRANSFORMS = {"identity": 0, "dense": 1, "hadamard": 2}
_ASSIGNMENTS = {"radius": 0, "l1": 1}
_DIMENSIONS = (8, 32, 64)


def canonical_matmul(left, matrix):
    """Fixed input-dimension order; separate float64 multiply and add ufuncs.

    Neither BLAS nor a reduction tree is used.  Batch size/segmentation cannot
    alter the order of operations for a coordinate.  IEEE float64 round-to-
    nearest behavior and NumPy's explicit non-fused multiply/add are required.
    """
    left = np.asarray(left, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if left.ndim != 2 or matrix.ndim != 2 or left.shape[1] != matrix.shape[0]:
        raise ValueError("incompatible canonical matrix dimensions")
    result = np.zeros((left.shape[0], matrix.shape[1]), dtype=np.float64)
    for coordinate in range(left.shape[1]):
        product = np.multiply(left[:, coordinate:coordinate + 1], matrix[coordinate:coordinate + 1, :])
        np.add(result, product, out=result)
    return result


def canonical_row_dot(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("incompatible canonical dot dimensions")
    result = np.zeros(len(left), dtype=np.float64)
    for coordinate in range(left.shape[1]):
        product = np.multiply(left[:, coordinate], right[:, coordinate])
        np.add(result, product, out=result)
    return result


def _procedural_rotation(dimension, transform):
    if transform == "identity":
        return np.eye(dimension, dtype=np.float64)
    if transform != "hadamard":
        raise ValueError("dense rotation requires its explicit serialized matrix")
    result = np.ones((1, 1), dtype=np.float64)
    while len(result) < dimension:
        result = np.block([[result, result], [result, -result]])
    return np.divide(result, np.sqrt(np.float64(dimension)))


def _validate_rotation(rotation, dimension):
    if rotation.shape != (dimension, dimension) or rotation.dtype != np.float64 or not np.all(np.isfinite(rotation)):
        raise ValueError("invalid rotation matrix")
    gram = canonical_matmul(rotation.T, rotation)
    if np.max(np.abs(gram - np.eye(dimension))) > 1e-12:
        raise ValueError("rotation is not orthogonal within the stated float64 tolerance")


@dataclass(frozen=True)
class BinaryState:
    dimension: int
    transform: str
    seed: int
    model_id: str
    rotation: np.ndarray

    @property
    def state_id(self):
        return hashlib.sha256(state_bytes(self)).hexdigest()

    @property
    def runtime_rotation_array_bytes(self):
        return int(self.rotation.nbytes)


@dataclass(frozen=True)
class EncodedMatrix:
    data: bytes
    stats: dict


def make_state(model, dimension=8, transform="dense", seed=104729):
    """Create one global transform state per D; never fit it to KV inputs.

    Dense QR generation may depend on the NumPy/LAPACK build, so the complete
    resulting float64 matrix is serialized.  Decoder loading never reruns QR.
    Identity/Hadamard are procedural and require only the state descriptor.
    """
    if dimension not in _DIMENSIONS or transform not in _TRANSFORMS or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("unsupported binary transform configuration")
    if transform == "dense":
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((dimension, dimension))
        rotation, triangular = np.linalg.qr(matrix)
        signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
        rotation = np.multiply(rotation, signs[None, :]).astype(np.float64)
    else:
        rotation = _procedural_rotation(dimension, transform)
    rotation = np.ascontiguousarray(rotation)
    _validate_rotation(rotation, dimension)
    rotation.flags.writeable = False
    return BinaryState(dimension, transform, seed, model.model_id, rotation)


def state_bytes(state):
    if state.dimension not in _DIMENSIONS or state.transform not in _TRANSFORMS:
        raise ValueError("invalid transform state")
    if state.rotation.shape != (state.dimension, state.dimension) or state.rotation.dtype != np.float64:
        raise ValueError("invalid runtime rotation array")
    if state.transform == "dense":
        payload = state.rotation.astype("<f8").tobytes()
    else:
        expected = _procedural_rotation(state.dimension, state.transform)
        if not np.array_equal(expected, state.rotation):
            raise ValueError("procedural runtime rotation differs from its descriptor")
        payload = b""
    header = _STATE.pack(b"KVBQ", _VERSION, _TRANSFORMS[state.transform], state.dimension, 0,
        state.seed, bytes.fromhex(state.model_id), len(payload))
    body = header + payload
    return body + struct.pack("<I", zlib.crc32(body))


def save_state(state, path):
    data = state_bytes(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(state_id=hashlib.sha256(data).hexdigest(), model_id=state.model_id,
        dimension=state.dimension, transform=state.transform, serialized_state_bytes=len(data),
        runtime_rotation_array_bytes=state.runtime_rotation_array_bytes)


def load_state(source):
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if len(data) < _STATE.size + 4:
        raise ValueError("truncated binary transform state")
    if zlib.crc32(data[:-4]) != struct.unpack_from("<I", data, len(data) - 4)[0]:
        raise ValueError("binary state checksum mismatch")
    magic, version, transform_id, dimension, flags, seed, model_id, length = _STATE.unpack_from(data)
    if (magic != b"KVBQ" or version != _VERSION or transform_id not in _TRANSFORMS.values()
            or dimension not in _DIMENSIONS or flags or len(data) != _STATE.size + length + 4):
        raise ValueError("invalid binary transform state")
    transform = next(name for name, identifier in _TRANSFORMS.items() if identifier == transform_id)
    if transform == "dense":
        if length != dimension * dimension * 8:
            raise ValueError("incorrect dense rotation payload length")
        rotation = np.frombuffer(data, dtype="<f8", count=dimension * dimension,
            offset=_STATE.size).reshape(dimension, dimension).astype(np.float64).copy()
    else:
        if length:
            raise ValueError("procedural transform has an unexpected matrix payload")
        rotation = _procedural_rotation(dimension, transform)
    _validate_rotation(rotation, dimension)
    rotation.flags.writeable = False
    result = BinaryState(dimension, transform, seed, model_id.hex(), rotation)
    if state_bytes(result) != data:
        raise ValueError("noncanonical transform state")
    return result


def direction_from_bits(sign_bits, state):
    bits = np.asarray(sign_bits)
    if bits.ndim != 2 or bits.shape[1] != state.dimension or np.any((bits != 0) & (bits != 1)):
        raise ValueError("expected binary directions with the transform dimension")
    signs = np.subtract(np.multiply(bits.astype(np.float64), 2.0), 1.0)
    direction = canonical_matmul(signs, state.rotation.T)
    return np.divide(direction, np.sqrt(np.float64(state.dimension)))


def reconstruct_prediction(prototypes, sign_bits, stored_amplitudes, state):
    """Canonical decoder predictor; shared verbatim with the encoder.

    Prediction is round-to-even(c + float64(stored_float32_a) * u), clipped to
    the 15-bit magnitude domain.  Rounding the combined sum pins odd/even-c
    half ties explicitly; no original residual or unrounded amplitude enters.
    """
    c = np.asarray(prototypes)
    amplitudes = np.asarray(stored_amplitudes)
    if c.ndim != 2 or c.shape[1] != state.dimension or amplitudes.shape != (len(c),) or amplitudes.dtype != np.float32:
        raise ValueError("incompatible binary predictor arrays")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
        raise ValueError("amplitudes must be finite nonnegative stored float32 values")
    direction = direction_from_bits(sign_bits, state)
    if direction.shape != c.shape:
        raise ValueError("binary direction count mismatch")
    contribution = np.multiply(amplitudes.astype(np.float64)[:, None], direction)
    prediction = np.rint(np.add(c.astype(np.float64), contribution))
    return np.clip(prediction, 0, 32767).astype(np.uint16)


def encode_matrix(words, model, state, key="global", assignment="radius", force=True,
                  amplitude_precision="f32"):
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2 or assignment not in _ASSIGNMENTS:
        raise ValueError("expected uint16 matrix and a supported assignment")
    if amplitude_precision not in ("f32", "bf16"):
        raise ValueError("unsupported stored amplitude precision")
    if key not in model.codebooks or state.model_id != model.model_id:
        raise ValueError("learned dictionary/transform model identity mismatch")
    state_id = state.state_id
    name = key.encode("utf8")
    if len(name) > 65535:
        raise ValueError("dictionary name is too long")
    rows, columns = words.shape
    flat = np.ascontiguousarray(words).reshape(-1)
    raw = flat.astype("<u2").tobytes()
    nonfinite = bool(np.any((flat & 0x7F80) == 0x7F80))
    components = {"matrix_header_and_identities": _RECORD.size + len(name), "matrix_checksum": 4}
    mechanism = dict(binary_records=0, raw_nonfinite_records=0, raw_fallback_records=0,
        learned_vectors=0, binary_vectors=0, binary_sign_bits=0, original_sign_bits=0,
        amplitudes=0, nonzero_amplitudes=0)
    correction_modes = {}
    residual_summary = {}
    candidate_bytes = 0
    if nonfinite or len(flat) == 0:
        mode, payload = 0, raw
        components["raw_words"] = len(payload)
        mechanism["raw_nonfinite_records"] = int(nonfinite)
    else:
        dimension = state.dimension
        padded = np.pad(flat, (0, (-len(flat)) % dimension))
        magnitudes = padded & 0x7FFF
        ids = _assign(magnitudes.reshape(-1, 8), model.codebooks[key], assignment)
        learned = model.codebooks[key][ids].reshape(-1, dimension)
        residual = magnitudes.reshape(-1, dimension).astype(np.int32) - learned.astype(np.int32)
        projected = canonical_matmul(residual, state.rotation)
        bits = (projected >= 0).astype(np.uint16)  # exact zero has positive sign
        direction = direction_from_bits(bits, state)
        amplitude64 = canonical_row_dot(residual, direction)
        # Tiny negative roundoff is not expected because a = ||R^T r||_1/sqrtD;
        # pin the nonnegative projection family if roundoff yields -0/tiny<0.
        amplitudes = np.maximum(amplitude64, 0.0).astype(np.float32)
        if amplitude_precision == "bf16":
            amplitude_words = float32_to_bf16_words(amplitudes)
            amplitudes = bf16_words_to_float32(amplitude_words)
            amplitude_data = amplitude_words.astype("<u2").tobytes()
        else:
            amplitude_data = amplitudes.astype("<f4").tobytes()
        prediction = reconstruct_prediction(learned, bits, amplitudes, state)
        remaining = magnitudes.astype(np.int32) - prediction.reshape(-1).astype(np.int32)
        correction = entropy.encode(remaining.astype(np.int32), strategy="best")
        ids_data = _pack(ids, (model.n_centroids - 1).bit_length())
        original_signs = _pack(padded >> 15, 1)
        binary_signs = _pack(bits.reshape(-1), 1)
        payload = ids_data + original_signs + binary_signs + amplitude_data + correction.data
        candidate_bytes = len(payload)
        if not force and len(payload) >= len(raw):
            mode, payload = 2, raw
            components["raw_words"] = len(payload)
            mechanism["raw_fallback_records"] = 1
        else:
            mode = 1
            components.update(learned_indices=len(ids_data), exact_original_signs=len(original_signs),
                binary_direction_signs=len(binary_signs))
            components["float32_amplitudes" if amplitude_precision == "f32" else "bf16_amplitudes"] = len(amplitude_data)
            components.update({"correction_" + k: v for k, v in correction.stats["component_bytes"].items()})
            mechanism.update(binary_records=1, learned_vectors=len(ids), binary_vectors=len(amplitudes),
                binary_sign_bits=padded.size, original_sign_bits=padded.size,
                amplitudes=len(amplitudes), nonzero_amplitudes=int(np.count_nonzero(amplitudes)))
            correction_modes[correction.stats["mode"]] = 1
            residual_summary = dict(values=int(padded.size),
                before_abs_sum=int(np.abs(residual.astype(np.int64)).sum()),
                after_abs_sum=int(np.abs(remaining.astype(np.int64)).sum()),
                before_squared_sum=int((residual.astype(np.int64) ** 2).sum()),
                after_squared_sum=int((remaining.astype(np.int64) ** 2).sum()))
    assignment_flags = _ASSIGNMENTS[assignment] | (2 if amplitude_precision == "bf16" else 0)
    header = _RECORD.pack(b"KVBR", _VERSION, mode, assignment_flags, state.dimension,
        rows, columns, bytes.fromhex(model.model_id), bytes.fromhex(state_id), len(name)) + name
    body = header + payload
    data = body + struct.pack("<I", zlib.crc32(body))
    assert sum(components.values()) == len(data)
    mode_name = {0: "raw_nonfinite_or_empty", 1: "binary", 2: "raw_smaller"}[mode]
    return EncodedMatrix(data, dict(method="learned_binary_residual", assignment=assignment,
        force=bool(force), dimension=state.dimension, transform=state.transform,
        amplitude_precision=amplitude_precision,
        model_id=model.model_id, state_id=state_id, dictionary_key=key,
        raw_payload_bytes=int(words.nbytes), serialized_bytes=len(data), component_bytes=components,
        mechanism_counts=mechanism, correction_mode_counts=correction_modes,
        record_modes={mode_name: 1},
        residual_summary=residual_summary, candidate_payload_bytes=candidate_bytes,
        serialized_transform_state_bytes=len(state_bytes(state)),
        runtime_rotation_array_bytes=state.runtime_rotation_array_bytes,
        math_contract="fixed-order float64 mul/add; actual stored amplitude expanded to float32; rint(c+a*u)"))


def record_info(encoded):
    """Check complete-record CRC and expose the actual stored mechanism mode."""
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _RECORD.size + 4:
        raise ValueError("truncated binary matrix")
    if zlib.crc32(data[:-4]) != struct.unpack_from("<I", data, len(data) - 4)[0]:
        raise ValueError("binary matrix checksum mismatch")
    magic, version, mode, assignment, dimension, rows, columns, model_id, state_id, keylen = _RECORD.unpack_from(data)
    if (magic != b"KVBR" or version != _VERSION or mode not in (0, 1, 2)
            or assignment not in (0, 1, 2, 3) or dimension not in _DIMENSIONS
            or len(data) < _RECORD.size + keylen + 4):
        raise ValueError("invalid binary record header")
    try:
        key = data[_RECORD.size:_RECORD.size + keylen].decode("utf8")
    except UnicodeDecodeError as error:
        raise ValueError("invalid binary dictionary key") from error
    return dict(mode={0: "raw_nonfinite_or_empty", 1: "binary", 2: "raw_smaller"}[mode],
        is_binary=mode == 1, assignment=next(k for k, v in _ASSIGNMENTS.items() if v == (assignment & 1)),
        amplitude_precision="bf16" if assignment & 2 else "f32",
        amplitude_itemsize=2 if assignment & 2 else 4,
        dimension=dimension, shape=[rows, columns], dictionary_key=key,
        model_id=model_id.hex(), state_id=state_id.hex(),
        binary_vectors=((rows * columns + dimension - 1) // dimension) if mode == 1 else 0,
        payload_bytes=len(data) - _RECORD.size - keylen - 4, serialized_bytes=len(data))


def decode_matrix(encoded, model, state):
    data = encoded.data if isinstance(encoded, EncodedMatrix) else bytes(encoded)
    if len(data) < _RECORD.size + 4:
        raise ValueError("truncated binary matrix")
    if zlib.crc32(data[:-4]) != struct.unpack_from("<I", data, len(data) - 4)[0]:
        raise ValueError("binary matrix checksum mismatch")
    body = data[:-4]
    magic, version, mode, assignment, dimension, rows, columns, model_id, state_id, keylen = _RECORD.unpack_from(body)
    if magic != b"KVBR" or version != _VERSION or mode not in (0, 1, 2) or assignment not in (0, 1, 2, 3):
        raise ValueError("invalid binary record header")
    if model_id.hex() != model.model_id or state.model_id != model.model_id or state_id.hex() != state.state_id or dimension != state.dimension:
        raise ValueError("binary model/transform state identity mismatch")
    cursor = _RECORD.size
    if cursor + keylen > len(body):
        raise ValueError("truncated binary dictionary key")
    try:
        key = body[cursor:cursor + keylen].decode("utf8")
    except UnicodeDecodeError as error:
        raise ValueError("invalid binary dictionary key") from error
    cursor += keylen
    if key not in model.codebooks:
        raise ValueError("unknown learned dictionary key")
    total = rows * columns
    if mode in (0, 2):
        if len(body) - cursor != 2 * total:
            raise ValueError("incorrect binary raw payload length")
        return np.frombuffer(body, dtype="<u2", count=total, offset=cursor).copy().reshape(rows, columns)
    padded = ((total + dimension - 1) // dimension) * dimension
    nlearned, nvectors = padded // 8, padded // dimension
    qbits = (model.n_centroids - 1).bit_length()
    amplitude_precision = "bf16" if assignment & 2 else "f32"
    idbytes, signbytes = (nlearned * qbits + 7) // 8, (padded + 7) // 8
    ampbytes = nvectors * (2 if amplitude_precision == "bf16" else 4)
    if len(body) < cursor + idbytes + 2 * signbytes + ampbytes:
        raise ValueError("truncated binary predictor fields")
    ids = _unpack(body[cursor:cursor + idbytes], nlearned, qbits)
    cursor += idbytes
    if np.any(ids >= model.n_centroids):
        raise ValueError("invalid learned prototype index")
    original_signs = _unpack(body[cursor:cursor + signbytes], padded, 1)
    cursor += signbytes
    binary_signs = _unpack(body[cursor:cursor + signbytes], padded, 1).reshape(nvectors, dimension)
    cursor += signbytes
    if amplitude_precision == "bf16":
        words = np.frombuffer(body, dtype="<u2", count=nvectors, offset=cursor)
        amplitudes = bf16_words_to_float32(words)
    else:
        amplitudes = np.frombuffer(body, dtype="<f4", count=nvectors, offset=cursor).astype(np.float32).copy()
    cursor += ampbytes
    learned = model.codebooks[key][ids].reshape(nvectors, dimension)
    prediction = reconstruct_prediction(learned, binary_signs, amplitudes, state).reshape(-1)
    remaining = entropy.decode(body[cursor:])
    if remaining.dtype != np.int32 or remaining.shape != (padded,):
        raise ValueError("invalid exact correction shape")
    recovered = prediction.astype(np.int64) + remaining.astype(np.int64)
    if np.any((recovered < 0) | (recovered > 32767)):
        raise ValueError("recovered magnitude outside BF16 domain")
    return (recovered.astype(np.uint16) | (original_signs << 15))[:total].reshape(rows, columns)

"""Framed CPU reference for the team's exact BF16 predictive transfer codec.

The block predictor and group/mantissa probability construction follow the
supplied ``combined.py``.  This module adds explicit frozen state, separate
packet decoding, matched predictor ablations and paid int8 side information.
It is neither a GPU codec nor a cross-platform determinism result: constriction
still converts float64 probability arrays into its finite-precision model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


FORMAT = "team_transfer_reference_v1"
MAGIC = b"KVTRF001"
PREFIX = struct.Struct("<8sI")
METADATA_DIVISOR = 128
_MAX_FIXED = 1 << 40
_MAX_VALUES = 1 << 28


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _constriction():
    try:
        import constriction
    except ImportError as error:
        raise RuntimeError("Install constriction to run the CPU transfer reference") from error
    return constriction


def from_bits(words):
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def words(capture, layer=None):
    k, v = np.asarray(capture["K_bits"]), np.asarray(capture["V_bits"])
    if layer is None:
        return np.concatenate((k, v), axis=2)
    return np.concatenate((k[layer], v[layer]), axis=1)


def _geometry(capture):
    k, v = np.asarray(capture["K_bits"]), np.asarray(capture["V_bits"])
    if k.dtype != np.uint16 or v.dtype != np.uint16 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("ACT K_bits and V_bits must be equal uint16 [layers,tokens,channels]")
    layers, tokens, channels = k.shape
    heads, dim = int(capture["Hk"]), int(capture["hd"])
    base = float(capture["base"])
    if min(layers, tokens, channels, heads, dim) <= 0 or dim % 2 or channels != heads * dim or not np.isfinite(base) or base <= 1:
        raise ValueError("invalid ACT geometry")
    result = dict(layers=layers, channels=channels, heads=heads, head_dim=dim, base=base)
    if "rope_inv_freq" in capture:
        frequency = np.asarray(capture["rope_inv_freq"], dtype=np.float64)
        if frequency.shape != (dim // 2,) or not np.isfinite(frequency).all() or np.any(frequency <= 0):
            raise ValueError("invalid captured RoPE inverse frequencies")
        result["rope_inv_freq"] = frequency.tolist()
    return result


def _rope_frequencies(geometry):
    if "rope_inv_freq" in geometry:
        return np.asarray(geometry["rope_inv_freq"], dtype=np.float64)
    return geometry["base"] ** (-np.arange(0, geometry["head_dim"], 2, dtype=np.float64) / geometry["head_dim"])


def _positions(capture):
    count = np.asarray(capture["K_bits"]).shape[1]
    result = np.asarray(capture.get("positions", np.arange(count)), dtype=np.int64)
    if result.shape != (count,) or np.any(result < 0):
        raise ValueError("ACT positions must be nonnegative [tokens]")
    return result


def _fixed(value, name="features"):
    value = np.asarray(value, dtype=np.float64)
    scaled = np.rint(value * 65536.0)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) > _MAX_FIXED):
        raise ValueError(name + " exceed the finite fixed-point reference domain")
    return scaled.astype(np.int64)


def _rope_float(values, geometry, positions, inverse=False):
    dim = geometry["head_dim"]
    k = values.reshape(len(values), geometry["heads"], dim)
    angles = positions[:, None] * _rope_frequencies(geometry)[None, :]
    cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
    if inverse:
        sine = -sine
    out = np.empty_like(k)
    out[..., 0::2] = k[..., 0::2] * cosine - k[..., 1::2] * sine
    out[..., 1::2] = k[..., 0::2] * sine + k[..., 1::2] * cosine
    return out.reshape(values.shape)


def canonical_features(capture, layer):
    geometry = _geometry(capture)
    value = from_bits(words(capture, layer)).astype(np.float64)
    _fixed(value)  # Explicitly reject unsupported/nonfinite fit inputs.
    channels = geometry["channels"]
    return np.concatenate((_rope_float(value[:, :channels], geometry, _positions(capture), True), value[:, channels:]), axis=1)


def _rope_int(value, ch0, positions, model, inverse=False):
    if value.shape[1] % 2 or ch0 % 2:
        raise ValueError("RoPE groups must contain complete adjacent pairs")
    if np.any(np.abs(value) > _MAX_FIXED):
        raise ValueError("RoPE fixed-point input overflow")
    dim = model.geometry["head_dim"]
    index = ((ch0 + np.arange(0, value.shape[1], 2)) % dim) // 2
    cosine = model.cosine[positions][:, index]
    sine = model.sine[positions][:, index]
    a, b = value[:, 0::2], value[:, 1::2]
    if inverse:
        sine = -sine
    result = np.empty_like(value)
    result[:, 0::2] = (a * cosine - b * sine + (1 << 14)) >> 15
    result[:, 1::2] = (a * sine + b * cosine + (1 << 14)) >> 15
    return result


def _all_fixed(word_matrix, positions, model):
    channels = model.geometry["channels"]
    value = _fixed(from_bits(word_matrix))
    return np.concatenate((_rope_int(value[:, :channels], 0, positions, model, True), value[:, channels:]), axis=1)


def _predict(features, block, ch0, positions, model):
    selected = features[:, block["columns"]]
    matrix = block["A"]
    # Conservative integer overflow proof before NumPy's wrapping int64 matmul.
    maximum = int(np.max(np.abs(selected))) if selected.size else 0
    row_l1 = max((sum(abs(int(x)) for x in row) for row in matrix), default=0)
    if maximum * row_l1 + (1 << 15) >= (1 << 62):
        raise ValueError("predictor int64 accumulation would overflow")
    accumulation = selected @ matrix.T if selected.shape[1] else np.zeros((len(features), len(block["b"])), dtype=np.int64)
    pre = ((accumulation + (1 << 15)) >> 16) + block["b"]
    if np.any(np.abs(pre) > _MAX_FIXED):
        raise ValueError("predictor fixed-point output overflow")
    if ch0 < model.geometry["channels"]:
        pre = _rope_int(pre, ch0, positions, model)
    return pre / 65536.0


def _meta(value, layers, tokens, rank):
    if rank == 0:
        if value is not None and np.asarray(value).size:
            raise ValueError("model has no metadata slots")
        return np.empty((layers, tokens, 0), dtype=np.int8)
    result = np.asarray(value)
    if result.dtype != np.int8 or result.shape != (layers, tokens, rank):
        raise ValueError("metadata must be already-quantized int8 [layers,tokens,rank]")
    return np.ascontiguousarray(result)


@dataclass
class TransferModel:
    geometry: dict
    mode: str
    block_size: int
    metadata_rank: int
    predictors: list
    cosine: np.ndarray
    sine: np.ndarray
    f_tab: np.ndarray
    state_sha256: str = ""


def _arrays(model):
    arrays = dict(cosine=model.cosine, sine=model.sine, f_tab=model.f_tab)
    for layer, blocks in enumerate(model.predictors):
        for index, block in enumerate(blocks):
            for name in ("columns", "A", "b", "inv_s"):
                arrays[f"l{layer:03d}_b{index:04d}_{name}"] = block[name]
    return arrays


def _description(model):
    return dict(format=FORMAT, geometry=model.geometry, mode=model.mode, block_size=model.block_size,
                metadata_rank=model.metadata_rank, metadata_divisor=METADATA_DIVISOR,
                nblocks=[len(blocks) for blocks in model.predictors])


def _state_digest(model):
    digest = hashlib.sha256(_canonical(_description(model)))
    for name, array in sorted(_arrays(model).items()):
        digest.update(name.encode()); digest.update(array.dtype.str.encode())
        digest.update(_canonical(list(array.shape))); digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def model_statistics(model):
    arrays = _arrays(model)
    active = sum(block["A"].size for blocks in model.predictors for block in blocks)
    fits_int32 = all(np.all(block["A"] >= -(1 << 31)) and np.all(block["A"] < (1 << 31))
                     for blocks in model.predictors for block in blocks)
    width = 2 * model.geometry["channels"]
    predicted_active = sum(block["A"].size for blocks in model.predictors[1:] for block in blocks)
    return dict(active_coefficients=int(active), coefficient_array_bytes=int(sum(block["A"].nbytes for blocks in model.predictors for block in blocks)),
                frozen_array_bytes=int(sum(array.nbytes for array in arrays.values())), state_sha256=model.state_sha256,
                active_INT32_hypothetical_bytes=int(active * 4) if fits_int32 else None,
                active_coefficients_fit_INT32=bool(fits_int32),
                predictor_MACs_per_value=float(active / (len(model.predictors) * width)),
                predicted_layers_MACs_per_value=float(predicted_active / ((len(model.predictors) - 1) * width)) if len(model.predictors) > 1 else None,
                layout="active causal coefficient slices stored as INT64; not the original padded dense arrays")


def save_model(model, path):
    model.state_sha256 = _state_digest(model)
    arrays = _arrays(model)
    description = dict(_description(model), state_sha256=model.state_sha256)
    arrays["description"] = np.frombuffer(_canonical(description), dtype=np.uint8)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=str(target), serialized_bytes=target.stat().st_size, **model_statistics(model))


def load_model(path):
    with np.load(path, allow_pickle=False) as source:
        description = json.loads(source["description"].tobytes())
        if description.get("format") != FORMAT or description.get("metadata_divisor") != METADATA_DIVISOR:
            raise ValueError("unsupported frozen transfer state")
        predictors = []
        for layer, count in enumerate(description["nblocks"]):
            blocks = []
            for index in range(count):
                blocks.append({name: source[f"l{layer:03d}_b{index:04d}_{name}"].copy() for name in ("columns", "A", "b", "inv_s")})
            predictors.append(blocks)
        model = TransferModel(description["geometry"], description["mode"], description["block_size"], description["metadata_rank"],
                              predictors, source["cosine"].copy(), source["sine"].copy(), source["f_tab"].copy(), description["state_sha256"])
    if _state_digest(model) != model.state_sha256:
        raise ValueError("frozen transfer state checksum mismatch")
    return model


def context_layers(mode, layer):
    """Earlier same-token layers available to a packet-only decoder.

    The strong-layer ablation adds layer 0 and l-2, deduplicating coincident
    inputs in the first layers. It has no dependency on another token/page.
    """
    if mode == "strong_layers":
        return tuple(sorted({0, layer - 2, layer - 1} & set(range(layer))))
    return (layer - 1,) if layer else ()


def _context_columns(mode, layer, ch0, channels, rank):
    previous = 2 * channels if layer else 0
    metadata = list(range(previous, previous + rank))
    current = previous + rank
    if mode == "mean":
        return metadata
    if mode == "previous":
        return list(range(previous)) + metadata
    if mode == "aqua":
        if ch0 < channels:
            return (list(range(channels)) if layer else []) + metadata
        return (list(range(channels, 2 * channels)) if layer else []) + metadata + list(range(current, current + channels))
    if mode == "strong_layers":
        current = len(context_layers(mode, layer)) * 2 * channels + rank
    return list(range(current + ch0))


def fit(train, scale, mode="block", block_size=32, metadata_train=None, metadata_scale=None,
        max_positions=None, ridge=1e-3):
    """Fit on TRAIN only; estimate scale on a separate calibration capture.

    ``metadata_*`` contain transmitted int8 codes, never raw hidden tensors.
    Their reconstructed numeric features are codes /128.  They must have the
    same layer/token shape as their corresponding ACT capture.
    """
    if not train or mode not in ("mean", "previous", "aqua", "block", "strong_layers") or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("invalid predictor fit configuration")
    geometry = _geometry(train[0]); channels = geometry["channels"]; nv = 2 * channels
    if block_size <= 0 or block_size % 2 or channels % block_size:
        raise ValueError("block_size must divide K width and preserve adjacent RoPE pairs")
    for capture in [*train, scale]:
        if _geometry(capture) != geometry:
            raise ValueError("all captures must share model geometry")
    rank = 0 if metadata_train is None else int(np.asarray(metadata_train[0]).shape[-1])
    if metadata_train is not None and len(metadata_train) != len(train):
        raise ValueError("metadata TRAIN count mismatch")
    metas = [_meta(None if metadata_train is None else metadata_train[i], geometry["layers"], np.asarray(d["K_bits"]).shape[1], rank)
             for i, d in enumerate(train)]
    scale_meta = _meta(metadata_scale, geometry["layers"], np.asarray(scale["K_bits"]).shape[1], rank)
    needed = max(int(_positions(d).max()) + 1 for d in [*train, scale])
    max_positions = needed if max_positions is None else int(max_positions)
    if max_positions < needed or max_positions > 1 << 20:
        raise ValueError("max_positions does not cover calibration positions")
    geometry = dict(geometry, max_positions=max_positions)
    angles = np.arange(max_positions, dtype=np.float64)[:, None] * _rope_frequencies(geometry)[None, :]
    cosine = np.rint(np.cos(angles) * (1 << 15)).astype(np.int64)
    sine = np.rint(np.sin(angles) * (1 << 15)).astype(np.int64)
    f_tab = np.rint(2.0 ** 32 / (1 + np.exp(-(-32.0 + np.arange(16385) / 256)))).astype(np.int64)
    f_tab[0], f_tab[-1] = 0, 1 << 32
    model = TransferModel(geometry, mode, int(block_size), rank, [], cosine, sine, f_tab)
    for layer in range(geometry["layers"]):
        cur = np.concatenate([canonical_features(d, layer) for d in train])
        sources = context_layers(mode, layer)
        previous = np.concatenate([np.concatenate([canonical_features(d, s) for s in sources], axis=1) for d in train]) if sources else np.empty((len(cur), 0))
        meta = np.concatenate([m[layer] for m in metas]).astype(np.float64) / METADATA_DIVISOR
        joint = np.concatenate((previous, meta, cur), axis=1)
        mean = joint.mean(axis=0); centered = joint - mean
        covariance = centered.T @ centered / len(centered)
        regularization = ridge * max(float(np.trace(covariance)) / len(covariance), 1e-16)
        covariance += regularization * np.eye(len(covariance))
        npast = previous.shape[1] + rank
        full_a = None
        if mode in ("block", "strong_layers"):
            lower = np.linalg.cholesky(covariance)
            inverse = np.linalg.solve(lower, np.eye(len(lower)))
            rows = np.arange(npast, npast + nv)
            starts = npast + (np.arange(nv) // block_size) * block_size
            full_a = (lower[rows] * (np.arange(len(lower))[None, :] < starts[:, None])) @ inverse
        scale_words = words(scale, layer)
        scale_positions = _positions(scale)
        scale_cur = _all_fixed(scale_words, scale_positions, model)
        scale_previous = np.concatenate([_all_fixed(words(scale, s), scale_positions, model) for s in sources], axis=1) if sources else np.empty((len(scale_cur), 0), dtype=np.int64)
        scale_features = np.concatenate((scale_previous, scale_meta[layer].astype(np.int64) * (65536 // METADATA_DIVISOR), scale_cur), axis=1)
        blocks = []
        solved_contexts = {}
        for ch0 in range(0, nv, block_size):
            columns = np.asarray(_context_columns(mode, layer, ch0, channels, rank), dtype=np.int32)
            rows = np.arange(npast + ch0, npast + ch0 + block_size)
            if full_a is not None:
                coefficient = full_a[ch0:ch0 + block_size, columns]
            elif len(columns):
                # Previous-only and AQUA-style groups repeat the same context.
                # Factor each distinct system once, for all output channels,
                # rather than repeating a dense solve for every channel group.
                key = tuple(columns)
                if key not in solved_contexts:
                    targets = np.arange(npast, npast + nv)
                    solved_contexts[key] = np.linalg.solve(covariance[np.ix_(columns, columns)],
                        covariance[np.ix_(columns, targets)]).T
                coefficient = solved_contexts[key][ch0:ch0 + block_size]
            else:
                coefficient = np.empty((block_size, 0))
            bias = mean[rows] - coefficient @ mean[columns]
            block = dict(columns=columns, A=_fixed(coefficient, "coefficients"), b=_fixed(bias, "bias"))
            prediction = _predict(scale_features, block, ch0, scale_positions, model)
            residual = np.abs(from_bits(scale_words[:, ch0:ch0 + block_size]).astype(np.float64) - prediction)
            spread = residual.mean(axis=0) / (2 * np.log(2)) + 1e-12
            scale_class = np.rint(8 * np.log2(spread)).astype(np.int64)
            block["inv_s"] = 2.0 ** (-scale_class / 8.0)
            if not np.isfinite(block["inv_s"]).all() or np.any(block["inv_s"] <= 0):
                raise ValueError("invalid calibrated logistic scale")
            blocks.append(block)
        model.predictors.append(blocks)
    model.state_sha256 = _state_digest(model)
    return model


def _shape(model, shape, start):
    if len(shape) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0 for x in shape):
        raise ValueError("invalid transfer tensor shape")
    layers, tokens, width = map(int, shape)
    if layers != model.geometry["layers"] or width != 2 * model.geometry["channels"] or layers * tokens * width > _MAX_VALUES:
        raise ValueError("transfer tensor geometry differs from frozen model")
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start < 0 or start + tokens > model.geometry["max_positions"]:
        raise ValueError("transfer positions outside frozen RoPE tables")
    return layers, tokens, width


def _run(model, shape, start, metadata, correction, residue_bits, source=None, compressed=None):
    from .transfer_coset import code_block
    constriction = _constriction()
    encoding = source is not None
    coder = constriction.stream.queue.RangeEncoder() if encoding else constriction.stream.queue.RangeDecoder(compressed)
    layers, tokens, nv = shape
    positions = np.arange(start, start + tokens)
    previous = np.empty((tokens, 0), dtype=np.int64)
    history = {}
    output = np.empty(shape, dtype=np.uint16)
    ideal = 0.0; layer_bits = []; per_block = []
    for layer in range(layers):
        if model.mode == "strong_layers":
            sources = context_layers(model.mode, layer)
            previous = np.concatenate([history[s] for s in sources], axis=1) if sources else np.empty((tokens, 0), dtype=np.int64)
        current = np.zeros((tokens, nv), dtype=np.int64)
        offset = previous.shape[1] + model.metadata_rank
        features = np.concatenate((previous, metadata[layer].astype(np.int64) * (65536 // METADATA_DIVISOR), current), axis=1)
        total_layer = 0.0
        for index, block in enumerate(model.predictors[layer]):
            ch0 = index * model.block_size
            selection = slice(ch0, ch0 + model.block_size)
            prediction = _predict(features, block, ch0, positions, model)
            inverse = np.broadcast_to(block["inv_s"], prediction.shape)
            target = source[layer, :, selection].ravel() if encoding else None
            recovered, bits = code_block(coder, prediction.ravel(), inverse.ravel(), target, mode=correction,
                                         residue_bits=residue_bits, f_tab=model.f_tab)
            recovered = recovered.reshape(tokens, model.block_size)
            output[layer, :, selection] = recovered
            features[:, offset + ch0:offset + ch0 + model.block_size] = _rope_int(_fixed(from_bits(recovered)), ch0, positions, model, True) if ch0 < model.geometry["channels"] else _fixed(from_bits(recovered))
            total_layer += bits
            per_block.append(dict(layer=layer, channel_start=ch0, values=int(recovered.size), ideal_bits=float(bits)))
        previous = features[:, offset:].copy()
        if model.mode == "strong_layers":
            history[layer] = previous
            history = {s: v for s, v in history.items() if s == 0 or s >= layer - 1}
        ideal += total_layer; layer_bits.append(float(total_layer))
    stream = coder.get_compressed().astype("<u4").tobytes() if encoding else None
    return output, stream, dict(ideal_bits=float(ideal), layer_ideal_bits=layer_bits, block_diagnostics=per_block)


def encode(model, source, start=0, metadata=None, correction="direct", residue_bits=0):
    """Return a self-framed packet; all metadata bytes and framing are charged."""
    begin = time.perf_counter()
    source = np.asarray(source)
    if source.dtype != np.uint16:
        raise ValueError("source must contain uint16 original BF16 words")
    shape = _shape(model, source.shape, start)
    metadata = _meta(metadata, shape[0], shape[1], model.metadata_rank)
    if (correction not in ("direct", "coset_uniform", "coset_model") or isinstance(residue_bits, (bool, np.bool_))
            or not isinstance(residue_bits, (int, np.integer)) or not 0 <= residue_bits <= 7
            or (correction == "direct" and residue_bits != 0)):
        raise ValueError("unknown correction or residue width")
    raw = np.ascontiguousarray(source, dtype="<u2").tobytes()
    fallback = None
    try:
        _fixed(from_bits(source))
        recovered, stream, detail = _run(model, shape, start, metadata, correction, int(residue_bits), source=source)
        if not np.array_equal(recovered, source):
            raise AssertionError("encoder symbol pass altered source")
        kind = "range"; meta_bytes = metadata.tobytes()
    except ValueError as error:
        if "fixed-point" not in str(error) and "overflow" not in str(error):
            raise
        kind = "raw"; stream = raw; meta_bytes = b""; fallback = str(error)
        detail = dict(ideal_bits=float(len(raw) * 8), layer_ideal_bits=[], block_diagnostics=[])
    header = dict(format=FORMAT, state_sha256=model.state_sha256, shape=list(shape), start=int(start), kind=kind,
                  correction=correction, residue_bits=int(residue_bits), metadata_bytes=len(meta_bytes), stream_bytes=len(stream),
                  decoded_sha256=hashlib.sha256(raw).hexdigest(), payload_sha256=hashlib.sha256(meta_bytes + stream).hexdigest())
    framing = _canonical(header)
    packet = PREFIX.pack(MAGIC, len(framing)) + framing + meta_bytes + stream
    stats = dict(detail, packet_bytes=len(packet), header_bytes=PREFIX.size + len(framing), framing_bytes=PREFIX.size + len(framing),
                 payload_bytes=len(meta_bytes) + len(stream), metadata_bytes=len(meta_bytes),
                 entropy_stream_bytes=len(stream) if kind == "range" else 0, raw_fallback_bytes=len(stream) if kind == "raw" else 0,
                 fallback=fallback, values=int(source.size), packet_bpv=8 * len(packet) / source.size,
                 encode_seconds=time.perf_counter() - begin, state_sha256=model.state_sha256)
    return packet, stats


def decode(model, packet):
    """Decode from packet plus frozen state, without originals or hidden tensors."""
    begin = time.perf_counter()
    if len(packet) < PREFIX.size:
        raise ValueError("truncated transfer packet")
    magic, length = PREFIX.unpack_from(packet)
    if magic != MAGIC or length > 1 << 20 or len(packet) < PREFIX.size + length:
        raise ValueError("invalid transfer packet prefix")
    header = json.loads(packet[PREFIX.size:PREFIX.size + length])
    if header.get("format") != FORMAT or header.get("state_sha256") != model.state_sha256:
        raise ValueError("packet requires different frozen state")
    shape = _shape(model, header["shape"], header["start"])
    payload = packet[PREFIX.size + length:]
    if len(payload) != header["metadata_bytes"] + header["stream_bytes"] or hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise ValueError("transfer payload length or checksum mismatch")
    if header["kind"] == "raw":
        if header["metadata_bytes"] != 0 or len(payload) != np.prod(shape) * 2:
            raise ValueError("invalid raw fallback transfer")
        output = np.frombuffer(payload, dtype="<u2").reshape(shape).copy()
        detail = dict(ideal_bits=float(len(payload) * 8), layer_ideal_bits=[], block_diagnostics=[])
    elif header["kind"] == "range":
        metadata_size = shape[0] * shape[1] * model.metadata_rank
        if header["metadata_bytes"] != metadata_size or header["stream_bytes"] % 4:
            raise ValueError("invalid transfer metadata or range stream")
        metadata = np.frombuffer(payload[:metadata_size], dtype=np.int8).reshape(shape[0], shape[1], model.metadata_rank)
        stream = np.frombuffer(payload[metadata_size:], dtype="<u4").astype(np.uint32)
        output, _, detail = _run(model, shape, header["start"], metadata, header["correction"], header["residue_bits"], compressed=stream)
    else:
        raise ValueError("unknown transfer packet kind")
    if hashlib.sha256(output.astype("<u2", copy=False).tobytes()).hexdigest() != header["decoded_sha256"]:
        raise ValueError("decoded BF16 checksum mismatch")
    return output, dict(detail, values=int(output.size), decode_seconds=time.perf_counter() - begin,
                        packet_bytes=len(packet), exact_checksum_verified=True, state_sha256=model.state_sha256)

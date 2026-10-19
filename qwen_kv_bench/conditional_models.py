"""TRAIN-only conditional entropy and joint full-BF16 producer predictors.

Every fitted array enters a canonical, checksummed artifact. Anchor predictors
retain actual selected pre-RoPE K/V coordinates. Latent predictors retain a small
BF16 projection of the actual normalized hidden state. Both jointly predict
pre-RoPE K and V; exact corrections and verified RoPE replay remain mandatory.
Continuous fits start from ridge MSE, then choose a whole-layer shrinkage from
four candidates by actual rounded NumPy BF16 correction-byte proxies. This is
not a GPU wire-cost optimizer and does not establish held-out savings.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib

import numpy as np

from . import frontier_models as frontier
from . import producer_models as producer

_MAGIC = b"CMKV"
_VERSION = 1
_HEADER = struct.Struct("<4sBIII")
_FORMAT = "qwen_conditional_models_v1"
_DTYPES = {"|u1": np.dtype("u1"), "<i4": np.dtype("<i4"), "<f4": np.dtype("<f4")}
_BOOKS = ("exp_K", "exp_V", "sign_K", "sign_V")
_BOOK_ARRAYS = ("code", "length", "lookup_symbol", "lookup_length")
_MAX_BF16 = np.array([0x7f7f0000], dtype=np.uint32).view(np.float32)[0]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _float(words):
    value = np.asarray(words)
    if value.dtype != np.uint16 or value.ndim != 2:
        raise ValueError("conditional inputs must be uint16[T,C] BF16 words")
    result = (value.astype(np.uint32) << 16).view(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("conditional producer inputs must be finite")
    return result


def _bf16_words(value):
    value = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("nonfinite conditional predictor intermediate")
    value = np.clip(value, -_MAX_BF16, _MAX_BF16).astype(np.float32)
    bits = value.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _source(record, name):
    if name in ("hidden", "v"):
        words = producer._word_array(record, name)
    else:
        if name not in record:
            raise ValueError("conditional producer record missing " + name)
        words = np.asarray(record[name])
    _float(words)
    return words


def _joint(record):
    k, v = _source(record, "k_pre_rope"), _source(record, "v")
    if k.shape != v.shape:
        raise ValueError("conditional pre-RoPE K/V shapes differ")
    return np.concatenate((k, v), axis=1)


def _readonly(params):
    for layer in params.values():
        for value in layer.values():
            value.flags.writeable = False


@dataclass
class Model:
    kind: str
    params: dict[str, dict[str, np.ndarray]]
    metadata: dict
    path: str = ""

    @property
    def config(self):
        return self.metadata["config"]

    @property
    def state_bytes(self):
        return len(model_bytes(self))

    @property
    def model_id(self):
        return hashlib.sha256(model_bytes(self)).hexdigest()

    def layer_params(self, layer):
        key = producer._key(layer)
        if key not in self.params:
            raise ValueError("conditional model lacks layer")
        return self.params[key]

    def book(self, prefix):
        if self.kind != "conditional_entropy" or prefix not in _BOOKS:
            raise ValueError("unknown conditional entropy book")
        result = dict(self.metadata["books"][prefix])
        result.update({name: self.params["global"][prefix + "_" + name] for name in _BOOK_ARRAYS})
        return result


def _validate_model(model):
    meta = model.metadata
    if (meta.get("format") != _FORMAT or meta.get("kind") != model.kind
            or model.kind not in ("conditional_entropy", "conditional_anchor", "conditional_latent", "conditional_mean")
            or meta.get("training_only") is not True or not meta.get("model_identity")):
        raise ValueError("invalid conditional model identity")
    sets = meta.get("training_record_sets", [])
    if (len(sets) != 3 or any(not isinstance(row, dict) or row.get("split") != "train"
            or not row.get("records") for row in sets)
            or meta.get("training_hashes") != [row.get("sha256") for row in sets]):
        raise ValueError("invalid conditional TRAIN provenance")
    for params in model.params.values():
        for array in params.values():
            value = np.asarray(array)
            if value.dtype.newbyteorder("<").str not in _DTYPES or value.ndim < 1 or min(value.shape) < 1:
                raise ValueError("unsupported conditional parameter array")
            if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
                raise ValueError("nonfinite conditional model parameter")
    if model.kind == "conditional_entropy":
        if set(model.params) != {"global"} or set(meta.get("books", {})) != set(_BOOKS):
            raise ValueError("invalid conditional entropy model schema")
        expected = {prefix + "_" + name for prefix in _BOOKS for name in _BOOK_ARRAYS}
        if set(model.params["global"]) != expected:
            raise ValueError("conditional entropy parameter fields mismatch")
        from . import conditional_entropy as entropy
        for prefix in _BOOKS:
            book = model.book(prefix)
            alphabet = 512 if prefix.startswith("exp") else 256
            if len(book["length"]) != alphabet or len(book["code"]) != alphabet:
                raise ValueError("conditional entropy alphabet mismatch")
            codes, lengths = book["code"], book["length"]
            if (book.get("format") != entropy.FORMAT or book.get("alphabet_size") != alphabet
                    or codes.dtype != np.int32 or lengths.dtype != np.uint8
                    or codes.shape != (alphabet,) or lengths.shape != (alphabet,)
                    or np.any(lengths < 1) or np.any(lengths > 15)
                    or book.get("max_bits") != int(lengths.max())
                    or book.get("min_bits") != int(lengths.min())
                    or not np.array_equal(codes, entropy._canonical(lengths))
                    or book.get("codebook_id") != entropy._identity(codes, lengths)):
                raise ValueError("invalid conditional canonical entropy book")
            size = 1 << int(lengths.max())
            lookup_symbol, lookup_length = book["lookup_symbol"], book["lookup_length"]
            if (lookup_symbol.dtype != np.int32 or lookup_length.dtype != np.uint8
                    or lookup_symbol.shape != (size,) or lookup_length.shape != (size,)):
                raise ValueError("conditional entropy lookup geometry mismatch")
            symbols, widths = np.full(size, -1, dtype=np.int32), np.zeros(size, dtype=np.uint8)
            for symbol, (code, width) in enumerate(zip(codes, lengths)):
                suffix = int(lengths.max()) - int(width)
                start = int(code) << suffix
                symbols[start:start + (1 << suffix)] = symbol
                widths[start:start + (1 << suffix)] = width
            if not np.array_equal(symbols, lookup_symbol) or not np.array_equal(widths, lookup_length):
                raise ValueError("conditional entropy lookup differs from canonical book")
        return
    config = meta.get("config", {})
    requested = config.get("rank_requested")
    if (requested not in ((0,) if model.kind == "conditional_mean" else (8, 32)) or config.get("feature_dtype") != "bf16"
            or config.get("target_source") != "concat_actual_pre_rope_K_and_V"
            or config.get("fit_objective") not in (
                "ridge real-value MSE on TRAIN; not a correction-bit objective",
                "ridge-MSE initializer; TRAIN NumPy BF16 whole-layer shrinkage selected by packed full-word correction proxy")):
        raise ValueError("invalid conditional numerical predictor config")
    schema = meta.get("layer_schema", {})
    if not schema or set(schema) != set(model.params):
        raise ValueError("conditional model layer schema mismatch")
    for layer, params in model.params.items():
        if re.fullmatch(r"layer_\d+", layer) is None:
            raise ValueError("invalid conditional layer key")
        shape = schema[layer]
        channels, hidden, actual = (shape.get(key) for key in ("channels", "hidden_channels", "actual_rank"))
        if (any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (channels, hidden))
                or isinstance(actual, bool) or not isinstance(actual, int) or actual < (0 if model.kind == "conditional_mean" else 1)
                or actual > min(requested, hidden, 2 * channels)
                or not isinstance(shape.get("head_dim"), int) or shape["head_dim"] < 1
                or channels % shape["head_dim"]):
            raise ValueError("invalid conditional predictor geometry")
        expected = ({"intercept"} if model.kind == "conditional_mean" else
            {"coeff", "intercept", "anchor_indices"} if model.kind == "conditional_anchor" else
            {"coeff", "intercept", "feature_center", "feature_projection"})
        if set(params) != expected:
            raise ValueError("conditional predictor parameter fields mismatch")
        for name, expected_shape in (("coeff", (actual, 2 * channels)), ("intercept", (2 * channels,))):
            if name == "coeff" and model.kind == "conditional_mean":
                continue
            if params[name].dtype != np.float32 or params[name].shape != expected_shape:
                raise ValueError("conditional regression shape/dtype mismatch")
        if model.kind == "conditional_anchor":
            indices = params["anchor_indices"]
            if (indices.dtype != np.int32 or indices.shape != (actual,) or len(np.unique(indices)) != actual
                    or np.any(indices < 0) or np.any(indices >= 2 * channels)):
                raise ValueError("invalid conditional anchor selectors")
        elif model.kind == "conditional_latent":
            for name, expected_shape in (("feature_center", (hidden,)), ("feature_projection", (hidden, actual))):
                if params[name].dtype != np.float32 or params[name].shape != expected_shape:
                    raise ValueError("conditional latent projection shape/dtype mismatch")


def model_bytes(model):
    _validate_model(model)
    metadata = producer._json(model.metadata)
    payload, manifest = bytearray(), []
    for layer in sorted(model.params):
        for name in sorted(model.params[layer]):
            array = np.asarray(model.params[layer][name])
            dtype = array.dtype.newbyteorder("<")
            canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
            manifest.append(dict(layer=layer, name=name, dtype=dtype.str, shape=list(array.shape),
                                 offset=len(payload), nbytes=canonical.nbytes))
            payload.extend(canonical.tobytes())
    metadata["array_manifest"] = manifest
    encoded = _canonical(metadata)
    body = encoded + bytes(payload)
    return _HEADER.pack(_MAGIC, _VERSION, len(encoded), len(payload), zlib.crc32(body)) + body


def load(path, *, expected_model_identity=None, expected_training_hashes=None,
         expected_model_id=None, expected_kind=None):
    path = Path(path)
    data = path.read_bytes()
    if data[:4] == b"PMKV":
        return producer.load(path, expected_model_identity=expected_model_identity,
            expected_training_hashes=expected_training_hashes, expected_model_id=expected_model_id,
            expected_kind=expected_kind)
    if len(data) < _HEADER.size:
        raise ValueError("truncated conditional model")
    magic, version, metadata_size, payload_size, checksum = _HEADER.unpack_from(data)
    body = data[_HEADER.size:]
    if (magic != _MAGIC or version != _VERSION or metadata_size + payload_size != len(body)
            or zlib.crc32(body) != checksum):
        raise ValueError("invalid conditional model header/CRC")
    try:
        metadata = json.loads(body[:metadata_size])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid conditional model JSON") from error
    if _canonical(metadata) != body[:metadata_size]:
        raise ValueError("noncanonical conditional model metadata")
    if expected_model_identity is not None and producer._json(expected_model_identity) != metadata.get("model_identity"):
        raise ValueError("conditional source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes) != metadata.get("training_hashes"):
        raise ValueError("conditional TRAIN source identity mismatch")
    if expected_kind is not None and expected_kind != metadata.get("kind"):
        raise ValueError("conditional predictor kind mismatch")
    params, offset = {}, 0
    manifest = metadata.pop("array_manifest", None)
    if not isinstance(manifest, list):
        raise ValueError("missing conditional array manifest")
    for entry in manifest:
        try:
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if (not isinstance(shape, list) or not shape or any(isinstance(n, bool) or not isinstance(n, int)
                    or n < 1 for n in shape) or entry["offset"] != offset
                    or entry["nbytes"] != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    or offset + entry["nbytes"] > payload_size):
                raise ValueError("invalid conditional array bounds")
            layer, name = entry["layer"], entry["name"]
            if not isinstance(layer, str) or not isinstance(name, str) or name in params.get(layer, {}):
                raise ValueError("duplicate/invalid conditional array name")
            start = metadata_size + offset
            value = np.frombuffer(body[start:start + entry["nbytes"]], dtype=dtype).reshape(shape).copy()
            params.setdefault(layer, {})[name] = value
            offset += entry["nbytes"]
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("invalid conditional array manifest") from error
    if offset != payload_size:
        raise ValueError("unreferenced conditional model payload")
    model = Model(metadata.get("kind"), params, metadata, str(path.resolve()))
    _validate_model(model)
    if model_bytes(model) != data:
        raise ValueError("noncanonical conditional array order")
    if expected_model_id is not None and model.model_id != expected_model_id:
        raise ValueError("conditional model SHA256 mismatch")
    _readonly(params)
    return model


def _anchor_indices(target, rank):
    """Greedy pivoted residual energy, using actual joint TRAIN K/V values."""
    residual = target.astype(np.float64) - target.mean(axis=0, dtype=np.float64)
    indices = []
    for _ in range(rank):
        energy = np.sum(residual * residual, axis=0)
        energy[indices] = -1
        chosen = int(np.argmax(energy))
        indices.append(chosen)
        direction = residual[:, chosen].copy()
        norm = float(direction @ direction)
        if norm > 0:
            residual -= np.outer(direction, (direction @ residual) / norm)
    return np.asarray(indices, dtype=np.int32)


def _latent_projection(hidden, target, rank, seed):
    """Randomized supervised cross-covariance range; never form H-by-2C SVD."""
    center = hidden.mean(axis=0, dtype=np.float64).astype(np.float32)
    x = hidden.astype(np.float64) - center
    y = target.astype(np.float64) - target.mean(axis=0, dtype=np.float64)
    sketch_rank = min(rank + 8, x.shape[1], y.shape[1], max(1, len(x) - 1))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((y.shape[1], sketch_rank)) / np.sqrt(sketch_rank)
    sketch = x.T @ (y @ omega)
    q, _ = np.linalg.qr(sketch, mode="reduced")
    # Rank the small supervised sketch; only a <=40 by <=40 eigensystem.
    reduced = x @ q
    correlation = reduced.T @ y
    eigenvalues, vectors = np.linalg.eigh(correlation @ correlation.T)
    order = np.argsort(eigenvalues, kind="stable")[::-1][:rank]
    projection = q @ vectors[:, order]
    for column in range(projection.shape[1]):
        peak = int(np.argmax(np.abs(projection[:, column])))
        if projection[peak, column] < 0:
            projection[:, column] *= -1
    return center, projection.astype(np.float32)


def _regression(features, target):
    x, y = features.astype(np.float64), target.astype(np.float64)
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    x = x - mean_x
    gram = x.T @ x
    ridge = max(float(np.trace(gram)) / max(1, gram.shape[0]) * 1e-4, 1e-8)
    coeff = np.linalg.solve(gram + np.eye(gram.shape[0]) * ridge, x.T @ (y - mean_y))
    intercept = mean_y - mean_x @ coeff
    return coeff.astype(np.float32), intercept.astype(np.float32), ridge


def _predict_numpy(kind, params, feature_words):
    features = _float(feature_words)
    prediction = _bf16_words(features @ params["coeff"] + params["intercept"])
    if kind == "conditional_anchor":
        prediction[:, params["anchor_indices"]] = feature_words
    return prediction


def _proxy(target_words, predicted_words, feature_words, record_rows=None, anchor_indices=None):
    error = ((target_words.astype(np.int64) - predicted_words.astype(np.int64) + 32768) & 65535) - 32768
    zigzag = np.where(error >= 0, 2 * error, -2 * error - 1)
    # The real pair codec codes K/V independently and resets every record.
    # Zero padding finds the last group's width; padded symbols are NOT sent.
    record_rows = [len(target_words)] if record_rows is None else record_rows
    if sum(record_rows) != len(target_words) or any(n < 1 for n in record_rows):
        raise ValueError("conditional proxy record boundaries mismatch")
    channels = target_words.shape[1] // 2
    retained = set() if anchor_indices is None else set(np.asarray(anchor_indices).tolist())
    if any(not isinstance(index, int) or not 0 <= index < 2 * channels for index in retained):
        raise ValueError("invalid conditional proxy anchor selectors")
    payload_bytes, width_bytes, groups, raw_bits, offset = 0, 0, 0, 0, 0
    for count in record_rows:
        for first in (0, channels):
            indices = [index for index in range(first, first + channels) if index not in retained]
            flat = zigzag[offset:offset + count][:, indices].reshape(-1)
            if not len(flat):
                continue
            padding = (-len(flat)) % 64
            grouped = np.pad(flat, (0, padding)).reshape(-1, 64)
            maximum = grouped.max(axis=1)
            width = np.where(maximum > 0, np.floor(np.log2(np.maximum(maximum, 1))).astype(np.int64) + 1, 0)
            lengths = np.full(len(width), 64, dtype=np.int64)
            lengths[-1] -= padding
            bits = int(np.sum(width * lengths))
            raw_bits += bits
            payload_bytes += (bits + 7) // 8
            width_bytes += (len(width) * 5 + 7) // 8
            groups += len(width)
        offset += count
    target_exp = (target_words.astype(np.uint32) >> 7) & 255
    pred_exp = (predicted_words.astype(np.uint32) >> 7) & 255
    return dict(training_only=True, numerical_scope="NumPy float32 predictor proxy; GPU runtime can round differently and must compute its own exact corrections",
        original_words=int(target_words.size), feature_bytes=int(feature_words.nbytes),
        feature_bits_per_target_word=float(feature_words.nbytes * 8 / target_words.size),
        exact_word_fraction=float(np.mean(target_words == predicted_words)),
        sign_agreement_fraction=float(np.mean((target_words >> 15) == (predicted_words >> 15))),
        mean_abs_exponent_error=float(np.abs(target_exp.astype(np.int16) - pred_exp.astype(np.int16)).mean()),
        correction_group_size=64, correction_scope="signed circular full-16-bit BF16 word difference; independently reset K/V records; exact retained anchor coordinates excluded; five-bit widths; only real symbols sent, payload/width byte padding charged",
        anchors_excluded_from_corrections=len(retained),
        correction_records=len(record_rows), correction_groups=groups,
        correction_payload_data_bits=raw_bits,
        correction_payload_bytes=payload_bytes, correction_width_bytes=width_bytes,
        correction_payload_bits_per_word=float(payload_bytes * 8 / target_words.size),
        correction_width_bits_per_word=float(width_bytes * 8 / target_words.size),
        selection_proxy_total_bits=int((payload_bytes + width_bytes + feature_words.nbytes) * 8),
        zero_residual_fraction=float(np.mean(error == 0)))


def _select_shrinkage(kind, params, feature_words, target_words, target_mean, record_rows):
    """Select four whole-layer TRAIN candidates by actually rounded BF16 errors.

    State fields, features, headers and indexes are identical across candidates;
    only predictor weights and correction byte lengths differ. This is a host
    rate proxy, not measured GPU wire size or a held-out optimization result.
    """
    candidates = []
    for alpha in (0., .5, .75, 1.):
        adjusted = {key: value for key, value in params.items() if key not in ("coeff", "intercept")}
        adjusted["coeff"] = (params["coeff"] * np.float32(alpha)).astype(np.float32)
        adjusted["intercept"] = (target_mean + (params["intercept"] - target_mean) * np.float32(alpha)).astype(np.float32)
        prediction = _predict_numpy(kind, adjusted, feature_words)
        stats = _proxy(target_words, prediction, feature_words, record_rows,
            params["anchor_indices"] if kind == "conditional_anchor" else None)
        candidates.append((adjusted, stats, alpha))
    winner = min(range(len(candidates)), key=lambda i: (candidates[i][1]["selection_proxy_total_bits"], -candidates[i][2]))
    adjusted, stats, alpha = candidates[winner]
    stats.update(shrinkage_alpha=alpha, shrinkage_candidates=[dict(alpha=a,
        correction_payload_bytes=s["correction_payload_bytes"], correction_width_bytes=s["correction_width_bytes"],
        feature_bytes=s["feature_bytes"], total_proxy_bits=s["selection_proxy_total_bits"]) for _, s, a in candidates],
        selection_scope="Whole-layer alpha selected on TRAIN only using NumPy float32 then BF16; per-record K/V full-word circular correction payload and packed five-bit widths plus fixed features. Same parameter shapes/state bytes; headers/indexes excluded because identical across alpha candidates. GPU actual wire can differ due runtime arithmetic.")
    return adjusted, stats


def torch_features_params(params, kind, fields):
    """Derive BF16 feature words using already resident parameters/producer fields.

    This encoder-only operation needs actual producer data. Decoder calls only
    torch_predict_params with the feature words retained in its packet.
    """
    import torch
    if kind not in ("conditional_anchor", "conditional_latent"):
        raise ValueError("unknown conditional numerical predictor")
    def words(name):
        value = fields[name]
        if not isinstance(value, torch.Tensor) or value.dtype not in (torch.uint16, torch.int16) or value.ndim != 2:
            raise ValueError("expected resident BF16 word tensor for " + name)
        return value.contiguous().view(torch.uint16)
    if kind == "conditional_anchor":
        # Anchors are actual retained words and have NO correction payload.
        # Copy their bits directly, preserving signed zero/subnormals regardless
        # of a backend's floating-point flush-to-zero or conversion behavior.
        joint = torch.cat((words("k_pre_rope").view(torch.int16), words("v").view(torch.int16)), dim=1)
        return joint[:, params["anchor_indices"].to(torch.int64)].contiguous().view(torch.uint16)
    else:
        hidden = words("hidden").view(torch.bfloat16).to(torch.float32)
        features = (hidden - params["feature_center"]) @ params["feature_projection"]
    if not bool(torch.isfinite(features).all()):
        raise ValueError("nonfinite conditional producer features")
    return features.clamp(-float(_MAX_BF16), float(_MAX_BF16)).to(torch.bfloat16).contiguous().view(torch.uint16)


def torch_predict_params(params, kind, feature_words):
    """Predict joint pre-RoPE K/V from packet features and complete resident state.

    FP32 matrix products are runtime-specific; the enclosing codec MUST bind
    backend, versions and TF32/autocast controls before it stores corrections.
    """
    import torch
    if kind not in ("conditional_anchor", "conditional_latent"):
        raise ValueError("unknown conditional numerical predictor")
    if (not isinstance(feature_words, torch.Tensor) or feature_words.dtype not in (torch.uint16, torch.int16)
            or feature_words.ndim != 2 or feature_words.shape[1] != params["coeff"].shape[0]):
        raise ValueError("conditional stored feature geometry mismatch")
    feature_words = feature_words.contiguous().view(torch.uint16)
    features = feature_words.view(torch.bfloat16).to(torch.float32)
    prediction = features @ params["coeff"] + params["intercept"]
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("nonfinite conditional predictor output")
    result = prediction.clamp(-float(_MAX_BF16), float(_MAX_BF16)).to(torch.bfloat16).contiguous().view(torch.uint16)
    if kind == "conditional_anchor":
        # uint16 advanced-index assignment is missing on some torch backends.
        signed = result.view(torch.int16).clone()
        signed[:, params["anchor_indices"].to(torch.int64)] = feature_words.view(torch.int16)
        result = signed.view(torch.uint16)
    return result


def _fit_entropy(grouped, frozen):
    from . import conditional_entropy as entropy
    counts = {prefix: np.zeros(512 if prefix.startswith("exp") else 256, dtype=np.int64) for prefix in _BOOKS}
    sign_padding = {side: dict(records=0, transition_bits=0, packed_symbols=0, padding_bits=0,
        records_with_padding=0) for side in ("K", "V")}
    for layer, records in sorted(grouped.items()):
        base = frozen.params[layer]
        for record in records:
            for side, source in (("K", "k_pre_rope"), ("V", "v")):
                words = _source(record, source)
                exponents = ((words.astype(np.int64) >> 7) & 255)
                error = exponents - base["base_" + side][None, :]
                coded = np.where(error >= 0, 2 * error, -2 * error - 1)
                counts["exp_" + side] += np.bincount(coded.reshape(-1), minlength=512)
                signs = (words >> 15).astype(np.uint8)
                transitions = (signs[1:] ^ signs[:-1]).T.reshape(-1)
                packed = np.packbits(transitions, bitorder="little")
                counts["sign_" + side] += np.bincount(packed, minlength=256)
                padding = (-len(transitions)) % 8
                observed = sign_padding[side]
                observed["records"] += 1
                observed["transition_bits"] += len(transitions)
                observed["packed_symbols"] += len(packed)
                observed["padding_bits"] += padding
                observed["records_with_padding"] += int(padding > 0)
    params, books, stats = {"global": {}}, {}, {}
    for prefix, histogram in counts.items():
        book = entropy.fit_codebook(histogram, max_bits=15)
        if any(name not in book for name in _BOOK_ARRAYS):
            raise ValueError("conditional entropy codebook missing canonical arrays")
        for name in _BOOK_ARRAYS:
            dtype = np.uint8 if name in ("length", "lookup_length") else np.int32
            params["global"][prefix + "_" + name] = np.asarray(book[name], dtype=dtype).copy()
        books[prefix] = producer._json(_plain({key: value for key, value in book.items() if key not in _BOOK_ARRAYS}))
        size = int(histogram.sum())
        stats[prefix] = dict(training_symbols=size, training_symbol_histogram=histogram.tolist(),
            count_sha256=hashlib.sha256(histogram.astype("<i8").tobytes()).hexdigest(),
            observed_symbols=int(np.count_nonzero(histogram)),
            training_code_bits=int(histogram @ np.asarray(book["length"], dtype=np.int64)),
            training_bits_per_symbol=float(histogram @ np.asarray(book["length"], dtype=np.int64) / size) if size else None,
            scope="Actual TRAIN symbols only; unseen symbols receive the entropy fitter's smoothing; segment headers/seeds/indexes are charged by the codec")
        if prefix.startswith("sign_"):
            stats[prefix]["packed_sign_padding_observations"] = dict(sign_padding[prefix[-1]],
                rule="Previous-row XOR sign transitions transpose channel-major, pack8LSB; each independent record's final symbol is zero-padded. Padding-containing symbols are included in the histogram; first-row sign seeds are excluded.")
    return params, books, stats


def fit(training_record_sets, output_dir, rank=8, *, training_metadata, model_identity, seed=71):
    """Fit legacy frontier controls plus five new independent TRAIN-only artifacts."""
    if (not isinstance(training_record_sets, Sequence) or isinstance(training_record_sets, (str, bytes))
            or len(training_record_sets) != 3 or any(not isinstance(records, Sequence) or not records for records in training_record_sets)):
        raise ValueError("conditional fitting requires three nonempty TRAIN capture sets")
    before = [producer._source_hash(records) for records in training_record_sets]
    metadata_before = producer._source_hash(dict(training_metadata=producer._json(training_metadata), model_identity=producer._json(model_identity)))
    grouped, schema = {}, {}
    for records in training_record_sets:
        for record in records:
            layer, shape = producer._validate_record(record)
            if _joint(record).shape != (record["tokens"], shape["channels"] * 2):
                raise ValueError("conditional target geometry mismatch")
            if layer in schema and schema[layer] != shape:
                raise ValueError("conditional TRAIN geometry changes within layer")
            schema[layer] = shape
            grouped.setdefault(layer, []).append(record)
    output = Path(output_dir).resolve()
    names = ["conditional_entropy", "conditional_mean"] + [f"conditional_{kind}_r{r}" for kind in ("anchor", "latent") for r in (8, 32)]
    if any((output / (name + ".bin")).exists() for name in names):
        raise FileExistsError("conditional artifacts exist; use a fresh output directory")
    models = frontier.fit(training_record_sets, output, rank=rank,
        training_metadata=training_metadata, model_identity=model_identity, seed=seed)
    provenance = models["cost_pre_rope"].metadata
    def metadata(kind, config):
        return dict(format=_FORMAT, kind=kind, training_only=True,
            model_identity=producer._json(model_identity), training_hashes=list(before),
            training_record_sets=provenance["training_record_sets"],
            layer_schema={key: dict(value) for key, value in schema.items()}, config=config,
            seed=int(seed), scope="TRAIN-only fit; full serialized model state must be charged; no held-out result or speed claim")
    def save(name, kind, params, meta):
        _readonly(params)
        model = Model(kind, params, meta)
        path = output / (name + ".bin")
        path.write_bytes(model_bytes(model))
        models[name] = load(path, expected_model_identity=model_identity,
            expected_training_hashes=before, expected_model_id=model.model_id, expected_kind=kind)
    params, books, stats = _fit_entropy(grouped, models["cost_pre_rope"])
    meta = metadata("conditional_entropy", dict(source_K="k_pre_rope", source_V="v",
        exponent_base_model_id=models["cost_pre_rope"].model_id,
        exponent_rule="zigzag(actual_exponent - per-channel frozen TRAIN cost base); alphabet512",
        sign_rule="pack8LSB(previous-row XOR signs transposed channel-major); reset/seed each independent record; alphabet256",
        objective="canonical Huffman TRAIN symbol frequencies with positive smoothing increased until maximum code length<=15; not optimal length-limited coding; independent K/V books"))
    meta.update(books=books, training_stats=stats)
    save("conditional_entropy", "conditional_entropy", params, meta)
    fitted = {(kind, requested): ({}, {}) for kind in ("anchor", "latent") for requested in (8, 32)}
    fitted["mean", 0] = ({}, {})
    for layer, records in sorted(grouped.items()):
        words = np.concatenate([_joint(record) for record in records])
        target = _float(words)
        hidden = np.concatenate([_float(_source(record, "hidden")) for record in records])
        mean = target.mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_prediction = np.broadcast_to(_bf16_words(mean), words.shape).copy()
        record_rows = [len(_joint(record)) for record in records]
        mean_stats = _proxy(words, mean_prediction, np.empty((len(words), 0), dtype=np.uint16), record_rows)
        mean_stats.update(actual_rank=0, training_rows=len(target), ridge_lambda=0.,
            parameter_payload_bytes=mean.nbytes, fit_objective="ridge real-value MSE on TRAIN; not a correction-bit objective")
        fitted["mean", 0][0][layer] = {"intercept": mean}
        fitted["mean", 0][1][layer] = mean_stats
        for kind in ("anchor", "latent"):
            for requested in (8, 32):
                actual = min(requested, hidden.shape[1], target.shape[1], max(1, len(target) - 1))
                if kind == "anchor":
                    indices = _anchor_indices(target, actual)
                    params = dict(anchor_indices=indices)
                    features_words = words[:, indices].copy()
                else:
                    center, projection = _latent_projection(hidden, target, actual, seed + int(layer.split("_")[-1]))
                    params = dict(feature_center=center, feature_projection=projection)
                    features_words = _bf16_words((hidden - center) @ projection)
                coeff, intercept, ridge = _regression(_float(features_words), target)
                params.update(coeff=coeff, intercept=intercept)
                params, stats = _select_shrinkage("conditional_" + kind, params, features_words, words, mean, record_rows)
                stats.update(actual_rank=actual, training_rows=len(target), ridge_lambda=ridge,
                    parameter_payload_bytes=sum(array.nbytes for array in params.values()),
                    fit_objective="ridge-MSE initializer; TRAIN NumPy BF16 whole-layer shrinkage selected by packed full-word correction proxy")
                if kind == "anchor":
                    stats["anchor_indices"] = indices.tolist()
                    stats["anchor_selection"] = "greedy pivoted centered joint-K/V residual energy on TRAIN"
                else:
                    stats["latent_selection"] = "randomized joint-target cross-covariance range, <=40-dimensional supervised eigensystem; no full hidden-by-target SVD"
                fitted[kind, requested][0][layer] = params
                fitted[kind, requested][1][layer] = stats
    for (kind, requested), (params, stats) in fitted.items():
        meta = metadata("conditional_" + kind, dict(rank_requested=requested, feature_dtype="bf16",
            target_source="concat_actual_pre_rope_K_and_V",
            fit_objective=("ridge real-value MSE on TRAIN; not a correction-bit objective" if kind == "mean" else
                "ridge-MSE initializer; TRAIN NumPy BF16 whole-layer shrinkage selected by packed full-word correction proxy"),
            prediction_rule="stored BF16 features cast FP32, @ FP32 coefficients + FP32 intercept, clamp finite BF16 range then BF16; anchor coordinates overwritten from stored features",
            numerical_scope="runtime/backend-specific FP32 matmul; codec must bind backend/TF32/autocast and preserve exact full-word corrections",
            source_K="k_pre_rope", source_V="v"))
        for layer in meta["layer_schema"]:
            meta["layer_schema"][layer]["actual_rank"] = stats[layer]["actual_rank"]
        meta["training_stats"] = stats
        name = "conditional_mean" if kind == "mean" else f"conditional_{kind}_r{requested}"
        save(name, "conditional_" + kind, params, meta)
    after = [producer._source_hash(records) for records in training_record_sets]
    metadata_after = producer._source_hash(dict(training_metadata=producer._json(training_metadata), model_identity=producer._json(model_identity)))
    if before != after or metadata_before != metadata_after or before != models["cost_pre_rope"].metadata["training_hashes"]:
        raise AssertionError("conditional TRAIN source or provenance changed during fitting")
    return models

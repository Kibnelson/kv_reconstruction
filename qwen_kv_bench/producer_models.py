"""TRAIN-only predictors that turn prefill-producer metadata into exponents.

A frozen control stores per-channel TRAIN exponent medians. The hidden-feature
experiment retains only a few int8 channels of the normalized hidden states
already produced by the real forward pass. Both K and V reuse those features.
A q8 integer linear predictor recreates exponents; exact correction codecs must
still transmit every prediction error. These are small experimental predictors,
not replay of the full hidden state, model projections, or transformer layers.
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

_MAGIC = b"PMKV"
_VERSION = 1
_HEADER = struct.Struct("<4sBIII")
_FORMAT = "qwen_prefill_producer_models_v1"
_LAYER = re.compile(r"layer_(\d+)$")
_DTYPES = {"|u1": np.dtype("u1"), "<i2": np.dtype("<i2"), "<i4": np.dtype("<i4"), "<f4": np.dtype("<f4")}


def _json(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("producer model metadata must contain finite JSON values") from error


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source_description(value):
    """Hash every supplied record field, including fields unused by this fit."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object arrays cannot identify producer inputs")
        data = np.ascontiguousarray(value)
        return dict(array_dtype=data.dtype.str, shape=list(data.shape),
            array_sha256=hashlib.sha256(data.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return _source_description(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("producer record field names must be strings")
        return {key: _source_description(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_source_description(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return _json(value)
    raise ValueError("producer record fields must be arrays or finite JSON values")


def _source_hash(value):
    return hashlib.sha256(_canonical(_source_description(value))).hexdigest()


def _key(layer):
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or layer < 0:
        raise ValueError("producer layer must be a nonnegative integer")
    return f"layer_{int(layer):02d}"


def _word_array(record, name):
    aliases = (name, name.upper()) if name in ("k", "v") else (name, "hidden_states", "normalized_hidden")
    present = [field for field in aliases if field in record]
    if not present:
        raise ValueError("producer record missing " + name)
    # Never silently choose between conflicting aliases.
    array = np.asarray(record[present[0]])
    if any(not np.array_equal(array, np.asarray(record[field])) for field in present[1:]):
        raise ValueError("conflicting producer record aliases for " + name)
    if array.dtype != np.uint16 or array.ndim != 2:
        raise ValueError("producer " + name + " must be uint16[T,C] BF16 words")
    return array


def _float_hidden(words):
    value = (words.astype(np.uint32) << 16).view(np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("producer hidden features require finite actual hidden states")
    return value


def _validate_record(record):
    if not isinstance(record, Mapping):
        raise ValueError("producer fitting requires record mappings")
    if record.get("split", "train") not in ("train", "training"):
        raise ValueError("producer fitting accepts TRAIN records only")
    layer = _key(record.get("layer"))
    for name in ("start", "tokens", "head_dim"):
        value = record.get(name)
        minimum = 0 if name == "start" else 1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("invalid producer record " + name)
    k, v, hidden = (_word_array(record, name) for name in ("k", "v", "hidden"))
    if k.shape != v.shape or len(k) != record["tokens"] or len(hidden) != len(k) or min(hidden.shape) < 1:
        raise ValueError("producer K/V/hidden record shapes differ")
    if k.shape[1] < 1 or k.shape[1] % record["head_dim"]:
        raise ValueError("producer KV channels disagree with explicit head_dim")
    _float_hidden(hidden)  # Validate finite BF16 words before any fitting.
    return layer, dict(channels=k.shape[1], hidden_channels=hidden.shape[1], head_dim=int(record["head_dim"]))


def _quantize(hidden, indices, scales):
    # Float32 division and ties-to-even rounding match the GPU producer rule.
    selected = np.asarray(hidden, dtype=np.float32)[:, indices]
    normalized = selected / scales[None, :]
    return np.clip(np.rint(normalized), -127, 127).astype(np.int8)


def _fit_layer_hidden(records, base_k, base_v, rank, *, feature_transform="signed", shared=None):
    hidden = np.concatenate([_float_hidden(_word_array(record, "hidden")) for record in records])
    if feature_transform not in ("signed", "abs"):
        raise ValueError("producer feature transform must be signed or abs")
    variance = np.var(hidden.astype(np.float64), axis=0)
    if shared is None:
        indices = np.lexsort((np.arange(hidden.shape[1]), -variance))[:rank].astype(np.int32)
        maximum = np.max(np.abs(hidden[:, indices]), axis=0)
        scales = (maximum / np.float32(127)).astype(np.float32)
        scales = np.where(maximum == 0, np.float32(1), np.maximum(scales, np.finfo(np.float32).tiny)).astype(np.float32)
    else:
        # The abs ablation reuses the signed model's selector/scale fit.
        indices, scales = shared["indices"].copy(), shared["scales"].copy()
    features = _quantize(hidden, indices, scales)
    predictor_features = np.abs(features.astype(np.int16)) if feature_transform == "abs" else features
    design = predictor_features.astype(np.float64)
    gram = design.T @ design + np.eye(rank, dtype=np.float64)  # ridge lambda=1
    params = dict(indices=indices, scales=scales)
    stats = dict(selected_hidden_indices=indices.tolist(),
        selected_hidden_variances=variance[indices].tolist(),
        feature_scale_rule="TRAIN max(abs(hidden))/127 in float32; zero columns use 1.0",
        training_feature_saturated_values=int(np.count_nonzero(np.abs(features.astype(np.int16)) == 127)))
    for kind, base in (("K", base_k), ("V", base_v)):
        target = np.concatenate([((_word_array(record, kind.lower()).astype(np.uint32) >> 7) & 255)
                                 for record in records]).astype(np.float64) - base[None, :]
        weights = np.linalg.solve(gram, design.T @ target)
        quantized = np.clip(np.rint(weights * 256), -32768, 32767).astype(np.int16)
        params["coeff_" + kind] = quantized
        prediction = np.clip(base.astype(np.int64)[None, :] +
            (predictor_features.astype(np.int64) @ quantized.astype(np.int64) + 128) // 256, 0, 255)
        stats[kind] = dict(training_mean_abs_exponent_error=float(np.abs(target + base - prediction).mean()),
            coefficient_clipped_values=int(np.count_nonzero((np.rint(weights * 256) < -32768) | (np.rint(weights * 256) > 32767))),
            ridge_lambda=1.0, feature_transform=feature_transform)
    return params, stats


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
    def model_id(self):
        return hashlib.sha256(model_bytes(self)).hexdigest()

    @property
    def state_bytes(self):
        return len(model_bytes(self))

    @property
    def base(self):
        return {layer: {kind: params["base_" + kind] for kind in ("K", "V")} for layer, params in self.params.items()}

    @property
    def coeff(self):
        return {layer: {kind: params["coeff_" + kind] for kind in ("K", "V")} for layer, params in self.params.items()} if self.kind != "frozen" else {}

    @property
    def indices(self):
        return {layer: params["indices"] for layer, params in self.params.items()} if self.kind != "frozen" else {}

    @property
    def scales(self):
        return {layer: params["scales"] for layer, params in self.params.items()} if self.kind != "frozen" else {}

    def layer_params(self, layer):
        key = _key(layer)
        if key not in self.params:
            raise ValueError("producer model lacks layer")
        return self.params[key]

    def features(self, record):
        params = self.layer_params(record["layer"])
        if self.kind == "frozen":
            tokens = record.get("tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, (int, np.integer)) or tokens < 0:
                raise ValueError("invalid frozen producer token count")
            # This control needs no hidden-state payload, even at the API edge.
            return np.empty((int(tokens), 0), dtype=np.int8)
        words = _word_array(record, "hidden")
        schema = self.metadata["layer_schema"][_key(record["layer"])]
        if words.shape[1] != schema["hidden_channels"] or len(words) != record["tokens"]:
            raise ValueError("producer hidden feature geometry mismatch")
        return _quantize(_float_hidden(words), params["indices"], params["scales"])

    def predict_exponents(self, layer, kind, tokens, features=None):
        if kind not in ("K", "V") or isinstance(tokens, bool) or not isinstance(tokens, (int, np.integer)) or tokens < 0:
            raise ValueError("invalid producer exponent request")
        params = self.layer_params(layer)
        base = params["base_" + kind]
        if self.kind == "frozen":
            if features is not None:
                raise ValueError("frozen producer control takes no hidden features")
            return np.broadcast_to(base, (int(tokens), len(base))).astype(np.int16, copy=True)
        value = np.asarray(features)
        if value.dtype != np.int8 or value.shape != (tokens, self.config["rank"]):
            raise ValueError("hidden producer predictor requires stored int8[T,rank] features")
        design = value.astype(np.int64)
        if self.config["feature_transform"] == "abs":
            design = np.abs(design)
        integer = design @ params["coeff_" + kind].astype(np.int64)
        return np.clip(base.astype(np.int64)[None, :] + (integer + 128) // 256, 0, 255).astype(np.int16)


def _validate_model(model):
    meta = model.metadata
    if (meta.get("format") != _FORMAT or meta.get("kind") != model.kind or model.kind not in ("frozen", "hidden", "hidden_abs")
            or meta.get("training_only") is not True or not meta.get("model_identity")):
        raise ValueError("invalid training-only producer model identity")
    sets = meta.get("training_record_sets", [])
    if (len(sets) != 3 or any(row.get("split") != "train" or not row.get("records") for row in sets)
            or meta.get("training_hashes") != [row.get("sha256") for row in sets]):
        raise ValueError("invalid producer TRAIN provenance")
    config = meta.get("config", {})
    rank = config.get("rank")
    if (isinstance(rank, bool) or not isinstance(rank, int) or (rank != 0 if model.kind == "frozen" else not 1 <= rank <= 64)
            or config.get("coefficient_fractional_bits") != 8 or config.get("integer_rounding") != "floor((sum+128)/256)"
            or config.get("feature_transform") != {"frozen":"none", "hidden":"signed", "hidden_abs":"abs"}[model.kind]):
        raise ValueError("invalid producer predictor arithmetic")
    schema = meta.get("layer_schema", {})
    if not schema or set(schema) != set(model.params):
        raise ValueError("producer model layer schema mismatch")
    for layer, params in model.params.items():
        if not _LAYER.fullmatch(layer):
            raise ValueError("invalid producer model layer key")
        shape = schema[layer]
        channels, hidden_channels = shape.get("channels"), shape.get("hidden_channels")
        if (not isinstance(channels, int) or channels < 1 or not isinstance(hidden_channels, int)
                or hidden_channels < 1 or not isinstance(shape.get("head_dim"), int)
                or shape["head_dim"] < 1 or channels % shape["head_dim"]):
            raise ValueError("invalid producer model geometry")
        expected = {"base_K", "base_V"} if model.kind == "frozen" else {"base_K", "base_V", "indices", "scales", "coeff_K", "coeff_V"}
        if set(params) != expected:
            raise ValueError("producer model parameter fields mismatch")
        for kind in ("K", "V"):
            base = np.asarray(params["base_" + kind])
            if base.dtype != np.uint8 or base.shape != (channels,):
                raise ValueError("invalid frozen exponent base")
            if model.kind != "frozen":
                coeff = np.asarray(params["coeff_" + kind])
                if coeff.dtype != np.int16 or coeff.shape != (rank, channels):
                    raise ValueError("invalid producer q8 coefficient array")
        if model.kind != "frozen":
            indices, scales = np.asarray(params["indices"]), np.asarray(params["scales"])
            if (indices.dtype != np.int32 or indices.shape != (rank,) or len(np.unique(indices)) != rank
                    or np.any(indices < 0) or np.any(indices >= hidden_channels)):
                raise ValueError("invalid producer hidden selectors")
            if scales.dtype != np.float32 or scales.shape != (rank,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("invalid producer feature scales")


def model_bytes(model):
    _validate_model(model)
    metadata = _json(model.metadata)
    manifest, payload = [], bytearray()
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


def load(path, *, expected_model_identity=None, expected_training_hashes=None, expected_model_id=None, expected_kind=None):
    source = Path(path)
    data = source.read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError("truncated producer model")
    magic, version, metadata_bytes, payload_bytes, checksum = _HEADER.unpack_from(data)
    body = data[_HEADER.size:]
    if (magic != _MAGIC or version != _VERSION or metadata_bytes + payload_bytes != len(body)
            or zlib.crc32(body) != checksum):
        raise ValueError("invalid producer model header/CRC")
    try:
        metadata = json.loads(body[:metadata_bytes])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid producer model JSON") from error
    if _canonical(metadata) != body[:metadata_bytes]:
        raise ValueError("noncanonical producer model metadata")
    if expected_model_identity is not None and _json(expected_model_identity) != metadata.get("model_identity"):
        raise ValueError("producer source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes) != metadata.get("training_hashes"):
        raise ValueError("producer TRAIN source identity mismatch")
    kind = metadata.get("kind")
    if expected_kind is not None and expected_kind != kind:
        raise ValueError("producer predictor kind mismatch")
    params, offset = {}, 0
    manifest = metadata.pop("array_manifest", None)
    if not isinstance(manifest, list):
        raise ValueError("missing producer model array manifest")
    for entry in manifest:
        try:
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if (not isinstance(shape, list) or not shape or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape)
                    or entry["offset"] != offset or entry["nbytes"] != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    or offset + entry["nbytes"] > payload_bytes):
                raise ValueError("invalid producer array bounds")
            layer, name = entry["layer"], entry["name"]
            if not isinstance(layer, str) or not isinstance(name, str) or name in params.setdefault(layer, {}):
                raise ValueError("duplicate/invalid producer model array")
            array = np.frombuffer(body, dtype=dtype, count=int(np.prod(shape)), offset=metadata_bytes + offset).reshape(shape).copy()
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("invalid producer array manifest") from error
        array.flags.writeable = False
        params[layer][name] = array
        offset += entry["nbytes"]
    if offset != payload_bytes:
        raise ValueError("unaccounted producer state payload")
    result = Model(kind, params, metadata, str(source.resolve()))
    if model_bytes(result) != data:
        raise ValueError("producer model canonical payload mismatch")
    if expected_model_id is not None and result.model_id != expected_model_id:
        raise ValueError("producer model SHA256 mismatch")
    return result


def fit(training_record_sets, output_dir, *, rank=8, training_metadata, model_identity, seed=71):
    """Fit frozen, signed-feature and abs-feature controls on three TRAIN captures."""
    if (not isinstance(training_record_sets, Sequence) or isinstance(training_record_sets, (str, bytes))
            or len(training_record_sets) != 3 or any(not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records
                                                    for records in training_record_sets)):
        raise ValueError("producer fitting requires three explicit nonempty TRAIN record sets")
    if (isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 64
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("invalid producer model rank/seed")
    if training_metadata is None or len(training_metadata) != 3:
        raise ValueError("producer fitting requires metadata for all three TRAIN captures")
    metadata_rows = _json(training_metadata)
    if any(not isinstance(meta, dict) or meta.get("split") not in ("train", "training") for meta in metadata_rows):
        raise ValueError("producer fitting accepts explicitly labeled TRAIN captures only")
    identity = _json(model_identity)
    if not isinstance(identity, (dict, str)) or not identity:
        raise ValueError("explicit producer source model identity is required")
    before = [_source_hash(records) for records in training_record_sets]
    metadata_before = _source_hash(dict(training_metadata=metadata_rows, model_identity=identity))
    schema, grouped, descriptions = {}, {}, []
    expected_layers = None
    for index, records in enumerate(training_record_sets):
        layers = set()
        record_rows = []
        seen = set()
        for record in records:
            layer, geometry = _validate_record(record)
            if geometry["hidden_channels"] < rank:
                raise ValueError("producer feature rank exceeds hidden channel count")
            if layer in schema and schema[layer] != geometry:
                raise ValueError("producer TRAIN record geometries differ")
            schema[layer] = geometry
            layers.add(layer)
            address = (layer, int(record["start"]))
            if address in seen:
                raise ValueError("duplicate producer record in one TRAIN capture")
            seen.add(address)
            grouped.setdefault(layer, []).append(record)
            record_rows.append(dict(layer=int(record["layer"]), start=int(record["start"]),
                tokens=int(record["tokens"]), head_dim=int(record["head_dim"]), sha256=_source_hash(record)))
        if expected_layers is not None and layers != expected_layers:
            raise ValueError("three producer TRAIN captures must cover matching layers")
        expected_layers = layers
        descriptions.append(dict(name=metadata_rows[index].get("name", f"producer_train_{index}"), split="train",
            sha256=before[index], records=record_rows, supplied_metadata=metadata_rows[index]))
    architecture = identity.get("config", identity) if isinstance(identity, dict) else {}
    if isinstance(architecture, dict):
        if architecture.get("head_dim") is not None and any(int(architecture["head_dim"]) != g["head_dim"] for g in schema.values()):
            raise ValueError("producer record head_dim differs from source architecture")
        if architecture.get("hidden_size") is not None and any(int(architecture["hidden_size"]) != g["hidden_channels"] for g in schema.values()):
            raise ValueError("producer hidden width differs from source architecture")
        if architecture.get("num_key_value_heads") is not None and any(int(architecture["num_key_value_heads"]) * g["head_dim"] != g["channels"] for g in schema.values()):
            raise ValueError("producer KV projection width differs from source architecture")
    frozen_params, hidden_params, abs_params, fit_stats, abs_stats = {}, {}, {}, {}, {}
    for layer in sorted(grouped):
        records = grouped[layer]
        bases = {}
        for kind in ("K", "V"):
            exponents = np.concatenate([((_word_array(record, kind.lower()).astype(np.uint32) >> 7) & 255)
                                        for record in records])
            bases["base_" + kind] = np.floor(np.median(exponents, axis=0)).astype(np.uint8)
        learned, stats = _fit_layer_hidden(records, bases["base_K"], bases["base_V"], rank)
        frozen_params[layer] = bases
        hidden_params[layer] = {name: array.copy() for name, array in bases.items()} | learned
        fit_stats[layer] = stats
        abs_learned, layer_abs_stats = _fit_layer_hidden(records, bases["base_K"], bases["base_V"], rank,
            feature_transform="abs", shared=learned)
        abs_params[layer] = {name: array.copy() for name, array in bases.items()} | abs_learned
        abs_stats[layer] = layer_abs_stats
    after = [_source_hash(records) for records in training_record_sets]
    metadata_after = _source_hash(dict(training_metadata=_json(training_metadata), model_identity=_json(model_identity)))
    if before != after or metadata_before != metadata_after:
        raise AssertionError("producer TRAIN source fields changed during fitting")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = dict(frozen=output / "producer_frozen.bin", hidden=output / f"producer_hidden_r{rank}.bin",
        hidden_abs=output / f"producer_hidden_abs_r{rank}.bin")
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("producer model artifacts exist; use a fresh output directory")
    models = {}
    for kind, params in (("frozen", frozen_params), ("hidden", hidden_params), ("hidden_abs", abs_params)):
        for layer in params.values():
            for array in layer.values():
                array.flags.writeable = False
        metadata = dict(format=_FORMAT, kind=kind, training_only=True, model_identity=identity,
            training_hashes=before, training_record_sets=descriptions, layer_schema=schema,
            config=dict(rank=rank if kind != "frozen" else 0, requested_hidden_rank=rank, seed=seed,
                feature_transform={"frozen":"none", "hidden":"signed", "hidden_abs":"abs"}[kind],
                selector="TRAIN normalized-hidden float64 variance; descending variance, ascending index ties",
                feature_quantization="float32(hidden_selected/scales), round-nearest-even, clip[-127,127], int8",
                feature_scaling="TRAIN maxabs/127 float32; zero maxabs uses1.0; floor at float32 tiny",
                exponent_base="floor median of all TRAIN row exponents per layer/KV/channel",
                coefficient_fractional_bits=8, coefficient_fit="float64 ridge lambda1.0, no intercept beyond frozen base",
                coefficient_quantization="round-nearest-even(weight*256), clip int16",
                integer_rounding="floor((sum+128)/256)", prediction_clamp=[0,255],
                shared_features="one int8 feature vector per token and layer, reused for K and V"),
            training_stats={"frozen":{}, "hidden":fit_stats, "hidden_abs":abs_stats}[kind],
            scope="Three actual TRAIN producer captures only. Hidden recipe retains selected int8 features, never a full hidden state or transformer forward replay; exact exponent corrections remain required.")
        model = Model(kind, params, metadata, str(paths[kind]))
        paths[kind].write_bytes(model_bytes(model))
        models[kind] = load(paths[kind], expected_model_identity=identity,
            expected_training_hashes=before, expected_model_id=model.model_id, expected_kind=kind)
    return models

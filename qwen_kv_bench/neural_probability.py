"""Torch port of the supplied cheap previous-layer logistic probability model.

This learns a distribution, not a replacement KV tensor. Exact BF16 interval
evaluation and actual entropy streams belong to separate modules. The source
algorithm is approach2_improve.py, M=1, inputs=prev1, width=512, vk=0, dz=0.
Qwen uses explicit split-half RoPE, unlike the source's adjacent-pair Smol RoPE.
A shared network plus layer one-hot is a separate experimental ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import torch
from torch.nn import functional as F

from . import producer_models as producer

_MAGIC = b"NPKV"
_HEADER = struct.Struct("<4sBIII")
_FORMAT = "qwen_previous_layer_logistic_models_v1"
_LOG_SCALE_LIMIT = 30.


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _floats(words):
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.ndim != 2 or min(words.shape) < 1:
        raise ValueError("neural training inputs must be nonempty uint16[T,C] BF16 words")
    value = (words.astype(np.uint32) << 16).view(np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("neural probability model requires finite actual BF16 KV")
    return value


def _runtime(device):
    device = torch.device(device)
    result = dict(rule="previous-layer-FP32-GELU-tanh-logistic-split-half-RoPE-v1",
        torch_version=str(torch.__version__), torch_git_version=getattr(torch.version, "git_version", None),
        cuda_build=torch.version.cuda, device_type=device.type,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        autocast_enabled=bool(torch.is_autocast_enabled(device.type)),
        cpu_threads=torch.get_num_threads() if device.type == "cpu" else None)
    if device.type == "cuda":
        result.update(device_name=torch.cuda.get_device_name(device),
            device_capability=list(torch.cuda.get_device_capability(device)))
    return result


def _forward(params, normalizers, inverse, scaling, head_dim, previous, start, *, layer_index=None, layers=None):
    """The identical eager FP32 inference math used during fitting and decoding."""
    tokens, double_channels = previous.shape
    channels = double_channels // 2
    x = (previous - normalizers["prev_mu"]) / normalizers["prev_sd"]
    if layer_index is not None:
        onehot = torch.zeros((tokens, layers), device=previous.device, dtype=torch.float32)
        onehot[:, layer_index] = 1.
        x = torch.cat((x, onehot), dim=1)
    x = F.gelu(x @ params["W0"] + params["b0"], approximate="tanh")
    x = F.gelu(x @ params["W1"] + params["b1"], approximate="tanh")
    output = x @ params["W2"] + params["b2"]
    k = (output[:, :channels] * normalizers["k_scale"]).reshape(tokens, channels // head_dim, head_dim)
    positions = torch.arange(start, start + tokens, device=previous.device, dtype=torch.float32)
    frequencies = positions[:, None] * inverse[None, :]
    angles = torch.cat((frequencies, frequencies), dim=1)
    cosine, sine = (torch.cos(angles) * scaling)[:, None, :], (torch.sin(angles) * scaling)[:, None, :]
    rotated = torch.cat((-k[..., head_dim // 2:], k[..., :head_dim // 2]), dim=-1)
    mean_k = (k * cosine + rotated * sine).reshape(tokens, channels)
    mean_v = output[:, channels:2 * channels] * normalizers["v_sd"] + normalizers["v_mu"]
    log_scale = output[:, 2 * channels:] + torch.log(torch.cat((normalizers["k_scale"], normalizers["v_sd"]))) - 1.
    # The original has no scale clamp. This disclosed numerical guard is part
    # of both the fit and runtime rule; it is never hidden as exact reproduction.
    return torch.cat((mean_k, mean_v), dim=1), log_scale.clamp(-_LOG_SCALE_LIMIT, _LOG_SCALE_LIMIT)


def _loss(target_words, target, mu, log_scale):
    exponent = (target_words.to(torch.int64) >> 7) & 255
    log2_ulp = torch.where(exponent > 0, exponent - 134, -133).to(torch.float32)
    standardized = (target - mu) * torch.exp(-log_scale)
    absolute = standardized.abs()
    # Symmetric form avoids cancellation for negative standardized residuals.
    log_density = -absolute - log_scale - 2 * F.softplus(-absolute)
    return (-(log_density / np.log(2.) + log2_ulp)).clamp_min(0).mean()


def _schema(model):
    meta, params = model.metadata, model.params
    if (meta.get("format") != _FORMAT or meta.get("mode") != model.mode or model.mode not in ("per_layer", "shared")
            or meta.get("training_only") is not True or not meta.get("model_identity")):
        raise ValueError("invalid neural model identity")
    geometry = meta.get("geometry", {})
    channels, layers, head_dim, width = (geometry.get(key) for key in ("channels", "layers", "head_dim", "width"))
    if (any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (channels, layers, head_dim, width))
            or layers < 2 or head_dim % 2 or channels % head_dim):
        raise ValueError("invalid neural model geometry")
    training = meta.get("training_captures", [])
    if (not training or meta.get("training_hashes") != [row.get("sha256") for row in training]
            or any(row.get("split") != "train" for row in training)):
        raise ValueError("invalid neural TRAIN provenance")
    config = meta.get("config", {})
    if (config.get("activation") != "GELU approximate=tanh" or config.get("mixture_components") != 1
            or config.get("inputs") != "previous-layer exact final post-RoPE K and V"
            or config.get("log_scale_clamp") != [-_LOG_SCALE_LIMIT, _LOG_SCALE_LIMIT]
            or config.get("rope_layout") != "split-half"):
        raise ValueError("invalid neural inference rule")
    expected = {"global"} | {f"norm_{layer:02d}" for layer in range(1, layers)} | (
        {f"net_{layer:02d}" for layer in range(1, layers)} if model.mode == "per_layer" else {"shared"})
    if set(params) != expected or set(params["global"]) != {"rope_inv_freq"}:
        raise ValueError("neural model parameter groups mismatch")
    if params["global"]["rope_inv_freq"].shape != (head_dim // 2,):
        raise ValueError("neural RoPE frequency geometry mismatch")
    if not isinstance(config.get("rope_scaling"), (int, float)) or not np.isfinite(config["rope_scaling"]) or config["rope_scaling"] <= 0:
        raise ValueError("invalid neural RoPE scaling")
    for group, fields in params.items():
        for array in fields.values():
            if array.dtype != np.float32 or array.ndim < 1 or min(array.shape) < 1 or not np.all(np.isfinite(array)):
                raise ValueError("invalid neural parameter dtype/finite values")
        if group.startswith("norm_"):
            expected_shapes = dict(prev_mu=(2 * channels,), prev_sd=(2 * channels,),
                k_scale=(channels,), v_mu=(channels,), v_sd=(channels,))
        elif group != "global":
            inputs = 2 * channels + (layers - 1 if model.mode == "shared" else 0)
            expected_shapes = dict(W0=(inputs, width), b0=(width,), W1=(width, width), b1=(width,),
                                   W2=(width, 4 * channels), b2=(4 * channels,))
        else:
            continue
        if set(fields) != set(expected_shapes) or any(fields[name].shape != shape for name, shape in expected_shapes.items()):
            raise ValueError("neural model parameter shapes mismatch")
        if group.startswith("norm_") and any(np.any(fields[name] <= 0) for name in ("prev_sd", "k_scale", "v_sd")):
            raise ValueError("neural model scales must be strictly positive")


@dataclass
class NeuralModel:
    mode: str
    params: dict
    metadata: dict
    path: str = ""

    @property
    def model_id(self):
        return hashlib.sha256(model_bytes(self)).hexdigest()

    @property
    def state_bytes(self):
        return _artifact_manifest(self)[0]["total_serialized_bytes"]

    @property
    def serialized_bytes(self):
        return self.state_bytes

    @property
    def artifact_paths(self):
        if not self.path:
            return []
        base = Path(self.path).parent
        descriptor, _ = _artifact_manifest(self)
        return [str(Path(self.path))] + [str(base / row["file"]) for row in descriptor["shards"]]

    @property
    def parameter_count(self):
        return sum(array.size for name, group in self.params.items() if name.startswith("net_") or name == "shared" for array in group.values())

    @property
    def total_float32_values(self):
        return sum(array.size for group in self.params.values() for array in group.values())

    def macs_per_token(self, layer=None):
        geometry = self.metadata["geometry"]
        c, w, layers = (geometry[key] for key in ("channels", "width", "layers"))
        return (2 * c + (layers - 1 if self.mode == "shared" else 0)) * w + w * w + w * 4 * c

    @property
    def macs_per_value(self):
        return self.macs_per_token() / (2 * self.metadata["geometry"]["channels"])

    def prepare(self, device="cpu"):
        return PreparedNeuralModel(self, device)

    def predict(self, layer, prev_words, start=0, device="cpu"):
        """Convenience only: prepare once outside measured loops for real use."""
        return self.prepare(device).predict(layer, prev_words, start)


class PreparedNeuralModel:
    def __init__(self, model, device):
        _schema(model)
        self.device = torch.empty(0, device=device).device
        self.mode, self.metadata = model.mode, producer._json(model.metadata)
        self.model_id = model.model_id
        self.serialized_bytes = model.state_bytes
        self.params = {group: {name: torch.tensor(array.copy(), device=self.device)
            for name, array in fields.items()} for group, fields in model.params.items()}
        self.execution_contract = _runtime(self.device)
        self.state_id = hashlib.sha256(_canonical(dict(model_id=self.model_id, runtime=self.execution_contract))).hexdigest()
        self.parameter_count = model.parameter_count
        self.macs_per_token = model.macs_per_token()
        self.macs_per_value = model.macs_per_value

    @property
    def physical_nbytes(self):
        return sum(value.numel() * value.element_size() for group in self.params.values() for value in group.values())

    @property
    def state_bytes(self):
        return max(self.serialized_bytes, self.physical_nbytes)

    @torch.inference_mode()
    def predict(self, layer, prev_words, start=0):
        if _canonical(_runtime(self.device)) != _canonical(self.execution_contract):
            raise ValueError("neural predictor execution contract changed")
        geometry = self.metadata["geometry"]
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or not 1 <= layer < geometry["layers"]:
            raise ValueError("neural prediction requires target layer>=1; layer0 needs its anchor codec")
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start < 0:
            raise ValueError("invalid neural token start")
        if isinstance(prev_words, np.ndarray):
            if prev_words.dtype != np.uint16:
                raise ValueError("neural previous KV must contain BF16 word bits")
            previous = torch.from_numpy(np.ascontiguousarray(prev_words).copy()).to(self.device)
        elif isinstance(prev_words, torch.Tensor) and prev_words.dtype in (torch.uint16, torch.int16):
            previous = prev_words.to(self.device).contiguous().view(torch.uint16)
        else:
            raise ValueError("neural previous KV must be NumPy uint16 or Torch word bits")
        if previous.ndim != 2 or previous.shape[0] < 1 or previous.shape[1] != 2 * geometry["channels"]:
            raise ValueError("neural previous KV geometry mismatch")
        previous = previous.contiguous().view(torch.bfloat16).to(torch.float32)
        if not bool(torch.isfinite(previous).all()):
            raise ValueError("nonfinite neural previous KV")
        network = self.params[f"net_{layer:02d}" if self.mode == "per_layer" else "shared"]
        with torch.autocast(device_type=self.device.type, enabled=False):
            mu, log_scale = _forward(network, self.params[f"norm_{layer:02d}"],
                self.params["global"]["rope_inv_freq"], self.metadata["config"]["rope_scaling"],
                geometry["head_dim"], previous, int(start),
                layer_index=int(layer) - 1 if self.mode == "shared" else None, layers=geometry["layers"] - 1)
        if not bool(torch.isfinite(mu).all() & torch.isfinite(log_scale).all()):
            raise ValueError("nonfinite neural probability parameters")
        return mu, log_scale


def _shard_bytes(group, fields):
    payload, manifest = bytearray(), []
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name].astype("<f4", copy=False))
        manifest.append(dict(name=name, shape=list(array.shape), dtype="<f4", offset=len(payload), nbytes=array.nbytes))
        payload.extend(array.tobytes())
    meta = dict(format="neural_parameter_shard_v1", group=group, array_manifest=manifest)
    encoded = _canonical(meta)
    body = encoded + bytes(payload)
    return _HEADER.pack(_MAGIC, 1, len(encoded), len(payload), zlib.crc32(body)) + body


def _array_signature(model):
    return tuple((group, name, id(array), array.dtype.str, array.shape, bool(array.flags.writeable))
        for group in sorted(model.params) for name, array in sorted(model.params[group].items()))


def _seal(model, descriptor, encoded):
    for group in model.params.values():
        for array in group.values():
            array.flags.writeable = False
    model._frozen_descriptor = descriptor
    model._frozen_manifest_bytes = encoded
    model._frozen_array_signature = _array_signature(model)
    model._frozen_metadata_sha = hashlib.sha256(_canonical(model.metadata)).hexdigest()


def _artifact_manifest(model):
    if hasattr(model, "_frozen_manifest_bytes"):
        # Frozen arrays cannot be edited accidentally. Reject replaced arrays,
        # changed writeability/shape, or metadata changes before reusing hashes.
        # This checks small descriptors, not hundreds of MB of payload per call.
        if (_array_signature(model) != model._frozen_array_signature
                or model.mode != model._frozen_descriptor["metadata"]["mode"]
                or hashlib.sha256(_canonical(model.metadata)).hexdigest() != model._frozen_metadata_sha):
            raise ValueError("frozen neural artifact parameters or metadata changed")
        return model._frozen_descriptor, model._frozen_manifest_bytes
    _schema(model)
    shards = []
    for group in sorted(model.params):
        wire = _shard_bytes(group, model.params[group])
        if len(wire) >= 99 * 1024 ** 2:
            raise ValueError("one neural parameter group exceeds the 99 MiB artifact-shard limit; reduce width")
        shards.append(dict(file=f"neural_{model.mode}_{group}.npkv", group=group,
            bytes=len(wire), sha256=hashlib.sha256(wire).hexdigest()))
    descriptor = dict(format="neural_probability_sharded_artifact_v1", metadata=producer._json(model.metadata), shards=shards,
        total_shard_bytes=sum(row["bytes"] for row in shards), manifest_bytes=0, total_serialized_bytes=0)
    descriptor["metadata_and_shards_crc32"] = zlib.crc32(_canonical(dict(metadata=descriptor["metadata"], shards=shards)))
    for _ in range(8):
        encoded = _canonical(descriptor)
        if descriptor["manifest_bytes"] == len(encoded) and descriptor["total_serialized_bytes"] == len(encoded) + descriptor["total_shard_bytes"]:
            return descriptor, encoded
        descriptor["manifest_bytes"] = len(encoded)
        descriptor["total_serialized_bytes"] = len(encoded) + descriptor["total_shard_bytes"]
    raise AssertionError("neural manifest byte accounting failed to converge")


def model_bytes(model):
    """Canonical manifest bytes; all parameter shards are bound by SHA and CRC."""
    return _artifact_manifest(model)[1]


def save(model, path):
    path = Path(path)
    descriptor, encoded = _artifact_manifest(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in descriptor["shards"]:
        (path.parent / row["file"]).write_bytes(_shard_bytes(row["group"], model.params[row["group"]]))
    path.write_bytes(encoded)
    _seal(model, descriptor, encoded)
    return path


def _load_shard(data, expected_group):
    if len(data) < _HEADER.size:
        raise ValueError("truncated neural parameter shard")
    magic, version, metadata_size, payload_size, checksum = _HEADER.unpack_from(data)
    body = data[_HEADER.size:]
    if magic != _MAGIC or version != 1 or len(body) != metadata_size + payload_size or zlib.crc32(body) != checksum:
        raise ValueError("invalid neural parameter shard header/CRC")
    try:
        meta = json.loads(body[:metadata_size])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid neural shard JSON") from error
    if _canonical(meta) != body[:metadata_size]:
        raise ValueError("noncanonical neural shard metadata")
    if meta.get("format") != "neural_parameter_shard_v1" or meta.get("group") != expected_group:
        raise ValueError("neural parameter shard group mismatch")
    manifest = meta.pop("array_manifest", None)
    if not isinstance(manifest, list):
        raise ValueError("missing neural model manifest")
    params, offset = {}, 0
    for entry in manifest:
        try:
            shape, name = entry["shape"], entry["name"]
            if (entry["dtype"] != "<f4" or not isinstance(shape, list) or not shape
                    or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape)
                    or entry["offset"] != offset or entry["nbytes"] != int(np.prod(shape, dtype=np.int64)) * 4
                    or offset + entry["nbytes"] > payload_size
                    or not isinstance(name, str) or name in params):
                raise ValueError("invalid neural model array bounds")
            begin = metadata_size + offset
            params[name] = np.frombuffer(body[begin:begin + entry["nbytes"]], dtype="<f4").reshape(shape).copy()
            offset += entry["nbytes"]
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("invalid neural model manifest") from error
    if offset != payload_size:
        raise ValueError("unreferenced neural payload bytes")
    if _shard_bytes(expected_group, params) != data:
        raise ValueError("noncanonical neural shard array order")
    return params


def load(path, *, expected_model_identity=None, expected_training_hashes=None, expected_model_id=None, expected_mode=None):
    path = Path(path)
    data = path.read_bytes()
    try:
        descriptor = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid neural artifact manifest JSON") from error
    if (not isinstance(descriptor, dict) or descriptor.get("format") != "neural_probability_sharded_artifact_v1"
            or _canonical(descriptor) != data or not isinstance(descriptor.get("shards"), list)):
        raise ValueError("noncanonical neural artifact manifest")
    meta, shards = descriptor.get("metadata"), descriptor["shards"]
    if (not isinstance(meta, dict) or descriptor.get("metadata_and_shards_crc32") !=
            zlib.crc32(_canonical(dict(metadata=meta, shards=shards)))):
        raise ValueError("invalid neural manifest CRC")
    if expected_model_id is not None and hashlib.sha256(data).hexdigest() != expected_model_id:
        raise ValueError("neural model SHA256 mismatch")
    if expected_model_identity is not None and producer._json(expected_model_identity) != meta.get("model_identity"):
        raise ValueError("neural source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes) != meta.get("training_hashes"):
        raise ValueError("neural TRAIN source identity mismatch")
    if expected_mode is not None and expected_mode != meta.get("mode"):
        raise ValueError("neural predictor mode mismatch")
    params, total = {}, 0
    for row in shards:
        try:
            filename, group = row["file"], row["group"]
            if (not isinstance(filename, str) or Path(filename).name != filename or filename in (".", "..")
                    or not isinstance(group, str) or group in params):
                raise ValueError("invalid neural shard filename/group")
            wire = (path.parent / filename).read_bytes()
            if len(wire) != row["bytes"] or hashlib.sha256(wire).hexdigest() != row["sha256"]:
                raise ValueError("neural shard byte size/SHA256 mismatch")
            params[group] = _load_shard(wire, group)
            total += len(wire)
        except (KeyError, TypeError) as error:
            raise ValueError("invalid neural shard manifest") from error
    if (descriptor.get("total_shard_bytes") != total or descriptor.get("manifest_bytes") != len(data)
            or descriptor.get("total_serialized_bytes") != total + len(data)):
        raise ValueError("neural complete artifact byte accounting mismatch")
    model = NeuralModel(meta.get("mode"), params, meta, str(path.resolve()))
    if model_bytes(model) != data:
        raise ValueError("noncanonical neural complete artifact")
    for group in params.values():
        for array in group.values():
            array.flags.writeable = False
    _seal(model, descriptor, data)
    return model


def _init(inputs, width, output, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    params = {}
    for index, (a, b) in enumerate(((inputs, width), (width, width), (width, output))):
        params["W" + str(index)] = torch.nn.Parameter(torch.randn((a, b), device=device, generator=generator) * np.sqrt(2. / a))
        params["b" + str(index)] = torch.nn.Parameter(torch.zeros(b, device=device))
    return params


def _normalizers(captures, layer, channels, head_dim):
    previous = np.concatenate([np.concatenate((_floats(rows[layer - 1][0]), _floats(rows[layer - 1][1])), axis=1) for rows in captures])
    k = np.concatenate([_floats(rows[layer][0]) for rows in captures])
    v = np.concatenate([_floats(rows[layer][1]) for rows in captures])
    key = k.reshape(len(k), channels // head_dim, head_dim).astype(np.float64)
    half = head_dim // 2
    pair_rms = np.sqrt(((key[..., :half] ** 2 + key[..., half:] ** 2) * .5).mean(axis=0))
    k_scale = np.concatenate((pair_rms, pair_rms), axis=-1).reshape(channels).astype(np.float32) + np.float32(1e-6)
    return dict(prev_mu=previous.mean(axis=0, dtype=np.float64).astype(np.float32),
        prev_sd=previous.astype(np.float64).std(axis=0).astype(np.float32) + np.float32(1e-6),
        k_scale=k_scale, v_mu=v.mean(axis=0, dtype=np.float64).astype(np.float32),
        v_sd=v.astype(np.float64).std(axis=0).astype(np.float32) + np.float32(1e-6))


def fit(training_captures, output_dir, *, model_identity, training_metadata, head_dim,
        rope_inv_freq, rope_scaling=1., device="cpu", width=512, steps=2000,
        batch_tokens=256, seed=71, modes=("per_layer", "shared")):
    """Fit only explicit TRAIN captures; no DEV/TEST input enters this API."""
    if not isinstance(training_captures, (list, tuple)) or not training_captures:
        raise ValueError("neural fitting requires nonempty actual TRAIN captures")
    if (isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 2 or head_dim % 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (width, steps, batch_tokens))
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("invalid neural training geometry/budget")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ("per_layer", "shared") for mode in modes):
        raise ValueError("unknown or duplicate neural model modes")
    metadata_rows, identity = producer._json(training_metadata), producer._json(model_identity)
    if (not isinstance(metadata_rows, list) or len(metadata_rows) != len(training_captures)
            or any(not isinstance(row, dict) or row.get("split") not in ("train", "training") for row in metadata_rows)
            or not isinstance(identity, (str, dict)) or not identity):
        raise ValueError("neural fitting requires explicit TRAIN labels and source model identity")
    before = [producer._source_hash(rows) for rows in training_captures]
    provenance_before = producer._source_hash(dict(metadata=metadata_rows, identity=identity))
    layers, channels = len(training_captures[0]), None
    if layers < 2:
        raise ValueError("neural previous-layer model requires at least two layers")
    capture_rows = []
    for rows in training_captures:
        if not isinstance(rows, (list, tuple)) or len(rows) != layers:
            raise ValueError("neural TRAIN captures must cover the same layers")
        tokens = None
        for pair in rows:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("neural capture layer must contain actual K/V word matrices")
            k, v = pair
            _floats(k), _floats(v)
            if k.shape != v.shape:
                raise ValueError("neural K/V geometry mismatch")
            if channels is None:
                channels = k.shape[1]
            if tokens is None:
                tokens = len(k)
            if k.shape != (tokens, channels):
                raise ValueError("neural TRAIN layers have inconsistent geometry")
        capture_rows.append(tokens)
    inverse = np.asarray(rope_inv_freq)
    if (channels % head_dim or inverse.dtype != np.float32 or inverse.shape != (head_dim // 2,)
            or not np.all(np.isfinite(inverse)) or np.any(inverse <= 0)
            or not isinstance(rope_scaling, (int, float)) or not np.isfinite(rope_scaling) or rope_scaling <= 0):
        raise ValueError("invalid explicit Qwen RoPE configuration")
    architecture = identity.get("config", identity) if isinstance(identity, dict) else {}
    if isinstance(architecture, dict):
        for name, actual in (("num_hidden_layers", layers), ("head_dim", head_dim)):
            if name in architecture and architecture[name] != actual:
                raise ValueError("neural capture geometry differs from explicit source architecture")
        if "num_key_value_heads" in architecture and architecture["num_key_value_heads"] * head_dim != channels:
            raise ValueError("neural KV width differs from explicit source architecture")
    output = Path(output_dir).resolve()
    if any((output / ("neural_" + mode + ".json")).exists() for mode in modes):
        raise FileExistsError("neural artifacts exist; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    target_device = torch.empty(0, device=device).device
    norms = {layer: _normalizers(training_captures, layer, channels, head_dim) for layer in range(1, layers)}
    inverse_device = torch.tensor(inverse.copy(), device=target_device)
    results = {}
    for mode in modes:
        params = {"global": {"rope_inv_freq": inverse.copy()}}
        params.update({f"norm_{layer:02d}": {key: value.copy() for key, value in norm.items()} for layer, norm in norms.items()})
        logs, total_updates = {}, 0
        groups = [[layer] for layer in range(1, layers)] if mode == "per_layer" else [list(range(1, layers))]
        for layer_group in groups:
            inputs = 2 * channels + (layers - 1 if mode == "shared" else 0)
            network = _init(inputs, width, 4 * channels, target_device, seed + (layer_group[0] if mode == "per_layer" else 0))
            optimizer = torch.optim.Adam(list(network.values()), lr=1e-3, betas=(.9, .999), eps=1e-8)
            # Per-layer mode trains/releases one layer at a time. Shared mode
            # needs only small per-layer normalizers; no target tensors persist.
            resident_norms = {layer: {key: torch.tensor(value.copy(), device=target_device) for key, value in norms[layer].items()}
                              for layer in layer_group}
            rng = np.random.default_rng(seed + layer_group[0])
            checkpoints, losses, updates = [], [], {str(layer): 0 for layer in layer_group}
            initial_loss = None
            for update in range(1, steps + 1):
                layer = layer_group[int(rng.integers(len(layer_group)))]
                capture = int(rng.integers(len(training_captures)))
                rows = training_captures[capture]
                count = min(batch_tokens, capture_rows[capture])
                start = int(rng.integers(capture_rows[capture] - count + 1))
                previous_words = np.concatenate((rows[layer - 1][0][start:start + count], rows[layer - 1][1][start:start + count]), axis=1)
                target_words = np.concatenate((rows[layer][0][start:start + count], rows[layer][1][start:start + count]), axis=1)
                previous = torch.tensor(_floats(previous_words), device=target_device)
                target = torch.tensor(_floats(target_words), device=target_device)
                target_bits = torch.tensor(target_words.copy(), device=target_device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=target_device.type, enabled=False):
                    mu, log_scale = _forward(network, resident_norms[layer], inverse_device, float(rope_scaling), head_dim,
                        previous, start, layer_index=layer - 1 if mode == "shared" else None, layers=layers - 1)
                    loss = _loss(target_bits, target, mu, log_scale)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite neural TRAIN density/bin-width objective")
                value = float(loss.detach().cpu())
                if initial_loss is None:
                    initial_loss = value
                losses.append(value)
                if len(losses) > 100:
                    losses.pop(0)
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in network.values()):
                    raise ValueError("nonfinite neural TRAIN gradient")
                rate = 1e-3 * .5 * (1 + np.cos(np.pi * update / steps))
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.step()
                updates[str(layer)] += 1
                if update == 1 or update == steps or update % max(1, steps // 8) == 0:
                    checkpoints.append(dict(update=update, sampled_layer=layer, capture=capture,
                        window_start=start, tokens=count, density_proxy_bits_per_value=value, learning_rate=float(rate)))
            group_name = f"net_{layer_group[0]:02d}" if mode == "per_layer" else "shared"
            params[group_name] = {name: value.detach().to("cpu").numpy().astype(np.float32, copy=True) for name, value in network.items()}
            logs[group_name] = dict(initial_batch_loss=initial_loss, final_batch_loss=losses[-1],
                final_mean_last_100_batch_losses=float(np.mean(losses)), optimizer_updates=steps,
                updates_by_layer=updates, checkpoints=checkpoints,
                scope="TRAIN density/bin-width proxy only; initial/final losses use sampled batches, not a fixed validation set")
            total_updates += steps
            del optimizer, network, resident_norms
        meta = dict(format=_FORMAT, mode=mode, training_only=True, model_identity=identity,
            geometry=dict(layers=layers, channels=channels, head_dim=head_dim, width=width),
            training_hashes=before, training_captures=[dict(split="train", sha256=digest, tokens=tokens,
                metadata=metadata) for digest, tokens, metadata in zip(before, capture_rows, metadata_rows)],
            config=dict(inputs="previous-layer exact final post-RoPE K and V", mixture_components=1,
                stored_latent=False, condition_V_on_current_K=False, activation="GELU approximate=tanh",
                weight_dtype="float32", target_dtype="BF16 word intervals", rope_layout="split-half",
                rope_scaling=float(rope_scaling), log_scale_clamp=[-_LOG_SCALE_LIMIT, _LOG_SCALE_LIMIT],
                normalizer_rule="TRAIN mean/std float64 accumulation then float32; +1e-6 scales; split-half K-pair RMS ties actual rotated coordinate pairs",
                training_objective="mean max(0, -log2(logistic density * BF16 ULP)); density/bin-width approximation, not exact interval likelihood or actual compressed bytes",
                source_port="Torch adaptation of supplied approach2_improve.py M=1/prev1/vk=0/z=0; Qwen split-half RoPE replaces Smol adjacent pairs; disclosed log-scale clamp; no numerical reproduction claim",
                shared_rule="one MLP with (layers-1)-dimensional target-layer one-hot and separate per-layer TRAIN normalizers" if mode == "shared" else None),
            training=dict(seed=seed, steps_per_network=steps, batch_tokens=batch_tokens,
                optimizer="Adam betas=(0.9,0.999),eps=1e-8", learning_rate="1e-3*0.5*(1+cos(pi*update/steps))",
                sampling="uniform capture, uniform target layer in shared mode, uniform valid consecutive token window including final possible window",
                total_optimizer_updates=total_updates, networks=logs,
                budget_note="Per-layer mode performs (layers-1)*steps updates; shared mode performs steps total. These are explicitly unequal training budgets.",
                runtime=_runtime(target_device)),
            scope="Learned probability model only; complete weights/normalizers/RoPE state are charged. Exact entropy stream roundtrips and continuation must be measured separately. Layer0 uses a separate anchor codec.")
        for fields in params.values():
            for array in fields.values():
                array.flags.writeable = False
        model = NeuralModel(mode, params, meta)
        meta["compute"] = dict(trainable_parameter_count=model.parameter_count,
            complete_float32_state_values=model.total_float32_values,
            network_MACs_per_token=model.macs_per_token(), network_MACs_per_KV_value=model.macs_per_value,
            scope="Dense-layer MAC counts only; GELU, normalization, RoPE, probability CDF and entropy coding add work. Full codec state includes every normalizer and explicit inverse frequency.")
        path = output / ("neural_" + mode + ".json")
        save(model, path)
        results[mode] = load(path, expected_model_identity=identity, expected_training_hashes=before,
            expected_model_id=model.model_id, expected_mode=mode)
    if (before != [producer._source_hash(rows) for rows in training_captures]
            or provenance_before != producer._source_hash(dict(metadata=producer._json(training_metadata), identity=producer._json(model_identity)))):
        raise AssertionError("neural TRAIN captures or metadata changed during fitting")
    return results

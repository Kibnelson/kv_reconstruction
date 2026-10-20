"""Rate-trained prefill side codes for independently decodable exact KV pages.

Four matched shared decoders use no code, a code learned from the same KV,
a code from inverse-RoPE-canonicalized final KV, or a code learned from the
actual pre-projection normalized hidden tensor.
An encoder is optimized jointly with the decoder using a BF16 straight-through
latent and a disclosed logistic density/bin-width rate proxy. Actual exact
interval probabilities and transmitted bytes are measured by side_codec.
No previous layer or original producer activation is an inference dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np
import torch
from torch.nn import functional as F

from . import neural_probability as neural, producer_models as producer

FORMAT = "qwen_page_local_bf16_side_probability_v1"
MODES = ("none", "kv", "kv_canonical", "hidden")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("invalid side probability " + name)
    return int(value)


def _words(value, name):
    array = np.asarray(value)
    if array.dtype != np.uint16 or array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(name + " must be nonempty uint16[T,C] BF16 words")
    return array


def _floats(value):
    return neural._floats(value)


def joint(record):
    k, v = _words(record.get("k"), "k"), _words(record.get("v"), "v")
    if k.shape != v.shape:
        raise ValueError("side probability K/V geometry mismatch")
    return np.concatenate((k, v), axis=1)


def canonical_source(record, inverse, rope_scaling, head_dim):
    """BF16 encoder input derived only from final produced KV and fixed RoPE.

    FP32 inverse rotation is an approximate canonicalization of BF16 final K,
    never access to the actual pre-RoPE hook. This CPU encoder-only reference is
    included in feature production time. Exact target symbols stay unchanged.
    """
    source = joint(record)
    k, v = np.split(source, 2, axis=1)
    inverse = np.asarray(inverse)
    head_dim = _integer(head_dim, "head_dim")
    start = _integer(record.get("start"), "start", 0)
    if (head_dim % 2 or k.shape[1] % head_dim or inverse.dtype != np.float32
            or inverse.shape != (head_dim // 2,) or not np.isfinite(inverse).all()
            or np.any(inverse <= 0) or not np.isfinite(rope_scaling) or rope_scaling <= 0):
        raise ValueError("invalid side probability canonical encoder RoPE")
    key = _floats(k).reshape(len(k), -1, head_dim)
    position = np.arange(start, start + len(k), dtype=np.float32)
    frequency = position[:, None] * inverse[None, :]
    angles = np.concatenate((frequency, frequency), axis=1)
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    rotated = np.concatenate((-key[..., head_dim // 2:], key[..., :head_dim // 2]), axis=-1)
    canonical = ((key * cosine - rotated * sine) / np.float32(rope_scaling)).reshape(k.shape)
    if not np.isfinite(canonical).all():
        raise ValueError("nonfinite side probability canonical encoder source")
    bits = np.ascontiguousarray(canonical).view(np.uint32)
    canonical_words = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.concatenate((canonical_words, v), axis=1)


def _source(record, mode, *, inverse=None, rope_scaling=1., head_dim=None):
    if mode == "kv_canonical":
        return canonical_source(record, inverse, rope_scaling, head_dim)
    return joint(record) if mode == "kv" else _words(record.get("hidden"), "hidden")


def _runtime(device):
    result = neural._runtime(device)
    result["rule"] = "page-local-BF16-sidecode-FP32-GELU-tanh-logistic-split-half-RoPE-v1"
    return result


def _network_keys(mode):
    return {"W0", "b0", "W1", "b1", "W2", "b2"} | ({"encoder_W", "encoder_b"} if mode != "none" else set())


def _schema(model):
    meta, p = model.metadata, model.params
    if (model.mode not in MODES or meta.get("format") != FORMAT or meta.get("mode") != model.mode
            or meta.get("training_only") is not True or not meta.get("model_identity")):
        raise ValueError("invalid side probability identity")
    g = meta.get("geometry", {})
    l, c, h, d, r, w = (_integer(g.get(k), k, 0 if k == "rank" else 1)
                        for k in ("layers", "channels", "hidden_size", "head_dim", "rank", "width"))
    if d % 2 or c % d or (model.mode == "none") != (r == 0):
        raise ValueError("invalid side probability geometry")
    source_width = 2 * c if model.mode in ("kv", "kv_canonical") else h
    expected = dict(rope_inv_freq=(d // 2,), k_scale=(l, c), v_mu=(l, c), v_sd=(l, c),
                    W0=(r + l, w), b0=(w,), W1=(w, w), b1=(w,), W2=(w, 4 * c), b2=(4 * c,))
    if model.mode != "none":
        expected.update(source_mu=(l, source_width), source_sd=(l, source_width),
                        encoder_W=(source_width, r), encoder_b=(r,))
    if set(p) != set(expected):
        raise ValueError("side probability parameter fields mismatch")
    for name, shape in expected.items():
        a = p[name]
        if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
            raise ValueError("invalid side probability parameter " + name)
    for name in ("rope_inv_freq", "k_scale", "v_sd", "source_sd"):
        if name in p and np.any(p[name] <= 0):
            raise ValueError("side probability scale/frequency must be positive")
    config = meta.get("config", {})
    if (config.get("activation") != "GELU approximate=tanh" or config.get("latent_dtype") != "BF16"
            or config.get("rope_layout") != "split-half"
            or config.get("log_scale_clamp") != [-30., 30.]
            or config.get("feature_clamp") != [-10000., 10000.]
            or not isinstance(config.get("rope_scaling"), (int, float))
            or not np.isfinite(config["rope_scaling"]) or config["rope_scaling"] <= 0):
        raise ValueError("invalid side probability inference contract")
    captures = meta.get("training_captures", [])
    if (not captures or meta.get("training_hashes") != [row.get("sha256") for row in captures]
            or any(row.get("split") != "train" for row in captures)):
        raise ValueError("invalid side probability TRAIN provenance")


def _descriptor(model):
    _schema(model)
    wire = neural._shard_bytes("side_parameters", model.params)
    if len(wire) >= 99 * 1024 ** 2:
        raise ValueError("side probability artifact exceeds 99 MiB; reduce width/rank")
    shard = dict(file=f"side_{model.mode}.npkv", bytes=len(wire), sha256=hashlib.sha256(wire).hexdigest())
    desc = dict(format="side_probability_artifact_v1", metadata=producer._json(model.metadata), shard=shard,
                manifest_bytes=0, total_serialized_bytes=0)
    desc["metadata_and_shard_crc32"] = zlib.crc32(_canonical(dict(metadata=desc["metadata"], shard=shard)))
    for _ in range(8):
        data = _canonical(desc)
        if desc["manifest_bytes"] == len(data) and desc["total_serialized_bytes"] == len(data) + len(wire):
            return desc, data, wire
        desc["manifest_bytes"] = len(data)
        desc["total_serialized_bytes"] = len(data) + len(wire)
    raise AssertionError("side artifact byte accounting failed to converge")


@dataclass
class Model:
    mode: str
    params: dict
    metadata: dict
    path: str = ""

    def _frozen(self):
        if not hasattr(self, "_descriptor"):
            desc, data, _ = _descriptor(self)
            self._descriptor, self._manifest = desc, data
            self._metadata_sha = hashlib.sha256(_canonical(self.metadata)).hexdigest()
            self._signature = tuple((key, id(a), a.shape, a.dtype.str) for key, a in sorted(self.params.items()))
            for a in self.params.values():
                a.flags.writeable = False
        elif (hashlib.sha256(_canonical(self.metadata)).hexdigest() != self._metadata_sha
              or tuple((key, id(a), a.shape, a.dtype.str) for key, a in sorted(self.params.items())) != self._signature
              or any(a.flags.writeable for a in self.params.values())):
            raise ValueError("frozen side probability model changed")
        return self._descriptor, self._manifest

    @property
    def model_id(self):
        return hashlib.sha256(self._frozen()[1]).hexdigest()

    @property
    def state_bytes(self):
        return self._frozen()[0]["total_serialized_bytes"]

    @property
    def artifact_paths(self):
        return [self.path, str(Path(self.path).parent / self._frozen()[0]["shard"]["file"])] if self.path else []

    @property
    def parameter_count(self):
        return sum(self.params[name].size for name in _network_keys(self.mode))

    @property
    def total_float32_values(self):
        return sum(a.size for a in self.params.values())

    def prepare(self, device="cpu"):
        return Prepared(self, device)


def save(model, path):
    path = Path(path)
    desc, data, wire = _descriptor(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / desc["shard"]["file"]).write_bytes(wire)
    path.write_bytes(data)
    model.path = str(path.resolve())
    model._frozen()
    return path


def load(path, *, expected_model_identity=None, expected_training_hashes=None, expected_model_id=None, expected_mode=None):
    path = Path(path)
    data = path.read_bytes()
    try:
        desc = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid side probability manifest JSON") from error
    if not isinstance(desc, dict) or desc.get("format") != "side_probability_artifact_v1" or _canonical(desc) != data:
        raise ValueError("invalid canonical side probability manifest")
    meta, shard = desc.get("metadata"), desc.get("shard")
    if not isinstance(meta, dict) or not isinstance(shard, dict):
        raise ValueError("missing side probability metadata/shard")
    name = shard.get("file")
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError("invalid side probability shard filename")
    if desc.get("metadata_and_shard_crc32") != zlib.crc32(_canonical(dict(metadata=meta, shard=shard))):
        raise ValueError("side probability manifest CRC mismatch")
    if expected_model_identity is not None and producer._json(expected_model_identity) != meta.get("model_identity"):
        raise ValueError("side probability source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes) != meta.get("training_hashes"):
        raise ValueError("side probability TRAIN hashes mismatch")
    if expected_mode is not None and expected_mode != meta.get("mode"):
        raise ValueError("side probability mode mismatch")
    if expected_model_id is not None and expected_model_id != hashlib.sha256(data).hexdigest():
        raise ValueError("side probability model SHA mismatch")
    wire = (path.parent / name).read_bytes()
    if len(wire) != shard.get("bytes") or hashlib.sha256(wire).hexdigest() != shard.get("sha256"):
        raise ValueError("side probability shard bytes/SHA mismatch")
    params = neural._load_shard(wire, "side_parameters")
    model = Model(meta.get("mode"), params, meta, str(path.resolve()))
    if _descriptor(model)[1] != data:
        raise ValueError("noncanonical side probability complete artifact")
    model._frozen()
    return model


def _encode(params, source, layer, *, straight_through):
    x = (source - params["source_mu"][layer]) / params["source_sd"][layer]
    z = (x @ params["encoder_W"] + params["encoder_b"]).clamp(-10000., 10000.)
    rounded = z.to(torch.bfloat16).to(torch.float32)
    # Exact rounded forward value; only the unrounded branch carries gradient.
    return rounded.detach() + (z - z.detach()) if straight_through else rounded


def _forward(params, geometry, scaling, layer, features, start):
    tokens = len(features)
    l, c, d = (geometry[key] for key in ("layers", "channels", "head_dim"))
    identity = torch.zeros((tokens, l), dtype=torch.float32, device=features.device)
    identity[:, layer] = 1.
    x = torch.cat((features, identity), dim=1)
    x = F.gelu(x @ params["W0"] + params["b0"], approximate="tanh")
    x = F.gelu(x @ params["W1"] + params["b1"], approximate="tanh")
    output = x @ params["W2"] + params["b2"]
    key = (output[:, :c] * params["k_scale"][layer]).reshape(tokens, c // d, d)
    position = torch.arange(start, start + tokens, device=features.device, dtype=torch.float32)
    frequency = position[:, None] * params["rope_inv_freq"][None, :]
    angles = torch.cat((frequency, frequency), dim=1)
    cosine, sine = (torch.cos(angles) * scaling)[:, None, :], (torch.sin(angles) * scaling)[:, None, :]
    rotated = torch.cat((-key[..., d // 2:], key[..., :d // 2]), dim=-1)
    mean_key = (key * cosine + rotated * sine).reshape(tokens, c)
    mean_value = output[:, c:2 * c] * params["v_sd"][layer] + params["v_mu"][layer]
    logs = output[:, 2 * c:] + torch.log(torch.cat((params["k_scale"][layer], params["v_sd"][layer]))) - 1.
    return torch.cat((mean_key, mean_value), dim=1), logs.clamp(-30., 30.)


class Prepared:
    def __init__(self, model, device="cpu"):
        _schema(model)
        model._frozen()
        self.mode, self.metadata = model.mode, producer._json(model.metadata)
        self.device = torch.empty(0, device=device).device
        self.params = {name: torch.tensor(a.copy(), device=self.device) for name, a in model.params.items()}
        self.model_id, self.serialized_bytes = model.model_id, model.state_bytes
        self.execution_contract = _runtime(self.device)
        self.state_id = hashlib.sha256(_canonical(dict(model_id=self.model_id, runtime=self.execution_contract))).hexdigest()
        self.parameter_count = model.parameter_count

    @property
    def physical_nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.params.values())

    @property
    def state_bytes(self):
        return max(self.serialized_bytes, self.physical_nbytes)

    def _check(self, layer, start, tokens):
        if _canonical(_runtime(self.device)) != _canonical(self.execution_contract):
            raise ValueError("side probability predictor execution contract changed")
        layer, start, tokens = (_integer(layer, "layer", 0), _integer(start, "start", 0), _integer(tokens, "tokens"))
        if layer >= self.metadata["geometry"]["layers"]:
            raise ValueError("side probability layer outside model")
        return layer, start, tokens

    @torch.inference_mode()
    def encode_features(self, record):
        target = joint(record)
        layer, _, tokens = self._check(record.get("layer"), record.get("start"), len(target))
        geometry = self.metadata["geometry"]
        if target.shape[1] != 2 * geometry["channels"] or record.get("tokens") != tokens:
            raise ValueError("side probability record target geometry mismatch")
        rank = geometry["rank"]
        if self.mode == "none":
            return np.empty((tokens, 0), dtype=np.uint16)
        source = _source(record, self.mode, inverse=self.params["rope_inv_freq"].cpu().numpy(),
            rope_scaling=self.metadata["config"]["rope_scaling"], head_dim=geometry["head_dim"])
        if source.shape != (tokens, self.params["encoder_W"].shape[0]):
            raise ValueError("side probability encoder source geometry mismatch")
        inputs = torch.tensor(_floats(source), device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            z = _encode(self.params, inputs, layer, straight_through=False)
        if not bool(torch.isfinite(z).all()):
            raise ValueError("nonfinite side probability encoded feature")
        return z.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy().reshape(tokens, rank)

    @torch.inference_mode()
    def predict(self, layer, feature_words, start=0):
        features = np.asarray(feature_words)
        if features.dtype != np.uint16 or features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("side probability features must be uint16[T,rank]")
        layer, start, tokens = self._check(layer, start, len(features))
        geometry = self.metadata["geometry"]
        if features.shape != (tokens, geometry["rank"]):
            raise ValueError("side probability feature rank mismatch")
        if geometry["rank"]:
            x = torch.tensor(_floats(features), device=self.device)
        else:
            x = torch.empty((tokens, 0), device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            mean, logs = _forward(self.params, geometry, self.metadata["config"]["rope_scaling"], layer, x, start)
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(logs).all()):
            raise ValueError("nonfinite side probability prediction")
        return mean.cpu().numpy().copy(), logs.cpu().numpy().copy()


def _normalizers(records, mode, channels, head_dim, inverse, rope_scaling):
    k = np.concatenate([_floats(r["k"]) for r in records])
    v = np.concatenate([_floats(r["v"]) for r in records])
    key = k.reshape(len(k), channels // head_dim, head_dim).astype(np.float64)
    half = head_dim // 2
    rms = np.sqrt(((key[..., :half] ** 2 + key[..., half:] ** 2) * .5).mean(axis=0))
    result = dict(k_scale=np.concatenate((rms, rms), axis=-1).reshape(channels).astype(np.float32) + np.float32(1e-6),
                  v_mu=v.mean(axis=0, dtype=np.float64).astype(np.float32),
                  v_sd=v.astype(np.float64).std(axis=0).astype(np.float32) + np.float32(1e-6))
    if mode != "none":
        source = np.concatenate([_floats(_source(r, mode, inverse=inverse,
            rope_scaling=rope_scaling, head_dim=head_dim)) for r in records])
        result.update(source_mu=source.mean(axis=0, dtype=np.float64).astype(np.float32),
                      source_sd=source.astype(np.float64).std(axis=0).astype(np.float32) + np.float32(1e-6))
    return result


def fit(training_record_sets, output_dir, *, training_metadata, model_identity, inverse,
        rope_scaling, head_dim, rank=8, width=128, steps=200, batch_tokens=256,
        device="cpu", seed=71, modes=MODES):
    """Fit only explicitly labelled TRAIN records; no DEV/TEST selection occurs."""
    rank, width, steps, batch_tokens = (_integer(v, k) for v, k in
        ((rank, "rank"), (width, "width"), (steps, "steps"), (batch_tokens, "batch_tokens")))
    head_dim, seed = _integer(head_dim, "head_dim"), _integer(seed, "seed", 0)
    if head_dim % 2:
        raise ValueError("side probability requires even split-half head_dim")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("invalid side probability modes")
    metadata, identity = producer._json(training_metadata), producer._json(model_identity)
    if (not isinstance(training_record_sets, (list, tuple)) or not training_record_sets
            or not isinstance(metadata, list) or len(metadata) != len(training_record_sets)
            or any(not isinstance(row, dict) or row.get("split") != "train" for row in metadata)
            or not isinstance(identity, (str, dict)) or not identity):
        raise ValueError("side probability fitting requires explicit TRAIN sets and source identity")
    before = [producer._source_hash(rows) for rows in training_record_sets]
    provenance = producer._source_hash(dict(metadata=metadata, identity=identity))
    all_records, grouped, channels, hidden_size = [], {}, None, None
    layer_sets = []
    for records in training_record_sets:
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("side probability TRAIN record set must be nonempty")
        seen, intervals = set(), {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("side probability TRAIN record must be a dictionary")
            layer = _integer(record.get("layer"), "layer", 0)
            start = _integer(record.get("start"), "start", 0)
            tokens = _integer(record.get("tokens"), "tokens")
            target, hidden = joint(record), _words(record.get("hidden"), "hidden")
            _floats(target), _floats(hidden)
            if channels is None:
                channels, hidden_size = target.shape[1] // 2, hidden.shape[1]
            if (record.get("head_dim") != head_dim or target.shape != (tokens, 2 * channels)
                    or hidden.shape != (tokens, hidden_size)):
                raise ValueError("side probability TRAIN record geometry mismatch")
            if any(start < end and begin < start + tokens for begin, end in intervals.get(layer, [])):
                raise ValueError("side probability TRAIN records overlap within a capture")
            intervals.setdefault(layer, []).append((start, start + tokens))
            grouped.setdefault(layer, []).append(record)
            all_records.append(record)
            seen.add(layer)
        layer_sets.append(seen)
    layers = len(grouped)
    if set(grouped) != set(range(layers)) or any(seen != set(range(layers)) for seen in layer_sets):
        raise ValueError("side probability every TRAIN capture must cover every contiguous layer")
    inverse = np.asarray(inverse)
    if (channels % head_dim or inverse.dtype != np.float32 or inverse.shape != (head_dim // 2,)
            or not np.isfinite(inverse).all() or np.any(inverse <= 0)
            or not isinstance(rope_scaling, (int, float)) or not np.isfinite(rope_scaling) or rope_scaling <= 0):
        raise ValueError("invalid side probability fixed RoPE geometry")
    architecture = identity.get("config", identity) if isinstance(identity, dict) else {}
    for name, actual in (("num_hidden_layers", layers), ("head_dim", head_dim), ("hidden_size", hidden_size)):
        if isinstance(architecture, dict) and name in architecture and architecture[name] != actual:
            raise ValueError("side probability source architecture mismatch: " + name)
    if isinstance(architecture, dict) and "num_key_value_heads" in architecture and architecture["num_key_value_heads"] * head_dim != channels:
        raise ValueError("side probability source KV head geometry mismatch")
    output = Path(output_dir).resolve()
    if any((output / f"side_{mode}.json").exists() or (output / f"side_{mode}.npkv").exists() for mode in modes):
        raise FileExistsError("side probability artifacts exist; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    target_device = torch.empty(0, device=device).device
    results = {}
    for mode in modes:
        started = time.perf_counter()
        actual_rank = 0 if mode == "none" else rank
        geometry = dict(layers=layers, channels=channels, hidden_size=hidden_size,
                        head_dim=head_dim, rank=actual_rank, width=width)
        norm_rows = [_normalizers(grouped[layer], mode, channels, head_dim, inverse, rope_scaling)
                     for layer in range(layers)]
        arrays = {name: np.stack([row[name] for row in norm_rows]) for name in norm_rows[0]}
        arrays["rope_inv_freq"] = inverse.copy()
        constants = {name: torch.tensor(a.copy(), device=target_device) for name, a in arrays.items()}
        network = neural._init(actual_rank + layers, width, 4 * channels, target_device, seed)
        if mode != "none":
            source_width = 2 * channels if mode in ("kv", "kv_canonical") else hidden_size
            generator = torch.Generator(device=target_device)
            generator.manual_seed(seed + 1)
            network["encoder_W"] = torch.nn.Parameter(torch.randn((source_width, rank), device=target_device,
                generator=generator) / np.sqrt(source_width))
            network["encoder_b"] = torch.nn.Parameter(torch.zeros(rank, device=target_device))
        params = dict(constants, **network)
        optimizer = torch.optim.Adam(list(network.values()), lr=1e-3)
        rng = np.random.default_rng(seed)
        checkpoints, recent, updates = [], [], {str(layer): 0 for layer in range(layers)}
        token_examples = {str(layer): 0 for layer in range(layers)}
        batch_histogram = {}
        initial_loss = None
        for update in range(1, steps + 1):
            record_index = int(rng.integers(len(all_records)))
            record = all_records[record_index]
            count = min(batch_tokens, record["tokens"])
            offset = int(rng.integers(record["tokens"] - count + 1))
            layer, start = int(record["layer"]), int(record["start"]) + offset
            target_words = joint(record)[offset:offset + count]
            target = torch.tensor(_floats(target_words), device=target_device)
            bits = torch.tensor(target_words.copy(), device=target_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=target_device.type, enabled=False):
                if mode == "none":
                    features = torch.empty((count, 0), device=target_device)
                else:
                    source = torch.tensor(_floats(_source(record, mode, inverse=inverse,
                        rope_scaling=rope_scaling, head_dim=head_dim)[offset:offset + count]), device=target_device)
                    features = _encode(params, source, layer, straight_through=True)
                mean, logs = _forward(params, geometry, float(rope_scaling), layer, features, start)
                loss = neural._loss(bits, target, mean, logs)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite side probability TRAIN rate proxy")
            value = float(loss.detach().cpu())
            if initial_loss is None:
                initial_loss = value
            recent.append(value)
            recent = recent[-100:]
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in network.values()):
                raise ValueError("nonfinite side probability TRAIN gradient")
            rate = float(1e-3 * .5 * (1 + np.cos(np.pi * update / steps)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in network.values()):
                raise ValueError("nonfinite side probability TRAIN parameter after update")
            updates[str(layer)] += 1
            token_examples[str(layer)] += count
            batch_histogram[str(count)] = batch_histogram.get(str(count), 0) + 1
            if update == 1 or update == steps or update % max(1, steps // 8) == 0:
                checkpoints.append(dict(update=update, sampled_layer=layer, record=record_index, offset=offset,
                    start=start, tokens=count, density_proxy_bits_per_value=value,
                    side_bits_per_value=16 * actual_rank / (2 * channels), learning_rate=rate))
        arrays.update({name: p.detach().cpu().numpy().astype(np.float32, copy=True) for name, p in network.items()})
        metadata_out = dict(format=FORMAT, mode=mode, training_only=True, model_identity=identity, geometry=geometry,
            training_hashes=before, training_captures=[dict(split="train", sha256=digest, metadata=row)
                for digest, row in zip(before, metadata)],
            config=dict(source={"none": "no side information; target layer identity only",
                "kv": "same-page produced exact final K and V",
                "kv_canonical": "same-page produced exact final K inverse-rotated in CPU FP32 using fixed RoPE and BF16-rounded, plus final V; no pre-RoPE hook",
                "hidden": "actual producer input_layernorm normalized hidden"}[mode],
                activation="GELU approximate=tanh", latent_dtype="BF16", feature_clamp=[-10000., 10000.],
                latent_training="BF16 rounding with identity straight-through gradient",
                probability_target="exact final BF16 KV word intervals", rope_layout="split-half",
                rope_scaling=float(rope_scaling), log_scale_clamp=[-30., 30.],
                dependency="own packet feature words + shared immutable probability state; no previous-layer KV",
                objective="mean max(0,-log2(logistic density * BF16 ULP)); approximate density/bin-width rate, not exact interval likelihood or transmitted bytes",
                side_rate="fixed BF16 rank: 16*rank/(2*channels) bits per KV value; constant in optimizer, fully charged in packets",
                normalizers="per-layer TRAIN mean/std with float64 accumulation then float32; K split-half pair-RMS; +1e-6 scales"),
            training=dict(seed=seed, total_optimizer_updates=steps, batch_tokens=batch_tokens, updates_by_layer=updates,
                total_token_examples=sum(token_examples.values()), token_examples_by_layer=token_examples,
                effective_batch_tokens_min=min(map(int, batch_histogram)), effective_batch_tokens_max=max(map(int, batch_histogram)),
                batch_tokens_histogram=batch_histogram,
                initial_batch_loss=initial_loss, final_batch_loss=recent[-1], final_mean_last_100_batch_losses=float(np.mean(recent)),
                checkpoints=checkpoints, sampling="identical seeded uniform record/window schedule across modes; every layer included in TRAIN records",
                optimizer="Adam defaults; cosine learning rate 1e-3*0.5*(1+cos(pi*update/steps))",
                runtime=_runtime(target_device), fit_seconds=time.perf_counter() - started),
            scope="Page-local learned entropy model. Encoder, decoder, all normalizers and RoPE frequencies conservatively charged once per model. CPU entropy reference and same-runtime predictor only; no GPU throughput/serving claim.")
        model = Model(mode, arrays, metadata_out)
        metadata_out["compute"] = dict(trainable_parameter_count=model.parameter_count,
            complete_float32_state_values=model.total_float32_values,
            encoder_only_float32_state_values=sum(array.size for name, array in arrays.items()
                if name in ("encoder_W", "encoder_b", "source_mu", "source_sd")),
            decoder_float32_state_values=sum(array.size for name, array in arrays.items()
                if name not in ("encoder_W", "encoder_b", "source_mu", "source_sd")),
            encoder_MACs_per_token=(2 * channels if mode in ("kv", "kv_canonical") else hidden_size) * actual_rank if mode != "none" else 0,
            decoder_MACs_per_token=(actual_rank + layers) * width + width * width + width * 4 * channels)
        save(model, output / f"side_{mode}.json")
        results[mode] = load(model.path, expected_model_identity=identity, expected_training_hashes=before,
            expected_model_id=model.model_id, expected_mode=mode)
        del optimizer, network, params, constants
    if before != [producer._source_hash(rows) for rows in training_record_sets] or provenance != producer._source_hash(dict(metadata=producer._json(training_metadata), identity=producer._json(model_identity))):
        raise AssertionError("side probability TRAIN records/provenance changed during fitting")
    return results

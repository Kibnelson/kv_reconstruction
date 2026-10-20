"""TRAIN-only scalar prefill hints and small page-local joint-bit predictors.

Hints are fixed-rate discrete IDs, not BF16 values.  A shared scalar projection
uses per-layer *post-projection* calibration and subtracts a learned prediction
from decoder-known coarse KV statistics.  Exact KV reconstruction belongs to
compact_codec; CE and hint entropy are diagnostics, not measured packet sizes.
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

FORMAT = "compact_prefill_joint_models_v1"
TOTAL = 65536
_INPUTS = 33
_CHUNK_WORDS = 16384


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _positive(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("invalid compact model " + name)
    return int(value)


def default_specs():
    def spec(name, source="none", side_bits=0, *, pooled=True, page_shared=False,
             key_first=False, layout="lsb"):
        return dict(name=name, source=source, side_bits=side_bits, pooled=pooled,
                    page_shared=page_shared, key_first=key_first, layout=layout, base_bits=2)
    return [spec("lsb_local_none", pooled=False), spec("lsb_pool_none"),
        spec("lsb_pool_kv2", "kv", 2),
        *[spec("lsb_pool_hidden" + str(b), "hidden", b) for b in (1, 2, 4)],
        spec("lsb_page_hidden2", "hidden", 2, page_shared=True),
        spec("lsb_page_kv2", "kv", 2, page_shared=True),
        spec("lsb_kfirst_none", key_first=True),
        spec("lsb_kfirst_kv2", "kv", 2, key_first=True),
        spec("lsb_kfirst_hidden2", "hidden", 2, key_first=True),
        spec("msb_pool_none", layout="msb"),
        spec("msb_pool_kv2", "kv", 2, layout="msb"),
        spec("msb_pool_hidden2", "hidden", 2, layout="msb")]


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"name", "source", "side_bits", "pooled",
            "page_shared", "key_first", "layout", "base_bits"}:
        raise ValueError("invalid compact specification fields")
    name = spec["name"]
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
        raise ValueError("invalid compact specification name")
    if (spec["source"] not in ("none", "kv", "hidden")
            or type(spec["side_bits"]) is not int or spec["side_bits"] not in (0, 1, 2, 4)
            or (spec["source"] == "none") != (spec["side_bits"] == 0)
            or any(type(spec[k]) is not bool for k in ("pooled", "page_shared", "key_first"))
            or spec["layout"] not in ("lsb", "msb") or spec["base_bits"] != 2):
        raise ValueError("inconsistent compact specification")
    if spec["source"] == "none" and spec["page_shared"]:
        raise ValueError("no-code compact specification cannot emit a page code")
    return dict(spec)


def stage_plan(spec):
    return [(2, 2), (4, 3)] if spec["layout"] == "lsb" else [(3, 2), (0, 3)]


def initial_mask(spec):
    return 0xff80 | (3 if spec["layout"] == "lsb" else 0x60)


def stage_mask(spec, stage):
    if type(stage) is not int or not 0 <= stage < 2:
        raise ValueError("compact stage must be 0 or 1")
    mask = initial_mask(spec)
    for shift, width in stage_plan(spec)[:stage]:
        mask |= ((1 << width) - 1) << shift
    return mask


def _words(value, name):
    value = np.asarray(value)
    if value.dtype != np.uint16 or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(name + " must be nonempty uint16[T,C]")
    return value


def _known(words, spec, stage, channels):
    """Re-mask untrusted callers, allowing exact K only in K-first mode."""
    result = words.to(torch.int64) & stage_mask(spec, stage)
    if spec["key_first"]:
        result = torch.cat((words[:, :channels].to(torch.int64), result[:, channels:]), dim=1)
    return result


def _summary(known, head_dim):
    """Head and global summaries made solely from transmitted bit fields.

    No float reinterpretation is used, so NaN/subnormal/signed-zero BF16 fields
    are safe.  Six global features describe mean exponent/sign/mantissa for K
    and V. Eight per-channel features combine own-head means and K-head means.
    """
    tokens, columns = known.shape
    channels = columns // 2
    exponent = (((known >> 7) & 255).float() - 127.) / 16.
    sign = ((known >> 15) & 1).float() * 2. - 1.
    mantissa = (known & 127).float() / 127.
    fields = torch.stack((exponent, sign, mantissa, exponent.square().clamp_max(64.) / 8.), dim=-1)
    heads = fields.reshape(tokens, columns // head_dim, head_dim, 4).mean(dim=2)
    own = heads.repeat_interleave(head_dim, dim=1)
    kh = heads[:, :channels // head_dim].repeat_interleave(head_dim, dim=1)
    k_for_all = torch.cat((kh, kh), dim=1)
    pooled = torch.cat((own, k_for_all), dim=-1)
    global_fields = fields[..., :3].reshape(tokens, 2, channels, 3).mean(dim=2).reshape(tokens, 6)
    return pooled, global_fields


def _hint(params, spec, source, layer, coarse, *, straight_through):
    if spec["source"] == "none":
        return torch.zeros((len(coarse), 1), dtype=torch.float32, device=coarse.device), None
    projected = (source @ params["encoder_W"] + params["encoder_b"])
    z = (projected - params["projection_mu"][layer]) / params["projection_sd"][layer]
    _, summary = _summary(coarse, params["_head_dim"])
    z = z - summary @ params["innovation_W"]
    if spec["page_shared"]:
        z = z.mean(dim=0, keepdim=True)
    levels = 1 << spec["side_bits"]
    continuous = ((z.clamp(-3., 3.) + 3.) * ((levels - 1) / 6.))
    rounded = continuous.round()
    quantized = continuous + (rounded - continuous).detach() if straight_through else rounded
    value = quantized * (2. / (levels - 1)) - 1.
    if spec["page_shared"]:
        value = value.expand(len(coarse), 1)
    return value, rounded


def _decode_hint(ids, spec, tokens, device):
    if spec["source"] == "none":
        return torch.zeros((tokens, 1), device=device)
    z = torch.tensor(ids.copy(), dtype=torch.float32, device=device).reshape(-1, 1)
    z = z * (2. / ((1 << spec["side_bits"]) - 1)) - 1.
    return z.expand(tokens, 1) if spec["page_shared"] else z


def _forward(params, g, spec, layer, start, known, hint, stage, channel_indices=None):
    integer = _known(known, spec, stage, g["channels"])
    pooled, _ = _summary(integer, g["head_dim"])
    if channel_indices is None:
        channel_indices = torch.arange(2 * g["channels"], device=known.device)
    integer = integer[:, channel_indices]
    tokens, selected = integer.shape
    local = torch.cat(((((integer >> 7) & 255).float() - 127.)[..., None] / 16.,
        (((integer >> 15) & 1).float() * 2. - 1.)[..., None],
        torch.stack([((integer >> j) & 1).float() for j in range(7)], dim=-1)), dim=-1)
    p = pooled[:, channel_indices] if spec["pooled"] else torch.zeros((tokens, selected, 8), device=known.device)
    positions = (torch.arange(tokens, device=known.device, dtype=torch.float32) + start) / 1024.
    pos = torch.stack((positions.sin(), positions.cos()), dim=-1)
    x = torch.cat((local, p, hint[:, None, :].expand(tokens, selected, -1),
        params["layer_embedding"][layer][None, None, :].expand(tokens, selected, -1),
        params["channel_embedding"][channel_indices][None, :, :].expand(tokens, selected, -1),
        pos[:, None, :].expand(tokens, selected, -1),
        torch.full((tokens, selected, 1), float(stage), device=known.device)), dim=-1)
    x = F.gelu(x @ params["W0"] + params["b0"], approximate="tanh")
    x = F.gelu(x @ params["W1"] + params["b1"], approximate="tanh")
    return (x @ params["W2"][:, stage, :] + params["b2"][stage])[..., :1 << stage_plan(spec)[stage][1]]


def _shapes(g, spec):
    c, l, w = g["channels"], g["layers"], g["width"]
    result = dict(layer_embedding=(l, 4), channel_embedding=(2*c, 8),
        W0=(_INPUTS, w), b0=(w,), W1=(w, w), b1=(w,), W2=(w, 2, 8), b2=(2, 8))
    if spec["source"] != "none":
        result.update(encoder_W=(g["source_width"], 1), encoder_b=(1,),
            projection_mu=(l, 1), projection_sd=(l, 1), innovation_W=(6, 1))
    return result


def _schema(model):
    spec = validate_spec(model.spec)
    meta, g = model.metadata, model.metadata.get("geometry", {})
    if (meta.get("format") != FORMAT or meta.get("spec") != spec or meta.get("training_only") is not True
            or not meta.get("model_identity")):
        raise ValueError("invalid compact model identity")
    for key in ("layers", "channels", "head_dim", "hidden_size", "width"):
        _positive(g.get(key), key)
    source = 0 if spec["source"] == "none" else (2*g["channels"] if spec["source"] == "kv" else g["hidden_size"])
    if g.get("source_width") != source or g["channels"] % g["head_dim"]:
        raise ValueError("invalid compact source geometry")
    shapes = _shapes(g, spec)
    if set(model.params) != set(shapes):
        raise ValueError("compact model array fields differ")
    for name, shape in shapes.items():
        a = model.params[name]
        if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
            raise ValueError("invalid compact parameter " + name)
    if source and np.any(model.params["projection_sd"] <= 0):
        raise ValueError("compact projection calibration needs positive scales")
    rows = meta.get("training_captures")
    if (not isinstance(rows, list) or not rows or any(r.get("split") != "train" for r in rows)
            or len(meta.get("training_hashes", [])) != len(rows)):
        raise ValueError("invalid compact TRAIN provenance")


def _descriptor(model):
    _schema(model)
    wire = neural._shard_bytes("compact_parameters", model.params)
    if len(wire) >= 64 * 1024**2:
        raise ValueError("compact parameter shard exceeds safety bound")
    desc = dict(format="compact_prefill_artifact_v1", metadata=model.metadata,
        shard=dict(file="compact_" + model.spec["name"] + ".bin", bytes=len(wire),
                   sha256=hashlib.sha256(wire).hexdigest()), manifest_bytes=0, total_serialized_bytes=0)
    for _ in range(8):
        data = _canonical(desc)
        if desc["manifest_bytes"] == len(data) and desc["total_serialized_bytes"] == len(data) + len(wire):
            return desc, data, wire
        desc["manifest_bytes"], desc["total_serialized_bytes"] = len(data), len(data) + len(wire)
    raise AssertionError("compact state accounting failed to converge")


@dataclass
class Model:
    spec: dict
    params: dict
    metadata: dict
    path: str = ""

    @property
    def model_id(self):
        return hashlib.sha256(_descriptor(self)[1]).hexdigest()

    @property
    def state_bytes(self):
        return _descriptor(self)[0]["total_serialized_bytes"]

    @property
    def parameter_count(self):
        return sum(a.size for a in self.params.values())

    @property
    def artifact_paths(self):
        return [self.path, str(Path(self.path).parent / _descriptor(self)[0]["shard"]["file"])] if self.path else []

    def prepare(self, device="cpu"):
        return Prepared(self, device)


def save(model, path):
    path = Path(path); desc, data, wire = _descriptor(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / desc["shard"]["file"]).write_bytes(wire)
    path.write_bytes(data); model.path = str(path.resolve())
    for a in model.params.values():
        a.flags.writeable = False
    return path


def load(path, *, expected_model_id=None, expected_model_identity=None, expected_training_hashes=None):
    path = Path(path); data = path.read_bytes()
    try:
        desc = json.loads(data); meta, shard = desc["metadata"], desc["shard"]
        filename = shard["file"]
        if (desc["format"] != "compact_prefill_artifact_v1" or _canonical(desc) != data
                or not isinstance(filename, str) or Path(filename).name != filename or filename in (".", "..")):
            raise ValueError("invalid compact canonical manifest")
        if expected_model_id is not None and hashlib.sha256(data).hexdigest() != expected_model_id:
            raise ValueError("compact model SHA mismatch")
        if expected_model_identity is not None and producer._json(expected_model_identity) != meta["model_identity"]:
            raise ValueError("compact source identity mismatch")
        if expected_training_hashes is not None and list(expected_training_hashes) != meta["training_hashes"]:
            raise ValueError("compact TRAIN hash mismatch")
        wire = (path.parent / filename).read_bytes()
        if len(wire) != shard["bytes"] or hashlib.sha256(wire).hexdigest() != shard["sha256"]:
            raise ValueError("compact shard bytes/SHA mismatch")
        model = Model(meta["spec"], neural._load_shard(wire, "compact_parameters"), meta, str(path.resolve()))
        if _descriptor(model)[1] != data:
            raise ValueError("noncanonical complete compact artifact")
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid compact model artifact") from error
    for a in model.params.values():
        a.flags.writeable = False
    return model


def hint_diagnostics(ids, spec):
    ids = np.asarray(ids)
    alphabet = 1 << spec["side_bits"] if spec["side_bits"] else 0
    if ids.dtype != np.uint8 or ids.ndim != 1 or (ids.size and (not alphabet or int(ids.max()) >= alphabet)):
        raise ValueError("invalid compact hint IDs")
    hist = np.bincount(ids, minlength=alphabet).tolist() if alphabet else []
    p = np.asarray(hist, dtype=np.float64) / len(ids) if len(ids) else np.array([])
    p = p[p > 0]
    return dict(ids=int(len(ids)), histogram=hist, entropy_bits_per_id=float(-np.sum(p*np.log2(p))),
        fixed_bits_per_id=spec["side_bits"], page_shared=spec["page_shared"],
        transitions=int(np.count_nonzero(ids[1:] != ids[:-1])),
        distinct_ids=int(np.count_nonzero(hist)), sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        qualification="Observed histogram entropy is diagnostic, not the fixed packed wire length or a held-out entropy-code gain")


class Prepared:
    def __init__(self, model, device="cpu"):
        _schema(model)
        self.spec = dict(model.spec)
        self.metadata = producer._json(model.metadata)
        self.device = torch.device(device)
        self.params = {n: torch.tensor(a.copy(), device=self.device) for n, a in model.params.items()}
        self.params["_head_dim"] = self.metadata["geometry"]["head_dim"]
        self.model_id, self.serialized_bytes = model.model_id, model.state_bytes
        self.execution_contract = neural._runtime(self.device)
        self.execution_contract["rule"] = "compact-joint-symbol-FP32-CPU-positive-counts-v1"
        self.state_id = hashlib.sha256(_canonical(dict(model_id=self.model_id, runtime=self.execution_contract))).hexdigest()
        self.last_hint_diagnostics = {}

    @property
    def physical_nbytes(self):
        return sum(t.numel()*t.element_size() for t in self.params.values() if isinstance(t, torch.Tensor))

    @property
    def state_bytes(self):
        return max(self.serialized_bytes, self.physical_nbytes)

    def _check(self, layer, start, target):
        execution = neural._runtime(self.device)
        execution["rule"] = "compact-joint-symbol-FP32-CPU-positive-counts-v1"
        if _canonical(execution) != _canonical(self.execution_contract):
            raise ValueError("compact predictor execution contract changed")
        layer, start = _positive(layer, "layer", 0), _positive(start, "start", 0)
        g = self.metadata["geometry"]
        if layer >= g["layers"] or target.shape[1] != 2*g["channels"]:
            raise ValueError("compact layer/channel geometry mismatch")
        return layer, start, g

    @torch.inference_mode()
    def encode_hints(self, record):
        target = np.concatenate((_words(record["k"], "k"), _words(record["v"], "v")), axis=1)
        layer, start, g = self._check(record["layer"], record["start"], target)
        if record.get("tokens") != len(target) or record["k"].shape != record["v"].shape:
            raise ValueError("compact record geometry differs")
        if self.spec["source"] == "none":
            result = np.empty(0, np.uint8)
        else:
            source = target if self.spec["source"] == "kv" else _words(record["hidden"], "hidden")
            value = neural._floats(source)
            if value.shape != (len(target), g["source_width"]):
                raise ValueError("compact source shape differs")
            with torch.autocast(device_type=self.device.type, enabled=False):
                coarse = _known(torch.tensor(target.copy(), device=self.device), self.spec, 0, g["channels"])
                _, ids = _hint(self.params, self.spec, torch.tensor(value.copy(), device=self.device), layer, coarse, straight_through=False)
            result = ids.to(torch.uint8).flatten().cpu().numpy().copy()
        self.last_hint_diagnostics = hint_diagnostics(result, self.spec)
        return result

    @torch.inference_mode()
    def predict_counts(self, layer, start, known_words, hint_ids, stage_index):
        known = _words(known_words, "known_words")
        layer, start, g = self._check(layer, start, known)
        stage_mask(self.spec, stage_index)
        ids = np.asarray(hint_ids)
        expected = 0 if self.spec["source"] == "none" else (1 if self.spec["page_shared"] else len(known))
        if ids.dtype != np.uint8 or ids.shape != (expected,) or (ids.size and ids.max() >= 1 << self.spec["side_bits"]):
            raise ValueError("compact retained hint geometry/range mismatch")
        hint = _decode_hint(ids, self.spec, len(known), self.device)
        chunks, per = [], max(1, _CHUNK_WORDS // known.shape[1])
        with torch.autocast(device_type=self.device.type, enabled=False):
            for offset in range(0, len(known), per):
                words = torch.tensor(known[offset:offset+per].copy(), device=self.device)
                logits = _forward(self.params, g, self.spec, layer, start+offset, words, hint[offset:offset+per], stage_index)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite compact probability logits")
                chunks.append(logits.cpu().numpy().astype(np.float64))
        logits = np.concatenate(chunks)
        logits -= logits.max(axis=-1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=-1, keepdims=True)
        size = p.shape[-1]
        counts = np.floor(p*(TOTAL-size)).astype(np.int32)+1
        # Fixed last-symbol remainder is simple and fully specified; it is
        # not largest-remainder optimization and its small rate cost counts.
        counts[..., -1] += TOTAL - counts.sum(axis=-1)
        return counts


def fit(training_records, output, *, training_metadata, model_identity, head_dim, steps=200,
        batch_tokens=32, channels_per_step=64, width=32, device="cpu", seed=71, specs=None):
    """Fit only TRAIN captures. Every specification receives the same schedule."""
    specs = default_specs() if specs is None else [validate_spec(s) for s in specs]
    if not specs or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("compact specs must be nonempty and uniquely named")
    for n, v in (("steps", steps), ("batch_tokens", batch_tokens), ("channels_per_step", channels_per_step),
                 ("width", width), ("head_dim", head_dim)):
        _positive(v, n)
    _positive(seed, "seed", 0)
    metadata = producer._json(training_metadata); identity = producer._json(model_identity)
    if (not isinstance(training_records, (list, tuple)) or not training_records or
            not isinstance(metadata, list) or len(metadata) != len(training_records)
            or any(r.get("split") != "train" for r in metadata) or not identity):
        raise ValueError("compact fitting requires explicit TRAIN captures and identity")
    capture_fields = {"split", "name", "tokens", "sha256", "producer_records_sha256",
        "producer_inputs_sha256", "input_ids_sha256", "kv_words_sha256", "geometry"}
    minimal_metadata = [{k:v for k,v in r.items() if k in capture_fields} for r in metadata]
    hashes = [producer._source_hash(r) for r in training_records]
    flat, grouped, addresses = [], {}, []
    channels = hidden_size = None
    for capture, rows in enumerate(training_records):
        seen = set()
        for r in rows:
            if r.get("split", "train") != "train":
                raise ValueError("compact fitting received non-TRAIN record")
            k, v, h = (_words(r[n], n) for n in ("k", "v", "hidden"))
            if k.shape != v.shape or len(h) != len(k) or r.get("tokens") != len(k) or r.get("head_dim") != head_dim:
                raise ValueError("compact TRAIN record geometry mismatch")
            channels = k.shape[1] if channels is None else channels
            hidden_size = h.shape[1] if hidden_size is None else hidden_size
            if k.shape[1] != channels or h.shape[1] != hidden_size or channels % head_dim:
                raise ValueError("compact TRAIN geometry differs")
            layer, start = _positive(r["layer"], "layer", 0), _positive(r["start"], "start", 0)
            if any(l == layer and max(s, start) < min(e, start+len(k)) for l, s, e in seen):
                raise ValueError("compact TRAIN records overlap")
            seen.add((layer, start, start+len(k)))
            neural._floats(k); neural._floats(v); neural._floats(h)
            row = dict(record=r, target=np.concatenate((k, v), axis=1), hidden=neural._floats(h))
            flat.append(row); grouped.setdefault(layer, []).append(row); addresses.append((capture, layer, start))
        if not rows:
            raise ValueError("empty compact TRAIN capture")
    layers = max(grouped)+1
    if set(grouped) != set(range(layers)) or any({r["layer"] for r in rows} != set(range(layers)) for rows in training_records):
        raise ValueError("compact TRAIN must cover every layer")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device); result = {}
    for spec in specs:
        g = dict(layers=layers, channels=channels, head_dim=head_dim, hidden_size=hidden_size,
            width=width, source_width=0 if spec["source"] == "none" else (2*channels if spec["source"] == "kv" else hidden_size))
        generator = torch.Generator(device=device); generator.manual_seed(seed)
        params = {}
        for name, shape in _shapes(g, spec).items():
            if name.endswith("_sd"):
                value = torch.ones(shape, device=device)
            elif name.endswith("_mu") or name.startswith("b") or name in ("encoder_b", "innovation_W"):
                value = torch.zeros(shape, device=device)
            else:
                value = torch.randn(shape, device=device, generator=generator) * (0.1 if "embedding" in name else np.sqrt(1./shape[0]))
            params[name] = torch.nn.Parameter(value, requires_grad=not name.startswith("projection_"))
        params["_head_dim"] = head_dim
        if spec["source"] != "none":
            with torch.no_grad():
                for layer, rows in grouped.items():
                    source = np.concatenate([neural._floats(r["target"]) if spec["source"] == "kv" else r["hidden"] for r in rows])
                    projection = torch.tensor(source.copy(), device=device) @ params["encoder_W"] + params["encoder_b"]
                    params["projection_mu"][layer].copy_(projection.mean(dim=0))
                    params["projection_sd"][layer].copy_(projection.std(dim=0, unbiased=False).clamp_min(1e-5))
        optimizer = torch.optim.Adam([p for p in params.values() if isinstance(p, torch.Tensor) and p.requires_grad], lr=0.003)
        rng = np.random.default_rng(seed)
        channel_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x434f4d50]))
        losses = []; exposure = np.zeros((layers, 2*channels, 2), np.int64)
        sampling, data_sampling = hashlib.sha256(), hashlib.sha256(); started = time.perf_counter()
        queues = {(l, st):np.empty(0, np.int64) for l in range(layers) for st in range(2)}
        for update in range(steps):
            layer = update % layers; stage = (update // layers) % 2
            rows = grouped[layer]; row_index = int(rng.integers(len(rows))); row = rows[row_index]
            t = len(row["target"]); count = min(batch_tokens, t)
            offset = int(rng.integers(t-count+1))
            begin = channels if spec["key_first"] else 0
            eligible = np.arange(begin, 2*channels)
            size = min(channels_per_step, len(eligible)); key = (layer, stage)
            if len(queues[key]) < size:
                # Finish a cycle before repeating previously unused channels.
                previous = queues[key]
                rest = channel_rng.permutation(eligible)
                rest = rest[~np.isin(rest, previous)]
                queues[key] = np.concatenate((previous, rest))
            columns = queues[key][:size]; queues[key] = queues[key][size:]
            sample = dict(update=update, layer=layer, stage=stage, record_index=row_index,
                          start=int(row["record"]["start"])+offset, tokens=count)
            data_sampling.update(_canonical(sample))
            sampling.update(_canonical(dict(sample, channels=columns.tolist())))
            target = torch.tensor(row["target"].copy(), device=device, dtype=torch.int64)
            coarse = _known(target, spec, 0, channels)
            source = None if spec["source"] == "none" else torch.tensor(
                (neural._floats(row["target"]) if spec["source"] == "kv" else row["hidden"]).copy(), device=device)
            with torch.autocast(device_type=device.type, enabled=False):
                hint, _ = _hint(params, spec, source, layer, coarse, straight_through=True)
                channel_indices = torch.tensor(columns.copy(), device=device)
                logits = _forward(params, g, spec, layer, int(row["record"]["start"])+offset,
                    target[offset:offset+count], hint[offset:offset+count], stage, channel_indices)
                shift, bits = stage_plan(spec)[stage]
                labels = ((target[offset:offset+count, channel_indices] >> shift) & ((1 << bits)-1)).long()
                loss = F.cross_entropy(logits.reshape(-1, 1 << bits), labels.reshape(-1)) / np.log(2.)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite compact TRAIN loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in params.values() if isinstance(p, torch.Tensor) and p.requires_grad], 5.)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
            exposure[layer, columns, stage] += count
        arrays = {n:p.detach().cpu().numpy().astype(np.float32, copy=True) for n,p in params.items() if isinstance(p, torch.Tensor)}
        exposure_rows = {str(l):{str(st):dict(eligible_channels=(channels if spec["key_first"] else 2*channels),
            channels_without_examples=int(np.sum(exposure[l, channels if spec["key_first"] else 0:,st]==0)),
            minimum_examples=int(np.min(exposure[l, channels if spec["key_first"] else 0:,st])),
            median_examples=float(np.median(exposure[l, channels if spec["key_first"] else 0:,st])),
            maximum_examples=int(np.max(exposure[l, channels if spec["key_first"] else 0:,st])),
            word_examples=int(exposure[l,:,st].sum())) for st in range(2)} for l in range(layers)}
        training_summary = dict(updates=steps, initial_loss_bits=losses[0], final_loss_bits=losses[-1],
            mean_last_20_loss_bits=float(np.mean(losses[-20:])), sampling_sha256=sampling.hexdigest(),
            data_sampling_sha256=data_sampling.hexdigest(), channels_per_update=channels_per_step,
            batch_tokens=batch_tokens, seed=seed, eligible_channel_range=[channels if spec["key_first"] else 0, 2*channels],
            minimum_updates_for_one_complete_channel_stage_cycle=layers*2*int(np.ceil(
                (channels if spec["key_first"] else 2*channels)/min(channels_per_step, channels if spec["key_first"] else 2*channels))),
            channel_stage_slots_without_examples=sum(r["channels_without_examples"] for rows in exposure_rows.values() for r in rows.values()))
        diagnostic = dict(format="compact_prefill_fit_diagnostics_v1", spec=spec,
            training_hashes=hashes, training=dict(training_summary, losses_bits=losses,
                exposure_by_layer_stage=exposure_rows, wall_seconds=time.perf_counter()-started),
            scope="Experiment diagnostics only; neither inference input nor runtime codec state")
        diagnostic_name = "compact_"+spec["name"]+"_training.json"
        diagnostic_wire = _canonical(diagnostic)
        (output/diagnostic_name).write_bytes(diagnostic_wire)
        training_summary["diagnostics"] = dict(file=diagnostic_name, bytes=len(diagnostic_wire),
            sha256=hashlib.sha256(diagnostic_wire).hexdigest(), required_for_decode=False)
        model = Model(spec, arrays, dict(format=FORMAT, spec=spec, geometry=g, training_only=True,
            model_identity=identity, training_captures=minimal_metadata, training_hashes=hashes,
            config=dict(activation="GELU approximate=tanh", hint="scalar postprojection TRAIN calibration, known-field innovation, uniform STE grid[-3,3]",
                objective="Joint-symbol cross entropy surrogate; fixed hint bits charged by packet codec, no variable-rate entropy bonus",
                projection_calibration="Frozen TRAIN-only scalar mean/std at initialization, not per-input-channel normalizers",
                counts="CPU float64 softmax; positive floor and fixed last-symbol remainder; total65536",
                stage_plan=[list(s) for s in stage_plan(spec)], known_context="Exact field statistics only, remasked at every call",
                runtime_scope="same selected-device FP32 predictor and CPU integerization; cross-device CDF portability not promised"),
            training=training_summary))
        save(model, output / ("compact_"+spec["name"]+".json")); result[spec["name"]] = model
    if hashes != [producer._source_hash(r) for r in training_records]:
        raise AssertionError("compact TRAIN input changed during fitting")
    return result

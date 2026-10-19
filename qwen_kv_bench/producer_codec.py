"""Exact KV pair recipes from actual forward intermediates.

Producer-only inputs can include normalized hidden states or pre-normalization
K projections.  Decoders see only packet fields and frozen, completely charged
state.  No original hidden tensor, original KV target, dense per-record predictor
or RoPE table is retained by a decoder.  This eager tensor implementation is a
functional experiment; its allocations and reconstructed nested headers are
part of measured restoration cost, and it is not a fused serving kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from . import exponent_recipe as exponent
from . import gpu_ops as g


def recipes():
    result = [dict(name="producer_kv_median", kind="kv_median"),
              dict(name="producer_kv_fixed127", kind="kv_fixed127"),
              dict(name="producer_frozen_exponent", kind="frozen"),
              dict(name="producer_hidden_exponent", kind="hidden"),
              dict(name="producer_hidden_abs_exponent", kind="hidden_abs")]
    for source in ("pre_rope", "kproj_norm_rope"):
        for correction in ("residual", "coset"):
            result.append(dict(name=f"producer_{source}_{correction}", kind=source,
                               correction=correction))
    for spec in result:
        spec["method"] = "producer_pair"
    return result


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()).encode()


def _array_artifact(metadata, arrays):
    descriptions, chunks, offset = {}, [], 0
    for name, array in sorted(arrays.items()):
        array = np.ascontiguousarray(array)
        blob = array.tobytes()
        descriptions[name] = dict(dtype=array.dtype.str, shape=list(array.shape),
                                  offset=offset, nbytes=len(blob))
        chunks.append(blob)
        offset += len(blob)
    header = _json(dict(metadata=metadata, arrays=descriptions))
    return b"PRODUCER_STATE_V1\0" + struct.pack("<Q", len(header)) + header + b"".join(chunks)


@dataclass
class ProducerState:
    spec: dict
    fields: dict
    metadata: dict
    wire_bytes: bytes
    state_id: str
    device: torch.device

    @property
    def physical_nbytes(self):
        return g._storage_bytes(self.fields.values())

    @property
    def serialized_bytes(self):
        return len(self.wire_bytes)

    @property
    def state_bytes(self):
        return max(self.serialized_bytes, self.physical_nbytes)

    @property
    def charged_bytes(self):
        return self.state_bytes

    def dump(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.wire_bytes)
        return path


def state_bytes(state):
    return state.wire_bytes


def state_info(state):
    return dict(state_id=state.state_id, serialized_bytes=state.serialized_bytes,
                physical_tensor_bytes=state.physical_nbytes, charged_bytes=state.state_bytes,
                accounting="maximum of full serialized state and distinct device tensor storage",
                decoder_state_metadata=state.metadata)


def make_state(spec, models, decoder_context, device="cpu"):
    """Prepare every decoder tensor before timed decode; charge complete state."""
    known = {item["name"]: item for item in recipes()}
    if spec.get("name") not in known or spec != known[spec["name"]]:
        raise ValueError("unknown or modified producer recipe")
    device = torch.device(device)
    arrays, metadata, wire = {}, {}, b""
    kind = spec["kind"]
    if kind in ("frozen", "hidden", "hidden_abs"):
        model = models.get(kind) if isinstance(models, dict) else next((m for m in models if m.kind == kind), None)
        if model is None or model.kind != kind:
            raise ValueError("frozen TRAIN producer model is required: " + kind)
        if (not model.params or model.config.get("feature_transform") !=
                {"frozen": "none", "hidden": "signed", "hidden_abs": "abs"}[kind]):
            raise ValueError("producer model feature transform differs from recipe")
        for layer, params in sorted(model.params.items()):
            base_k, base_v = np.asarray(params["base_K"]), np.asarray(params["base_V"])
            if base_k.dtype != np.uint8 or base_k.ndim != 1 or base_v.dtype != np.uint8 or base_v.shape != base_k.shape:
                raise ValueError("producer base arrays must be matching uint8 channels")
            if kind != "frozen":
                indices, scales = np.asarray(params["indices"]), np.asarray(params["scales"])
                if (indices.dtype != np.int32 or indices.ndim != 1 or not len(indices)
                        or np.any(indices < 0) or scales.dtype != np.float32 or scales.shape != indices.shape
                        or not np.isfinite(scales).all() or np.any(scales <= 0)
                        or any(np.asarray(params["coeff_" + kv]).dtype != np.int16 or
                               np.asarray(params["coeff_" + kv]).shape != (len(indices), len(base_k)) for kv in ("K", "V"))):
                    raise ValueError("invalid producer feature selector, scale or coefficient arrays")
            for name, array in sorted(params.items()):
                arrays[f"{layer}/{name}"] = np.ascontiguousarray(array)
        metadata = dict(kind=kind, model_id=model.model_id, config=model.config,
                        training_metadata=model.metadata)
        metadata["layer_geometry"] = {layer: dict(channels=len(params["base_K"]),
            feature_rank=len(params["indices"]) if kind != "frozen" else 0,
            maximum_hidden_index=int(np.max(params["indices"])) if kind != "frozen" else -1)
            for layer, params in model.params.items()}
        path = _get(model, "path")
        if path and Path(path).exists():
            wire = Path(path).read_bytes()
            if hashlib.sha256(wire).hexdigest() != model.model_id:
                raise ValueError("producer model file does not match frozen identity")
        else:
            # Actual Model artifacts use the public complete serializer.  Small
            # synthetic test models can supply the same numerical contract.
            if model.__class__.__module__.endswith("producer_models"):
                from .producer_models import model_bytes
                wire = model_bytes(model)
            else:
                wire = _array_artifact(metadata, arrays)
    elif kind in ("pre_rope", "kproj_norm_rope"):
        if not _get(decoder_context, "rope_stable", False):
            raise ValueError("replay requires frozen stable RoPE frequencies")
        inv = np.asarray(_get(decoder_context, "rope_inv_freq"), dtype=np.float32)
        scale = float(_get(decoder_context, "rope_attention_scaling", 1.0))
        if inv.ndim != 1 or not len(inv) or not np.isfinite(inv).all() or not np.isfinite(scale):
            raise ValueError("invalid frozen RoPE decoder parameters")
        arrays["rope_inv_freq"] = inv.copy()
        metadata = dict(kind=kind, rope_attention_scaling=scale, rope_stable=True,
                        positional_source="global integer position, float32 multiply, BF16 cos/sin")
        if kind == "kproj_norm_rope":
            weights = _get(decoder_context, "norm_weights", {})
            eps = _get(decoder_context, "norm_eps", {})
            metadata["norm_eps"] = {}
            for layer, value in sorted(weights.items(), key=lambda x: int(x[0])):
                array = np.asarray(value)
                if array.dtype != np.uint16 or array.ndim != 1 or len(array) != 2 * len(inv):
                    raise ValueError("RMSNorm weights must be BF16 words matching head dimension")
                epsilon = float(eps.get(layer, eps.get(str(layer), 0.0)))
                if not np.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("invalid RMSNorm epsilon")
                arrays[f"norm_weight/{int(layer)}"] = array.copy()
                metadata["norm_eps"][str(int(layer))] = epsilon
            if not weights:
                raise ValueError("K projection replay requires all frozen K RMSNorm weights")
        wire = _array_artifact(metadata, arrays)
    else:
        metadata = dict(kind=kind, external_state="none")
    fields = {name: torch.from_numpy(np.ascontiguousarray(array).copy()).to(device)
              for name, array in arrays.items()}
    # Resolve an unindexed 'cuda' request to the actual current CUDA device;
    # packet tensors report cuda:0, which does not compare equal to cuda.
    device = next(iter(fields.values())).device if fields else torch.empty(0, device=device).device
    identity_bytes = wire if wire else _json(spec)
    return ProducerState(dict(spec), fields, metadata, wire,
                         hashlib.sha256(identity_bytes).hexdigest(), device)


def _words(value, device):
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value).copy())
    if tensor.dtype != torch.uint16 or tensor.ndim != 2:
        raise ValueError("producer values must be uint16[T,C] BF16 words")
    return tensor.to(device=device).contiguous()


def prepare_record(record, device, spec=None):
    """Stage only fields required by the selected producer recipe."""
    needed = {"k", "v"}
    kind = spec["kind"] if spec else None
    if kind in ("hidden", "hidden_abs"):
        needed.add("hidden")
    elif kind == "pre_rope":
        needed.add("k_pre_rope")
    elif kind == "kproj_norm_rope":
        needed.add("k_proj")
    elif spec is None:
        needed.update(name for name in ("hidden", "k_pre_rope", "k_proj") if name in record)
    result = {key: record[key] for key in ("layer", "start", "tokens", "head_dim")}
    for key in needed:
        if key not in record:
            raise ValueError("missing actual producer intermediate: " + key)
        result[key] = _words(record[key], device)
    return result


def _embed(packet, fields, nested, label):
    nested[label] = packet.metadata
    fields.update({label + "__" + name: value for name, value in packet.fields.items()})


def _restore_nested(meta, fields, label, device):
    descriptor = dict(meta["nested"][label])
    components = {name: fields[label + "__" + name] for name in descriptor["field_bytes"]}
    # This real header reconstruction allocates and is included in decode time.
    return exponent.decode(g._finish(components, descriptor, device))


def _param(state, layer, name):
    key = f"layer_{layer:02d}/{name}"
    if key not in state.fields:
        raise ValueError("missing frozen producer model parameter: " + key)
    return state.fields[key]


def _features(hidden, state, layer):
    geometry = state.metadata["layer_geometry"].get(f"layer_{layer:02d}")
    if geometry is None or hidden.shape[1] <= geometry["maximum_hidden_index"]:
        raise ValueError("actual hidden state does not match frozen feature selector")
    indices = _param(state, layer, "indices").to(torch.int64)
    scales = _param(state, layer, "scales").to(torch.float32)
    values = hidden.view(torch.bfloat16).to(torch.float32)[:, indices]
    # Actual model prefill is finite.  Define a deterministic feature for unusual
    # synthetic nonfinite inputs without ever modifying the exact KV target.
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.round(values / scales[None, :]).clamp(-127, 127).to(torch.int8)


def _predict(state, layer, kind, tokens, features=None):
    base = _param(state, layer, "base_" + kind).to(torch.int64)
    prediction = base[None, :].expand(tokens, len(base))
    if features is not None:
        if state.metadata.get("config", {}).get("feature_transform", "signed") == "abs":
            features = features.to(torch.int64).abs()
        coefficient = _param(state, layer, "coeff_" + kind).to(torch.int64)
        if coefficient.shape != (features.shape[1], len(base)):
            raise ValueError("hidden feature coefficient geometry mismatch")
        delta = torch.zeros((tokens, len(base)), device=base.device, dtype=torch.int64)
        for i in range(features.shape[1]):
            delta = delta + features[:, i:i + 1].to(torch.int64) * coefficient[i:i + 1, :]
        prediction = prediction + torch.div(delta + 128, 256, rounding_mode="floor")
    return prediction.clamp(0, 255)


def _encode_predicted(words, prediction, fields, label, segment_size=256):
    integer = words.to(torch.int64)
    if prediction.shape != words.shape:
        raise ValueError("producer exponent prediction shape mismatch")
    z = g._zigzag(exponent._ordered(((integer >> 7) & 255) - prediction, "time"))
    second, rare = (z[z >= 3] - 3).clamp_max(7), z[z >= 10]
    fields[label + "__lower"] = (((integer >> 8) & 128) | (integer & 127)).to(torch.uint8).reshape(-1)
    for name, values, width in (("primary", z.clamp_max(3), 2), ("secondary", second, 3), ("rare", rare, 9)):
        fields[label + "__" + name], _ = g._pack_fixed(values, width)
    fields[label + "__secondary_offsets"] = g._wire(exponent._segment_offsets(z >= 3, segment_size))
    fields[label + "__rare_offsets"] = g._wire(exponent._segment_offsets(z >= 10, segment_size))
    return dict(shape=list(words.shape), secondary_count=second.numel(), rare_count=rare.numel(),
                segment_size=segment_size, layout="time", correction="signed_2_3_9_tiers")


def _decode_predicted(description, fields, label, prediction):
    shape = description["shape"]
    count = int(np.prod(shape, dtype=np.int64))
    primary = g._unpack_fixed(fields[label + "__primary"], count, 2)
    second = g._unpack_fixed(fields[label + "__secondary"], description["secondary_count"], 3)
    rare = g._unpack_fixed(fields[label + "__rare"], description["rare_count"], 9)
    mask = primary == 3
    positions = exponent._segment_ranks(mask, fields[label + "__secondary_offsets"].view(torch.uint32), description["segment_size"])
    selected = second[positions[mask]]
    rare_mask = selected == 7
    full_rare = torch.zeros(count, device=primary.device, dtype=torch.bool)
    full_rare[mask] = rare_mask
    positions = exponent._segment_ranks(full_rare, fields[label + "__rare_offsets"].view(torch.uint32), description["segment_size"])
    correction = selected + 3
    correction[rare_mask] = rare[positions[full_rare]]
    primary[mask] = correction
    exp = prediction + exponent._canonical(g._unzigzag(primary), shape, "time")
    lower = fields[label + "__lower"].to(torch.int64)
    return (((lower & 128) << 8) | (lower & 127) | (exp.reshape(-1) << 7)).to(torch.uint16).reshape(shape)


def _rope(source, state, start, head_dim, layer, normalize=False):
    tokens, channels = source.shape
    inv = state.fields["rope_inv_freq"]
    if head_dim < 2 or head_dim % 2 or channels % head_dim or len(inv) * 2 != head_dim:
        raise ValueError("RoPE replay geometry mismatch")
    value = source.view(torch.bfloat16).reshape(tokens, channels // head_dim, head_dim)
    if normalize:
        name = f"norm_weight/{layer}"
        if name not in state.fields:
            raise ValueError("missing frozen K RMSNorm weights")
        numeric = value.to(torch.float32)
        variance = (numeric * numeric).mean(dim=-1, keepdim=True)
        value = (numeric * torch.rsqrt(variance + state.metadata["norm_eps"][str(layer)])).to(torch.bfloat16)
        value = value * state.fields[name].view(torch.bfloat16)
    position = torch.arange(start, start + tokens, device=source.device, dtype=torch.float32)
    frequencies = position[:, None] * inv[None, :]
    angles = torch.cat((frequencies, frequencies), dim=-1)
    scale = state.metadata["rope_attention_scaling"]
    cosine = (torch.cos(angles) * scale).to(torch.bfloat16)[:, None, :]
    sine = (torch.sin(angles) * scale).to(torch.bfloat16)[:, None, :]
    half = torch.cat((-value[..., head_dim // 2:], value[..., :head_dim // 2]), dim=-1)
    return (value * cosine + half * sine).contiguous().view(torch.uint16).reshape(tokens, channels)


def _encode_full_correction(actual, prediction, fields, label, correction, group_size=64):
    actual, prediction = actual.to(torch.int64).reshape(-1), prediction.to(torch.int64).reshape(-1)
    residual = ((actual - prediction + 32768) & 65535) - 32768
    temporary, descriptions = {}, {}
    g._adaptive(residual, temporary, descriptions, "data", group_size, True, 16)
    widths = temporary["data_widths"]
    description = descriptions["data"]
    if correction == "coset":
        expanded = widths.to(torch.int64).repeat_interleave(group_size)[:actual.numel()]
        temporary["data"], nbits = g._pack_variable(actual & ((1 << expanded) - 1), widths, group_size, 16)
        if nbits != description["nbits"]:
            raise AssertionError("matched coset and signed residual widths diverged")
    fields[label + "__correction"] = temporary["data"]
    fields[label + "__correction_widths"], _ = g._pack_fixed(widths, 5)
    return dict(count=actual.numel(), group_size=group_size, max_width=16,
                nbits=description["nbits"], groups=widths.numel(), correction=correction,
                domain_bits=16, width_storage_bits=5)


def _apply_full_correction(prediction, fields, label, description):
    widths = g._unpack_fixed(fields[label + "__correction_widths"], description["groups"], 5)
    coded = g._unpack_variable(fields[label + "__correction"], widths, description["count"],
                               description["group_size"], 16, description["nbits"])
    base = prediction.to(torch.int64).reshape(-1)
    if description["correction"] == "coset":
        modulus = 1 << widths.to(torch.int64).repeat_interleave(description["group_size"])[:len(base)]
        half = modulus >> 1
        delta = ((coded - base + half) & (modulus - 1)) - half
    else:
        delta = g._unzigzag(coded)
    return ((base + delta) & 65535).to(torch.uint16).reshape(prediction.shape)


def encode(record, spec, state):
    if spec != state.spec:
        raise ValueError("producer recipe and prepared state differ")
    record = prepare_record(record, state.device, spec)
    if any(isinstance(record[name], bool) or not isinstance(record[name], (int, np.integer))
           or record[name] < (1 if name == "head_dim" else 0)
           for name in ("layer", "start", "tokens", "head_dim")):
        raise ValueError("producer record requires nonnegative integer geometry")
    k, v = record["k"], record["v"]
    if k.shape != v.shape or record["tokens"] != len(k) or record["start"] < 0:
        raise ValueError("producer KV pair geometry mismatch")
    tokens, channels = k.shape
    layer, kind = int(record["layer"]), spec["kind"]
    fields, nested = {}, {}
    meta = dict(version=1, method="producer_pair", shape=[tokens, 2 * channels],
                kv_shape=[tokens, channels], recipe=spec["name"], kind=kind,
                state_id=state.state_id, layer=layer, start=int(record["start"]),
                head_dim=int(record["head_dim"]), nested=nested)
    if kind in ("kv_median", "kv_fixed127"):
        predictor = "median" if kind == "kv_median" else "fixed127"
        for label, words in (("K", k), ("V", v)):
            _embed(exponent.encode(words, predictor=predictor, layout="time"), fields, nested, label)
    elif kind in ("frozen", "hidden", "hidden_abs"):
        features = _features(record["hidden"], state, layer) if kind != "frozen" else None
        if features is not None:
            if len(features) != tokens:
                raise ValueError("hidden features do not align with KV tokens")
            fields["shared_features"] = g._wire(features)
            meta["feature_rank"] = features.shape[1]
        meta["predicted_exponents"] = {label: _encode_predicted(words,
            _predict(state, layer, label, tokens, features), fields, label)
            for label, words in (("K", k), ("V", v))}
    else:
        source = record["k_pre_rope" if kind == "pre_rope" else "k_proj"]
        if source.shape != k.shape:
            raise ValueError("K producer intermediate shape differs from target")
        _embed(exponent.encode(source, predictor="median", layout="time"), fields, nested, "K_source")
        _embed(exponent.encode(v, predictor="median", layout="time"), fields, nested, "V")
        predicted = _rope(source, state, meta["start"], meta["head_dim"], layer,
                          normalize=kind == "kproj_norm_rope")
        meta["full_word_correction"] = _encode_full_correction(k, predicted, fields, "K", spec["correction"])
    return g._finish(fields, meta, state.device)


def decode(packet, state):
    g.validate_packet(packet, check_header=not packet._header_verified)
    meta, fields = packet.metadata, packet.fields
    if (meta["method"] != "producer_pair" or state is None or meta.get("state_id") != state.state_id
            or meta.get("recipe") != state.spec["name"] or meta.get("kind") != state.spec["kind"]):
        raise ValueError("producer decoder state or recipe identity mismatch")
    if packet.header.device != state.device:
        raise ValueError("producer packet and state must share device")
    tokens, channels = meta["kv_shape"]
    if meta["shape"] != [tokens, 2 * channels]:
        raise ValueError("producer outer and KV shapes differ")
    kind, layer = meta["kind"], meta["layer"]
    if (not isinstance(layer, int) or layer < 0 or not isinstance(meta["start"], int) or meta["start"] < 0
            or not isinstance(meta["head_dim"], int) or meta["head_dim"] < 1):
        raise ValueError("invalid producer replay geometry")
    nested_labels = set(meta.get("nested", {}))
    expected_nested = ({"K", "V"} if kind in ("kv_median", "kv_fixed127") else
                       {"K_source", "V"} if kind in ("pre_rope", "kproj_norm_rope") else set())
    if nested_labels != expected_nested or any(meta["nested"][name]["shape"] != [tokens, channels]
                                               for name in nested_labels):
        raise ValueError("producer nested source geometry mismatch")
    if kind in ("kv_median", "kv_fixed127"):
        return tuple(_restore_nested(meta, fields, label, state.device) for label in ("K", "V"))
    if kind in ("frozen", "hidden", "hidden_abs"):
        if _param(state, layer, "base_K").numel() != channels or _param(state, layer, "base_V").numel() != channels:
            raise ValueError("frozen exponent center does not match KV channels")
        features = None
        if kind != "frozen":
            rank = meta["feature_rank"]
            if rank != _param(state, layer, "indices").numel() or fields["shared_features"].numel() != tokens * rank:
                raise ValueError("producer shared feature geometry mismatch")
            features = fields["shared_features"].view(torch.int8).reshape(tokens, rank)
        if set(meta.get("predicted_exponents", {})) != {"K", "V"} or any(
                meta["predicted_exponents"][name]["shape"] != [tokens, channels] for name in ("K", "V")):
            raise ValueError("producer exponent geometry mismatch")
        return tuple(_decode_predicted(meta["predicted_exponents"][label], fields, label,
            _predict(state, layer, label, tokens, features)) for label in ("K", "V"))
    source = _restore_nested(meta, fields, "K_source", state.device)
    v = _restore_nested(meta, fields, "V", state.device)
    prediction = _rope(source, state, meta["start"], meta["head_dim"], layer,
                       normalize=kind == "kproj_norm_rope")
    description = meta["full_word_correction"]
    if (description["correction"] != state.spec["correction"] or description["count"] != tokens * channels
            or description["groups"] != (tokens * channels + description["group_size"] - 1) // description["group_size"]
            or description.get("width_storage_bits") != 5 or description.get("domain_bits") != 16):
        raise ValueError("invalid full-word producer correction contract")
    k = _apply_full_correction(prediction, fields, "K", meta["full_word_correction"])
    return k, v


def describe(packet):
    meta, fields = packet.metadata, packet.fields
    result = dict(recipe=meta["recipe"], kind=meta["kind"], pair_tokens=meta["kv_shape"][0],
                  pair_channels=meta["kv_shape"][1], header_bytes=packet.header.numel(),
                  shared_feature_bytes=fields["shared_features"].numel() if "shared_features" in fields else 0,
                  original_hidden_stored=False, original_raw_KV_target_stored=False,
                  full_RoPE_table_stored=False)
    if "full_word_correction" in meta:
        result.update(correction=meta["full_word_correction"]["correction"],
                      correction_data_bits=meta["full_word_correction"]["nbits"],
                      correction_payload_bytes=fields["K__correction"].numel(),
                      correction_width_bytes=fields["K__correction_widths"].numel())
    return result

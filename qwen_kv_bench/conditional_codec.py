"""Exact producer pair recipes: frozen entropy models and shared joint predictors.

Each record is independently reconstructable from its actual fields and frozen
state. Producer activations are used only when creating small retained features;
decoders never read an original hidden state or unretained KV target. All eager
float prediction/replay is bound to the recorded backend and matmul settings.
Diagnostic sidecars contain only bounded aggregate counts, never decoder inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch

from . import gpu_ops as g
from . import producer_codec as producer
from . import frontier_codec as frontier
from . import exponent_recipe as exponent

_LEGACY = ("producer_frozen_exponent", "frontier_cost_final_raw8")


def recipes():
    result = []
    for name in _LEGACY:
        registry = producer.recipes() if name.startswith("producer_") else frontier.recipes()
        result.append(next(dict(row, family="control") for row in registry if row["name"] == name))
    result += [dict(name="conditional_fast_pre_rope", family="control", kind="tier"),
        dict(name="conditional_huffman_exponent", family="exponent", kind="huffman"),
        dict(name="conditional_huffman_exponent_sign", family="sign", kind="huffman_sign"),
        dict(name="conditional_tier_sign", family="sign", kind="tier_sign"),
        dict(name="conditional_raw_pair", family="control", kind="raw_pair"),
        dict(name="conditional_raw_fullword", family="control", kind="zero", fallback=False),
        dict(name="conditional_mean_predictor", family="control", kind="mean", model="conditional_mean", fallback=True)]
    for family in ("anchor", "latent"):
        for rank in (8, 32):
            result.append(dict(name=f"conditional_{family}_r{rank}", family=family, kind="predictor",
                model=f"conditional_{family}_r{rank}", rank=rank, fallback=True))
    for row in result:
        if row["name"] not in _LEGACY:
            row["method"] = "conditional_pair"
    return result


def generation_representatives():
    return [row["name"] for row in recipes()]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _legacy_spec(spec):
    registry = producer.recipes() if spec["name"].startswith("producer_") else frontier.recipes()
    return next(row for row in registry if row["name"] == spec["name"])


def _execution_contract(device):
    result = frontier._execution_contract(device)
    try:
        autocast_enabled = bool(torch.is_autocast_enabled(torch.device(device).type))
        autocast_dtype = str(torch.get_autocast_dtype(torch.device(device).type))
    except (TypeError, RuntimeError):
        autocast_enabled, autocast_dtype = False, None
    result.update(autocast_enabled=autocast_enabled, autocast_dtype=autocast_dtype,
        conditional_prediction_rule="stored-BF16-features, eager-FP32-matmul-plus-intercept, BF16-round-v1",
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
    return result


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensors(item)]
    if hasattr(value, "__dict__"):
        return _tensors(vars(value))
    return []


@dataclass
class ConditionalState:
    spec: dict
    cost_state: producer.ProducerState
    rope_state: producer.ProducerState | None
    predictor_params: dict
    predictor_kind: str | None
    entropy_books: dict
    wire_bytes: bytes
    state_id: str
    execution_contract: dict | None
    model_metadata: dict

    @property
    def device(self):
        return self.cost_state.device

    @property
    def physical_nbytes(self):
        return g._storage_bytes(list(self.cost_state.fields.values()) + (list(self.rope_state.fields.values()) if self.rope_state else [])
            + _tensors(self.predictor_params) + _tensors(self.entropy_books))

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
    if isinstance(state, ConditionalState):
        return state.wire_bytes
    return frontier.state_bytes(state) if isinstance(state, frontier.FrontierState) else producer.state_bytes(state)


def state_info(state):
    if not isinstance(state, ConditionalState):
        return frontier.state_info(state) if isinstance(state, frontier.FrontierState) else producer.state_info(state)
    encoder_only = [array for params in state.predictor_params.values()
                    for name, array in params.items() if name in ("feature_center", "feature_projection")]
    decoder_predictor = [array for params in state.predictor_params.values()
                         for name, array in params.items() if name not in ("feature_center", "feature_projection")]
    encoder_only_bytes = g._storage_bytes(encoder_only)
    return dict(state_id=state.state_id, serialized_bytes=state.serialized_bytes,
        physical_tensor_bytes=state.physical_nbytes, charged_bytes=state.state_bytes,
        encoder_only_predictor_tensor_bytes=encoder_only_bytes,
        decoder_predictor_tensor_bytes=g._storage_bytes(decoder_predictor),
        required_decoder_numerical_bytes_excluding_encoder_only_predictor=state.physical_nbytes-encoder_only_bytes,
        encoder_parameters_in_headline_charge=True,
        accounting="Conservative maximum of complete artifacts/runtime contract and ALL resident GPU tensors, including encoder-only latent projection/center. Numerical role breakdown is reported; this benchmark does not serialize a separate minimal decoder artifact.",
        replay_execution_contract=state.execution_contract, model_metadata=state.model_metadata,
        cost_base_state=producer.state_info(state.cost_state), replay_state=producer.state_info(state.rope_state) if state.rope_state else None)


def _model_wire(model):
    if model is None:
        return b""
    if model.path and Path(model.path).exists():
        wire = Path(model.path).read_bytes()
    else:
        from . import conditional_models
        wire = conditional_models.model_bytes(model)
    if hashlib.sha256(wire).hexdigest() != model.model_id:
        raise ValueError("conditional model artifact and frozen identity differ")
    return wire


def make_state(spec, models, context, device="cpu"):
    known = {row["name"]: row for row in recipes()}
    if spec.get("name") not in known or spec != known[spec["name"]]:
        raise ValueError("unknown or modified conditional recipe")
    if spec["name"] in _LEGACY:
        legacy = _legacy_spec(spec)
        return (producer.make_state(legacy, models, context, device) if legacy["method"] == "producer_pair"
                else frontier.make_state(legacy, models, context, device))
    if spec["kind"] == "raw_pair":
        target_device = torch.empty(0, device=device).device
        identity = hashlib.sha256(_json(spec)).hexdigest()
        empty = producer.ProducerState(dict(spec), {}, {"kind":"raw_pair", "external_state":"none"}, b"", identity, target_device)
        return ConditionalState(dict(spec), empty, None, {}, None, {}, b"", identity, None, {})
    if "cost_pre_rope" not in models:
        raise ValueError("conditional recipes require the TRAIN pre-RoPE cost base")
    cost_model = models["cost_pre_rope"]
    if cost_model.config.get("source_k") != "k_pre_rope" or cost_model.config.get("source_v") != "v":
        raise ValueError("conditional cost bases must match actual pre-RoPE K and V")
    base_spec = next(row for row in producer.recipes() if row["name"] == "producer_frozen_exponent")
    cost = producer.make_state(base_spec, {"frozen": cost_model}, context, device)
    replay_spec = next(row for row in producer.recipes() if row["name"] == "producer_pre_rope_coset")
    replay = producer.make_state(replay_spec, models, context, device)
    predictor = models.get(spec.get("model")) if spec.get("model") else None
    if spec.get("model") and predictor is None:
        raise ValueError("missing frozen conditional predictor model")
    predictor_params = {}
    if predictor:
        expected_kind = "conditional_mean" if spec["kind"] == "mean" else "conditional_" + spec["family"]
        expected_rank = 0 if spec["kind"] == "mean" else spec["rank"]
        if (predictor.kind != expected_kind or predictor.config.get("rank_requested") != expected_rank
                or predictor.config.get("source_K") != "k_pre_rope" or predictor.config.get("source_V") != "v"
                or predictor.config.get("target_source") != "concat_actual_pre_rope_K_and_V"):
            raise ValueError("conditional predictor source, family or rank differs from recipe")
        predictor_params = {layer: {name: torch.from_numpy(np.asarray(array).copy()).to(cost.device)
                            for name, array in params.items()} for layer, params in predictor.params.items()}
        if predictor.kind == "conditional_anchor":
            for layer, params in predictor_params.items():
                channels = len(predictor.params[layer]["intercept"]) // 2
                anchors = set(int(value) for value in predictor.params[layer]["anchor_indices"])
                for label, offset in (("K", 0), ("V", channels)):
                    keep = [index for index in range(channels) if index + offset not in anchors]
                    params["_complement_" + label] = torch.tensor(keep, device=cost.device, dtype=torch.int64)
    entropy_model = None
    books = {}
    if spec["kind"] in ("huffman", "huffman_sign", "tier_sign"):
        from . import conditional_entropy
        entropy_model = models.get("conditional_entropy")
        if entropy_model is None:
            raise ValueError("missing frozen TRAIN conditional entropy tables")
        if (entropy_model.kind != "conditional_entropy"
                or entropy_model.config.get("exponent_base_model_id") != cost_model.model_id
                or entropy_model.config.get("source_K") != "k_pre_rope" or entropy_model.config.get("source_V") != "v"):
            raise ValueError("conditional entropy TRAIN source/base identity mismatch")
        prefixes = (["exp_K", "exp_V"] if spec["kind"] != "tier_sign" else [])
        if spec["kind"] in ("huffman_sign", "tier_sign"):
            prefixes += ["sign_K", "sign_V"]
        books = {prefix: conditional_entropy.prepare(entropy_model.book(prefix), cost.device) for prefix in prefixes}
    metadata = dict(predictor_kind=predictor.kind if predictor else None,
        predictor_model_id=predictor.model_id if predictor else None,
        predictor_config=predictor.config if predictor else None,
        entropy_model_id=entropy_model.model_id if entropy_model else None,
        entropy_config=entropy_model.config if entropy_model else None)
    contract = _execution_contract(cost.device)
    parts = [cost.wire_bytes, replay.wire_bytes, _model_wire(predictor), _model_wire(entropy_model)]
    descriptor = _json(dict(format="conditional_complete_state_v1", spec=spec,
        cost_state_id=cost.state_id, replay_state_id=replay.state_id,
        execution_contract=contract, models=metadata))
    wire = b"CONDITIONAL_STATE_V1\0" + struct.pack("<5Q", len(descriptor), *[len(part) for part in parts]) + descriptor + b"".join(parts)
    return ConditionalState(dict(spec), cost, replay, predictor_params,
        predictor.kind if predictor else None, books, wire, hashlib.sha256(wire).hexdigest(), contract, metadata)


def prepare_record(record, device, spec=None):
    if spec is not None and spec["name"] in _LEGACY:
        legacy = _legacy_spec(spec)
        return (producer.prepare_record(record, device, legacy) if legacy["method"] == "producer_pair"
                else frontier.prepare_record(record, device, legacy))
    needed = {"k", "v"} if spec and spec.get("kind") == "raw_pair" else {"k", "v", "k_pre_rope"}
    if spec and spec.get("family") == "latent":
        needed.add("hidden")
    result = {key: record[key] for key in ("layer", "start", "tokens", "head_dim")}
    for key in needed:
        if key not in record:
            raise ValueError("missing actual conditional producer source: " + key)
        result[key] = producer._words(record[key], device)
    return result


def _check_state(state):
    if state.execution_contract is not None and _json(_execution_contract(state.device)) != _json(state.execution_contract):
        raise ValueError("conditional predictor/replay execution contract changed")


def _base(state, layer, label, shape):
    base = producer._param(state.cost_state, layer, "base_" + label).to(torch.int64)
    if len(base) != shape[1]:
        raise ValueError("conditional cost-base geometry mismatch")
    return base[None, :].expand(shape)


def _tier_encode(words, prediction, fields, label):
    description = producer._encode_predicted(words, prediction, fields, label)
    frontier._compact(fields, description, label)
    return description


def _tier_decode(description, fields, label, prediction):
    tokens, channels = description["shape"]
    count = tokens * channels
    segments = (count + description["segment_size"] - 1) // description["segment_size"]
    dtype = torch.uint16 if description["offset_storage_bits"] == 16 else torch.uint32
    restored = dict(fields)
    for suffix, terminal in (("secondary_offsets", "secondary_count"), ("rare_offsets", "rare_count")):
        starts = fields[label + "__" + suffix].view(dtype).to(torch.int64)
        if starts.numel() != segments:
            raise ValueError("conditional compact tier offset length mismatch")
        complete = torch.cat((starts, torch.tensor([description[terminal]], device=starts.device, dtype=torch.int64))).to(torch.uint32)
        restored[label + "__" + suffix] = g._wire(complete)
    return producer._decode_predicted(description, restored, label, prediction)


def _embed(fields, prefix, child):
    fields.update({prefix + "__" + name: value for name, value in child.items()})


def _children(fields, prefix):
    start = prefix + "__"
    return {name[len(start):]: value for name, value in fields.items() if name.startswith(start)}


def _sign_encode(words, state, label, fields):
    from . import conditional_entropy
    signs = words.to(torch.int64) >> 15
    fields[label + "__sign_seed"], _ = g._pack_fixed(signs[0], 1)
    transitions = exponent._ordered(signs[1:] ^ signs[:-1], "time")
    packed, nbits = g._pack_fixed(transitions, 1)
    child, descriptor = conditional_entropy.encode(packed.to(torch.int64), state.entropy_books["sign_" + label])
    _embed(fields, label + "__sign_huff", child)
    return dict(seed_count=words.shape[1], transition_bits=nbits, entropy=descriptor)


def _sign_decode(description, state, label, fields, shape):
    from . import conditional_entropy
    tokens, channels = shape
    if description["seed_count"] != channels or description["transition_bits"] != max(tokens - 1, 0) * channels:
        raise ValueError("conditional sign reset geometry mismatch")
    packed = conditional_entropy.decode(_children(fields, label + "__sign_huff"), description["entropy"],
        state.entropy_books["sign_" + label]).to(torch.uint8)
    transitions = g._unpack_fixed(packed, description["transition_bits"], 1)
    transitions = exponent._canonical(transitions, [max(tokens - 1, 0), channels], "time")
    seed = g._unpack_fixed(fields[label + "__sign_seed"], channels, 1)
    return torch.cat((seed[None, :], (transitions.cumsum(dim=0) + seed[None, :]) & 1), dim=0)


def _encode_side(words, state, layer, label, fields, kind):
    from . import conditional_entropy
    prediction = _base(state, layer, label, list(words.shape))
    signed = kind in ("huffman_sign", "tier_sign")
    if kind in ("huffman", "huffman_sign"):
        values = g._zigzag(exponent._ordered(((words.to(torch.int64) >> 7) & 255) - prediction, "time"))
        child, descriptor = conditional_entropy.encode(values, state.entropy_books["exp_" + label])
        _embed(fields, label + "__exp_huff", child)
        description = dict(codec="huffman", shape=list(words.shape), entropy=descriptor)
        if not signed:
            integer = words.to(torch.int64)
            fields[label + "__lower"] = (((integer >> 8) & 128) | (integer & 127)).to(torch.uint8).reshape(-1)
    else:
        description = dict(codec="tier", tier=_tier_encode(words, prediction, fields, label), shape=list(words.shape))
    if signed:
        fields.pop(label + "__lower", None)
        fields[label + "__mantissa7"], _ = g._pack_fixed(words.reshape(-1).to(torch.int64) & 127, 7)
        description["sign"] = _sign_encode(words, state, label, fields)
    description["lower_mode"] = "previous_sign_huffman_plus_mantissa7" if signed else "raw8"
    return description


def _decode_side(description, fields, state, layer, label):
    from . import conditional_entropy
    shape = description["shape"]
    prediction = _base(state, layer, label, shape)
    restored = dict(fields)
    if description["lower_mode"] == "previous_sign_huffman_plus_mantissa7":
        signs = _sign_decode(description["sign"], state, label, fields, shape)
        mantissa = g._unpack_fixed(fields[label + "__mantissa7"], shape[0] * shape[1], 7)
        lower = (mantissa | (signs.reshape(-1) << 7)).to(torch.uint8)
    elif description["lower_mode"] == "raw8":
        lower = fields[label + "__lower"]
    else:
        raise ValueError("unknown conditional lower-plane mode")
    if description["codec"] == "tier":
        restored[label + "__lower"] = lower
        return _tier_decode(description["tier"], restored, label, prediction)
    if description["codec"] != "huffman":
        raise ValueError("unknown conditional exponent codec")
    values = conditional_entropy.decode(_children(fields, label + "__exp_huff"), description["entropy"],
        state.entropy_books["exp_" + label])
    exponents = prediction + exponent._canonical(g._unzigzag(values), shape, "time")
    lo = lower.to(torch.int64).reshape(shape)
    return (((lo & 128) << 8) | (lo & 127) | (exponents << 7)).to(torch.uint16)


def _params(state, layer):
    key = f"layer_{layer:02d}"
    if key not in state.predictor_params:
        raise ValueError("conditional predictor lacks actual layer")
    return state.predictor_params[key]


def _features_and_prediction(record, state):
    from . import conditional_models
    tokens, channels = record["v"].shape
    if state.spec["kind"] == "zero":
        return None, torch.zeros((tokens, 2 * channels), device=state.device, dtype=torch.uint16)
    params = _params(state, int(record["layer"]))
    if state.spec["kind"] == "mean":
        prediction = params["intercept"].to(torch.float32).to(torch.bfloat16).view(torch.uint16)[None, :].expand(tokens, 2 * channels).contiguous()
        return None, prediction
    features = conditional_models.torch_features_params(params, state.predictor_kind, record)
    prediction = conditional_models.torch_predict_params(params, state.predictor_kind, features)
    return features, prediction


def _restore_prediction(meta, fields, state):
    from . import conditional_models
    tokens, channels = meta["kv_shape"]
    if state.spec["kind"] == "zero":
        return torch.zeros((tokens, 2 * channels), device=state.device, dtype=torch.uint16)
    params = _params(state, meta["layer"])
    if state.spec["kind"] == "mean":
        return params["intercept"].to(torch.float32).to(torch.bfloat16).view(torch.uint16)[None, :].expand(tokens, 2 * channels).contiguous()
    rank = meta["feature_rank"]
    if fields["shared_features"].numel() != tokens * rank * 2:
        raise ValueError("conditional retained feature geometry mismatch")
    features = fields["shared_features"].view(torch.uint16).reshape(tokens, rank)
    return conditional_models.torch_predict_params(params, state.predictor_kind, features)


def _correction_columns(state, layer, label, channels):
    if state.predictor_kind == "conditional_anchor":
        return _params(state, layer)["_complement_" + label]
    return torch.arange(channels, device=state.device, dtype=torch.int64)


def _select_columns(words, indices):
    # Signed views avoid missing uint16 indexing kernels on several backends.
    return words.contiguous().view(torch.int16).index_select(1, indices).contiguous().view(torch.uint16)


def _replay_encode(k, source, fields, meta, state):
    prediction = producer._rope(source, state.rope_state, meta["start"], meta["head_dim"], meta["layer"])
    if bool(torch.equal(k.to(torch.int32), prediction.to(torch.int32))):
        meta["replay"] = dict(mode="exact")
    else:
        meta["replay"] = dict(mode="coset", correction=producer._encode_full_correction(k, prediction,
            fields, "replay_K", "coset"))
    return prediction


def _replay_decode(source, fields, meta, state):
    prediction = producer._rope(source, state.rope_state, meta["start"], meta["head_dim"], meta["layer"])
    if meta["replay"]["mode"] == "exact":
        return prediction
    if meta["replay"]["mode"] != "coset":
        raise ValueError("unknown conditional replay correction")
    return producer._apply_full_correction(prediction, fields, "replay_K", meta["replay"]["correction"])


def _meta(record, state, mode):
    tokens, channels = record["v"].shape
    return dict(version=1, method="conditional_pair", recipe=state.spec["name"], kind=state.spec["kind"],
        shape=[tokens, 2 * channels], kv_shape=[tokens, channels], state_id=state.state_id,
        layer=int(record["layer"]), start=int(record["start"]), head_dim=int(record["head_dim"]),
        mode=mode, source="actual_pre_rope_K_and_actual_V")


def _baseline_packet(record, state, mode):
    fields, meta = {}, _meta(record, state, mode)
    if mode == "raw":
        fields.update(source_K=g._wire(record["k_pre_rope"].clone()), source_V=g._wire(record["v"].clone()))
    else:
        meta["sides"] = {label: _encode_side(words, state, meta["layer"], label, fields, "tier")
            for label, words in (("K", record["k_pre_rope"]), ("V", record["v"]))}
    _replay_encode(record["k"], record["k_pre_rope"], fields, meta, state)
    return g._finish(fields, meta, state.device)


def _quality(actual, prediction):
    a, p = actual.to(torch.int64), prediction.to(torch.int64)
    difference = a ^ p
    pop = difference - ((difference >> 1) & 0x5555)
    pop = (pop & 0x3333) + ((pop >> 2) & 0x3333)
    pop = (pop + (pop >> 4)) & 0x0f0f
    pop = (pop + (pop >> 8)) & 0x1f
    error = ((a - p + 32768) & 65535) - 32768
    counts = torch.stack(((a == p).sum(), ((a >> 15) == (p >> 15)).sum(),
        (((a >> 7) & 255) == ((p >> 7) & 255)).sum(), pop.sum(), error.abs().sum())).detach().cpu().tolist()
    return dict(values=actual.numel(), word_matches=int(counts[0]), sign_matches=int(counts[1]),
        exponent_matches=int(counts[2]), matching_bits=int(actual.numel() * 16 - counts[3]),
        absolute_circular_word_error_sum=int(counts[4]))


def _width_hist(fields, meta):
    result = torch.zeros(17, device=next(iter(fields.values())).device, dtype=torch.int64)
    for label in ("K", "V"):
        if "corrections" not in meta:
            continue
        description = meta["corrections"][label]
        widths = g._unpack_fixed(fields[label + "__correction_widths"], description["groups"], 5)
        result += torch.bincount(widths, minlength=17)
    return {str(i): int(value) for i, value in enumerate(result.detach().cpu().tolist())}


def _stream_statistics(values, lengths, alphabet):
    values, lengths = values.reshape(-1).to(torch.int64), lengths.reshape(-1).to(torch.int64)
    histogram = torch.bincount(values, minlength=alphabet)
    width_histogram = torch.bincount(lengths, minlength=16)
    counts = torch.cat((histogram, width_histogram)).detach().cpu().tolist()
    symbols, width_counts = counts[:alphabet], counts[alphabet:]
    n = values.numel()
    entropy_bits = -sum((count / n) * math.log2(count / n) for count in symbols if count) if n else None
    encoded_bits = sum(width * count for width, count in enumerate(width_counts))
    return dict(symbols=n, symbol_histogram={str(i):int(count) for i,count in enumerate(symbols)},
        code_length_histogram={str(i):int(count) for i,count in enumerate(width_counts)},
        sum_code_bits=int(encoded_bits), empirical_entropy_bits_per_symbol=entropy_bits,
        actual_mean_code_bits_per_symbol=encoded_bits/n if n else None,
        entropy_scope="Finite-record descriptive plug-in entropy; not a lower bound, held-out rate prediction or fitted TEST policy")


def _symbol_diagnostics(record, state, kind):
    result = {}
    for label, field in (("K", "k_pre_rope"), ("V", "v")):
        words = record[field]
        prediction = _base(state, int(record["layer"]), label, list(words.shape))
        values = g._zigzag(exponent._ordered(((words.to(torch.int64) >> 7) & 255) - prediction, "time"))
        if "exp_" + label in state.entropy_books:
            widths = state.entropy_books["exp_" + label].length[values]
        else:
            widths = 2 + 3 * (values >= 3).to(torch.int64) + 9 * (values >= 10).to(torch.int64)
        result[label + "_exponent"] = _stream_statistics(values, widths, 512)
        if kind in ("huffman_sign", "tier_sign"):
            signs = words.to(torch.int64) >> 15
            transitions = exponent._ordered(signs[1:] ^ signs[:-1], "time")
            packed, bit_count = g._pack_fixed(transitions, 1)
            symbols = packed.to(torch.int64)
            widths = state.entropy_books["sign_" + label].length[symbols]
            stats = _stream_statistics(symbols, widths, 256)
            stats.update(source_values=words.numel(), transition_bits=bit_count,
                seed_signs=words.shape[1], seed_storage_bytes=(words.shape[1]+7)//8)
            result[label + "_sign"] = stats
    return result


def _entropy_diagnostics(packet):
    result = {}
    for label, side in packet.metadata.get("sides", {}).items():
        if side.get("entropy"):
            d = side["entropy"]
            result[label + "_exponent"] = dict(encoded_bits=d["nbits"], symbols=d["num_symbols"])
        if side.get("sign"):
            d = side["sign"]["entropy"]
            result[label + "_sign"] = dict(encoded_bits=d["nbits"], symbols=d["num_symbols"],
                seed_bytes=packet.fields[label + "__sign_seed"].numel())
    return result


def encode(record, spec, state):
    if spec["name"] in _LEGACY:
        legacy = _legacy_spec(spec)
        return producer.encode(record, legacy, state) if legacy["method"] == "producer_pair" else frontier.encode(record, legacy, state)
    if spec != state.spec:
        raise ValueError("conditional recipe and prepared state differ")
    _check_state(state)
    record = prepare_record(record, state.device, spec)
    if any(isinstance(record[name], bool) or not isinstance(record[name], (int, np.integer))
           or record[name] < (1 if name in ("tokens", "head_dim") else 0)
           for name in ("layer", "start", "tokens", "head_dim")):
        raise ValueError("conditional records require actual nonnegative integer geometry")
    if spec["kind"] == "raw_pair":
        if record["k"].shape != record["v"].shape or record["tokens"] != len(record["k"]) or record["tokens"] < 1:
            raise ValueError("raw paired KV geometry mismatch")
        fields = {"final_K":g._wire(record["k"].clone()), "final_V":g._wire(record["v"].clone())}
        meta = _meta(record, state, "raw_pair")
        meta.update(source="actual_final_K_and_actual_V", replay={"mode":"none"}, materialization="clone raw output tensors")
        return g._finish(fields, meta, state.device)
    if (any(isinstance(record[k], bool) or not isinstance(record[k], (int, np.integer)) or record[k] < 0 for k in ("layer", "start", "tokens"))
            or record["tokens"] < 1 or record["head_dim"] < 1
            or record["k"].shape != record["v"].shape or record["k_pre_rope"].shape != record["v"].shape
            or record["tokens"] != record["v"].shape[0]):
        raise ValueError("conditional producer geometry mismatch")
    kind = spec["kind"]
    if kind in ("tier", "huffman", "huffman_sign", "tier_sign"):
        fields, meta = {}, _meta(record, state, "entropy" if kind != "tier" else "tier")
        meta["sides"] = {label: _encode_side(words, state, meta["layer"], label, fields, kind)
            for label, words in (("K", record["k_pre_rope"]), ("V", record["v"]))}
        replay = _replay_encode(record["k"], record["k_pre_rope"], fields, meta, state)
        packet = g._finish(fields, meta, state.device)
        packet.analysis = dict(replay_quality=_quality(record["k"], replay),
            entropy=_entropy_diagnostics(packet), stream_statistics=_symbol_diagnostics(record, state, kind),
            diagnostics_cost="bounded tensor aggregate calculations and count-only GPU-to-host histograms included in encode timing")
        return packet
    features, prediction = _features_and_prediction(record, state)
    source = torch.cat((record["k_pre_rope"], record["v"]), dim=1)
    if prediction.shape != source.shape or prediction.dtype != torch.uint16:
        raise ValueError("conditional joint predictor output geometry mismatch")
    fields, meta = {}, _meta(record, state, "predicted")
    if features is not None:
        if features.dtype != torch.uint16 or len(features) != record["tokens"]:
            raise ValueError("conditional retained features must be actual BF16 word rows")
        fields["shared_features"] = g._wire(features.clone())
        meta["feature_rank"] = features.shape[1]
    channels = record["v"].shape[1]
    meta["correction_layout"] = "complement_of_retained_anchors" if state.predictor_kind == "conditional_anchor" else "all_source_coordinates"
    meta["corrections"] = {}
    for label, actual, predicted in (("K", record["k_pre_rope"], prediction[:, :channels]),
                                     ("V", record["v"], prediction[:, channels:])):
        columns = _correction_columns(state, meta["layer"], label, channels)
        meta["corrections"][label] = producer._encode_full_correction(_select_columns(actual, columns),
            _select_columns(predicted, columns), fields, label, "residual")
    replay = _replay_encode(record["k"], record["k_pre_rope"], fields, meta, state)
    predicted_packet = g._finish(fields, meta, state.device)
    width_hist = _width_hist(fields, meta)
    quality = _quality(source, prediction)
    candidates = {"predicted": predicted_packet}
    if spec.get("fallback", False):
        candidates["tier"] = _baseline_packet(record, state, "tier")
        candidates["raw"] = _baseline_packet(record, state, "raw")
    sizes = {name: value.physical_nbytes for name, value in candidates.items()}
    # Actual complete packet bytes decide; stable order retains predictor ties.
    chosen = min(candidates, key=lambda name: (sizes[name], {"predicted":0,"tier":1,"raw":2}[name]))
    packet = candidates[chosen]
    packet.analysis = dict(predictor_quality=quality, replay_quality=_quality(record["k"], replay),
        predictor_width_histogram=width_hist, candidate_complete_bytes=sizes,
        chosen_coder=chosen, predictor_fallback=chosen != "predicted",
        candidate_feature_bytes=fields.get("shared_features", torch.empty(0, dtype=torch.uint8)).numel(),
        candidate_anchor_bytes=fields.get("shared_features", torch.empty(0, dtype=torch.uint8)).numel() if spec["family"] == "anchor" else 0,
        diagnostics_cost="bounded tensor aggregate calculations included in encode timing; sidecar is not a decoder input")
    return packet


def decode(packet, state):
    if not isinstance(state, ConditionalState):
        return frontier.decode(packet, state) if isinstance(state, frontier.FrontierState) else producer.decode(packet, state)
    _check_state(state)
    g.validate_packet(packet, check_header=not packet._header_verified)
    meta, fields = packet.metadata, packet.fields
    if (meta.get("method") != "conditional_pair" or meta.get("state_id") != state.state_id
            or meta.get("recipe") != state.spec["name"] or meta.get("kind") != state.spec["kind"]
            or packet.header.device != state.device):
        raise ValueError("conditional packet/state identity mismatch")
    tokens, channels = meta["kv_shape"]
    if (tokens < 1 or channels < 1 or meta["shape"] != [tokens, 2 * channels]
            or not isinstance(meta["layer"], int) or meta["layer"] < 0
            or not isinstance(meta["start"], int) or meta["start"] < 0 or meta["head_dim"] < 1):
        raise ValueError("conditional packet geometry mismatch")
    mode = meta["mode"]
    if mode == "raw_pair":
        if state.spec["kind"] != "raw_pair" or set(fields) != {"final_K", "final_V"}:
            raise ValueError("raw conditional field contract mismatch")
        return tuple(fields[label].view(torch.uint16).reshape(tokens, channels).clone() for label in ("final_K", "final_V"))
    if mode == "raw":
        source = fields["source_K"].view(torch.uint16).reshape(tokens, channels)
        v = fields["source_V"].view(torch.uint16).reshape(tokens, channels)
    elif mode in ("tier", "entropy"):
        if set(meta["sides"]) != {"K", "V"} or any(meta["sides"][label]["shape"] != [tokens, channels] for label in ("K", "V")):
            raise ValueError("conditional side geometry mismatch")
        source, v = tuple(_decode_side(meta["sides"][label], fields, state, meta["layer"], label) for label in ("K", "V"))
    elif mode == "predicted":
        prediction = _restore_prediction(meta, fields, state)
        corrected = []
        expected_layout = "complement_of_retained_anchors" if state.predictor_kind == "conditional_anchor" else "all_source_coordinates"
        if meta.get("correction_layout") != expected_layout:
            raise ValueError("conditional correction coordinate layout mismatch")
        for label, predicted in (("K", prediction[:, :channels]), ("V", prediction[:, channels:])):
            columns = _correction_columns(state, meta["layer"], label, channels)
            description = meta["corrections"][label]
            if description["count"] != tokens * len(columns) or description["domain_bits"] != 16:
                raise ValueError("conditional correction geometry mismatch")
            recovered = producer._apply_full_correction(_select_columns(predicted, columns), fields, label, description)
            signed = predicted.contiguous().view(torch.int16).clone()
            signed[:, columns] = recovered.view(torch.int16)
            corrected.append(signed.view(torch.uint16))
        source, v = corrected
    else:
        raise ValueError("unknown conditional materialization mode")
    return _replay_decode(source, fields, meta, state), v


def describe(packet):
    if packet.metadata["method"] == "producer_pair":
        return producer.describe(packet)
    if packet.metadata["method"] == "frontier_pair":
        return frontier.describe(packet)
    meta = packet.metadata
    result = dict(recipe=meta["recipe"], kind=meta["kind"], source=meta["source"], chosen_coder=meta["mode"],
        replay_correction_mode=meta["replay"]["mode"], header_bytes=packet.header.numel(),
        retained_feature_bytes=packet.fields["shared_features"].numel() if "shared_features" in packet.fields else 0,
        original_hidden_stored=False, independent_tile=True, analysis_sidecar_in_wire=False)
    result.update(getattr(packet, "analysis", {}))
    return result

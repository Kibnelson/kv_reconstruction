"""Exact forward-producer frontier: cost bases, compact indexes, lower planes.

The three retained producer recipes are controls.  New recipes share one outer
packet schema, source-specific frozen TRAIN bases, 2/3/9-bit exponent tiers and
actual compact start-only indexes.  Optional tile/head shifts are retained only
when their actual field and JSON-descriptor bytes beat zero shifts.  Producer
replay omits corrections only after verifying every final K bit; otherwise an
exact full-word coset fallback is stored.  No savings or GPU speed is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
import struct

import torch

from . import exponent_recipe as exponent
from . import gpu_ops as g
from . import producer_codec as producer


_CONTROLS = ("producer_kv_median", "producer_frozen_exponent", "producer_pre_rope_coset")


def recipes():
    result = [spec for spec in producer.recipes() if spec["name"] in _CONTROLS]
    result.append(dict(name="frontier_cost_final_raw8", method="frontier_pair",
                       source="final", head_shifts=False, lower_mode="raw8", primary_mode="auto_plane"))
    for source in ("final", "pre_rope", "kproj"):
        for mode in ("raw8", "plane_raw", "plane_first", "plane_prev"):
            result.append(dict(name=f"frontier_shift_{source}_{mode}", method="frontier_pair",
                               source=source, head_shifts=True, lower_mode=mode, primary_mode="auto_plane"))
    return result


def generation_representatives():
    return ["producer_kv_median", "producer_frozen_exponent", "frontier_shift_final_plane_prev",
            "frontier_shift_pre_rope_plane_prev", "frontier_shift_kproj_plane_prev"]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass
class FrontierState:
    spec: dict
    cost_state: producer.ProducerState
    rope_state: producer.ProducerState | None
    wire_bytes: bytes
    state_id: str
    execution_contract: dict | None

    @property
    def device(self):
        return self.cost_state.device

    @property
    def physical_nbytes(self):
        tensors = list(self.cost_state.fields.values())
        if self.rope_state:
            tensors.extend(self.rope_state.fields.values())
        return g._storage_bytes(tensors)

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
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.wire_bytes)
        return path


def state_bytes(state):
    return state.wire_bytes


def state_info(state):
    if isinstance(state, producer.ProducerState):
        return producer.state_info(state)
    return dict(state_id=state.state_id, serialized_bytes=state.serialized_bytes,
                physical_tensor_bytes=state.physical_nbytes, charged_bytes=state.state_bytes,
                accounting="maximum of complete cost-base/replay serialized state and distinct device tensor storage",
                cost_base_state=producer.state_info(state.cost_state),
                replay_state=producer.state_info(state.rope_state) if state.rope_state else None,
                replay_execution_contract=state.execution_contract)


def _execution_contract(device):
    """Bind replay to this eager backend, not a universal canonical trig rule."""
    device = torch.device(device)
    contract = dict(replay_rule_version="qwen3-eager-fp32-rope-rms-bf16-v1",
        numerical_scope="same recorded eager backend; not universal canonical trigonometry",
        torch_version=str(torch.__version__), torch_git_version=getattr(torch.version, "git_version", None),
        cuda_build=getattr(torch.version, "cuda", None), hip_build=getattr(torch.version, "hip", None),
        build_configuration_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        device_type=device.type, deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if device.type == "cuda":
        contract.update(device_name=torch.cuda.get_device_name(device),
                        device_capability=list(torch.cuda.get_device_capability(device)))
    else:
        capability = torch.backends.cpu.get_cpu_capability() if device.type == "cpu" else "unknown"
        contract.update(device_name=platform.processor() or platform.machine(), device_capability=capability,
                        CPU_threads=torch.get_num_threads() if device.type == "cpu" else None)
    return contract


def _check_execution_contract(state):
    if state.rope_state and _json(_execution_contract(state.device)) != _json(state.execution_contract):
        raise ValueError("frontier replay execution contract changed; refit/re-encode on this backend")


def make_state(spec, models, context, device="cpu"):
    known = {row["name"]: row for row in recipes()}
    if spec.get("name") not in known or known[spec["name"]] != spec:
        raise ValueError("unknown or modified frontier recipe")
    if spec["name"] in _CONTROLS:
        return producer.make_state(spec, models, context, device)
    source = spec["source"]
    key = "cost_" + source
    if key not in models:
        raise ValueError("source-specific TRAIN cost model missing: " + key)
    model = models[key]
    expected = {"final": "k", "pre_rope": "k_pre_rope", "kproj": "k_proj"}[source]
    if model.config.get("source_k") != expected or model.config.get("source_v") != "v":
        raise ValueError("TRAIN cost-model source differs from frontier source")
    base_spec = next(row for row in producer.recipes() if row["name"] == "producer_frozen_exponent")
    cost = producer.make_state(base_spec, {"frozen": model}, context, device)
    replay = None
    if source != "final":
        name = "producer_pre_rope_coset" if source == "pre_rope" else "producer_kproj_norm_rope_coset"
        replay_spec = next(row for row in producer.recipes() if row["name"] == name)
        replay = producer.make_state(replay_spec, models, context, device)
    base_bytes, replay_bytes = cost.wire_bytes, replay.wire_bytes if replay else b""
    execution = _execution_contract(cost.device) if replay else None
    descriptor = _json(dict(format="frontier_complete_state_v1", source=source,
                            cost_state_id=cost.state_id, replay_state_id=replay.state_id if replay else None,
                            replay_execution_contract=execution))
    wire = (b"FRONTIER_STATE_V1\0" + struct.pack("<QQQ", len(descriptor), len(base_bytes), len(replay_bytes))
            + descriptor + base_bytes + replay_bytes)
    return FrontierState(dict(spec), cost, replay, wire, hashlib.sha256(wire).hexdigest(), execution)


def prepare_record(record, device, spec):
    if spec["name"] in _CONTROLS:
        return producer.prepare_record(record, device, spec)
    kind = {"final": "kv_median", "pre_rope": "pre_rope", "kproj": "kproj_norm_rope"}[spec["source"]]
    return producer.prepare_record(record, device, dict(kind=kind))


def _prediction(state, layer, label, shape, head_dim, shifts=None):
    tokens, channels = shape
    base = producer._param(state.cost_state, layer, "base_" + label).to(torch.int64)
    if base.numel() != channels:
        raise ValueError("frozen cost base differs from source channels")
    if shifts is not None:
        if channels % head_dim or shifts.numel() != channels // head_dim:
            raise ValueError("frontier head-shift geometry mismatch")
        base = (base + shifts.to(torch.int64).repeat_interleave(head_dim)).clamp(0, 255)
    return base[None, :].expand(tokens, channels)


def _compact(fields, description, label):
    count = description["shape"][0] * description["shape"][1]
    segment = description["segment_size"]
    last_start = ((count - 1) // segment) * segment if count else 0
    bits = 16 if last_start <= 65535 else 32
    dtype = torch.uint16 if bits == 16 else torch.uint32
    for suffix in ("secondary_offsets", "rare_offsets"):
        name = label + "__" + suffix
        # clone is essential for the uint32 fallback: retaining a view would
        # keep the omitted terminal's actual backing allocation alive.
        starts = fields[name].view(torch.uint32)[:-1].to(dtype=dtype, copy=True).contiguous()
        fields[name] = g._wire(starts)
    description.update(offset_storage_bits=bits, offset_starts_only=True)


def _component_size(fields, description):
    descriptor = dict(description=description,
                      field_bytes={name: tensor.numel() * tensor.element_size() for name, tensor in fields.items()})
    return g._storage_bytes(fields.values()) + len(_json(descriptor))


def _head_shift(words, state, layer, label, head_dim):
    tokens, channels = words.shape
    if head_dim < 1 or channels % head_dim:
        raise ValueError("head shifts require actual integer head geometry")
    candidates = [0] + [value for magnitude in range(1, 17) for value in (-magnitude, magnitude)]
    offsets = torch.tensor(candidates, device=words.device, dtype=torch.int64)
    base = producer._param(state.cost_state, layer, "base_" + label).to(torch.int64)
    pred = (base.reshape(channels // head_dim, head_dim, 1) + offsets).clamp(0, 255)
    actual = ((words.to(torch.int64) >> 7) & 255).reshape(tokens, channels // head_dim, head_dim, 1)
    zigzag = g._zigzag(actual - pred[None, ...])
    score = (2 + 3 * (zigzag >= 3).to(torch.int64) + 9 * (zigzag >= 10).to(torch.int64)).sum(dim=(0, 2))
    return offsets[score.argmin(dim=1)].to(torch.int8)


def _encode_side(words, state, layer, label, head_dim, spec, fields):
    from . import frontier_lower
    integer = words.to(torch.int64)
    lower = (((integer >> 8) & 128) | (integer & 127)).to(torch.uint8)
    raw_lower = frontier_lower.encode(lower, mode="raw8")
    requested_lower = (frontier_lower.encode(lower, mode=spec["lower_mode"])
                       if spec["lower_mode"] != "raw8" else raw_lower)

    def primary_plan(part, description):
        description.update(primary_storage="fixed2", primary_descriptor=None)
        if spec["primary_mode"] != "auto_plane":
            return part, description
        count = words.numel()
        primary = g._unpack_fixed(part[label + "__primary"], count, 2)
        canonical = exponent._canonical(primary.to(torch.uint8), list(words.shape), "time")
        plane_fields, plane_meta = frontier_lower.encode(canonical, mode="plane_raw")
        candidate = {name: value for name, value in part.items() if name != label + "__primary"}
        candidate.update({label + "__primary__" + name: value for name, value in plane_fields.items()})
        candidate_description = dict(description, primary_storage="bitplanes", primary_descriptor=plane_meta)
        if _component_size(candidate, candidate_description) < _component_size(part, description):
            return candidate, candidate_description
        return part, description

    def lower_plan(part, description):
        options = []
        for (low_fields, low_meta), fallback in ((raw_lower, spec["lower_mode"] != "raw8"),
                                                 (requested_lower, False)):
            candidate = dict(part)
            candidate.update({label + "__lower__" + name: value for name, value in low_fields.items()})
            candidate_description = dict(description, lower=low_meta,
                lower_requested_mode=spec["lower_mode"], lower_complete_fallback_to_raw=fallback)
            options.append((candidate, candidate_description))
        # This compares the COMPLETE side fields and their field_bytes entries,
        # not only the child codec's locally calculated descriptor.
        return min(options, key=lambda option: _component_size(*option))

    def encode_exponents(shifts):
        part = {}
        description = producer._encode_predicted(words,
            _prediction(state, layer, label, list(words.shape), head_dim, shifts), part, label)
        part.pop(label + "__lower")
        _compact(part, description, label)
        description["head_shift_applied"] = shifts is not None
        if shifts is not None:
            part[label + "__head_shift"] = g._wire(shifts)
        part, description = primary_plan(part, description)
        return lower_plan(part, description)
    chosen, description = encode_exponents(None)
    if spec["head_shifts"] and words.numel():
        candidate = _head_shift(words, state, layer, label, head_dim)
        shifted, shifted_description = encode_exponents(candidate)
        # Compare all real fields and their descriptor bytes, including the
        # shift tensor and field-byte entry. Ties retain the zero-shift plan.
        if _component_size(shifted, shifted_description) < _component_size(chosen, description):
            chosen, description = shifted, shifted_description
    fields.update(chosen)
    return description


def _decode_side(description, fields, label, state, layer, head_dim):
    from . import frontier_lower
    count = description["shape"][0] * description["shape"][1]
    segments = (count + description["segment_size"] - 1) // description["segment_size"]
    bits = description["offset_storage_bits"]
    if bits not in (16, 32) or not description.get("offset_starts_only"):
        raise ValueError("invalid compact frontier index contract")
    suffixes = ("secondary", "rare")
    restored = {label + "__" + name: fields[label + "__" + name] for name in suffixes}
    if description["primary_storage"] == "fixed2":
        primary = g._unpack_fixed(fields[label + "__primary"], count, 2)
    elif description["primary_storage"] == "bitplanes":
        prefix = label + "__primary__"
        parts = {name[len(prefix):]: value for name, value in fields.items() if name.startswith(prefix)}
        primary = frontier_lower.decode(parts, description["primary_descriptor"])
        if list(primary.shape) != description["shape"]:
            raise ValueError("frontier primary-plane shape mismatch")
        primary = exponent._ordered(primary, "time").to(torch.int64)
    else:
        raise ValueError("unknown frontier primary storage")
    dtype = torch.uint16 if bits == 16 else torch.uint32
    for suffix, terminal in (("secondary_offsets", "secondary_count"), ("rare_offsets", "rare_count")):
        array = fields[label + "__" + suffix].view(dtype).to(torch.int64)
        if array.numel() != segments:
            raise ValueError("frontier compact index length mismatch")
        array = torch.cat((array, torch.tensor([description[terminal]], device=array.device, dtype=torch.int64))).to(torch.uint32)
        restored[label + "__" + suffix] = g._wire(array)
    prefix = label + "__lower__"
    low_fields = {name[len(prefix):]: value for name, value in fields.items() if name.startswith(prefix)}
    lower = frontier_lower.decode(low_fields, description["lower"])
    if list(lower.shape) != description["shape"] or lower.dtype != torch.uint8:
        raise ValueError("frontier lower-byte geometry mismatch")
    shifts = fields[label + "__head_shift"].view(torch.int8) if description["head_shift_applied"] else None
    prediction = _prediction(state, layer, label, description["shape"], head_dim, shifts)
    second = g._unpack_fixed(restored[label + "__secondary"], description["secondary_count"], 3)
    rare = g._unpack_fixed(restored[label + "__rare"], description["rare_count"], 9)
    mask = primary == 3
    positions = exponent._segment_ranks(mask, restored[label + "__secondary_offsets"].view(torch.uint32), description["segment_size"])
    selected = second[positions[mask]]
    rare_mask = selected == 7
    full_rare = torch.zeros(count, device=primary.device, dtype=torch.bool)
    full_rare[mask] = rare_mask
    positions = exponent._segment_ranks(full_rare, restored[label + "__rare_offsets"].view(torch.uint32), description["segment_size"])
    correction = selected + 3
    correction[rare_mask] = rare[positions[full_rare]]
    primary[mask] = correction
    exponents = prediction + exponent._canonical(g._unzigzag(primary), description["shape"], "time")
    lower = lower.reshape(-1).to(torch.int64)
    return (((lower & 128) << 8) | (lower & 127) | (exponents.reshape(-1) << 7)).to(torch.uint16).reshape(description["shape"])


def encode(record, spec, state):
    if spec["name"] in _CONTROLS:
        return producer.encode(record, spec, state)
    if spec != state.spec:
        raise ValueError("frontier recipe and prepared state differ")
    _check_execution_contract(state)
    record = prepare_record(record, state.device, spec)
    k, v = record["k"], record["v"]
    if (k.shape != v.shape or len(k) != record["tokens"] or record["start"] < 0
            or record["head_dim"] < 1 or k.shape[1] % record["head_dim"]):
        raise ValueError("invalid frontier pair geometry")
    source = spec["source"]
    source_k = k if source == "final" else record["k_pre_rope" if source == "pre_rope" else "k_proj"]
    if source_k.shape != k.shape:
        raise ValueError("frontier K source does not match final target")
    fields = {}
    layer, tokens, channels = int(record["layer"]), len(k), k.shape[1]
    descriptions = {label: _encode_side(words, state, layer, label, record["head_dim"], spec, fields)
                    for label, words in (("K", source_k), ("V", v))}
    correction = dict(mode="direct", count=k.numel(), nbits=0, groups=0)
    if source != "final":
        predicted = producer._rope(source_k, state.rope_state, int(record["start"]),
                                   int(record["head_dim"]), layer, normalize=source == "kproj")
        if torch.equal(predicted.to(torch.int32), k.to(torch.int32)):
            correction["mode"] = "verified_zero"
        else:
            correction = producer._encode_full_correction(k, predicted, fields, "K_replay", "coset")
            correction["mode"] = "coset_fallback"
    metadata = dict(version=1, method="frontier_pair", shape=[tokens, 2 * channels],
        kv_shape=[tokens, channels], recipe=spec["name"], source=source, lower_mode=spec["lower_mode"],
        head_shifts=spec["head_shifts"], state_id=state.state_id, layer=layer,
        primary_mode=spec["primary_mode"],
        start=int(record["start"]), head_dim=int(record["head_dim"]),
        encoded_sides=descriptions, replay_correction=correction)
    return g._finish(fields, metadata, state.device)


def decode(packet, state):
    if packet.metadata.get("method") == "producer_pair":
        return producer.decode(packet, state)
    g.validate_packet(packet, check_header=not packet._header_verified)
    meta, fields = packet.metadata, packet.fields
    if (not isinstance(state, FrontierState) or meta["method"] != "frontier_pair"
            or meta["state_id"] != state.state_id or meta["recipe"] != state.spec["name"]
            or meta["source"] != state.spec["source"] or meta["lower_mode"] != state.spec["lower_mode"]
            or meta["head_shifts"] != state.spec["head_shifts"] or meta["primary_mode"] != state.spec["primary_mode"]
            or packet.header.device != state.device):
        raise ValueError("frontier packet/state recipe or identity mismatch")
    _check_execution_contract(state)
    shape = meta["kv_shape"]
    if meta["shape"] != [shape[0], 2 * shape[1]] or set(meta["encoded_sides"]) != {"K", "V"}:
        raise ValueError("frontier pair shape mismatch")
    if any(description["shape"] != shape for description in meta["encoded_sides"].values()):
        raise ValueError("frontier side shape mismatch")
    k, v = (_decode_side(meta["encoded_sides"][label], fields, label, state, meta["layer"], meta["head_dim"])
            for label in ("K", "V"))
    correction = meta["replay_correction"]
    if correction["count"] != shape[0] * shape[1]:
        raise ValueError("frontier replay correction count mismatch")
    if meta["source"] == "final":
        if correction["mode"] != "direct":
            raise ValueError("direct frontier recipe has replay correction")
        return k, v
    prediction = producer._rope(k, state.rope_state, meta["start"], meta["head_dim"],
                               meta["layer"], normalize=meta["source"] == "kproj")
    if correction["mode"] == "verified_zero":
        if any(name.startswith("K_replay__") for name in fields):
            raise ValueError("zero replay mode has hidden correction payload")
        return prediction, v
    if correction["mode"] != "coset_fallback" or correction.get("correction") != "coset":
        raise ValueError("invalid frontier replay correction mode")
    return producer._apply_full_correction(prediction, fields, "K_replay", correction), v


def describe(packet):
    if packet.metadata["method"] == "producer_pair":
        return producer.describe(packet)
    meta, fields = packet.metadata, packet.fields
    return dict(recipe=meta["recipe"], source=meta["source"], lower_mode=meta["lower_mode"],
        primary_mode=meta["primary_mode"],
        header_bytes=packet.header.numel(), replay_correction_mode=meta["replay_correction"]["mode"],
        head_shift_bytes=sum(value.numel() for name, value in fields.items() if name.endswith("__head_shift")),
        compact_index_bytes=sum(value.numel() for name, value in fields.items() if name.endswith("_offsets")),
        replay_correction_data_bits=meta["replay_correction"]["nbits"],
        side_diagnostics={label: dict(head_shift_applied=description["head_shift_applied"],
            offset_storage_bits=description["offset_storage_bits"],
            secondary_escape_count=description["secondary_count"], rare_escape_count=description["rare_count"],
            primary_storage=description["primary_storage"],
            lower_storage=description["lower"].get("storage"), lower_requested_mode=description["lower_requested_mode"],
            lower_complete_fallback_to_raw=description["lower_complete_fallback_to_raw"])
            for label, description in meta["encoded_sides"].items()},
        original_hidden_stored=False, full_RoPE_table_stored=False,
        shift_selection="TRAIN bases and candidate search use logical 2/3/9 tier cost; final whole-side physical bytes including primary/lower coding, shifts and descriptors must beat zero")

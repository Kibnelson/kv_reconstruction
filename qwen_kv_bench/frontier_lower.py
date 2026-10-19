"""Exact device-tensor coding of BF16 sign plus seven fraction bits.

The byte stream is complementary to exponent coding.  Three bitplane variants
use no prediction, XOR with the first token, or XOR with the previous token.
Each 256-bit group stores a 2-bit mode: raw, zero, one, or minority positions.
Minority groups additionally store a 9-bit count, 1-bit minority value and
8-bit positions. Short terminal groups encode only their actual valid bits.

All payload operations use PyTorch tensors on the input device. Dynamic boolean
selection can synchronize GPU allocation; this eager reference is not a fused
kernel. Descriptor sizes and tensor lengths are host metadata, never host reads
of byte values. A complete raw-byte fallback includes every lower byte.
"""
from __future__ import annotations

import json
import math

import torch

from . import gpu_ops as g


MODES = ("raw8", "plane_raw", "plane_first", "plane_prev")
GROUP_BITS = 256
_FORMAT = "frontier_lower_bitplanes_v1"
_PLANES = 8


def _descriptor_size(meta):
    return len(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())


def _field_bytes(fields):
    return sum(tensor.numel() * tensor.element_size() for tensor in fields.values())


def _raw_meta(shape, mode):
    return dict(format=_FORMAT, version=1, mode=mode, storage="raw8", shape=list(shape))


def _lengths(count, device):
    per_plane = (count + GROUP_BITS - 1) // GROUP_BITS
    length = (count - GROUP_BITS * torch.arange(per_plane, device=device,
        dtype=torch.int64)).clamp(0, GROUP_BITS)
    return length.repeat(_PLANES)


def _packed(values, width):
    return g._pack_fixed(values, width)[0]


def _tensor_input(lower):
    if not isinstance(lower, torch.Tensor) or lower.dtype != torch.uint8 or lower.ndim != 2:
        raise ValueError("Expected a torch.uint8 [tokens,channels] lower-byte tensor")
    return lower


def encode(lower, mode="raw8"):
    """Return actual byte fields and a JSON-only authoritative descriptor.

    ``plane_raw`` means unpredicted bitplanes, not forced raw groups. Every
    requested plane mode can select a complete raw8 fallback if its actual
    fields plus its own descriptor would be at least as large. The enclosing
    packet must still charge its full final header and field indexing costs.
    """
    lower = _tensor_input(lower)
    if mode not in MODES:
        raise ValueError("Unsupported lower-byte mode")
    raw_meta = _raw_meta(lower.shape, mode)
    if mode == "raw8" or lower.numel() == 0:
        return {"lower_raw": lower.contiguous().reshape(-1).clone()}, raw_meta
    tokens, channels = lower.shape
    fields = {}
    if mode in ("plane_first", "plane_prev"):
        fields["lower_first_row"] = lower[0].contiguous().clone()
        residual = lower[1:] ^ (lower[:1] if mode == "plane_first" else lower[:-1])
    else:
        residual = lower
    # Each channel's adjacent token rows remain adjacent before bit slicing.
    ordered = residual.T.contiguous().reshape(-1).to(torch.int64)
    count = ordered.numel()
    groups_per_plane = (count + GROUP_BITS - 1) // GROUP_BITS
    shifts = torch.arange(_PLANES, device=lower.device, dtype=torch.int64)
    planes = ((ordered[None, :] >> shifts[:, None]) & 1).to(torch.uint8)
    padding = (-count) % GROUP_BITS
    padded = torch.cat((planes, torch.zeros((_PLANES, padding), device=lower.device,
        dtype=torch.uint8)), dim=1)
    grouped = padded.reshape(-1, GROUP_BITS)
    lengths = _lengths(count, lower.device)
    positions = torch.arange(GROUP_BITS, device=lower.device, dtype=torch.int64)
    valid = positions[None, :] < lengths[:, None]
    ones = grouped.sum(dim=1, dtype=torch.int64)
    zeros = lengths - ones
    minority_counts = torch.minimum(ones, zeros)
    # Tie breaks to minority ones; a 128/128 group always selects raw anyway.
    minority_values = (ones <= zeros).to(torch.uint8)
    use_minority = 10 + 8 * minority_counts < lengths
    modes = torch.where(ones == 0, 1, torch.where(zeros == 0, 2,
        torch.where(use_minority, 3, 0))).to(torch.uint8)
    minority_groups = modes == 3
    raw_groups = modes == 0
    raw_bits = grouped[raw_groups[:, None] & valid]
    minority_mask = (minority_groups[:, None] & valid &
        (grouped == minority_values[:, None]))
    minority_positions = positions[None, :].expand(grouped.shape[0], -1)[minority_mask].to(torch.uint8)
    selected_counts = minority_counts[minority_groups]
    selected_values = minority_values[minority_groups]
    fields.update(lower_modes=_packed(modes, 2), lower_raw_bits=_packed(raw_bits, 1),
        lower_minority_counts=_packed(selected_counts, 9),
        lower_minority_values=_packed(selected_values, 1),
        lower_minority_positions=minority_positions.contiguous())
    # Counts obtained from tensor shapes, with no .cpu/.item/.numpy value read.
    mode_counts = {"raw": raw_groups.nonzero().numel(),
        "zero": (modes == 1).nonzero().numel(), "one": (modes == 2).nonzero().numel(),
        "minority": selected_counts.numel()}
    meta = dict(format=_FORMAT, version=1, mode=mode, storage="adaptive_bitplanes",
        shape=list(lower.shape), residual_shape=list(residual.shape), order="channel_then_time",
        plane_count=_PLANES, group_bits=GROUP_BITS, groups_per_plane=groups_per_plane,
        group_count=modes.numel(), value_count=count, raw_bit_count=raw_bits.numel(),
        minority_group_count=selected_counts.numel(), minority_position_count=minority_positions.numel(),
        mode_counts=mode_counts, mode_width_bits=2, minority_count_width_bits=9,
        minority_value_width_bits=1, minority_position_width_bits=8,
        bit_order="least-significant bit first; planes 0 through 7",
        count_selection="minority count; ties choose ones; strict bit-cost improvement over raw group")
    candidate_cost = _field_bytes(fields) + _descriptor_size(meta)
    raw_cost = lower.numel() + _descriptor_size(raw_meta)
    if candidate_cost >= raw_cost:
        return {"lower_raw": lower.contiguous().reshape(-1).clone()}, raw_meta
    return fields, meta


def _validate(fields, meta):
    if not isinstance(meta, dict) or meta.get("format") != _FORMAT or meta.get("version") != 1 or \
            meta.get("mode") not in MODES or meta.get("storage") not in ("raw8", "adaptive_bitplanes"):
        raise ValueError("Invalid lower-byte descriptor")
    shape = meta.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape):
        raise ValueError("Invalid lower-byte shape")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Missing lower-byte fields")
    device = next(iter(fields.values())).device
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or
            not value.is_contiguous() or value.device != device for value in fields.values()):
        raise ValueError("Lower-byte fields must be flat contiguous uint8 tensors on one device")
    if meta["storage"] == "raw8":
        if set(fields) != {"lower_raw"} or fields["lower_raw"].numel() != math.prod(shape):
            raise ValueError("Raw lower-byte field size mismatch")
        return device
    mode = meta["mode"]
    if mode == "raw8":
        raise ValueError("Raw8 mode cannot have adaptive bitplanes")
    residual_shape = [shape[0] - 1, shape[1]] if mode in ("plane_first", "plane_prev") else shape
    if residual_shape[0] < 0:
        raise ValueError("Predicted lower stream needs a first token")
    count = math.prod(residual_shape)
    per_plane = (count + GROUP_BITS - 1) // GROUP_BITS
    if meta.get("residual_shape") != residual_shape or meta.get("value_count") != count or \
            meta.get("plane_count") != _PLANES or meta.get("group_bits") != GROUP_BITS or \
            meta.get("groups_per_plane") != per_plane or meta.get("group_count") != per_plane * _PLANES or \
            meta.get("order") != "channel_then_time" or meta.get("mode_width_bits") != 2 or \
            meta.get("minority_count_width_bits") != 9 or meta.get("minority_value_width_bits") != 1 or \
            meta.get("minority_position_width_bits") != 8:
        raise ValueError("Lower-byte bitplane geometry mismatch")
    names = {"lower_modes", "lower_raw_bits", "lower_minority_counts", "lower_minority_values", "lower_minority_positions"}
    if mode in ("plane_first", "plane_prev"):
        names.add("lower_first_row")
    if set(fields) != names:
        raise ValueError("Lower-byte field set mismatch")
    for name in ("raw_bit_count", "minority_group_count", "minority_position_count"):
        if not isinstance(meta.get(name), int) or isinstance(meta[name], bool) or meta[name] < 0:
            raise ValueError("Invalid lower-byte stream count")
    if meta["raw_bit_count"] > count * _PLANES or meta["minority_group_count"] > per_plane * _PLANES or \
            meta["minority_position_count"] > meta["minority_group_count"] * (GROUP_BITS // 2):
        raise ValueError("Oversized lower-byte stream count")
    sizes = dict(lower_modes=(per_plane * _PLANES * 2 + 7) // 8,
        lower_raw_bits=(meta["raw_bit_count"] + 7) // 8,
        lower_minority_counts=(meta["minority_group_count"] * 9 + 7) // 8,
        lower_minority_values=(meta["minority_group_count"] + 7) // 8,
        lower_minority_positions=meta["minority_position_count"])
    if "lower_first_row" in names:
        sizes["lower_first_row"] = shape[1]
    if any(fields[name].numel() != size for name, size in sizes.items()):
        raise ValueError("Lower-byte stream field size mismatch")
    return device


def decode(fields, meta):
    """Restore the exact uint8 matrix from fields and descriptor only."""
    device = _validate(fields, meta)
    if meta["storage"] == "raw8":
        return fields["lower_raw"].reshape(meta["shape"]).clone()
    count = meta["value_count"]
    modes = g._unpack_fixed(fields["lower_modes"], meta["group_count"], 2)
    lengths = _lengths(count, device)
    positions = torch.arange(GROUP_BITS, device=device, dtype=torch.int64)
    valid = positions[None, :] < lengths[:, None]
    grouped = (modes == 2).to(torch.int64)[:, None].expand(-1, GROUP_BITS).clone()
    raw_mask = (modes == 0)[:, None] & valid
    raw = g._unpack_fixed(fields["lower_raw_bits"], meta["raw_bit_count"], 1)
    grouped[raw_mask] = raw
    minority_ids = (modes == 3).nonzero().reshape(-1)
    if minority_ids.numel() != meta["minority_group_count"]:
        raise ValueError("Lower-byte minority mode count mismatch")
    counts = g._unpack_fixed(fields["lower_minority_counts"], meta["minority_group_count"], 9)
    values = g._unpack_fixed(fields["lower_minority_values"], meta["minority_group_count"], 1)
    grouped[minority_ids] = 1 - values[:, None]
    group_ids = minority_ids.repeat_interleave(counts)
    per_position_values = values.repeat_interleave(counts)
    if group_ids.numel() != meta["minority_position_count"]:
        raise ValueError("Lower-byte minority positions/count mismatch")
    grouped[group_ids, fields["lower_minority_positions"].to(torch.int64)] = per_position_values
    planes = grouped.reshape(_PLANES, -1)[:, :count]
    shifts = torch.arange(_PLANES, device=device, dtype=torch.int64)
    ordered = (planes << shifts[:, None]).sum(dim=0).to(torch.uint8)
    rows, channels = meta["residual_shape"]
    residual = ordered.reshape(channels, rows).T.contiguous()
    if meta["mode"] == "plane_raw":
        return residual
    anchor = fields["lower_first_row"].reshape(1, channels)
    if meta["mode"] == "plane_first":
        restored = residual ^ anchor
    else:
        # XOR prefix scan is independent per channel and bit. Integer cumsum
        # parity avoids a Python loop over tokens and preserves every bit.
        bits = (residual.to(torch.int64)[..., None] >> shifts) & 1
        parity = bits.cumsum(dim=0) & 1
        prefix = (parity << shifts).sum(dim=-1).to(torch.uint8)
        restored = prefix ^ anchor
    return torch.cat((anchor, restored), dim=0)

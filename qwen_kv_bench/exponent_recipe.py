"""Exact prefill exponent recipes with bounded, independently indexed tiers.

Each original BF16 word contributes an unchanged sign/fraction byte.  Its
eight-bit exponent is predicted using this tile alone, then the signed error is
zigzag coded into 2-bit primary, 3-bit secondary and 9-bit rare streams.  The
secondary and rare streams are compact; uint32 count-prefix offsets identify
the part belonging to each fixed-size segment.  Offset and byte padding costs
are actual packet fields, never omitted from the measured size.

This is an eager tensor reference, not a fused CUDA kernel.  It executes on the
input device and never transfers exponent values to the host.  Boolean-index
output allocation may synchronize CUDA.  A later kernel can launch independent
segments using the stored offsets, while this reference decodes all segments
with tensorized prefix ranks.  Median means floor(mean of the two central
exponents), matching NumPy median(...).astype(uint8), including even lengths.
"""
from __future__ import annotations

import math

import torch


_PREDICTORS = ("zero", "fixed127", "median", "affine")
_LAYOUTS = ("channel", "time")
_TIER_NAMES = ("exponent_primary", "exponent_secondary", "exponent_rare")


def options():
    """Matched unpredicted control plus the two metadata-derived predictors."""
    configurations = [("zero", "channel"), ("fixed127", "channel"),
                      ("fixed127", "time"), ("median", "channel"),
                      ("median", "time"), ("affine", "channel"), ("affine", "time")]
    return [dict(name=f"native_prefill_exponent_{predictor}_{layout}_tiered",
                 method="prefill_exponent", options=dict(predictor=predictor,
                 layout=layout, segment_size=256))
            for predictor, layout in configurations]


def _ordered(values, layout):
    return (values if layout == "channel" else values.T).contiguous().reshape(-1)


def _canonical(values, shape, layout):
    tokens, channels = shape
    return (values.reshape(tokens, channels) if layout == "channel"
            else values.reshape(channels, tokens).T.contiguous())


def _prediction(exponents, fields, predictor):
    tokens, channels = exponents.shape
    if exponents.numel() == 0 or predictor == "zero":
        return torch.zeros_like(exponents)
    if predictor == "fixed127":
        # Source-independent exponent-bias control: no prefill information.
        return torch.full_like(exponents, 127)
    if predictor == "median":
        ordered = torch.sort(exponents, dim=0).values
        anchor = torch.div(ordered[(tokens - 1) // 2] + ordered[tokens // 2],
                           2, rounding_mode="floor")
        fields["exponent_anchor"] = anchor.to(torch.uint8).contiguous()
        return anchor[None, :].expand(tokens, channels)
    first, last = exponents[0], exponents[-1]
    fields["exponent_first"] = first.to(torch.uint8).contiguous()
    fields["exponent_last"] = last.to(torch.uint8).contiguous()
    rows = torch.arange(tokens, device=exponents.device, dtype=torch.int64)[:, None]
    return first[None, :] + torch.div((last - first)[None, :] * rows,
                                      max(tokens - 1, 1), rounding_mode="floor")


def _restore_prediction(meta, fields):
    tokens, channels = meta["shape"]
    device = fields["sign_mantissa"].device
    if tokens * channels == 0 or meta["predictor"] == "zero":
        return torch.zeros((tokens, channels), device=device, dtype=torch.int64)
    if meta["predictor"] == "fixed127":
        return torch.full((tokens, channels), 127, device=device, dtype=torch.int64)
    if meta["predictor"] == "median":
        return fields["exponent_anchor"].to(torch.int64)[None, :].expand(tokens, channels)
    first = fields["exponent_first"].to(torch.int64)
    last = fields["exponent_last"].to(torch.int64)
    rows = torch.arange(tokens, device=device, dtype=torch.int64)[:, None]
    return first[None, :] + torch.div((last - first)[None, :] * rows,
                                      max(tokens - 1, 1), rounding_mode="floor")


def _segment_offsets(mask, segment_size):
    padding = (-mask.numel()) % segment_size
    padded = torch.cat((mask, torch.zeros(padding, device=mask.device, dtype=torch.bool)))
    counts = padded.reshape(-1, segment_size).sum(dim=1, dtype=torch.int64)
    return torch.cat((torch.zeros(1, device=mask.device, dtype=torch.int64),
                      counts.cumsum(0))).to(torch.uint32)


def _segment_ranks(mask, offsets, segment_size):
    """Positions in a compact stream, calculated independently per segment."""
    count = mask.numel()
    padding = (-count) % segment_size
    padded = torch.cat((mask, torch.zeros(padding, device=mask.device, dtype=torch.bool)))
    local = padded.reshape(-1, segment_size).cumsum(1, dtype=torch.int64) - 1
    return (local + offsets[:-1].to(torch.int64)[:, None]).reshape(-1)[:count]


def encode(words, *, predictor="median", layout="time", segment_size=256, key="global"):
    """Create a self-contained exact packet on ``words.device`` from [T,C] words."""
    from . import gpu_ops as g

    if words.dtype != torch.uint16 or words.ndim != 2:
        raise ValueError("prefill exponent recipe expects uint16[T,C] word bits")
    if predictor not in _PREDICTORS or layout not in _LAYOUTS:
        raise ValueError("unsupported prefill exponent predictor or layout")
    if not isinstance(segment_size, int) or isinstance(segment_size, bool) or not 1 <= segment_size <= 65536:
        raise ValueError("segment_size must be an integer in 1..65536")
    if words.numel() > 0xffffffff:
        raise ValueError("uint32 segment offsets support at most 2**32-1 values per packet")
    integer = words.to(torch.int64)
    exponent = (integer >> 7) & 255
    fields = {"sign_mantissa": (((integer >> 8) & 128) | (integer & 127)).to(torch.uint8).reshape(-1).contiguous()}
    prediction = _prediction(exponent, fields, predictor)
    zigzag = g._zigzag(_ordered(exponent - prediction, layout))
    primary = zigzag.clamp_max(3)
    secondary = (zigzag[zigzag >= 3] - 3).clamp_max(7)
    rare = zigzag[zigzag >= 10]
    for name, values, width in zip(_TIER_NAMES, (primary, secondary, rare), (2, 3, 9)):
        fields[name], _ = g._pack_fixed(values, width)
    fields["secondary_offsets"] = g._wire(_segment_offsets(zigzag >= 3, segment_size))
    fields["rare_offsets"] = g._wire(_segment_offsets(zigzag >= 10, segment_size))
    metadata = dict(version=1, method="prefill_exponent", state_id="no_learned_state",
                    shape=list(words.shape), key=str(key), predictor=predictor, layout=layout,
                    segment_size=segment_size, correction="tiered_2_3_9",
                    median_rule="floor_average_of_two_central_exponents",
                    secondary_count=secondary.numel(), rare_count=rare.numel(),
                    segment_count=(zigzag.numel() + segment_size - 1) // segment_size)
    return g._finish(fields, metadata, words.device)


def _validate_contract(meta, fields):
    if (meta.get("method") != "prefill_exponent" or meta.get("state_id") != "no_learned_state"
            or meta.get("predictor") not in _PREDICTORS or meta.get("layout") not in _LAYOUTS
            or meta.get("correction") != "tiered_2_3_9"
            or meta.get("median_rule") != "floor_average_of_two_central_exponents"):
        raise ValueError("invalid prefill exponent recipe descriptor")
    size = meta.get("segment_size")
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 65536:
        raise ValueError("invalid exponent segment size")
    count = math.prod(meta["shape"])
    second, rare = meta.get("secondary_count"), meta.get("rare_count")
    if (not isinstance(second, int) or not isinstance(rare, int)
            or not 0 <= rare <= second <= count):
        raise ValueError("invalid exponent tier counts")
    segments = (count + size - 1) // size
    if meta.get("segment_count") != segments:
        raise ValueError("invalid exponent segment count")
    expected = dict(sign_mantissa=count, exponent_primary=(2 * count + 7) // 8,
                    exponent_secondary=(3 * second + 7) // 8, exponent_rare=(9 * rare + 7) // 8,
                    secondary_offsets=4 * (segments + 1), rare_offsets=4 * (segments + 1))
    if count and meta["predictor"] == "median":
        expected["exponent_anchor"] = meta["shape"][1]
    if count and meta["predictor"] == "affine":
        expected.update(exponent_first=meta["shape"][1], exponent_last=meta["shape"][1])
    if set(fields) != set(expected) or any(fields[name].numel() != n for name, n in expected.items()):
        raise ValueError("invalid exponent recipe field geometry")


def decode(packet):
    """Materialize exact BF16 words; no KV payload or header D2H per decode."""
    from . import gpu_ops as g

    g.validate_packet(packet, check_header=not packet._header_verified)
    meta, fields = packet.metadata, packet.fields
    _validate_contract(meta, fields)
    count = math.prod(meta["shape"])
    primary = g._unpack_fixed(fields["exponent_primary"], count, 2)
    secondary = g._unpack_fixed(fields["exponent_secondary"], meta["secondary_count"], 3)
    rare = g._unpack_fixed(fields["exponent_rare"], meta["rare_count"], 9)
    escape = primary == 3
    secondary_positions = _segment_ranks(escape, fields["secondary_offsets"].view(torch.uint32), meta["segment_size"])
    second_values = secondary[secondary_positions[escape]]
    rare_mask = second_values == 7
    rare_positions_mask = torch.zeros(count, device=primary.device, dtype=torch.bool)
    rare_positions_mask[escape] = rare_mask
    rare_positions = _segment_ranks(rare_positions_mask, fields["rare_offsets"].view(torch.uint32), meta["segment_size"])
    corrections = second_values + 3
    corrections[rare_mask] = rare[rare_positions[rare_positions_mask]]
    primary[escape] = corrections
    error = _canonical(g._unzigzag(primary), meta["shape"], meta["layout"])
    exponent = (_restore_prediction(meta, fields) + error).reshape(-1)
    lower = fields["sign_mantissa"].to(torch.int64)
    return (((lower & 128) << 8) | (lower & 127) | (exponent << 7)).to(torch.uint16).reshape(meta["shape"])


def describe(packet):
    """Actual stored components and escape rates, without reading device data."""
    meta, fields = packet.metadata, packet.fields
    _validate_contract(meta, fields)
    count = math.prod(meta["shape"])
    second, rare = meta["secondary_count"], meta["rare_count"]
    predictor_bytes = sum(fields[name].numel() for name in
                          ("exponent_anchor", "exponent_first", "exponent_last") if name in fields)
    scope = ("source_independent_constant_exponent_control" if meta["predictor"] in ("zero", "fixed127")
             else "tile_local_per_channel_exponents_during_prefill")
    return dict(predictor=meta["predictor"], layout=meta["layout"], correction=meta["correction"],
                prediction_scope=scope,
                external_state_bytes=0, predictor_payload_bytes=predictor_bytes,
                sign_mantissa_bytes=count, correction_payload_bytes=sum(fields[name].numel() for name in _TIER_NAMES),
                segment_offset_bytes=fields["secondary_offsets"].numel() + fields["rare_offsets"].numel(),
                header_bytes=packet.header.numel(), secondary_escape_count=second, rare_escape_count=rare,
                secondary_escape_fraction=second / count if count else 0.0,
                rare_escape_fraction=rare / count if count else 0.0,
                correction_bits_before_byte_padding=2 * count + 3 * second + 9 * rare,
                symbol_domain="signed noncircular differences of 8-bit BF16 exponent fields")

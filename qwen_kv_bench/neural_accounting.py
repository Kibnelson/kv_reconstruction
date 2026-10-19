"""Byte accounting for the neural KV reconstruction study.

Only the sampled raw and actual packet byte counts are measurements. Scaling
that packet/raw ratio to a full cache is an ESTIMATE. This module neither
measures allocator overhead nor establishes a GPU capacity or serving benefit.
Neural and anchor decoder state are each charged in full as the larger of
their serialized and physical resident sizes. Shared-state arithmetic assumes
the same compatible decoder state can serve every counted context.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Integral


def _bytes(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer byte count")
    value = int(value)
    if value < (1 if positive else 0):
        rule = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {rule}")
    return value


def _fraction(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite numeric value")
    try:
        # Decimal spelling makes the requested 0.4 exactly 2/5, rather than
        # importing the binary floating-point approximation into boundaries.
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric value") from error
    return result


def _exact(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _number(value):
    """Keep integral byte results integral in the JSON-compatible report."""
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else float(value)


def _ratio_fields(reduction):
    reduction = Fraction(reduction)
    return {
        "net_reduction_fraction": float(reduction),
        "net_reduction_pct": float(100 * reduction),
        "net_reduction_exact": _exact(reduction),
    }


def break_even_contexts(raw_cache_bytes, estimated_packet_bytes, state_bytes, *,
                        target_reduction=0.4):
    """Return minimum integer context counts under shared decoder state.

    ``estimated_packet_bytes`` may be a Fraction so extrapolation does not round
    before evaluating a boundary. Positive reduction is strictly greater than
    zero; the target is reached inclusively (at least 40% by default). A None
    count means the specified reduction is impossible under these assumptions.
    At least one context is always required, including when state is free.
    """
    raw = _bytes(raw_cache_bytes, "raw_cache_bytes", positive=True)
    payload = _fraction(estimated_packet_bytes, "estimated_packet_bytes")
    state = _bytes(state_bytes, "state_bytes")
    target = _fraction(target_reduction, "target_reduction")
    if payload < 0:
        raise ValueError("estimated_packet_bytes must be nonnegative")
    if not 0 <= target <= 1:
        raise ValueError("target_reduction must be between 0 and 1 inclusive")

    saving = Fraction(raw) - payload
    positive = (state // saving + 1) if saving > 0 else None
    margin = (1 - target) * raw - payload
    if margin > 0:
        quotient = Fraction(state) / margin
        target_count = max(1, -(-quotient.numerator // quotient.denominator))
    elif margin == 0 and state == 0:
        target_count = 1
    else:
        target_count = None

    return {
        "scope": "ESTIMATE: identical payload ratio and compatible shared decoder state",
        "positive_reduction_contexts": int(positive) if positive is not None else None,
        "positive_reduction_rule": "strictly greater than zero",
        "target_reduction_fraction": float(target),
        "target_reduction_pct": float(100 * target),
        "target_reduction_exact": _exact(target),
        "target_reduction_contexts": target_count,
        "target_reduction_rule": "greater than or equal to target",
        "per_context_payload_saving_bytes_exact": _exact(saving),
        "per_context_target_margin_bytes_exact": _exact(margin),
    }


def account(raw_sample_bytes, packet_bytes, raw_cache_bytes,
            neural_serialized_bytes, neural_resident_bytes,
            anchor_serialized_bytes, anchor_resident_bytes, *,
            concurrencies=(1, 4, 16, 64)):
    """Account for sampled packets and explicitly estimated full-cache savings.

    ``packet_bytes`` must include every byte in the measured sample's actual
    packet representation, including anchors, correction streams and framing.
    Reusable anchor decoder state is additionally charged through the anchor
    state arguments. Do not pass an entropy bound in place of packet bytes.
    Resident sizes are supplied measurements, not inferred from serialization.
    All outputs are JSON-compatible; rational forms preserve exact arithmetic.
    """
    raw_sample = _bytes(raw_sample_bytes, "raw_sample_bytes", positive=True)
    packets = _bytes(packet_bytes, "packet_bytes")
    raw_cache = _bytes(raw_cache_bytes, "raw_cache_bytes", positive=True)
    neural_disk = _bytes(neural_serialized_bytes, "neural_serialized_bytes")
    neural_memory = _bytes(neural_resident_bytes, "neural_resident_bytes")
    anchor_disk = _bytes(anchor_serialized_bytes, "anchor_serialized_bytes")
    anchor_memory = _bytes(anchor_resident_bytes, "anchor_resident_bytes")
    contexts = tuple(_bytes(n, "concurrency", positive=True) for n in concurrencies)

    neural_charge = max(neural_disk, neural_memory)
    anchor_charge = max(anchor_disk, anchor_memory)
    state = neural_charge + anchor_charge
    payload_ratio = Fraction(packets, raw_sample)
    sample_reduction = 1 - payload_ratio
    estimated_payload = payload_ratio * raw_cache
    single_total = estimated_payload + state

    concurrent = []
    for n in contexts:
        total_raw = n * raw_cache
        total_payload = n * estimated_payload
        total_stored = total_payload + state
        concurrent.append({
            "scope": "ESTIMATE: decoder state shared once across compatible contexts",
            "contexts": n,
            "raw_bytes": total_raw,
            "estimated_payload_bytes": _number(total_payload),
            "estimated_payload_bytes_exact": _exact(total_payload),
            "charged_state_bytes": state,
            "estimated_total_bytes": _number(total_stored),
            "estimated_total_bytes_exact": _exact(total_stored),
            **_ratio_fields(1 - total_stored / total_raw),
        })

    return {
        "format": "neural_kv_byte_accounting_v1",
        "scope": "Sample packet bytes measured; all full-cache and concurrency results are ESTIMATES",
        "sample": {
            "scope": "MEASURED: actual sampled raw and complete packet bytes",
            "raw_bytes": raw_sample,
            "packet_bytes": packets,
            "payload_fraction_of_raw": float(payload_ratio),
            "payload_fraction_of_raw_exact": _exact(payload_ratio),
            "payload_reduction_fraction": float(sample_reduction),
            "payload_reduction_pct": float(100 * sample_reduction),
            "payload_reduction_exact": _exact(sample_reduction),
            "state_amortized_over_sample": False,
        },
        "state": {
            "charging_rule": "max(serialized_bytes, resident_bytes) per family, then sum both families",
            "neural": {"serialized_bytes": neural_disk, "resident_bytes": neural_memory,
                       "charged_bytes": neural_charge},
            "anchor": {"serialized_bytes": anchor_disk, "resident_bytes": anchor_memory,
                       "charged_bytes": anchor_charge},
            "total_charged_bytes": state,
        },
        "full_cache_estimate": {
            "scope": "ESTIMATE: sampled packet/raw ratio scaled to full raw cache; full decoder state charged once",
            "raw_bytes": raw_cache,
            "estimated_packet_bytes": _number(estimated_payload),
            "estimated_packet_bytes_exact": _exact(estimated_payload),
            "charged_state_bytes": state,
            "estimated_total_bytes": _number(single_total),
            "estimated_total_bytes_exact": _exact(single_total),
            **_ratio_fields(1 - single_total / raw_cache),
        },
        "concurrent_shared_state_estimates": concurrent,
        "break_even": break_even_contexts(raw_cache, estimated_payload, state),
        "limitations": [
            "Sample extrapolation assumes the measured ratio represents the full cache; no confidence bound is implied.",
            "Concurrency arithmetic assumes one compatible neural state and one compatible anchor state are shared across contexts.",
            "Byte arithmetic does not establish GPU allocator use, materialization latency, attention compatibility or serving concurrency.",
        ],
    }

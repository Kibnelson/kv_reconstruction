"""Descriptive bit statistics and actual CPU lower-plane coding controls.

No predictor is fitted or selected here. Reports contain aggregate counts,
never original tokens, producer activations or KV arrays. Empirical entropy is
a finite-sample estimate, not a lower bound or an achievable codec size.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import struct
import time
import zlib

import numpy as np


_DOMAINS = dict(final_K="k", final_V="v", pre_rope_K="k_pre_rope", kproj_K="k_proj")
_CONTROL = struct.Struct("<4sBBHIIIIII")  # 32 actual bytes, independently parsed.
_MODES = ("raw_channel_major", "first_row_xor", "previous_row_xor")
_LN2 = math.log(2)


def _nlogn(counts):
    values = np.asarray(counts, dtype=np.int64)
    values = values[values > 0].astype(np.float64)
    return float(np.sum(values * np.log2(values)))


def _reliability(counts, support):
    counts, support = np.asarray(counts), np.asarray(support)
    good = (counts >= 64) & (counts >= 5 * support)
    mass = int(counts.sum())
    fraction = float(counts[good].sum() / mass) if mass else 0.
    adequate = bool(mass and fraction >= .95)
    return dict(adequate_for_descriptive_estimate=adequate,
        observations_in_contexts_passing_rule_fraction=fraction,
        rule="At least 95% of observations have >=64 samples and >=5 samples per occupied target symbol in their context. This heuristic is not a confidence interval and cannot detect unseen support.",
        warning=None if adequate else "Sparse conditioning contexts: plug-in entropy can be severely downward biased. Miller-Madow is only a first-order heuristic; do not interpret the estimate as a compression target.")


def _entropy(counts, alphabet_size):
    positive = np.asarray(counts, dtype=np.int64)
    positive = positive[positive > 0]
    n, occupied = int(positive.sum()), len(positive)
    estimate = math.log2(n) - _nlogn(positive) / n if n else None
    correction = (occupied - 1) / (2 * n * _LN2) if n else None
    maximum = math.log2(alphabet_size)
    return dict(observations=n, alphabet_size=alphabet_size, occupied_symbols=occupied,
        singleton_symbols=int(np.count_nonzero(positive == 1)),
        empirical_bits_per_value=estimate,
        miller_madow_uncapped_bits_per_value=estimate + correction if n else None,
        miller_madow_bits_per_value=min(maximum, estimate + correction) if n else None,
        mathematical_alphabet_range_bits=[0., maximum],
        reliability=_reliability(np.array([n]), np.array([occupied])),
        scope="Empirical marginal estimate; Miller-Madow is a clipped first-order bias correction, not a confidence bound or coded size.")


class _Conditional:
    def __init__(self):
        self.n = self.context_nlogn = self.joint_nlogn = 0
        self.context_symbols = self.joint_symbols = self.singletons = 0
        self.context_counts = []
        self.support_counts = []

    def add(self, context_counts, joint_counts, support_counts):
        context_counts = np.asarray(context_counts, dtype=np.int64)
        joint_counts = np.asarray(joint_counts, dtype=np.int64)
        support_counts = np.asarray(support_counts, dtype=np.int64)
        if len(context_counts) != len(support_counts) or context_counts.sum() != joint_counts.sum():
            raise AssertionError("Conditional count accounting differs")
        self.n += int(context_counts.sum())
        self.context_nlogn += _nlogn(context_counts)
        self.joint_nlogn += _nlogn(joint_counts)
        self.context_symbols += len(context_counts)
        self.joint_symbols += len(joint_counts)
        self.singletons += int(np.count_nonzero(joint_counts == 1))
        self.context_counts.append(context_counts)
        self.support_counts.append(support_counts)

    def matrix(self, counts):
        active = counts.sum(axis=1) > 0
        self.add(counts.sum(axis=1)[active], counts[counts > 0],
                 np.count_nonzero(counts, axis=1)[active])

    def sparse(self, keys):
        # Low eight bits are the target; the other bits identify a context.
        unique, joint = np.unique(keys, return_counts=True)
        context, counts = np.unique(keys >> 8, return_counts=True)
        support_context, support = np.unique(unique >> 8, return_counts=True)
        if not np.array_equal(context, support_context):
            raise AssertionError("Sparse target/context keys differ")
        self.add(counts, joint, support)

    def report(self, conditioning):
        n = self.n
        estimate = max(0., (self.context_nlogn - self.joint_nlogn) / n) if n else None
        correction = (self.joint_symbols - self.context_symbols) / (2 * n * _LN2) if n else None
        counts = np.concatenate(self.context_counts) if self.context_counts else np.empty(0, np.int64)
        support = np.concatenate(self.support_counts) if self.support_counts else np.empty(0, np.int64)
        return dict(conditioning=conditioning, observations=n, occupied_contexts=self.context_symbols,
            occupied_joint_symbols=self.joint_symbols, singleton_joint_symbols=self.singletons,
            empirical_conditional_bits_per_value=estimate,
            miller_madow_uncapped_bits_per_value=estimate + correction if n else None,
            miller_madow_bits_per_value=min(8., estimate + correction) if n else None,
            context_observation_counts=dict(minimum=int(counts.min()) if n else None,
                median=float(np.median(counts)) if n else None,
                maximum=int(counts.max()) if n else None),
            mathematical_alphabet_range_bits=[0., 8.], reliability=_reliability(counts, support),
            scope="Weighted empirical conditional entropy on observed contexts only. High-dimensional layer/channel conditioning with 64-128 rows is particularly biased; unseen outcomes are not counted. Neither the plug-in estimate nor Miller-Madow establishes a lower bound or achievable code length.")


def _lower(words):
    integer = words.astype(np.uint16, copy=False)
    return (((integer >> 8) & 128) | (integer & 127)).astype(np.uint8)


def _encode_lower_control(lower, mode):
    """Actual independent CPU record, with charged seed and validated framing."""
    if mode not in _MODES or lower.dtype != np.uint8 or lower.ndim != 2 or min(lower.shape) < 1:
        raise ValueError("Expected a nonempty uint8 lower plane and a known control")
    tokens, channels = lower.shape
    if lower.size > 0xffffffff or max(lower.shape) > 0xffffffff:
        raise ValueError("CPU lower-plane control exceeds uint32 record geometry")
    seed = b""
    if mode == "raw_channel_major":
        coded = lower
    else:
        seed = lower[0].tobytes()
        coded = lower[1:] ^ (lower[:1] if mode == "first_row_xor" else lower[:-1])
    payload = zlib.compress(np.ascontiguousarray(coded.T).tobytes(), level=6)
    raw = np.ascontiguousarray(lower.T).tobytes()
    header = _CONTROL.pack(b"FLC1", 1, _MODES.index(mode), 0, tokens, channels,
                           len(seed), len(payload), len(raw), zlib.crc32(raw))
    return header + seed + payload


def _decode_lower_control(blob):
    if len(blob) < _CONTROL.size:
        raise ValueError("Truncated CPU lower-plane control")
    magic, version, mode, flags, tokens, channels, seed_size, payload_size, raw_size, checksum = _CONTROL.unpack_from(blob)
    if (magic != b"FLC1" or version != 1 or flags or mode >= len(_MODES) or not tokens or not channels
            or raw_size != tokens * channels or seed_size != (channels if mode else 0)
            or len(blob) != _CONTROL.size + seed_size + payload_size):
        raise ValueError("Invalid CPU lower-plane control framing")
    seed = blob[_CONTROL.size:_CONTROL.size + seed_size]
    decoder = zlib.decompressobj()
    try:
        plain = decoder.decompress(blob[_CONTROL.size + seed_size:]) + decoder.flush()
    except zlib.error as error:
        raise ValueError("Invalid CPU lower-plane compressed payload") from error
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(plain) != channels * (tokens - bool(mode)):
        raise ValueError("CPU lower-plane payload length or stream termination differs")
    matrix = np.frombuffer(plain, np.uint8).reshape(channels, tokens - bool(mode)).T.copy()
    if mode:
        first = np.frombuffer(seed, np.uint8)[None, :]
        matrix = (np.concatenate((first, matrix ^ first)) if mode == 1
                  else np.bitwise_xor.accumulate(np.concatenate((first, matrix)), axis=0))
    if zlib.crc32(np.ascontiguousarray(matrix.T).tobytes()) != checksum:
        raise ValueError("CPU lower-plane control CRC differs")
    return matrix


def _validate(records):
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("Expected nonempty selected producer records")
    checked = []
    occupied = defaultdict(list)
    shapes = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Producer diagnostic records must be mappings")
        layer, start = record.get("layer"), record.get("start", 0)
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in (layer, start)):
            raise ValueError("Invalid producer diagnostic layer/start")
        arrays = {}
        for name in _DOMAINS.values():
            if name not in record:
                if name in ("k", "v"):
                    raise ValueError("Actual K and V targets are required")
                continue
            value = np.asarray(record[name])
            if value.dtype != np.uint16 or value.ndim != 2 or min(value.shape) < 1:
                raise ValueError("Producer diagnostic fields must be nonempty uint16[T,C] BF16 words")
            arrays[name] = value
        shape = arrays["k"].shape
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("Producer K/V/source geometries differ")
        if "tokens" in record and (isinstance(record["tokens"], bool) or record["tokens"] != shape[0]):
            raise ValueError("Producer diagnostic token count differs")
        if layer in shapes and shapes[layer] != shape[1]:
            raise ValueError("Producer diagnostic channel geometry changed within a layer")
        shapes[layer] = shape[1]
        occupied[layer].append((int(start), int(start) + shape[0]))
        checked.append((int(layer), int(start), arrays))
    for ranges in occupied.values():
        ranges.sort()
        if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("Overlapping selected records would duplicate diagnostic observations")
    return checked


def _domain(checked, field):
    grouped = defaultdict(list)
    for layer, start, arrays in checked:
        if field in arrays:
            grouped[layer].append((start, arrays[field]))
    count = sum(len(rows) for rows in grouped.values())
    if not count:
        return dict(status="unavailable", reason="Actual producer source was not supplied; no surrogate source is invented.", records=0, observations=0)
    exponent_hist = np.zeros(256, np.int64)
    lower_hist = np.zeros(256, np.int64)
    joint = np.zeros((256, 256), np.int64)
    temporal_joint = np.zeros((256, 256), np.int64)
    xor_hist = np.zeros(256, np.int64)
    pooled_layer, pooled_channel = _Conditional(), _Conditional()
    observations_per_layer_channel = []
    modes = {mode: dict(record_header_bytes=0, seed_bytes=0, compressed_payload_bytes=0,
        actual_record_bytes=0, roundtrip_verified_records=0, encode_cpu_seconds=0., decode_and_check_cpu_seconds=0.) for mode in _MODES}
    for layer in sorted(grouped):
        lower_parts, exponent_parts = [], []
        for _, words in sorted(grouped[layer], key=lambda item: item[0]):
            lower = _lower(words)
            exponents = ((words >> 7) & 255).astype(np.uint8)
            lower_parts.append(lower)
            exponent_parts.append(exponents)
            if len(words) > 1:
                previous, current = lower[:-1], lower[1:]
                temporal_joint += np.bincount((previous.astype(np.int64) * 256 + current).reshape(-1), minlength=65536).reshape(256, 256)
                xor_hist += np.bincount((current ^ previous).reshape(-1), minlength=256)
            for mode in _MODES:
                began = time.perf_counter()
                blob = _encode_lower_control(lower, mode)
                encoded = time.perf_counter() - began
                began = time.perf_counter()
                restored = _decode_lower_control(blob)
                if not np.array_equal(lower, restored):
                    raise AssertionError("CPU lower-plane diagnostic control differs")
                row = modes[mode]
                row["encode_cpu_seconds"] += encoded
                row["decode_and_check_cpu_seconds"] += time.perf_counter() - began
                row["record_header_bytes"] += _CONTROL.size
                row["seed_bytes"] += lower.shape[1] if mode != "raw_channel_major" else 0
                row["compressed_payload_bytes"] += len(blob) - _CONTROL.size - (lower.shape[1] if mode != "raw_channel_major" else 0)
                row["actual_record_bytes"] += len(blob)
                row["roundtrip_verified_records"] += 1
        lower, exponents = np.concatenate(lower_parts), np.concatenate(exponent_parts)
        rows, channels = lower.shape
        observations_per_layer_channel.extend([rows] * channels)
        exponent_hist += np.bincount(exponents.reshape(-1), minlength=256)
        lower_hist += np.bincount(lower.reshape(-1), minlength=256)
        layer_joint = np.bincount((exponents.astype(np.int64) * 256 + lower).reshape(-1), minlength=65536).reshape(256, 256)
        joint += layer_joint
        pooled_layer.matrix(layer_joint)
        # Sparse keys avoid allocating [channels,256,256] dense tables.
        keys = ((np.arange(channels, dtype=np.int64)[None, :] << 16)
                | (exponents.astype(np.int64) << 8) | lower)
        pooled_channel.sparse(keys.reshape(-1))
    n = int(lower_hist.sum())
    sign_hist = lower_hist.reshape(2, 128).sum(axis=1)
    mantissa_hist = lower_hist.reshape(2, 128).sum(axis=0)
    global_conditional, temporal_conditional = _Conditional(), _Conditional()
    global_conditional.matrix(joint)
    temporal_conditional.matrix(temporal_joint)
    transitions = int(xor_hist.sum())
    bitplanes = [float(lower_hist[(np.arange(256) & (1 << bit)) != 0].sum() / n) for bit in range(8)]
    xor_planes = [float(xor_hist[(np.arange(256) & (1 << bit)) != 0].sum() / transitions) if transitions else None for bit in range(8)]
    outer = 32 + 8 * (count + 1)
    for row in modes.values():
        row.update(calculated_outer_framing_bytes=outer, total_bytes_including_calculated_outer=row["actual_record_bytes"] + outer,
            bits_per_original_BF16_value=8 * (row["actual_record_bytes"] + outer) / n,
            lower8_raw_bytes=n, savings_percent_of_lower8_plane=100 * (1 - (row["actual_record_bytes"] + outer) / n))
    return dict(status="ok", records=count, missing_source_records=len(checked) - count,
        observations=n, layers=len(grouped),
        observations_per_layer_channel=dict(minimum=min(observations_per_layer_channel),
            median=float(np.median(observations_per_layer_channel)), maximum=max(observations_per_layer_channel)),
        entropy=dict(exponent8=_entropy(exponent_hist, 256), lower8=_entropy(lower_hist, 256),
                     sign=_entropy(sign_hist, 2), mantissa7=_entropy(mantissa_hist, 128)),
        conditional_lower8_given_exponent=dict(global_pooled=global_conditional.report("current exponent"),
            pooled_by_layer=pooled_layer.report("layer and current exponent"),
            pooled_by_layer_channel=pooled_channel.report("layer, channel and current exponent")),
        lower8_bitplane_one_fractions=bitplanes,
        temporal_lower8=dict(transition_values=transitions,
            zero_xor_fraction=float(xor_hist[0] / transitions) if transitions else None,
            xor_bitplane_one_fractions=xor_planes,
            conditional_on_previous_lower8=temporal_conditional.report("previous token lower8 in the same layer/channel and selected tile"),
            xor_marginal_entropy=_entropy(xor_hist, 256),
            scope="Within-record consecutive token rows only; sampled gaps and record boundaries never become temporal transitions. Conditional estimates do not pay decoder seeds, restart indexes or parallel reconstruction cost."),
        histograms=dict(exponent8=exponent_hist.tolist(), lower8=lower_hist.tolist(),
            sign=sign_hist.tolist(), mantissa7=mantissa_hist.tolist(), temporal_xor_lower8=xor_hist.tolist()),
        cpu_lower8_controls=dict(methods=modes, zlib_version=zlib.ZLIB_VERSION, zlib_level=6,
            record_header_bytes=_CONTROL.size, decoder_state_bytes=0, roundtrip_exact=True,
            scope="Actual CPU zlib lower-plane records, including 32-byte per-record geometry/length/CRC framing and raw first-row seeds for XOR modes. Specified outer header/index costs are calculated; no complete outer file is persisted. These code only eight lower bits, not a complete KV cache. They are CPU size controls, not GPU codecs or GPU timings; no candidate is selected or fitted."))


def analyze(records):
    """Aggregate actual selected records after capture; no fitting or raw export."""
    checked = _validate(records)
    return dict(format="qwen_frontier_diagnostics_v1", raw_arrays_included=False,
        fitting_performed=False, records_supplied=len(checked),
        target_ledger=dict(target_savings_percent=40., max_total_bits_per_BF16_value=9.6,
            fixed_original_lower8_bits_per_value=8., remaining_exponent_and_all_metadata_bits_per_value=1.6,
            old_mandatory_exponent_primary_bits_per_value=2.,
            old_raw_lower8_plus_two_bit_primary_min_payload_bits_per_value=10.,
            old_raw_lower8_plus_two_bit_primary_max_savings_percent=37.5,
            prior_exponent_payload_reference_bits_per_value_approx=2.8,
            reference_payload_bits_before_metadata=10.8, reference_savings_before_metadata_percent=32.5,
            reference_additional_net_reduction_needed_before_metadata_bits_per_value=1.2,
            scope="Algebra for the stated formats only. Keeping raw lower8 plus a mandatory two-bit exponent primary stream cannot reach 40%, even if every other byte were free. The ~2.8-bit exponent payload references earlier measurements, not a measurement by this function. New lower-plane coding or a different exponent format can change these assumptions; no universal information-theoretic impossibility is claimed."),
        domains={name: _domain(checked, field) for name, field in _DOMAINS.items()},
        scope="Descriptive statistics of actual selected TRAIN/DEV/TEST arrays after producer capture. No raw arrays, per-record symbols, learned predictors, candidate fitting or TEST-based policy are exported. Empirical entropy estimates can be severely biased under sparse conditioning; even a bias correction is neither a lower bound nor an achievable compressed size. CPU lower-plane controls serialize actual records and verify exactness, but do not measure a complete GPU KV representation or serving benefit.")

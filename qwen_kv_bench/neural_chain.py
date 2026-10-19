"""Actual CPU entropy packets with a bounded cross-layer neural dependency.

Only prediction runs on the prepared model's device. Entropy coding is an
explicit CPU reference. Every selected token tile contains all layers and starts
with an independently coded layer-zero anchor. Later layers use reconstructed
previous-layer words, never the original previous-layer capture.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import struct
import time
import zlib

import numpy as np
import torch

from . import gpu_ops as g, model_runtime as runtime

_CACHE = struct.Struct("<8s6I")  # 32 bytes, including header/index CRC.
_PAGE = struct.Struct("<8sIQI")
_CACHE_MAGIC = b"NKVCH001"
_PAGE_MAGIC = b"NKVPG001"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _word_sha(words):
    return _sha(np.asarray(words, dtype="<u2", order="C").tobytes())


def _prediction_sha(mu, log_scale):
    return _sha(np.asarray(mu, dtype="<f4", order="C").tobytes()
                + np.asarray(log_scale, dtype="<f4", order="C").tobytes())


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _identity(state):
    value = getattr(state, "state_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("prepared model/anchor requires a stable state_id")
    return value


def _page(metadata, payload):
    header = _json(metadata)
    payload = bytes(payload)
    return _PAGE.pack(_PAGE_MAGIC, len(header), len(payload),
                      zlib.crc32(header + payload)) + header + payload


def _read_page(wire):
    wire = bytes(wire)
    if len(wire) < _PAGE.size:
        raise ValueError("truncated neural page")
    magic, nh, npayload, crc = _PAGE.unpack_from(wire)
    if magic != _PAGE_MAGIC or _PAGE.size + nh + npayload != len(wire):
        raise ValueError("invalid neural page framing")
    body = wire[_PAGE.size:]
    if zlib.crc32(body) != crc:
        raise ValueError("neural page CRC mismatch")
    try:
        meta = json.loads(body[:nh])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid neural page descriptor") from error
    if not isinstance(meta, dict) or _json(meta) != body[:nh] or meta.get("version") != 1:
        raise ValueError("noncanonical neural page descriptor")
    return meta, body[nh:], dict(page_framing=_PAGE.size, page_descriptor=nh)


@dataclass(frozen=True)
class NeuralCache(Sequence):
    """Immutable actual pages; no arrays, predictor outputs or source closures."""
    pages: tuple[bytes, ...]
    layers: int
    tokens: int
    channels: int
    tile_tokens: int

    def __post_init__(self):
        object.__setattr__(self, "pages", tuple(bytes(page) for page in self.pages))
        for name in ("layers", "tokens", "channels", "tile_tokens"):
            _positive(getattr(self, name), name)
        _addresses(self)

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def to_bytes(self):
        _addresses(self)
        first = _CACHE.size + 8 * (len(self) + 1)
        offsets = [first]
        for page in self.pages:
            offsets.append(offsets[-1] + len(page))
        index = struct.pack("<" + "Q" * len(offsets), *offsets)
        values = (_CACHE_MAGIC, self.layers, self.tokens, self.channels,
                  self.tile_tokens, len(self), 0)
        crc = zlib.crc32(_CACHE.pack(*values) + index)
        return _CACHE.pack(*values[:-1], crc) + index + b"".join(self.pages)

    @classmethod
    def from_bytes(cls, wire):
        wire = bytes(wire)
        if len(wire) < _CACHE.size:
            raise ValueError("truncated neural cache")
        magic, layers, tokens, channels, tile, count, crc = _CACHE.unpack_from(wire)
        first = _CACHE.size + 8 * (count + 1)
        if magic != _CACHE_MAGIC or not count or first > len(wire):
            raise ValueError("invalid neural cache header/index")
        index = wire[_CACHE.size:first]
        empty_crc = _CACHE.pack(magic, layers, tokens, channels, tile, count, 0)
        if zlib.crc32(empty_crc + index) != crc:
            raise ValueError("neural cache header/index CRC mismatch")
        offsets = struct.unpack("<" + "Q" * (count + 1), index)
        if offsets[0] != first or offsets[-1] != len(wire) or any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("invalid neural cache offsets")
        return cls(tuple(wire[a:b] for a, b in zip(offsets, offsets[1:])),
                   layers, tokens, channels, tile)


def _addresses(cache):
    if not cache.pages:
        raise ValueError("empty neural cache")
    groups = {}
    for wire in cache.pages:
        meta, _, _ = _read_page(wire)
        layer, start, count = (meta.get(key) for key in ("layer", "start", "tokens"))
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in (layer, start, count))
                or not 0 <= layer < cache.layers or not 0 <= start < cache.tokens
                or start % cache.tile_tokens or count != min(cache.tile_tokens, cache.tokens - start)
                or meta.get("channels") != cache.channels):
            raise ValueError("neural page/cache geometry mismatch")
        address = groups.setdefault(start, {})
        if layer in address:
            raise ValueError("duplicate neural page address")
        address[layer] = wire
    if any(set(group) != set(range(cache.layers)) for group in groups.values()):
        raise ValueError("every selected neural tile requires all preceding layers")
    return groups


def _anchor_wire(packet):
    host = g.packet_to_cpu(packet)
    header = host.header.numpy().tobytes()
    fields = b"".join(host.fields[name].numpy().tobytes() for name in sorted(host.fields))
    payload = struct.pack("<I", len(header)) + header + fields
    return payload, dict(anchor_header_framing=4, anchor_native_header=len(header),
                         anchor_native_fields=len(fields))


def _anchor_packet(payload, device):
    if len(payload) < 4:
        raise ValueError("truncated neural anchor")
    nh, = struct.unpack_from("<I", payload)
    if nh > len(payload) - 4:
        raise ValueError("truncated neural anchor header")
    header = payload[4:4 + nh]
    try:
        descriptor = json.loads(header)
        sizes = descriptor["field_bytes"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("invalid neural anchor header") from error
    cursor, fields = 4 + nh, {}
    for name, size in sorted(sizes.items()):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or cursor + size > len(payload):
            raise ValueError("invalid neural anchor field length")
        fields[name] = torch.from_numpy(np.frombuffer(payload[cursor:cursor + size], np.uint8).copy())
        cursor += size
    if cursor != len(payload):
        raise ValueError("neural anchor trailing bytes")
    packet = g.packet_from_cpu_parts(torch.from_numpy(np.frombuffer(header, np.uint8).copy()), fields)
    return g.packet_to_device(packet, device)


def _codecs(anchor_codec, entropy_codec):
    if anchor_codec is None:
        from . import producer_codec as anchor_codec
    if entropy_codec is None:
        from . import logistic_reference as entropy_codec
    return anchor_codec, entropy_codec


def _predict(prepared, layer, previous, start):
    device = torch.device(prepared.device)
    staged, staging = runtime.timed_call(
        lambda: torch.from_numpy(np.asarray(previous).copy()).to(device), device)
    predicted, prediction = runtime.timed_call(lambda: prepared.predict(layer, staged, start), device)
    if not isinstance(predicted, (tuple, list)) or len(predicted) != 2:
        raise ValueError("neural predictor must return mu and log_scale")
    mu, log_scale = predicted
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or tuple(value.shape) != tuple(previous.shape) or value.device != device
           for value in (mu, log_scale)):
        raise ValueError("neural predictor must return aligned device FP32 parameters")
    output, copy = runtime.timed_call(lambda: (mu.detach().cpu().numpy().copy(),
                                             log_scale.detach().cpu().numpy().copy()), device)
    if any(not np.isfinite(value).all() for value in output):
        raise ValueError("nonfinite neural probability parameters")
    return *output, dict(input_staging=staging, neural_prediction=prediction,
                         prediction_output_to_cpu=copy, input_bytes=previous.nbytes,
                         output_bytes=sum(value.nbytes for value in output))


def _decode_one(wire, previous, previous_wire, prepared, anchor_state, anchor_codec, entropy_codec):
    meta, payload, _ = _read_page(wire)
    if meta.get("model_state_id") != _identity(prepared) or meta.get("anchor_state_id") != _identity(anchor_state):
        raise ValueError("neural page model/anchor state identity mismatch")
    timings = {}
    if meta["layer"] == 0:
        if meta.get("kind") != "anchor" or previous is not None or meta.get("previous_page_sha256") is not None:
            raise ValueError("invalid neural anchor dependency")
        started = time.perf_counter()
        packet = _anchor_packet(payload, anchor_state.device)
        pair, elapsed = runtime.timed_call(lambda: anchor_codec.decode(packet, anchor_state), anchor_state.device)
        words = np.concatenate([value.detach().cpu().numpy() for value in pair], axis=1)
        timings.update(anchor_decode=elapsed, anchor_restore_wall_seconds=time.perf_counter() - started)
    else:
        if (meta.get("kind") != "neural" or previous is None
                or meta.get("previous_page_sha256") != _sha(previous_wire)
                or meta.get("previous_words_sha256") != _word_sha(previous)):
            raise ValueError("neural page prerequisite identity mismatch")
        mu, log_scale, prediction = _predict(prepared, meta["layer"], previous, meta["start"])
        if _prediction_sha(mu, log_scale) != meta.get("prediction_sha256"):
            raise ValueError("neural prediction changed; entropy replay would be invalid")
        started = time.perf_counter()
        words = entropy_codec.decode(payload, meta["entropy"], mu, log_scale)
        timings.update(prediction=prediction, cpu_entropy_decode_seconds=time.perf_counter() - started)
    words = np.asarray(words)
    if (words.dtype != np.uint16 or words.shape != (meta["tokens"], 2 * meta["channels"])
            or _word_sha(words) != meta.get("words_sha256")):
        raise ValueError("neural decoded words/geometry checksum mismatch")
    return np.ascontiguousarray(words), timings


def _state_info(prepared, anchor_state, anchor_codec):
    neural_serialized = int(prepared.serialized_bytes)
    neural_resident = int(prepared.physical_nbytes)
    anchor_serialized = len(anchor_codec.state_bytes(anchor_state))
    anchor_resident = int(anchor_state.physical_nbytes)
    serialized = neural_serialized + anchor_serialized
    resident = neural_resident + anchor_resident
    return dict(neural_model_id=getattr(prepared, "model_id", None), neural_state_id=_identity(prepared),
        anchor_state_id=_identity(anchor_state), serialized_bytes=serialized,
        physical_tensor_bytes=resident,
        charged_bytes=max(neural_serialized, neural_resident) + max(anchor_serialized, anchor_resident),
        neural_serialized_bytes=neural_serialized, neural_resident_bytes=neural_resident,
        anchor_serialized_bytes=anchor_serialized, anchor_resident_bytes=anchor_resident,
        execution_contract=getattr(prepared, "execution_contract", None),
        accounting="Complete neural and anchor artifacts plus distinct resident tensors, sum of per-family max(serialized,resident), once per cache; caller owns persistent artifact export.")


def encode_cache(matrices, prepared_model, anchor_state, *, tile_tokens=64, max_pairs=72,
                 anchor_codec=None, entropy_codec=None, head_dim=1):
    """Encode selected aligned tile chains; verify by actually decoding each page."""
    anchor_codec, entropy_codec = _codecs(anchor_codec, entropy_codec)
    tile_tokens = _positive(tile_tokens, "tile_tokens")
    head_dim = _positive(head_dim, "head_dim")
    layers = len(matrices)
    if not layers:
        raise ValueError("empty neural source cache")
    first = np.asarray(matrices[0][0])
    if first.dtype != np.uint16 or first.ndim != 2:
        raise ValueError("neural sources must be CPU uint16[T,C] BF16 matrices")
    tokens, channels = first.shape
    _positive(tokens, "tokens"); _positive(channels, "channels")
    for pair in matrices:
        if len(pair) != 2 or any(np.asarray(value).dtype != np.uint16 or np.asarray(value).shape != first.shape for value in pair):
            raise ValueError("neural source layer geometry differs")
    tiles = (tokens + tile_tokens - 1) // tile_tokens
    if max_pairs is None:
        selected = tiles
    else:
        max_pairs = _positive(max_pairs, "max_pairs")
        if max_pairs < layers:
            raise ValueError("max_pairs must permit every layer of at least one aligned tile")
        selected = min(tiles, max_pairs // layers)
    starts = sorted(set(int(value) * tile_tokens for value in np.linspace(0, tiles - 1, selected, dtype=np.int64)))
    state = _state_info(prepared_model, anchor_state, anchor_codec)
    pages, traces, parts = [], [], Counter()
    raw = 0; dependency_peak = 0; started_all = time.perf_counter()
    for start in starts:
        count = min(tile_tokens, tokens - start)
        previous = previous_wire = None
        for layer in range(layers):
            words = np.concatenate([np.asarray(value)[start:start + count] for value in matrices[layer]], axis=1)
            meta = dict(version=1, kind="anchor" if layer == 0 else "neural", layer=layer,
                start=start, tokens=count, channels=channels, model_state_id=_identity(prepared_model),
                anchor_state_id=_identity(anchor_state), words_sha256=_word_sha(words),
                previous_page_sha256=_sha(previous_wire) if previous_wire else None,
                previous_words_sha256=_word_sha(previous) if previous is not None else None)
            timing = {}
            if layer == 0:
                record = dict(layer=0, start=start, tokens=count, head_dim=head_dim,
                              k=words[:, :channels], v=words[:, channels:])
                packet, elapsed = runtime.timed_call(lambda: anchor_codec.encode(record, anchor_state.spec, anchor_state), anchor_state.device)
                wire_started = time.perf_counter()
                payload, payload_parts = _anchor_wire(packet)
                timing.update(anchor_encode=elapsed, anchor_packet_to_cpu_seconds=time.perf_counter() - wire_started)
            else:
                mu, log_scale, prediction = _predict(prepared_model, layer, previous, start)
                meta["prediction_sha256"] = _prediction_sha(mu, log_scale)
                cpu_started = time.perf_counter()
                payload, description = entropy_codec.encode(words, mu, log_scale)
                meta["entropy"] = description
                payload_parts = dict(cpu_entropy_payload=len(payload))
                timing.update(prediction=prediction, cpu_entropy_encode_seconds=time.perf_counter() - cpu_started)
            wire = _page(meta, payload)
            restored, verification = _decode_one(wire, previous, previous_wire,
                prepared_model, anchor_state, anchor_codec, entropy_codec)
            if not np.array_equal(restored, words):
                raise AssertionError("neural page differs from actual original KV")
            dependency_peak = max(dependency_peak, restored.nbytes + (previous.nbytes if previous is not None else 0))
            _, _, framing = _read_page(wire)
            page_parts = dict(**framing, **payload_parts)
            if sum(page_parts.values()) != len(wire):
                raise AssertionError("neural page accounting differs from actual wire")
            parts.update(page_parts); pages.append(wire); raw += words.nbytes
            traces.append(dict(layer=layer, start=start, tokens=count, raw_bytes=words.nbytes,
                serialized_bytes=len(wire), component_bytes=page_parts, encode_timing=timing,
                verification_decode_timing=verification, roundtrip_exact=True,
                dependency_depth=layer, ideal_continuous_bits=meta.get("entropy", {}).get("ideal_continuous_bits"),
                integer_cdf_bits=meta.get("entropy", {}).get("integer_cdf_bits")))
            previous, previous_wire = restored, wire
    cache = NeuralCache(tuple(pages), layers, tokens, channels, tile_tokens)
    wire = cache.to_bytes()
    reopened = NeuralCache.from_bytes(wire)
    if reopened != cache:
        raise AssertionError("neural complete container did not reopen identically")
    parts.update(cache_header=_CACHE.size, page_offset_table=8 * (len(cache) + 1))
    if sum(parts.values()) != len(wire):
        raise AssertionError("neural complete cache byte accounting mismatch")
    full_pairs = layers * tiles
    full_raw = 4 * layers * tokens * channels
    # Match the study's shared accounting: scale the complete observed byte
    # density, including its outer envelope. This is an estimate, not a claimed
    # full-cache serialization; fixed header bytes can be overcounted.
    estimated = len(wire) * full_raw / raw
    full_stored = len(wire) if len(pages) == full_pairs else estimated
    metric = dict(records_measured=len(pages), records_total=full_pairs, selected_starts=starts,
        coverage="full" if len(pages) == full_pairs else "sampled", size_is_estimated=len(pages) != full_pairs,
        raw_payload_bytes=raw, full_raw_payload_bytes=full_raw, serialized_bytes=len(wire),
        actual_record_payload_bytes=sum(len(page) for page in pages), component_bytes=dict(parts),
        estimated_full_cache_serialized_bytes=estimated if len(pages) != full_pairs else None,
        decoder_state_info=state, decoder_state_bytes=state["charged_bytes"],
        savings_including_state_percent=100 * (1 - (len(wire) + state["charged_bytes"]) / raw),
        estimated_full_savings_including_state_percent=100 * (1 - (full_stored + state["charged_bytes"]) / full_raw),
        effective_bits_per_BF16_word_including_state=16 * (full_stored + state["charged_bytes"]) / full_raw,
        exact_verified=True, roundtrip_exact=True, container_roundtrip_verified=True,
        maximum_dependency_depth=layers - 1, peak_raw_dependency_context_bytes=dependency_peak,
        maximum_encoder_target_temporary_bytes=4 * min(tokens, tile_tokens) * channels,
        one_pass_wall_seconds=time.perf_counter() - started_all, record_diagnostics=traces,
        selection="Aligned evenly spaced token tiles; every selected tile includes all layers. No cross-token-tile dependency.",
        byte_accounting="Actual persisted/reopened 32-byte header, uint64 index, CRC-framed canonical page descriptors and all anchor/entropy payloads. Full-cache estimates scale the complete sampled container/raw byte density, including its outer envelope; fixed header bytes can be overcounted and unequal-tail samples need not represent unsampled pages. Complete state charged once.",
        scope="Neural inference on prepared.device; explicit CPU reference entropy coding. Encoder verification reruns prediction and actual decoding. Two raw dependency tiles exclude original capture, returned outputs, staged inputs, probability outputs, coder workspace and model state. No serving or GPU entropy-codec claim.")
    return cache, metric


def decode_cache(packets, prepared_model, anchor_state, *, layers, tokens, channels,
                 require_full=True, anchor_codec=None, entropy_codec=None, return_metrics=False):
    """Restore from actual pages/state only, in prerequisite order within tiles.

    Full restore returns the usual list of (K,V) CPU word matrices. Sampled
    restore returns record dictionaries. Returned outputs are separate from the
    at-most-two raw dependency tiles; entropy and neural workspace is additional.
    """
    anchor_codec, entropy_codec = _codecs(anchor_codec, entropy_codec)
    if isinstance(packets, (bytes, bytearray, memoryview)):
        packets = NeuralCache.from_bytes(packets)
    if not isinstance(packets, NeuralCache):
        raise ValueError("decode requires a NeuralCache or its serialized bytes")
    if (packets.layers, packets.tokens, packets.channels) != (layers, tokens, channels):
        raise ValueError("requested neural cache geometry differs from packet container")
    groups = _addresses(packets)
    expected_starts = set(range(0, tokens, packets.tile_tokens))
    if require_full and set(groups) != expected_starts:
        raise ValueError("full neural restore requires complete token-tile coverage")
    result = [(np.empty((tokens, channels), np.uint16), np.empty((tokens, channels), np.uint16))
              for _ in range(layers)] if require_full else []
    traces = []; peak = 0; started = time.perf_counter()
    for start, group in sorted(groups.items()):
        previous = previous_wire = None
        for layer in range(layers):
            wire = group[layer]
            current, timing = _decode_one(wire, previous, previous_wire,
                prepared_model, anchor_state, anchor_codec, entropy_codec)
            peak = max(peak, current.nbytes + (previous.nbytes if previous is not None else 0))
            count = len(current)
            if require_full:
                result[layer][0][start:start + count] = current[:, :channels]
                result[layer][1][start:start + count] = current[:, channels:]
            else:
                result.append(dict(layer=layer, start=start, tokens=count,
                    k=current[:, :channels].copy(), v=current[:, channels:].copy()))
            traces.append(dict(layer=layer, start=start, tokens=count, timing=timing))
            previous, previous_wire = current, wire
    metric = dict(records_decoded=len(traces), full_prefix=bool(require_full),
        wall_seconds=time.perf_counter() - started, record_diagnostics=traces,
        peak_raw_dependency_context_bytes=peak, maximum_dependency_depth=layers - 1,
        source_arrays_used=False, entropy_backend="CPU_reference_arithmetic",
        predictor_device=str(prepared_model.device),
        scope="Actual page-only dependency decode. Returned KV outputs and probability/coder/device workspace are additional to the two raw context tiles. No GPU entropy or compact-attention implementation.")
    return (result, metric) if return_metrics else result

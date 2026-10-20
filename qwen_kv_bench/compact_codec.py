"""Exact independent BF16 pages with tiny hints and grouped refinement.

Counts may be predicted on the selected device.  Actual arithmetic streams,
bitpacking and zlib are CPU references, not a GPU-throughput implementation.
Every hint, escape, descriptor, index and fixed-state byte remains payable.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
import zlib

import numpy as np

from . import logistic_reference as arithmetic

FORMAT = "compact_prefill_exact_page_v1"
_MAGIC = b"CPKVPG01"
_HEADER = struct.Struct("<8sIQI")
_MAX_HEADER = 1 << 20
_MAX_PAYLOAD = 64 << 20
_MAX_VALUES = 4 << 20
TOTAL = 65536


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _integer(value, name, low=0, high=(1 << 31) - 1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError("invalid compact page " + name)
    return int(value)


def stage_plan(spec):
    """Fixed two refinement passes, after two exact mantissa base bits."""
    if spec.get("base_bits", 2) != 2 or spec.get("layout") not in ("lsb", "msb"):
        raise ValueError("compact codec requires base_bits2 and lsb/msb layout")
    return [(2, 2), (4, 3)] if spec["layout"] == "lsb" else [(3, 2), (0, 3)]


def initial_mask(spec):
    stage_plan(spec)
    return np.uint16(0xff80 | (3 if spec["layout"] == "lsb" else 0x60))


def _contract(prepared):
    try:
        spec = dict(prepared.spec)
        g = prepared.metadata["geometry"]
        geometry = {k: _integer(g[k], k, 1) for k in ("layers", "channels", "head_dim")}
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("invalid compact prepared model") from error
    stage_plan(spec)
    if (not isinstance(spec.get("name"), str) or spec.get("source") not in ("none", "kv", "hidden")
            or spec.get("side_bits") not in (0, 1, 2, 4)
            or isinstance(spec.get("side_bits"), bool)
            or (spec["source"] == "none") != (spec["side_bits"] == 0)
            or any(type(spec.get(k)) is not bool for k in ("page_shared", "pooled", "key_first"))
            or geometry["head_dim"] % 2 or geometry["channels"] % geometry["head_dim"]):
        raise ValueError("inconsistent compact spec/geometry")
    for name in ("state_id", "model_id"):
        value = getattr(prepared, name, None)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("compact prepared model requires SHA256 " + name)
    return spec, geometry


def _pack(values, width):
    values = np.asarray(values)
    if (not 1 <= width <= 16 or values.dtype.kind not in "ui" or
            np.any(values.astype(np.uint64) >= (1 << width))):
        raise ValueError("invalid compact packed symbols")
    flat = values.astype(np.uint16, copy=False).ravel()
    bits = ((flat[:, None] >> np.arange(width, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bits.ravel(), bitorder="little").tobytes()


def _unpack(data, count, width):
    expected = (count * width + 7) // 8
    if len(data) != expected:
        raise ValueError("compact packed length mismatch")
    if count * width % 8 and data[-1] >> (count * width % 8):
        raise ValueError("nonzero compact padding")
    bits = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")[:count * width]
    return np.sum(bits.reshape(count, width).astype(np.uint16) << np.arange(width, dtype=np.uint16),
                  axis=1, dtype=np.uint16)


def _stored(raw):
    compressed = zlib.compress(raw, 6)
    mode, payload = ("zlib", compressed) if len(compressed) < len(raw) else ("raw", raw)
    return payload, dict(storage=mode, raw_bytes=len(raw), payload_bytes=len(payload))


def _open_stored(data, desc, expected):
    if (not isinstance(desc, dict) or desc.get("storage") not in ("raw", "zlib")
            or desc.get("raw_bytes") != expected or desc.get("payload_bytes") != len(data)):
        raise ValueError("invalid compact stored component")
    if desc["storage"] == "raw":
        raw = data
    else:
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(data, expected + 1)
        except zlib.error as error:
            raise ValueError("invalid compact zlib component") from error
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("invalid compact zlib framing")
    if len(raw) != expected:
        raise ValueError("compact uncompressed length mismatch")
    return raw


def _counts(prepared, layer, start, known, hints, stage, columns):
    tick = time.perf_counter()
    counts = np.asarray(prepared.predict_counts(layer, start, known, hints, stage))
    alphabet = 1 << stage_plan(prepared.spec)[stage][1]
    if (counts.dtype.kind not in "iu" or counts.shape != known.shape + (alphabet,)
            or np.any(counts <= 0) or np.any(counts > TOTAL)
            or np.any(counts.sum(axis=-1, dtype=np.int64) != TOTAL)):
        raise ValueError("compact predictor must return positive integer counts summing65536")
    counts = np.ascontiguousarray(counts[:, columns, :], dtype=np.int32)
    return counts, time.perf_counter() - tick


def _cdf(counts):
    flat = counts.reshape(-1, counts.shape[-1])
    return np.concatenate((np.zeros((len(flat), 1), np.int64), np.cumsum(flat, axis=-1, dtype=np.int64)), axis=1)


def _encode_symbols(symbols, counts):
    cdf = _cdf(counts)
    enc = arithmetic._ArithmeticEncoder(total=TOTAL)
    flat = symbols.ravel()
    for symbol, row in zip(flat, cdf):
        i = int(symbol)
        enc.update(int(row[i]), int(row[i + 1]))
    payload, coded_bits, nbits = enc.finish()
    selected = counts.reshape(-1, counts.shape[-1])[np.arange(len(flat)), flat]
    ideal = float(np.sum(-np.log2(selected.astype(np.float64) / TOTAL)))
    desc = dict(codec="arithmetic", payload_bytes=len(payload), arithmetic_bits=coded_bits,
        nbits=nbits, counts_sha256=_sha(counts.astype("<i4").tobytes()))
    return payload, desc, ideal


def _decode_symbols(payload, desc, counts):
    n = int(np.prod(counts.shape[:-1]))
    bits = _integer(desc.get("arithmetic_bits"), "arithmetic_bits", 2, 17 * n + 2)
    nbits = _integer(desc.get("nbits"), "nbits", 66, 17 * n + 66)
    if (nbits != bits + 64 or len(payload) != (nbits + 7) // 8
            or desc.get("payload_bytes") != len(payload)
            or desc.get("counts_sha256") != _sha(counts.astype("<i4").tobytes())):
        raise ValueError("compact arithmetic counts/length mismatch")
    allbits = np.unpackbits(np.frombuffer(payload, np.uint8))
    if np.any(allbits[bits:]):
        raise ValueError("nonzero compact arithmetic guard/padding")
    dec = arithmetic._ArithmeticDecoder(payload, nbits, total=TOTAL)
    out = np.empty(n, dtype=np.uint16)
    for i, row in enumerate(_cdf(counts)):
        symbol = int(np.searchsorted(row, dec.cumulative(), side="right") - 1)
        if not 0 <= symbol < counts.shape[-1]:
            raise ValueError("compact arithmetic symbol outside alphabet")
        out[i] = symbol
        dec.update(int(row[symbol]), int(row[symbol + 1]))
    return out.reshape(counts.shape[:-1])


def _hints(value, spec, tokens):
    expected = 0 if not spec["side_bits"] else 1 if spec["page_shared"] else tokens
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.shape != (expected,) or np.any(value >= (1 << spec["side_bits"])):
        raise ValueError("invalid compact hint IDs")
    return value.copy()


def encode_page(record, prepared, *, hint_ids=None):
    tick = time.perf_counter()
    spec, geometry = _contract(prepared)
    layer = _integer(record.get("layer"), "layer")
    start = _integer(record.get("start"), "start")
    tokens = _integer(record.get("tokens"), "tokens", 1)
    channels = geometry["channels"]
    if layer >= geometry["layers"] or start + tokens >= 1 << 31 or tokens * channels * 2 > _MAX_VALUES:
        raise ValueError("compact page address/size outside bounds")
    if record.get("head_dim") != geometry["head_dim"]:
        raise ValueError("compact source head geometry mismatch")
    kv = [np.asarray(record.get(k)) for k in ("k", "v")]
    if any(a.dtype != np.uint16 or a.shape != (tokens, channels) for a in kv):
        raise ValueError("compact KV must be uint16[T,C]")
    target = np.concatenate(kv, axis=1)
    before = time.perf_counter()
    supplied = hint_ids is not None
    hints = _hints(prepared.encode_hints(record) if hint_ids is None else hint_ids, spec, tokens)
    hint_seconds = time.perf_counter() - before
    hint_payload = _pack(hints, spec["side_bits"]) if hints.size else b""
    known = target & initial_mask(spec)
    columns = slice(channels, None) if spec["key_first"] else slice(None)
    sections = [hint_payload]
    components = dict(hint_bytes=len(hint_payload))
    key_desc = None
    if spec["key_first"]:
        known[:, :channels] = target[:, :channels]
        data, key_desc = _stored(np.ascontiguousarray(target[:, :channels], dtype="<u2").tobytes())
        sections.append(data); components["key_bytes"] = len(data)
    base_shift = 0 if spec["layout"] == "lsb" else 5
    base = ((target[:, columns] >> 7) << 2) | ((target[:, columns] >> base_shift) & 3)
    data, base_desc = _stored(_pack(base, 11))
    sections.append(data); components["base_bytes"] = len(data)
    stages, diagnostics = [], []
    for index, (shift, width) in enumerate(stage_plan(spec)):
        actual = ((target[:, columns] >> shift) & ((1 << width) - 1)).astype(np.uint16)
        counts, prediction_seconds = _counts(prepared, layer, start, known, hints, index, columns)
        began = time.perf_counter()
        coded, coded_desc, ideal_bits = _encode_symbols(actual, counts)
        raw, raw_desc = _stored(_pack(actual, width))
        raw_desc = dict(codec="packed", **raw_desc)
        use_coded = len(coded) + len(_canonical(coded_desc)) < len(raw) + len(_canonical(raw_desc))
        payload, desc = (coded, coded_desc) if use_coded else (raw, raw_desc)
        sections.append(payload); components[f"stage{index}_bytes"] = len(payload)
        stages.append(dict(index=index, shift=shift, width=width, **desc))
        diagnostic = dict(index=index, shift=shift, width=width, symbols=actual.size,
            alphabet=1 << width, retained_codec=desc["codec"], retained_payload_bytes=len(payload),
            retained_mode=desc["codec"] if use_coded else "packed_" + desc["storage"],
            rejected_or_retained_arithmetic_bytes=len(coded), packed_reference_bytes=len(raw),
            integer_cdf_bits=ideal_bits, prediction_seconds=prediction_seconds,
            entropy_encode_seconds=time.perf_counter() - began,
            argmax_errors=int(np.count_nonzero(np.argmax(counts, axis=-1) != actual)))
        diagnostic["by_kind"] = {}
        actual_columns = range(channels, 2 * channels) if spec["key_first"] else range(2 * channels)
        for kind in ("K", "V"):
            selected = np.array([i for i, col in enumerate(actual_columns) if (col < channels) == (kind == "K")])
            if selected.size:
                p = counts[:, selected, :].reshape(-1, 1 << width)
                a = actual[:, selected].ravel()
                diagnostic["by_kind"][kind] = dict(symbols=len(a), integer_cdf_bits=float(np.sum(-np.log2(p[np.arange(len(a)), a] / TOTAL))))
        diagnostics.append(diagnostic)
        known[:, columns] |= (actual << shift)
    if not np.array_equal(known, target):
        raise AssertionError("compact bit schedule failed to cover original KV")
    payload = b"".join(sections)
    meta = dict(format=FORMAT, version=1, spec=spec, model_id=prepared.model_id, state_id=prepared.state_id,
        geometry=geometry, layer=layer, start=start, tokens=tokens, hint_bytes=len(hint_payload),
        hint_count=len(hints), hints_sha256=_sha(hint_payload), key=key_desc, base=base_desc,
        stages=stages, words_sha256=_sha(np.ascontiguousarray(target, dtype="<u2").tobytes()),
        dependency_rule="own_page_and_immutable_state_only")
    header = _canonical(meta)
    if len(header) > _MAX_HEADER or len(payload) > _MAX_PAYLOAD:
        raise ValueError("compact page exceeds wire safety limit")
    body = header + payload
    wire = _HEADER.pack(_MAGIC, len(header), len(payload), zlib.crc32(body)) + body
    components["framing_bytes"] = _HEADER.size + len(header)
    return wire, dict(raw_bytes=target.nbytes, packet_bytes=len(wire), component_bytes=components,
        feature_bytes=len(hint_payload), hint_bytes=len(hint_payload), retained_hint_bits=len(hint_payload) * 8,
        hint_count=len(hints), hint_ids_supplied=supplied, hint_encode_seconds=0. if supplied else hint_seconds,
        encode_seconds=time.perf_counter() - tick, prediction_seconds=sum(x["prediction_seconds"] for x in diagnostics),
        stages=diagnostics, layer=layer, start=start, tokens=tokens,
        maximum_dependency_depth=0, intra_page_stages=2 + int(spec["key_first"]),
        raw_prerequisite_pages=0, codec_execution="CPU grouped arithmetic/zlib reference; selected-device predictions")


def _parse(wire):
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise ValueError("compact page must be bytes")
    wire = bytes(wire)
    if len(wire) < _HEADER.size:
        raise ValueError("truncated compact page")
    magic, nh, npayload, crc = _HEADER.unpack_from(wire)
    if (magic != _MAGIC or not 1 <= nh <= _MAX_HEADER or npayload > _MAX_PAYLOAD
            or _HEADER.size + nh + npayload != len(wire)):
        raise ValueError("invalid compact page framing")
    body = wire[_HEADER.size:]
    if zlib.crc32(body) != crc:
        raise ValueError("compact page CRC mismatch")
    try:
        meta = json.loads(body[:nh])
        if not isinstance(meta, dict) or _canonical(meta) != body[:nh] or meta.get("format") != FORMAT or meta.get("version") != 1:
            raise ValueError("invalid compact page metadata")
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        raise ValueError("invalid compact page metadata") from error
    return meta, body[nh:]


def inspect_page(wire):
    return _parse(wire)[0]


def page_metadata(wire):
    meta = inspect_page(wire)
    return dict(layer=meta["layer"], start=meta["start"], tokens=meta["tokens"], channels=meta["geometry"]["channels"])


def decode_page(wire, prepared, *, return_stats=False):
    began = time.perf_counter()
    meta, payload = _parse(wire)
    spec, geometry = _contract(prepared)
    if any(meta.get(k) != v for k, v in dict(spec=spec, geometry=geometry,
            model_id=prepared.model_id, state_id=prepared.state_id,
            dependency_rule="own_page_and_immutable_state_only").items()):
        raise ValueError("compact model/spec/state mismatch")
    layer = _integer(meta.get("layer"), "layer")
    start = _integer(meta.get("start"), "start")
    tokens = _integer(meta.get("tokens"), "tokens", 1)
    channels = geometry["channels"]
    if layer >= geometry["layers"] or start + tokens >= 1 << 31 or tokens * channels * 2 > _MAX_VALUES:
        raise ValueError("compact decoded page outside bounds")
    expected_hints = 0 if not spec["side_bits"] else 1 if spec["page_shared"] else tokens
    hint_bytes = (expected_hints * spec["side_bits"] + 7) // 8
    if meta.get("hint_bytes") != hint_bytes or meta.get("hint_count") != expected_hints:
        raise ValueError("compact hint length/count mismatch")
    offset = 0
    def take(size):
        nonlocal offset
        size = _integer(size, "component_size", 0, _MAX_PAYLOAD)
        if offset + size > len(payload):
            raise ValueError("truncated compact component")
        value = payload[offset:offset + size]; offset += size
        return value
    data = take(hint_bytes)
    if _sha(data) != meta.get("hints_sha256"):
        raise ValueError("compact hint hash mismatch")
    hints = _unpack(data, expected_hints, spec["side_bits"]).astype(np.uint8) if expected_hints else np.empty(0, np.uint8)
    known = np.zeros((tokens, 2 * channels), np.uint16)
    columns = slice(channels, None) if spec["key_first"] else slice(None)
    count = tokens * (channels if spec["key_first"] else 2 * channels)
    if spec["key_first"]:
        desc = meta.get("key", {})
        data = _open_stored(take(desc.get("payload_bytes")), desc, tokens * channels * 2)
        known[:, :channels] = np.frombuffer(data, dtype="<u2").reshape(tokens, channels)
    elif meta.get("key") is not None:
        raise ValueError("unexpected compact key component")
    desc = meta.get("base", {})
    data = _open_stored(take(desc.get("payload_bytes")), desc, (count * 11 + 7) // 8)
    base = _unpack(data, count, 11).reshape(known[:, columns].shape)
    base_shift = 0 if spec["layout"] == "lsb" else 5
    known[:, columns] = ((base >> 2) << 7) | ((base & 3) << base_shift)
    if not isinstance(meta.get("stages"), list) or len(meta["stages"]) != 2:
        raise ValueError("invalid compact stage list")
    prediction_seconds = entropy_seconds = 0.
    used = 0
    for index, (shift, width) in enumerate(stage_plan(spec)):
        desc = meta["stages"][index]
        if not isinstance(desc, dict) or any(desc.get(k) != v for k, v in dict(index=index, shift=shift, width=width).items()):
            raise ValueError("compact stage schedule mismatch")
        data = take(desc.get("payload_bytes"))
        if desc.get("codec") == "packed":
            raw = _open_stored(data, desc, (count * width + 7) // 8)
            actual = _unpack(raw, count, width).reshape(known[:, columns].shape)
        elif desc.get("codec") == "arithmetic":
            counts, seconds = _counts(prepared, layer, start, known, hints, index, columns)
            prediction_seconds += seconds
            tick = time.perf_counter(); actual = _decode_symbols(data, desc, counts)
            entropy_seconds += time.perf_counter() - tick; used += 1
        else:
            raise ValueError("unknown compact stage codec")
        known[:, columns] |= actual << shift
    if offset != len(payload):
        raise ValueError("trailing compact payload")
    if _sha(np.ascontiguousarray(known, dtype="<u2").tobytes()) != meta.get("words_sha256"):
        raise ValueError("compact restored words hash mismatch")
    stats = dict(wall_seconds=time.perf_counter() - began, prediction_seconds=prediction_seconds,
        entropy_decode_seconds=entropy_seconds, arithmetic_stages=used, stages_decoded=2,
        hint_bytes=hint_bytes, returned_output_bytes=known.nbytes,
        source_arrays_used=False, dependency_depth=0, raw_prerequisite_pages=0,
        codec_execution="CPU grouped arithmetic/zlib reference; selected-device predictions")
    return (known, stats) if return_stats else known

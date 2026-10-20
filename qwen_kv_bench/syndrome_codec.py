"""Independent exact BF16 pages with prefill-guided binary coset corrections.

NumPy/CPU reference, not a CUDA codec. Five-bit Hamming syndromes identify
zero/one prediction error among31 omitted bits. Encoder verification and raw
escapes guarantee exactness under the recorded predictor replay contract.
Matched sparse error-position coding has the same payload cost.
"""
from __future__ import annotations
import hashlib
import json
import struct
import time
import zlib
import numpy as np

FORMAT = "prefill_syndrome_page_v1"
_MAGIC = b"SYKVPG01"
_HEADER = struct.Struct("<8sIQI")
_MAX_HEADER = 1 << 20
_MAX_PAYLOAD = 64 << 20
_MAX_VALUES = 4 << 20
_COLUMNS = np.arange(1, 32, dtype=np.uint32)
_WEIGHTS31 = np.left_shift(np.uint32(1), np.arange(31, dtype=np.uint32))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _word_bytes(words):
    return np.ascontiguousarray(words, dtype="<u2").tobytes()


def _int(value, name, minimum=0, maximum=(1 << 31) - 1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError("invalid syndrome page " + name)
    return int(value)


def _geometry(prepared):
    try:
        g = prepared.metadata["geometry"]
        geometry = dict(layers=_int(g["layers"], "layers", 1), channels=_int(g["channels"], "channels", 1),
                        head_dim=_int(g["head_dim"], "head_dim", 2), rank=_int(g["rank"], "rank"))
        mode = prepared.mode
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("invalid prepared syndrome geometry") from error
    if mode not in ("none", "kv", "kv_canonical", "hidden", "hidden_norm") or geometry["channels"] % geometry["head_dim"]:
        raise ValueError("invalid syndrome metadata source/head geometry")
    if mode == "none":
        geometry["rank"] = 0
    elif geometry["rank"] < 1:
        raise ValueError("code-bearing syndrome mode needs positive rank")
    for name in ("model_id", "state_id"):
        value = getattr(prepared, name, None)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("prepared syndrome page needs SHA256 " + name)
    centers = getattr(prepared, "exponent_centers", None)
    if not isinstance(centers, np.ndarray) or centers.dtype != np.uint8 or centers.shape != (geometry["layers"], 2 * geometry["channels"]):
        raise ValueError("prepared syndrome page needs TRAIN exponent centers")
    return geometry, mode


def _pack(values, width):
    values = np.asarray(values).reshape(-1)
    if not 0 <= width <= 31:
        raise ValueError("invalid fixed width")
    if width == 0:
        if np.any(values):
            raise ValueError("nonzero zero-width symbols")
        return b""
    if np.any(values < 0) or np.any(values.astype(np.uint64) >= (1 << width)):
        raise ValueError("fixed-width symbol overflow")
    bits = ((values.astype(np.uint32)[:, None] >> np.arange(width, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def _unpack(data, count, width):
    count = _int(count, "symbol count", 0, _MAX_VALUES * 7)
    if not 0 <= width <= 31 or len(data) != (count * width + 7) // 8:
        raise ValueError("invalid fixed-width stream length")
    if width == 0:
        return np.zeros(count, dtype=np.uint32)
    bits = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")
    if np.any(bits[count * width:]):
        raise ValueError("nonzero fixed-width padding")
    bits = bits[:count * width].reshape(count, width).astype(np.uint32)
    return np.sum(bits << np.arange(width, dtype=np.uint32), axis=1, dtype=np.uint32)


def _choose_zlib(raw):
    compressed = zlib.compress(raw, level=6)
    return ("zlib", compressed) if len(compressed) < len(raw) else ("raw", raw)


def _inflate(data, mode, size):
    if mode == "raw":
        raw = data
    elif mode == "zlib":
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(data, size + 1)
            if len(raw) > size or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
                raise ValueError("invalid bounded zlib stream")
        except zlib.error as error:
            raise ValueError("invalid zlib syndrome field") from error
    else:
        raise ValueError("unknown syndrome field mode")
    if len(raw) != size:
        raise ValueError("syndrome field size mismatch")
    return raw


def _syndrome(bits):
    return np.bitwise_xor.reduce(np.where(bits, _COLUMNS, 0), axis=-1).astype(np.uint32)


def _group_words(bits):
    return np.sum(bits.astype(np.uint32) * _WEIGHTS31, axis=-1, dtype=np.uint32)


def _prediction(prepared, layer, features, start, known, residue_bits):
    value = prepared.predict_bits(layer, features, start, known, residue_bits)
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != known.shape + (7,) or np.any(value > 1):
        raise ValueError("syndrome predictor must return uint8 binary [tokens,2channels,7]")
    return value


def _subset_statistics(actual, predicted, full):
    counts = (actual.T[:, :full * 31].reshape(-1, 31) ^
              predicted.T[:, :full * 31].reshape(-1, 31)).sum(axis=1, dtype=np.int64)
    return dict(values=int(actual.size), bit_errors=int(np.count_nonzero(actual ^ predicted)),
        original_ones=int(actual.sum()), predicted_ones=int(predicted.sum()),
        errors_per_group=np.bincount(counts, minlength=32).astype(np.int64).tolist())


def _alternative_rate_diagnostic(actual, predicted, group_size, head_dim, layout="time"):
    """Count-only screen; no alternative packet or TEST-selected policy exists."""
    errors = actual ^ predicted
    tokens, columns = errors.shape
    if layout == "time":
        full, tail = divmod(tokens, group_size)
        grouped = errors.T[:, :full * group_size].reshape(-1, group_size)
        tail_bits = columns * tail
    else:
        full, tail = divmod(head_dim, group_size)
        grouped = errors.reshape(tokens, columns // head_dim, head_dim)[..., :full * group_size].reshape(-1, group_size)
        tail_bits = tokens * (columns // head_dim) * tail
    counts = grouped.sum(axis=1, dtype=np.int64)
    short = int(np.count_nonzero(counts <= 1))
    raw = len(counts) - short
    parity = (group_size + 1).bit_length() - 1
    projected_bytes = (len(counts) + 7) // 8 + (short * parity + 7) // 8 + (raw * group_size + 7) // 8 + (tail_bits + 7) // 8
    return dict(scope="Diagnostic only: alternative grouping is not encoded, no TEST selector; projected payload excludes descriptors/state.",
        layout=layout, group_size=group_size, groups=len(counts), tail_bits=tail_bits,
        errors_per_group=np.bincount(counts, minlength=group_size + 1).astype(np.int64).tolist(),
        zero_error_groups=int(np.count_nonzero(counts == 0)),
        one_error_groups=int(np.count_nonzero(counts == 1)),
        multi_error_groups=int(np.count_nonzero(counts > 1)), recoverable_groups=short,
        projected_payload_bytes=projected_bytes, raw_plane_bytes=(actual.size + 7) // 8,
        projected_payload_saving_bytes=(actual.size + 7) // 8 - projected_bytes)


def _plane_encode(actual, predicted, correction, head_dim):
    tokens, columns = actual.shape
    full, tail = divmod(tokens, 31)
    groups = actual.T[:, :full * 31].reshape(-1, 31)
    guess = predicted.T[:, :full * 31].reshape(-1, 31)
    errors = (groups ^ guess).astype(np.uint8)
    counts = errors.sum(axis=1, dtype=np.int64)
    short = counts <= 1
    modes = _pack(short, 1)
    labels = _syndrome(groups[short]) if correction == "syndrome" else _syndrome(errors[short])
    short_wire = _pack(labels, 5)
    raw_wire = _pack(_group_words(groups[~short]), 31)
    tail_wire = _pack(actual.T[:, full * 31:].reshape(-1), 1)
    candidate = modes + short_wire + raw_wire + tail_wire
    coded_desc = dict(mode="syndrome" if correction == "syndrome" else "position", short_count=int(short.sum()))
    raw = _pack(actual.T.reshape(-1), 1)
    raw_desc = dict(mode="raw")
    coded_complete = len(candidate) + len(_canonical(dict(bytes=len(candidate), **coded_desc)))
    raw_complete = len(raw) + len(_canonical(dict(bytes=len(raw), **raw_desc)))
    choose = coded_complete < raw_complete
    by_kind, by_head = {}, {}
    half = columns // 2
    for kind, begin, end in (("K", 0, half), ("V", half, columns)):
        by_kind[kind] = _subset_statistics(actual[:, begin:end], predicted[:, begin:end], full)
        for offset in range(begin, end, head_dim):
            by_head[kind + str((offset - begin) // head_dim)] = _subset_statistics(
                actual[:, offset:offset + head_dim], predicted[:, offset:offset + head_dim], full)
    stats = dict(values=int(actual.size), bit_errors=int(np.count_nonzero(actual ^ predicted)),
        original_ones=int(actual.sum()), predicted_ones=int(predicted.sum()),
        bit_error_rate=float(np.mean(actual ^ predicted)), groups=int(len(groups)),
        errors_per_group=np.bincount(counts, minlength=32).astype(np.int64).tolist(),
        zero_error_groups=int(np.count_nonzero(counts == 0)), one_error_groups=int(np.count_nonzero(counts == 1)),
        multi_error_groups=int(np.count_nonzero(counts > 1)), recoverable_groups=int(short.sum()),
        candidate_mode_bytes=len(modes), candidate_short_bytes=len(short_wire),
        candidate_raw_group_bytes=len(raw_wire), candidate_tail_bytes=len(tail_wire),
        candidate_payload_bytes=len(candidate), raw_plane_bytes=len(raw),
        retained_mode=correction if choose else "raw", retained_short_groups=int(short.sum()) if choose else 0,
        retained_raw_groups=int((~short).sum()) if choose else int(len(groups)),
        mode_bytes=len(modes) if choose else 0, short_bytes=len(short_wire) if choose else 0,
        raw_group_bytes=len(raw_wire) if choose else len(raw), tail_bytes=len(tail_wire) if choose else 0,
        tail_bits=int(columns * tail), retained_payload_bytes=len(candidate) if choose else len(raw),
        candidate_payload_saving_bytes=len(raw) - len(candidate),
        candidate_complete_saving_bytes=raw_complete - coded_complete,
        by_kind=by_kind, by_head=by_head,
        alternative_layout_diagnostic_not_encoded=_alternative_rate_diagnostic(actual, predicted, 31, head_dim, "channel_within_token_and_head"),
        alternative_group_diagnostics_not_encoded={str(size): _alternative_rate_diagnostic(actual, predicted, size, head_dim) for size in (15, 63)})
    return (candidate, coded_desc, stats) if choose else (raw, raw_desc, stats)


def _plane_decode(data, description, predicted, tokens, columns):
    count = tokens * columns
    full, tail = divmod(tokens, 31)
    groups = full * columns
    if description == {"mode": "raw"}:
        return _unpack(data, count, 1).astype(np.uint8).reshape(columns, tokens).T
    if set(description) != {"mode", "short_count"} or description["mode"] not in ("syndrome", "position"):
        raise ValueError("invalid syndrome bitplane description")
    short_count = _int(description["short_count"], "short groups", 0, groups)
    sizes = ((groups + 7) // 8, (short_count * 5 + 7) // 8,
             ((groups - short_count) * 31 + 7) // 8, (columns * tail + 7) // 8)
    if len(data) != sum(sizes):
        raise ValueError("invalid coded bitplane length")
    cursor, streams = 0, []
    for size in sizes:
        streams.append(data[cursor:cursor + size]); cursor += size
    short = _unpack(streams[0], groups, 1).astype(bool)
    if int(short.sum()) != short_count:
        raise ValueError("syndrome short-group count mismatch")
    labels = _unpack(streams[1], short_count, 5)
    guesses = predicted.T[:, :full * 31].reshape(-1, 31)
    selected = guesses[short]
    delta = labels ^ _syndrome(selected) if description["mode"] == "syndrome" else labels
    flip = np.where(delta != 0, np.left_shift(np.uint32(1), np.maximum(delta, 1) - 1), np.uint32(0))
    words = np.empty(groups, dtype=np.uint32)
    words[short] = _group_words(selected) ^ flip
    words[~short] = _unpack(streams[2], groups - short_count, 31)
    result = np.empty((columns, tokens), dtype=np.uint8)
    result[:, :full * 31] = ((words[:, None] >> np.arange(31, dtype=np.uint32)) & 1).astype(np.uint8).reshape(columns, full * 31)
    result[:, full * 31:] = _unpack(streams[3], columns * tail, 1).reshape(columns, tail)
    return result.T


def encode_page(record, prepared, *, residue_bits=2, correction="syndrome", feature_words=None):
    """Encode actual final K/V; supplied features are copied and transmitted."""
    began = time.perf_counter()
    g, mode = _geometry(prepared)
    residue_bits = _int(residue_bits, "residue_bits", 0, 7)
    if correction not in ("syndrome", "positions"):
        raise ValueError("unknown syndrome correction format")
    try:
        layer, start, tokens = (_int(record[k], k, 1 if k == "tokens" else 0) for k in ("layer", "start", "tokens"))
        k, v = record["k"], record["v"]
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete syndrome source record") from error
    shape = (tokens, g["channels"])
    if layer >= g["layers"] or start + tokens >= (1 << 31) or tokens * 2 * g["channels"] > _MAX_VALUES:
        raise ValueError("syndrome source geometry exceeds bounds")
    if "head_dim" in record and record["head_dim"] != g["head_dim"]:
        raise ValueError("syndrome head dimension mismatch")
    if any(not isinstance(x, np.ndarray) or x.dtype != np.uint16 or x.shape != shape for x in (k, v)):
        raise ValueError("syndrome source requires actual uint16 K/V matrices")
    target = np.concatenate((k, v), axis=1)
    supplied = feature_words is not None
    tick = time.perf_counter()
    features = prepared.encode_features(record) if feature_words is None else feature_words
    feature_seconds = time.perf_counter() - tick if not supplied else 0.
    if not isinstance(features, np.ndarray) or features.dtype != np.uint16 or features.shape != (tokens, g["rank"]):
        raise ValueError("syndrome retained feature geometry/dtype mismatch")
    if np.any((features & 0x7f80) == 0x7f80):
        raise ValueError("syndrome retained features must be finite BF16")
    features = features.copy()
    payloads = [features.astype("<u2").tobytes()]
    parts = dict(feature_bytes=len(payloads[0]))
    tick = time.perf_counter()
    exponents = ((target >> 7) & 255).astype(np.uint8)
    centered = ((exponents.astype(np.int16) - prepared.exponent_centers[layer][None] + 127) & 255).astype(np.uint8)
    exp_mode, exp_wire = _choose_zlib(centered.T.tobytes())
    sign_raw = _pack((target >> 15).T.reshape(-1), 1)
    sign_mode, sign_wire = _choose_zlib(sign_raw)
    residue_wire = _pack((target & ((1 << residue_bits) - 1)).T.reshape(-1), residue_bits)
    payloads.extend((exp_wire, sign_wire, residue_wire))
    parts.update(exponent_bytes=len(exp_wire), sign_bytes=len(sign_wire), residue_bytes=len(residue_wire))
    field_seconds = time.perf_counter() - tick
    mask = np.uint16(0xff80 | ((1 << residue_bits) - 1))
    known = target & mask
    tick = time.perf_counter()
    predictions = _prediction(prepared, layer, features, start, known, residue_bits) if residue_bits < 7 else np.zeros(target.shape + (7,), np.uint8)
    prediction_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    planes, diagnostics = [], []
    for bit in range(residue_bits, 7):
        plane, desc, diag = _plane_encode(((target >> bit) & 1).astype(np.uint8), predictions[:, :, bit], correction, g["head_dim"])
        payloads.append(plane); planes.append(dict(bit=bit, bytes=len(plane), **desc)); diagnostics.append(dict(bit=bit, **diag))
    coding_seconds = time.perf_counter() - tick
    parts["plane_bytes"] = sum(len(x) for x in payloads[4:])
    meta = dict(format=FORMAT, version=1, mode=mode, model_id=prepared.model_id, state_id=prepared.state_id,
        layer=layer, start=start, tokens=tokens, geometry=g, residue_bits=residue_bits, correction="syndrome" if correction == "syndrome" else "position",
        exponent_mode=exp_mode, sign_mode=sign_mode, feature_bytes=parts["feature_bytes"],
        exponent_bytes=parts["exponent_bytes"], sign_bytes=parts["sign_bytes"], residue_bytes=parts["residue_bytes"],
        planes=planes, features_sha256=_sha(payloads[0]), words_sha256=_sha(_word_bytes(target)),
        dependency_rule="own_page_and_shared_immutable_model_only")
    header, payload = _canonical(meta), b"".join(payloads)
    if len(header) > _MAX_HEADER or len(payload) > _MAX_PAYLOAD:
        raise ValueError("syndrome page exceeds wire bounds")
    body = header + payload
    wire = _HEADER.pack(_MAGIC, len(header), len(payload), zlib.crc32(body)) + body
    parts["framing_bytes"] = _HEADER.size + len(header)
    if sum(parts.values()) != len(wire):
        raise AssertionError("syndrome page byte accounting mismatch")
    tick = time.perf_counter()
    restored = decode_page(wire, prepared)
    if not np.array_equal(restored, target):
        raise AssertionError("syndrome encoder decoder verification failed")
    verification_seconds = time.perf_counter() - tick
    retained_short = sum(d["retained_short_groups"] for d in diagnostics)
    group_count = sum(d["groups"] for d in diagnostics)
    stats = dict(layer=layer, start=start, tokens=tokens, raw_bytes=target.nbytes, packet_bytes=len(wire),
        **parts, component_bytes=parts, residue_bits=residue_bits, correction=correction, bitplanes=diagnostics,
        diagnostics=diagnostics, feature_words_supplied=supplied, feature_encode_seconds=feature_seconds,
        field_encode_seconds=field_seconds, prediction_seconds=prediction_seconds,
        syndrome_encode_seconds=coding_seconds, verification_seconds=verification_seconds,
        encode_seconds=time.perf_counter() - began, roundtrip_exact=True, dependency_depth=0,
        raw_prerequisite_pages=0, retained_code_bits=parts["feature_bytes"] * 8,
        side_bits_per_KV_value=parts["feature_bytes"] * 8 / target.size,
        complete_packet_bits_per_KV_value=len(wire) * 8 / target.size,
        retained_short_groups=retained_short, candidate_groups=group_count,
        short_group_fraction=retained_short / group_count if group_count else 0.,
        planes_used_syndrome=sum(d["retained_mode"] == "syndrome" for d in diagnostics),
        planes_used_positions=sum(d["retained_mode"] == "positions" for d in diagnostics),
        prediction_used_for_retained_short_groups=retained_short > 0,
        retained_code_present=features.size > 0,
        metadata_effect_requires_ablation=True,
        candidate_plane_payload_saving_bytes=sum(d["candidate_payload_saving_bytes"] for d in diagnostics),
        retained_plane_payload_saving_bytes=sum(d["raw_plane_bytes"] - d["retained_payload_bytes"] for d in diagnostics),
        savings_after_feature_bytes=sum(d["raw_plane_bytes"] - d["retained_payload_bytes"] for d in diagnostics) - parts["feature_bytes"],
        savings_after_feature_scope="Plane payload minus feature bytes only; excludes page framing and frozen state. Complete packet accounting is authoritative.",
        codec_execution="NumPy CPU syndrome/position and zlib reference; prepared predictor uses its configured device")
    return wire, stats


def _parse(wire):
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise ValueError("syndrome page requires serialized bytes")
    wire = bytes(wire)
    if len(wire) < _HEADER.size:
        raise ValueError("truncated syndrome page")
    magic, nh, npayload, crc = _HEADER.unpack_from(wire)
    if magic != _MAGIC or not 1 <= nh <= _MAX_HEADER or npayload > _MAX_PAYLOAD or len(wire) != _HEADER.size + nh + npayload:
        raise ValueError("invalid syndrome page framing/bounds")
    body = wire[_HEADER.size:]
    if zlib.crc32(body) != crc:
        raise ValueError("syndrome page CRC mismatch")
    try:
        meta = json.loads(body[:nh])
        if not isinstance(meta, dict) or _canonical(meta) != body[:nh]:
            raise ValueError("noncanonical syndrome metadata")
    except (ValueError, UnicodeDecodeError, TypeError) as error:
        raise ValueError("invalid syndrome page metadata") from error
    return meta, body[nh:]


def inspect_page(wire):
    return _parse(wire)[0]


def page_metadata(wire):
    meta, _ = _parse(wire)
    try:
        return dict(layer=meta["layer"], start=meta["start"], tokens=meta["tokens"], channels=meta["geometry"]["channels"])
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete syndrome page address") from error


def decode_page(wire, prepared, *, return_stats=False):
    began = time.perf_counter()
    g, mode = _geometry(prepared)
    meta, payload = _parse(wire)
    try:
        if (meta["format"] != FORMAT or meta["version"] != 1 or meta["mode"] != mode or meta["geometry"] != g
                or meta["model_id"] != prepared.model_id or meta["state_id"] != prepared.state_id
                or meta["dependency_rule"] != "own_page_and_shared_immutable_model_only"):
            raise ValueError("syndrome page state/geometry identity mismatch")
        layer = _int(meta["layer"], "layer", 0, g["layers"] - 1)
        start = _int(meta["start"], "start")
        tokens = _int(meta["tokens"], "tokens", 1)
        b = _int(meta["residue_bits"], "residue_bits", 0, 7)
        if start + tokens >= (1 << 31) or tokens * 2 * g["channels"] > _MAX_VALUES:
            raise ValueError("syndrome decoded geometry exceeds bounds")
        if meta["correction"] not in ("syndrome", "position"):
            raise ValueError("unknown syndrome correction")
        columns, count = 2 * g["channels"], tokens * 2 * g["channels"]
        sizes = [_int(meta[name], name, 0, _MAX_PAYLOAD) for name in
                 ("feature_bytes", "exponent_bytes", "sign_bytes", "residue_bytes")]
        if sizes[0] != tokens * g["rank"] * 2 or sizes[3] != (count * b + 7) // 8:
            raise ValueError("syndrome feature/residue byte count mismatch")
        planes = meta["planes"]
        if not isinstance(planes, list) or len(planes) != 7 - b or [p.get("bit") for p in planes] != list(range(b, 7)):
            raise ValueError("syndrome bitplane coverage mismatch")
        plane_sizes = [_int(p["bytes"], "plane bytes", 0, _MAX_PAYLOAD) for p in planes]
        if sum(sizes) + sum(plane_sizes) != len(payload):
            raise ValueError("syndrome payload components mismatch")
        chunks, cursor = [], 0
        for size in sizes + plane_sizes:
            chunks.append(payload[cursor:cursor + size]); cursor += size
        if _sha(chunks[0]) != meta["features_sha256"]:
            raise ValueError("syndrome retained feature identity mismatch")
        features = np.frombuffer(chunks[0], dtype="<u2").reshape(tokens, g["rank"]).copy()
        if np.any((features & 0x7f80) == 0x7f80):
            raise ValueError("nonfinite retained syndrome feature")
        tick = time.perf_counter()
        raw_exp = _inflate(chunks[1], meta["exponent_mode"], count)
        centered = np.frombuffer(raw_exp, np.uint8).reshape(columns, tokens).T
        exponents = ((centered.astype(np.int16) + prepared.exponent_centers[layer][None] - 127) & 255).astype(np.uint16)
        sign_raw = _inflate(chunks[2], meta["sign_mode"], (count + 7) // 8)
        signs = _unpack(sign_raw, count, 1).astype(np.uint16).reshape(columns, tokens).T
        residues = _unpack(chunks[3], count, b).astype(np.uint16).reshape(columns, tokens).T
        result = (exponents << 7) | (signs << 15) | residues
        field_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        predictions = _prediction(prepared, layer, features, start, result.copy(), b) if b < 7 else np.zeros(result.shape + (7,), np.uint8)
        prediction_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        for description, chunk in zip(planes, chunks[4:]):
            desc = {k: v for k, v in description.items() if k not in ("bytes", "bit")}
            if desc.get("mode") != "raw" and desc.get("mode") != meta["correction"]:
                raise ValueError("syndrome plane correction identity mismatch")
            bit = description["bit"]
            values = _plane_decode(chunk, desc, predictions[:, :, bit], tokens, columns)
            result |= values.astype(np.uint16) << bit
        syndrome_seconds = time.perf_counter() - tick
        if _sha(_word_bytes(result)) != meta["words_sha256"]:
            raise ValueError("syndrome decoded original-word hash mismatch; predictor replay must be identical")
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed syndrome page") from error
    stats = dict(decode_seconds=time.perf_counter() - began, prediction_seconds=prediction_seconds,
        field_decode_seconds=field_seconds, syndrome_decode_seconds=syndrome_seconds,
        feature_bytes=sizes[0], dependency_depth=0, raw_prerequisite_pages=0,
        codec_execution="NumPy CPU syndrome/position and zlib reference")
    return (result, stats) if return_stats else result

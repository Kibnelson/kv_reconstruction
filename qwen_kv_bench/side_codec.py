"""Independent exact KV pages conditioned on their own retained BF16 code.

The probability network may run on a GPU.  This implementation deliberately
uses the existing CPU arithmetic reference, and makes no GPU throughput claim.
Decoding accepts only bytes plus immutable prepared model state; it never
consults original hidden states, original KV, or another page.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
import zlib

import numpy as np

from . import logistic_reference as entropy


FORMAT = "prefill_side_probability_page_v1"
_MAGIC = b"SPKVPG01"
_HEADER = struct.Struct("<8sIQI")
_MAX_HEADER = 4 * 1024 ** 2
_MAX_PAYLOAD = 64 * 1024 ** 2


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _words_bytes(words):
    return np.ascontiguousarray(words, dtype="<u2").tobytes()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value < 2 ** 31:
        raise ValueError("invalid side page " + name)
    return int(value)


def _geometry(prepared):
    try:
        geometry = prepared.metadata["geometry"]
        layers = _integer(geometry["layers"], "layers", 1)
        channels = _integer(geometry["channels"], "channels", 1)
        head_dim = _integer(geometry["head_dim"], "head_dim", 2)
        rank = _integer(geometry["rank"], "rank")
        mode = prepared.mode
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid prepared side page geometry") from error
    if mode not in ("none", "kv", "kv_canonical", "hidden") or (rank == 0) != (mode == "none") or head_dim % 2 or channels % head_dim:
        raise ValueError("inconsistent prepared side page geometry")
    for name in ("state_id", "model_id"):
        value = getattr(prepared, name, None)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("prepared side page needs a stable SHA256 " + name)
    return dict(layers=layers, channels=channels, head_dim=head_dim, rank=rank), mode


def _record(record, geometry):
    if not isinstance(record, dict):
        raise ValueError("side page source must be a producer record")
    try:
        layer = _integer(record["layer"], "layer")
        start = _integer(record["start"], "start")
        tokens = _integer(record["tokens"], "tokens", 1)
        head_dim = _integer(record["head_dim"], "head_dim", 2)
        k, v = record["k"], record["v"]
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete side page producer record") from error
    channels = geometry["channels"]
    if layer >= geometry["layers"] or head_dim != geometry["head_dim"] or tokens * 2 * channels > entropy.MAX_SYMBOLS:
        raise ValueError("side page source geometry outside model/safety bounds")
    if start + tokens >= 2 ** 31 or tokens * geometry["rank"] * 2 > _MAX_PAYLOAD:
        raise ValueError("side page address or retained code exceeds safety bounds")
    if any(not isinstance(value, np.ndarray) or value.dtype != np.uint16 or value.shape != (tokens, channels) for value in (k, v)):
        raise ValueError("side page K/V must contain actual uint16 BF16 words with matching geometry")
    return layer, start, tokens, np.concatenate((k, v), axis=1)


def _predict(prepared, layer, features, start, shape):
    mu, logs = prepared.predict(layer, features, start)
    if any(not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape
           or not np.all(np.isfinite(value)) for value in (mu, logs)):
        raise ValueError("side page predictor must return finite float32 probability arrays")
    return mu, logs


def encode_page(record, prepared, *, feature_words=None):
    """Return a complete independent wire page and separately charged timings.

    ``feature_words`` may supply the exact BF16 code already materialized by
    an inside-forward callback.  Its production cost belongs to the caller's
    prefill measurement; it is still transmitted and fully charged here.
    Probability prediction always consumes these transmitted word bits.
    """
    before = time.perf_counter()
    geometry, mode = _geometry(prepared)
    layer, start, tokens, target = _record(record, geometry)
    supplied = feature_words is not None
    if supplied:
        features, feature_seconds = feature_words, 0.
    else:
        tick = time.perf_counter()
        features = prepared.encode_features(record)
        feature_seconds = time.perf_counter() - tick
    if (not isinstance(features, np.ndarray) or features.dtype != np.uint16
            or features.shape != (tokens, geometry["rank"])):
        raise ValueError("side page encoder returned invalid BF16 retained code")
    if features.size and np.any((features & 0x7f80) == 0x7f80):
        raise ValueError("side page retained code must be finite BF16")
    features = features.copy()
    feature_payload = _words_bytes(features)
    tick = time.perf_counter()
    mu, logs = _predict(prepared, layer, features, start, target.shape)
    prediction_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    coded, descriptor = entropy.encode(target, mu, logs)
    entropy_seconds = time.perf_counter() - tick
    payload = feature_payload + coded
    meta = dict(format=FORMAT, version=1, mode=mode, model_id=prepared.model_id,
        state_id=prepared.state_id, layer=layer, start=start, tokens=tokens,
        geometry=geometry, feature_bytes=len(feature_payload), entropy_payload_bytes=len(coded),
        features_sha256=_sha(feature_payload), words_sha256=_sha(_words_bytes(target)),
        entropy=descriptor, dependency_rule="own_page_and_shared_immutable_model_only")
    header = _canonical(meta)
    if len(header) > _MAX_HEADER or len(payload) > _MAX_PAYLOAD:
        raise ValueError("side page exceeds serialized safety bounds")
    body = header + payload
    wire = _HEADER.pack(_MAGIC, len(header), len(payload), zlib.crc32(body)) + body
    stats = dict(raw_bytes=target.nbytes, feature_bytes=len(feature_payload),
        entropy_payload_bytes=len(coded), framing_bytes=_HEADER.size + len(header),
        packet_bytes=len(wire), ideal_bits=descriptor["ideal_continuous_bits"],
        integer_cdf_bits=descriptor["integer_cdf_bits"], arithmetic_bits=descriptor["arithmetic_bits"],
        retained_code_bits=8 * len(feature_payload),
        encode_seconds=time.perf_counter() - before, feature_encode_seconds=feature_seconds,
        feature_words_supplied=supplied,
        prediction_seconds=prediction_seconds, entropy_encode_seconds=entropy_seconds,
        layer=layer, start=start, tokens=tokens, dependency_depth=0,
        codec_execution="CPU arithmetic reference; neural inference uses prepared model device")
    return wire, stats


def _parse(wire):
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise ValueError("side page must be serialized bytes")
    wire = bytes(wire)
    if len(wire) < _HEADER.size:
        raise ValueError("truncated side page")
    magic, nh, npayload, checksum = _HEADER.unpack_from(wire)
    if (magic != _MAGIC or not 1 <= nh <= _MAX_HEADER or npayload > _MAX_PAYLOAD
            or _HEADER.size + nh + npayload != len(wire)):
        raise ValueError("invalid side page framing or safety bounds")
    body = wire[_HEADER.size:]
    if zlib.crc32(body) != checksum:
        raise ValueError("side page CRC mismatch")
    try:
        meta = json.loads(body[:nh])
        canonical = _canonical(meta)
    except (ValueError, UnicodeDecodeError, TypeError) as error:
        raise ValueError("invalid side page metadata") from error
    if not isinstance(meta, dict) or canonical != body[:nh]:
        raise ValueError("noncanonical side page metadata")
    return meta, body[nh:]


def inspect_page(wire):
    """Read integrity-checked addressing metadata; this does not restore KV."""
    return _parse(wire)[0]


def page_metadata(wire):
    """Return the page address used by cache assembly and random-order audits."""
    meta = inspect_page(wire)
    try:
        if meta.get("format") != FORMAT or meta.get("version") != 1:
            raise ValueError("unsupported side page")
        return dict(layer=_integer(meta["layer"], "layer"), start=_integer(meta["start"], "start"),
            tokens=_integer(meta["tokens"], "tokens", 1),
            channels=_integer(meta["geometry"]["channels"], "channels", 1))
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete side page address metadata") from error


def decode_page(wire, prepared, *, return_stats=False):
    """Restore exact joint K/V from this page and shared state, in any order."""
    before = time.perf_counter()
    geometry, mode = _geometry(prepared)
    meta, payload = _parse(wire)
    expected_keys = {"format", "version", "mode", "model_id", "state_id", "layer", "start", "tokens",
        "geometry", "feature_bytes", "entropy_payload_bytes", "features_sha256", "words_sha256",
        "entropy", "dependency_rule"}
    if (set(meta) != expected_keys or meta.get("format") != FORMAT or meta.get("version") != 1
            or meta.get("mode") != mode or meta.get("model_id") != prepared.model_id
            or meta.get("state_id") != prepared.state_id or meta.get("geometry") != geometry
            or meta.get("dependency_rule") != "own_page_and_shared_immutable_model_only"):
        raise ValueError("side page model, runtime, geometry or dependency contract mismatch")
    layer = _integer(meta["layer"], "layer")
    start = _integer(meta["start"], "start")
    tokens = _integer(meta["tokens"], "tokens", 1)
    feature_bytes = _integer(meta["feature_bytes"], "feature_bytes")
    coded_bytes = _integer(meta["entropy_payload_bytes"], "entropy_payload_bytes", 1)
    shape = (tokens, 2 * geometry["channels"])
    if (layer >= geometry["layers"] or start + tokens >= 2 ** 31 or np.prod(shape) > entropy.MAX_SYMBOLS
            or feature_bytes != tokens * geometry["rank"] * 2 or feature_bytes + coded_bytes != len(payload)
            or not isinstance(meta["entropy"], dict) or meta["entropy"].get("shape") != list(shape)):
        raise ValueError("side page payload/address geometry mismatch")
    feature_payload = payload[:feature_bytes]
    if _sha(feature_payload) != meta["features_sha256"]:
        raise ValueError("side page retained code SHA256 mismatch")
    features = np.frombuffer(feature_payload, dtype="<u2").astype(np.uint16, copy=True).reshape(tokens, geometry["rank"])
    if features.size and np.any((features & 0x7f80) == 0x7f80):
        raise ValueError("side page retained code must be finite BF16")
    tick = time.perf_counter()
    mu, logs = _predict(prepared, layer, features, start, shape)
    prediction_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    result = entropy.decode(payload[feature_bytes:], meta["entropy"], mu, logs)
    entropy_seconds = time.perf_counter() - tick
    if _sha(_words_bytes(result)) != meta["words_sha256"]:
        raise ValueError("side page restored words SHA256 mismatch")
    if return_stats:
        return result, dict(prediction_seconds=prediction_seconds,
            entropy_decode_seconds=entropy_seconds, decode_seconds=time.perf_counter() - before,
            feature_bytes=feature_bytes, dependency_depth=0)
    return result

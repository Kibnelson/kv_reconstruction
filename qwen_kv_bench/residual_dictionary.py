"""TRAIN-only dictionaries of errors left by prefill-derived affine prediction.

The dictionary learns signed 8D circular BF16-magnitude residuals, rather than
raw KV symbols. Decoder state is separately serialized, SHA256-bound, and fully
charged. The eager tensor codec manipulates integer symbols exactly; it neither
reruns transformer layers nor claims that a predictor creates lost information.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from .calibration import _cache_mapping, _case_identity, _json_copy

_MAGIC = b"RDKV"
_VERSION = 1
_HEADER = struct.Struct("<4sBIII")
_FORMAT = "qwen_prefill_residual_dictionary_v1"
_DOMAIN = "shifted_circular_residual"


def _canonical(metadata):
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _circular(values):
    return ((values + 16384) & 32767) - 16384


def _arrange(values, layout):
    if layout not in ("channel", "time"):
        raise ValueError("residual layout must be channel or time")
    return (values if layout == "channel" else values.T).reshape(-1)


def _np_affine(words):
    magnitude = words.astype(np.int64) & 32767
    rows = np.arange(len(words), dtype=np.int64)[:, None]
    prediction = magnitude[0:1] + (magnitude[-1:] - magnitude[0:1]) * rows // max(len(words) - 1, 1)
    return magnitude, prediction


def _vectors_from_cache(words, tile_tokens, layout):
    parts = []
    for start in range(0, len(words), tile_tokens):
        magnitude, prediction = _np_affine(words[start:start + tile_tokens])
        flat = _arrange(_circular(magnitude - prediction), layout)
        flat = np.pad(flat, (0, (-len(flat)) % 8))
        parts.append(flat.reshape(-1, 8))
    return np.concatenate(parts)


def _assign_np(vectors, centers, objective="bitcost"):
    residual = _circular(vectors[:, None, :] - centers[None, :, :])
    l1 = np.abs(residual).sum(axis=2)
    if objective == "l1":
        return l1.argmin(axis=1)
    zigzag = np.where(residual >= 0, 2 * residual, -2 * residual - 1)
    maximum = zigzag.max(axis=2)
    widths = (maximum[..., None] >= (1 << np.arange(15))).sum(axis=2)
    # Width is the actual payload-byte cost for an eight-value group. L1 is
    # only a deterministic tie-breaker and cannot outweigh a one-bit change.
    return (widths * 262145 + l1).argmin(axis=1)


def _fit(vectors, count, iterations, rng):
    # Keep a zero prototype, because affine anchors have exactly zero error.
    centers = np.zeros((count, 8), dtype=np.int64)
    if count > 1:
        centers[1] = vectors[int(rng.integers(len(vectors)))]
        minimum_score = np.full(len(vectors), np.iinfo(np.int64).max, dtype=np.int64)
        for j in range(2, count):
            # Update the nearest-center score incrementally: initialization is
            # O(samples * centers), not O(samples * centers squared).
            for center in (centers[:2] if j == 2 else centers[j - 1:j]):
                error = _circular(vectors - center)
                zigzag = np.where(error >= 0, 2 * error, -2 * error - 1)
                widths = (zigzag.max(axis=1)[:, None] >= (1 << np.arange(15))).sum(axis=1)
                score = widths * 262145 + np.abs(error).sum(axis=1)
                minimum_score = np.minimum(minimum_score, score)
            centers[j] = vectors[int(minimum_score.argmax())]
    for _ in range(iterations):
        ids = _assign_np(vectors, centers)
        for j in range(1, count):
            members = vectors[ids == j]
            if len(members):
                # Integer coordinate medians are an L1 update. Assignment is
                # bit-cost aware; this is not a globally optimal entropy fit.
                centers[j] = np.floor(np.median(members, axis=0)).astype(np.int64)
    ids = _assign_np(vectors, centers)
    remaining = _circular(vectors - centers[ids])
    zigzag = np.where(remaining >= 0, 2 * remaining, -2 * remaining - 1)
    widths = (zigzag.max(axis=1)[:, None] >= (1 << np.arange(15))).sum(axis=1)
    return centers.astype(np.int16), dict(sampled_vectors=len(vectors),
        training_mean_correction_width=float(widths.mean()),
        training_mean_absolute_circular_error=float(np.abs(remaining).mean()))


@dataclass
class ResidualModel:
    codebooks: dict[str, np.ndarray]
    metadata: dict
    path: str = ""

    @property
    def model_id(self):
        return hashlib.sha256(model_bytes(self)).hexdigest()

    @property
    def state_bytes(self):
        return len(model_bytes(self))

    @property
    def shifted_codebooks(self):
        return {key: (book.astype(np.int32) + 16384).astype(np.uint16)
                for key, book in self.codebooks.items()}

    @property
    def config(self):
        return self.metadata["configuration"]


def model_bytes(model):
    metadata = _json_copy(model.metadata)
    _validate_metadata(metadata)
    keys = metadata["matrix_keys"]
    count = metadata["configuration"]["n_centroids"]
    if set(keys) != set(model.codebooks):
        raise ValueError("residual dictionary keys disagree with metadata")
    payload = bytearray()
    for key in keys:
        book = np.asarray(model.codebooks[key])
        if book.dtype != np.int16 or book.shape != (count, 8) or np.any((book < -16384) | (book > 16383)):
            raise ValueError("invalid signed circular residual dictionary")
        payload.extend(book.astype("<i2").tobytes())
    meta = _canonical(metadata)
    data = meta + bytes(payload)
    return _HEADER.pack(_MAGIC, _VERSION, len(meta), len(payload), zlib.crc32(data)) + data


def _validate_metadata(metadata):
    if metadata.get("format") != _FORMAT or metadata.get("training_only") is not True:
        raise ValueError("not a training-only residual dictionary")
    cases = metadata.get("training_cases", [])
    if len(cases) != 3 or any(case.get("split") != "train" for case in cases):
        raise ValueError("residual dictionary requires three TRAIN sources")
    if metadata.get("training_hashes") != [case.get("identity_sha256") for case in cases]:
        raise ValueError("residual dictionary provenance mismatch")
    config = metadata.get("configuration", {})
    if (config.get("layout") not in ("channel", "time") or config.get("predictor") != "affine"
            or config.get("vector_dim") != 8 or config.get("book_domain") != _DOMAIN
            or not isinstance(config.get("tile_tokens"), int) or config["tile_tokens"] < 1
            or config.get("n_centroids") not in (1, 4, 16, 64, 256)):
        raise ValueError("invalid residual dictionary configuration")
    keys = metadata.get("matrix_keys")
    if not isinstance(keys, list) or not keys or sorted(set(keys)) != keys:
        raise ValueError("invalid residual dictionary key order")
    if not metadata.get("source_model_identity"):
        raise ValueError("missing source model identity")


def load(path, *, expected_model_identity=None, expected_training_hashes=None, expected_model_id=None):
    source = Path(path)
    data = source.read_bytes()
    if len(data) < _HEADER.size:
        raise ValueError("truncated residual dictionary")
    magic, version, metadata_bytes, payload_bytes, checksum = _HEADER.unpack_from(data)
    body = data[_HEADER.size:]
    if magic != _MAGIC or version != _VERSION or len(body) != metadata_bytes + payload_bytes or zlib.crc32(body) != checksum:
        raise ValueError("invalid residual dictionary header or CRC")
    try:
        metadata = json.loads(body[:metadata_bytes])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid residual dictionary JSON") from error
    _validate_metadata(metadata)
    if _canonical(metadata) != body[:metadata_bytes]:
        raise ValueError("noncanonical residual dictionary JSON")
    if expected_model_identity is not None and _json_copy(expected_model_identity) != metadata["source_model_identity"]:
        raise ValueError("residual dictionary source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes) != metadata["training_hashes"]:
        raise ValueError("residual dictionary training identity mismatch")
    count = metadata["configuration"]["n_centroids"]
    keys = metadata["matrix_keys"]
    if payload_bytes != len(keys) * count * 8 * 2:
        raise ValueError("residual dictionary array size mismatch")
    arrays = np.frombuffer(body, dtype="<i2", offset=metadata_bytes).reshape(len(keys), count, 8)
    books = {}
    for key, book in zip(keys, arrays):
        copy = book.astype(np.int16, copy=True)
        if np.any((copy < -16384) | (copy > 16383)):
            raise ValueError("out-of-domain residual dictionary prototype")
        copy.flags.writeable = False
        books[key] = copy
    result = ResidualModel(books, metadata, str(source.resolve()))
    if model_bytes(result) != data or (expected_model_id is not None and result.model_id != expected_model_id):
        raise ValueError("residual dictionary SHA256 mismatch")
    return result


def train(train_caches, output_dir, *, tile_tokens, layout="channel", n_centroids=64,
          seed=71, max_samples_per_key=1024, iterations=8, training_metadata=None, model_identity=None):
    """Freeze a separate dictionary for each (layer,K/V), using three TRAIN caches.

    The actual source tensors are hashed before and after fitting. Metadata and
    array bytes together form the model ID. No DEV/TEST fit or format selection
    occurs here. Fitting uses bit-width assignment plus integer L1 updates.
    """
    if not isinstance(train_caches, Sequence) or isinstance(train_caches, (str, bytes)) or len(train_caches) != 3:
        raise ValueError("residual fitting requires three explicit TRAIN captures")
    if layout not in ("channel", "time"):
        raise ValueError("residual layout must be channel or time")
    for value, minimum in ((tile_tokens, 1), (seed, 0), (iterations, 1), (max_samples_per_key, n_centroids)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("invalid deterministic residual training configuration")
    if isinstance(n_centroids, bool) or n_centroids not in (1, 4, 16, 64, 256):
        raise ValueError("supported residual dictionary sizes: 1,4,16,64,256")
    identity = _json_copy(model_identity)
    if not isinstance(identity, (dict, str)) or not identity:
        raise ValueError("explicit source model identity is required")
    if training_metadata is not None and len(training_metadata) != 3:
        raise ValueError("metadata required for all three TRAIN captures")
    metadata_rows = list(training_metadata) if training_metadata is not None else [None] * 3
    caches = [_cache_mapping(cache) for cache in train_caches]
    # Reject split labels carried by capture wrappers as well as metadata args.
    for cache in train_caches:
        if isinstance(cache, dict) and cache.get("split", "train") not in ("train", "training"):
            raise ValueError("residual fitting accepts TRAIN captures only")
    cases = [_case_identity(cache, meta, i) for i, (cache, meta) in enumerate(zip(caches, metadata_rows))]
    schema = (cases[0]["matrix_keys"], cases[0]["columns"], cases[0]["layers"])
    if any((case["matrix_keys"], case["columns"], case["layers"]) != schema for case in cases[1:]):
        raise ValueError("training cache architectures differ")
    architecture = identity.get("config", identity) if isinstance(identity, dict) else {}
    if isinstance(architecture, dict):
        if architecture.get("num_hidden_layers") is not None and int(architecture["num_hidden_layers"]) != schema[2]:
            raise ValueError("training layers disagree with source model")
        if architecture.get("head_dim") is not None and architecture.get("num_key_value_heads") is not None:
            if int(architecture["head_dim"]) * int(architecture["num_key_value_heads"]) != schema[1]:
                raise ValueError("training channels disagree with explicit model dimensions")
    rng = np.random.default_rng(seed)
    books, statistics = {}, {}
    for key in sorted(caches[0]):
        vectors = np.concatenate([_vectors_from_cache(cache[key], tile_tokens, layout) for cache in caches])
        available = len(vectors)
        if available > max_samples_per_key:
            vectors = vectors[rng.choice(available, max_samples_per_key, replace=False)]
        book, stats = _fit(vectors, n_centroids, iterations, rng)
        book.flags.writeable = False
        books[key] = book
        statistics[key] = dict(available_vectors=available, **stats)
    before = [case["identity_sha256"] for case in cases]
    after = [_case_identity(cache, meta, i)["identity_sha256"] for i, (cache, meta) in enumerate(zip(caches, metadata_rows))]
    if before != after:
        raise AssertionError("TRAIN caches changed during residual dictionary fitting")
    metadata = dict(format=_FORMAT, family="prefill_residual", training_only=True, source_model_identity=identity,
        training_cases=cases, training_hashes=before, matrix_keys=sorted(books),
        configuration=dict(tile_tokens=tile_tokens, layout=layout, n_centroids=n_centroids,
            seed=seed, max_samples_per_key=max_samples_per_key, iterations=iterations,
            predictor="affine", vector_dim=8, book_domain=_DOMAIN,
            fit_objective="circular zigzag group-width assignment; coordinate median L1 updates; fixed zero prototype",
            assignment_tie_break="minimum circular L1, then first dictionary index"),
        training_stats=statistics,
        scope="Frozen signed residual patterns after tile-local affine prefill prediction. Exact final corrections are always retained.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"recipe_residual{n_centroids}_{layout}_t{tile_tokens}.bin"
    if path.exists():
        raise FileExistsError("residual calibration exists; use a fresh output directory")
    result = ResidualModel(books, metadata, str(path))
    path.write_bytes(model_bytes(result))
    return load(path, expected_model_identity=identity, expected_training_hashes=before,
                expected_model_id=result.model_id)


def make_state(model, device="cpu"):
    from . import gpu_ops as g
    state = g.make_state(model.shifted_codebooks, device=device, state_id=model.model_id)
    state.residual_dictionary_config = _json_copy(model.config)
    state.residual_dictionary_serialized_bytes = model.state_bytes
    return state


def _torch_arrange(values, layout):
    return (values if layout == "channel" else values.T).contiguous().reshape(-1)


def _torch_unarrange(flat, shape, layout):
    return flat.reshape(shape) if layout == "channel" else flat.reshape(shape[1], shape[0]).T.contiguous()


def _assign_torch(vectors, book, assignment):
    import torch
    ids = []
    for start in range(0, len(vectors), 256):
        error = _circular(vectors[start:start + 256, None, :] - book[None, :, :])
        l1 = error.abs().sum(dim=2)
        if assignment == "l1":
            score = l1
        else:
            zz = torch.where(error >= 0, 2 * error, -2 * error - 1)
            maximum = zz.amax(dim=2)
            widths = (maximum[:, :, None] >= (1 << torch.arange(15, device=book.device))).sum(dim=2)
            score = widths * 262145 + l1
        ids.append(score.argmin(dim=1))
    return torch.cat(ids) if ids else torch.empty(0, device=book.device, dtype=torch.int64)


def encode(words, state, *, key, layout="channel", group_size=8, assignment="bitcost"):
    import torch
    from . import gpu_ops as g
    if words.dtype != torch.uint16 or words.ndim != 2:
        raise ValueError("expected torch.uint16 [tokens,channels] BF16 words")
    if layout not in ("channel", "time") or group_size != 8 or assignment not in ("bitcost", "l1"):
        raise ValueError("residual dictionaries require 8D channel/time groups and bitcost/l1 assignment")
    if state is None or key not in state.codebooks:
        raise ValueError("prefill residual dictionary state is required")
    config = getattr(state, "residual_dictionary_config", None)
    if config and (config["layout"] != layout or words.shape[0] > config["tile_tokens"]):
        raise ValueError("residual dictionary layout or tile differs from TRAIN configuration")
    book = state.codebooks[key].to(torch.int64) - 16384
    if book.device != words.device:
        raise ValueError("dictionary and KV words must share a device")
    count = words.numel()
    fields, descriptions = {}, {}
    magnitude = words.to(torch.int64) & 32767
    fields["original_signs"], _ = g._pack_fixed(words.reshape(-1).to(torch.int64) >> 15, 1)
    prediction = g._prefill_prediction(magnitude, fields, descriptions, "affine", 8, 6)
    flat = _torch_arrange(_circular(magnitude - prediction), layout)
    padded_count = ((count + 7) // 8) * 8
    padded = torch.cat((flat, torch.zeros(padded_count - count, device=words.device, dtype=torch.int64)))
    ids = _assign_torch(padded.reshape(-1, 8), book, assignment)
    index_bits = (len(book) - 1).bit_length()
    fields["residual_ids"], _ = g._pack_fixed(ids, index_bits)
    final = _circular(padded.reshape(-1, 8) - book[ids]).reshape(-1)[:count]
    g._adaptive(final, fields, descriptions, "prefill_correction", 8, True, 15)
    fields["prefill_correction_widths"], _ = g._pack_fixed(fields["prefill_correction_widths"], 4)
    descriptions["prefill_correction"]["width_storage_bits"] = 4
    meta = dict(version=1, method="prefill_dictionary", shape=list(words.shape), key=key,
        state_id=state.state_id, model_id=state.state_id, group_size=8, assignment=assignment,
        layout=layout, predictor="affine", correction="residual", circular_symbol_bits=15,
        book_domain=_DOMAIN, learned_index_bits=index_bits, n_centroids=len(book),
        padded_count=padded_count, descriptions=descriptions, original_cpu_format=False,
        prediction_scope="same-tile affine prefill metadata plus frozen TRAIN residual dictionary",
        dictionary_training_config=config,
        external_state_contract="entire frozen signed residual dictionary; shifted by 16384 on device")
    return g._finish(fields, meta, words.device)


def decode(packet, state):
    import math
    import torch
    from . import gpu_ops as g
    g.validate_packet(packet, check_header=not packet._header_verified)
    meta, fields = packet.metadata, packet.fields
    if (meta.get("method") != "prefill_dictionary" or state is None
            or meta.get("model_id") != state.state_id or meta.get("state_id") != state.state_id):
        raise ValueError("residual dictionary state identity mismatch")
    if meta.get("book_domain") != _DOMAIN or meta.get("layout") not in ("channel", "time") or meta.get("group_size") != 8:
        raise ValueError("invalid residual dictionary descriptor")
    config = getattr(state, "residual_dictionary_config", None)
    if meta.get("dictionary_training_config") != config:
        raise ValueError("residual dictionary configuration mismatch")
    if meta["key"] not in state.codebooks:
        raise ValueError("missing residual dictionary key")
    book = state.codebooks[meta["key"]].to(torch.int64) - 16384
    count = math.prod(meta["shape"])
    padded_count = ((count + 7) // 8) * 8
    if meta.get("padded_count") != padded_count or meta.get("n_centroids") != len(book) or meta.get("learned_index_bits") != (len(book) - 1).bit_length():
        raise ValueError("residual dictionary geometry mismatch")
    if book.device != packet.header.device:
        raise ValueError("dictionary and packet must share a device")
    ids = g._unpack_fixed(fields["residual_ids"], padded_count // 8, meta["learned_index_bits"])
    final = g._restore(fields, meta["descriptions"], "prefill_correction")
    residual_flat = (book[ids].reshape(-1)[:count] + final)
    residual = _torch_unarrange(residual_flat, meta["shape"], meta["layout"])
    prediction = g._restore_prefill_prediction(meta, fields)
    signs = g._unpack_fixed(fields["original_signs"], count, 1).reshape(meta["shape"])
    return (((prediction + residual) & 32767) | (signs << 15)).to(torch.uint16)

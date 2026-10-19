"""Qwen-native, training-only dictionaries for the local KV benchmark.

This reuses generic training algorithms, never a pretrained SmolLM dictionary.
Inputs are actual uint16 BF16 cache matrices supplied by the benchmark's three
training captures. Model identities bind the source model, training hashes and
configuration. Serialized model files are the complete external-state charges.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from typing import Any

import numpy as np

from kv_research import learned_codec
from kv_research.train_models import fixed_model
from kv_coset_fields import learned_lower

_FORMAT = "qwen_kv_calibration_v1"
_NAMESPACE = "qwen_kv_bench"
_KEY = re.compile(r"layer_(\d+)_(K|V)$")
_REQUIREMENTS = ("magnitude_sizes", "fixed_sizes", "lower_sizes", "conditional_lower_sizes")


@dataclass
class CalibrationArtifacts:
    models: dict[str, Any]
    paths: dict[str, str]
    state_bytes: dict[str, int]
    metadata: dict[str, Any]
    report_path: str

    @property
    def report_sha256(self):
        return _sha(Path(self.report_path))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("calibration identities/metadata must be finite JSON values") from error


def normalize_requirements(requirements=None):
    request = {key: [] for key in _REQUIREMENTS}
    request["magnitude_sizes"] = [64]
    if requirements is not None:
        if set(requirements) - set(_REQUIREMENTS):
            raise ValueError("unknown calibration requirement")
        request = {key: list(requirements.get(key, [])) for key in _REQUIREMENTS}
    for key, values in request.items():
        allowed = (16, 64, 256) if key == "magnitude_sizes" else (64,) if key == "fixed_sizes" else (16, 64)
        if any(isinstance(value, bool) or not isinstance(value, int) or value not in allowed for value in values):
            raise ValueError("unsupported calibration dictionary size")
        request[key] = sorted(set(values))
    return request


def _cache_mapping(cache):
    if isinstance(cache, Mapping):
        if cache and all(isinstance(key, str) and _KEY.fullmatch(key) for key in cache):
            return {key: np.asarray(array) for key, array in cache.items()}
        for field in ("cache", "matrices"):
            if field in cache:
                return _cache_mapping(cache[field])
    if hasattr(cache, "matrices"):
        return _cache_mapping(cache.matrices)
    if isinstance(cache, (list, tuple)):
        result = {}
        for layer, pair in enumerate(cache):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("cache layers must contain K/V pairs")
            for kind, array in zip(("K", "V"), pair):
                result[f"layer_{layer:02d}_{kind}"] = np.asarray(array)
        return result
    raise ValueError("expected a layer/KV matrix mapping or list of K/V pairs")


def _case_identity(cache, metadata, index):
    keys = sorted(cache, key=lambda key: (int(_KEY.fullmatch(key).group(1)), key[-1]))
    if not keys:
        raise ValueError("empty training cache")
    layer_ids = {int(_KEY.fullmatch(key).group(1)) for key in keys}
    expected = {f"layer_{layer:02d}_{kind}" for layer in range(len(layer_ids)) for kind in ("K", "V")}
    if layer_ids != set(range(len(layer_ids))) or set(keys) != expected:
        raise ValueError("training cache requires contiguous complete layer/KV keys")
    shape = cache[keys[0]].shape
    if len(shape) != 2 or min(shape) < 1 or shape[1] % 8:
        raise ValueError("training matrices need positive[T,C] dimensions and C multiple of8")
    digest = hashlib.sha256(); nonfinite = 0
    layout = []
    for key in keys:
        array = cache[key]
        if array.dtype != np.uint16 or array.ndim != 2 or array.shape != shape:
            raise ValueError("all training K/V matrices must be matching native uint16[T,C]")
        digest.update(array.astype("<u2").tobytes())
        nonfinite += int(np.count_nonzero((array & 0x7F80) == 0x7F80))
        layout.append({"key": key, "shape": list(array.shape), "dtype": "BF16 words uint16"})
    metadata = _json_copy(metadata or {})
    if metadata.get("split", "train") not in ("train", "training"):
        raise ValueError("calibration accepts TRAIN captures only")
    for field in ("raw_sha256", "canonical_raw_sha256"):
        if field in metadata and metadata[field] != digest.hexdigest():
            raise ValueError("training metadata raw hash disagrees with cache")
    ids = metadata.get("input_token_ids", metadata.get("input_ids"))
    if ids is not None:
        if len(ids) != shape[0] or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError("training token IDs do not match the cache length")
        token_sha = hashlib.sha256(np.asarray(ids, dtype="<u8").tobytes()).hexdigest()
    else:
        token_sha = None
    raw_sha = digest.hexdigest()
    layout_sha = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = hashlib.sha256(bytes.fromhex(raw_sha) + bytes.fromhex(layout_sha)).hexdigest()
    return dict(name=metadata.get("name", metadata.get("fixture", f"train_case_{index:02d}")), split="train",
                tokens=shape[0], columns=shape[1], layers=len(layer_ids), matrix_keys=keys,
                raw_payload_bytes=sum(array.nbytes for array in cache.values()), canonical_raw_sha256=raw_sha,
                layout_sha256=layout_sha, identity_sha256=identity, nonfinite_word_count=nonfinite,
                input_token_ids_sha256=token_sha, supplied_metadata=metadata)


def train_calibration(train_caches, output_dir, requirements=None, *, seed=71,
                      max_samples_per_key=1024, iterations=8, training_metadata=None,
                      model_identity=None, overwrite=False):
    """Train only on three supplied TRAIN caches and persist complete state.

    train_caches: sequence of {layer_00_K/V:uint16[T,C]} or lists of(K,V)pairs.
    requirements: registry descriptor with magnitude_sizes/fixed_sizes/
      lower_sizes/conditional_lower_sizes. Default fits learned64 only.
    model_identity: required finite JSON identity of the actual source model.
    No file containing KV or any pretrained dictionary is read by this function.
    """
    if not isinstance(train_caches, Sequence) or isinstance(train_caches, (str, bytes)) or len(train_caches) != 3:
        raise ValueError("calibration requires the three explicit training captures")
    identity = _json_copy(model_identity)
    if not isinstance(identity, (dict, str)) or not identity:
        raise ValueError("explicit nonempty source model_identity is required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("invalid deterministic training configuration")
    request = normalize_requirements(requirements)
    requested_sizes = request["magnitude_sizes"] + request["lower_sizes"] + request["conditional_lower_sizes"]
    if isinstance(max_samples_per_key, bool) or not isinstance(max_samples_per_key, int) or max_samples_per_key < max(requested_sizes, default=1):
        raise ValueError("sample budget must cover the requested dictionary sizes")
    if training_metadata is not None and len(training_metadata) != 3:
        raise ValueError("expected metadata for all three training captures")
    metadata_rows = list(training_metadata) if training_metadata is not None else [None] * 3
    caches = [_cache_mapping(cache) for cache in train_caches]
    cases = [_case_identity(cache, meta, i) for i, (cache, meta) in enumerate(zip(caches, metadata_rows))]
    schema = (cases[0]["matrix_keys"], cases[0]["columns"], cases[0]["layers"])
    if any((case["matrix_keys"], case["columns"], case["layers"]) != schema for case in cases[1:]):
        raise ValueError("training caches have different layer/head layouts")
    # Bind any explicitly supplied architecture fields as well as its identity.
    # In particular, Qwen's explicit head_dim must never be replaced by
    # hidden_size/num_attention_heads, which need not equal the projected width.
    architecture = identity.get("config", identity) if isinstance(identity, dict) else {}
    if isinstance(architecture, dict):
        if architecture.get("num_hidden_layers") is not None and int(architecture["num_hidden_layers"]) != schema[2]:
            raise ValueError("training layer count differs from source model configuration")
        if architecture.get("head_dim") is not None and architecture.get("num_key_value_heads") is not None:
            expected_columns = int(architecture["head_dim"]) * int(architecture["num_key_value_heads"])
            if expected_columns != schema[1]:
                raise ValueError("training columns differ from explicit source model KV head dimensions")
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    report_path = output / "calibration_manifest.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError("calibration already exists: validate/load it explicitly or use a fresh directory")
    training = {key: [cache[key] for cache in caches] for key in cases[0]["matrix_keys"]}
    training_hashes = [case["identity_sha256"] for case in cases]
    provenance = {"calibration_namespace": _NAMESPACE, "source_model_identity": identity,
                  "source_training_hashes": training_hashes}
    models, paths, state_bytes, model_reports = {}, {}, {}, {}
    specifications = [(f"learned{size}", "magnitude", size, False) for size in request["magnitude_sizes"]]
    specifications += [(f"fixed{size}", "fixed", size, False) for size in request["fixed_sizes"]]
    specifications += [(f"lower{size}", "lower", size, False) for size in request["lower_sizes"]]
    specifications += [(f"lower{size}_conditional", "lower", size, True) for size in request["conditional_lower_sizes"]]
    started = time.perf_counter()
    for name, family, count, conditional in specifications:
        begin = time.perf_counter()
        fit_config = {"n_centroids": count, "iterations": iterations, "seed": seed,
                      "max_samples_per_key": max_samples_per_key, **provenance}
        if family == "magnitude":
            model = learned_codec.train_model(training, fit_config)
        elif family == "fixed":
            model = fixed_model(cases[0]["matrix_keys"])
            model.config.update(calibration_namespace=_NAMESPACE, source_model_identity=identity,
                                calibration_kind="source_independent_fixed_control")
        else:
            fit_config.update(conditional=conditional, mean_prior_count=16)
            model = learned_lower.train_model(training, fit_config)
        fit_seconds = time.perf_counter() - begin
        path = output / (name + ".bin")
        if family == "lower":
            saved = learned_lower.save_model(model, path)
            arrays = sum(book.nbytes for book in model.codebooks.values()) + sum(mean.nbytes for mean in model.exponent_means.values())
        else:
            saved = learned_codec.save_model(model, path)
            arrays = sum(book.nbytes for book in model.codebooks.values())
        check_size = path.stat().st_size
        if check_size != saved["serialized_bytes"] or sha_model(model) != _sha(path):
            raise AssertionError("serialized calibration state accounting/identity mismatch")
        models[name], paths[name], state_bytes[name] = model, str(path), check_size
        model_reports[name] = dict(file=path.name, family=family, trained=family != "fixed", n_centroids=count,
            conditional=conditional, serialized_bytes=check_size, parameter_array_bytes=arrays,
            model_id=sha_model(model), sha256=_sha(path), training_seconds=fit_seconds,
            matrix_keys=len(model.codebooks), config=_json_copy(model.config))
    # Detect accidental mutation/asynchronous capture races before freezing.
    after = [_case_identity(cache, meta, i)["identity_sha256"] for i, (cache, meta) in enumerate(zip(caches, metadata_rows))]
    if after != training_hashes:
        raise AssertionError("training caches changed during calibration")
    report = dict(format=_FORMAT, calibration_namespace=_NAMESPACE,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), model_identity=identity,
        training_only=True, training_cases=cases, training_hashes=training_hashes,
        schema=dict(layers=schema[2], columns=schema[1], matrix_keys=schema[0], vector_dim=8,
                    note="Captured cache shapes; no hidden_size/num_attention_heads head-dimension inference."),
        requirements=request, configuration=dict(seed=seed, max_samples_per_key=max_samples_per_key, iterations=iterations),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, fit_device="CPU"),
        elapsed_seconds=time.perf_counter() - started, models=model_reports,
        scope="Actual source-model TRAIN KV only. No development/test KV, old SmolLM dictionary, or external activation used in fitting. Fixed control is explicitly untrained and data-independent.")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return CalibrationArtifacts(models, paths, state_bytes, report, str(report_path))


def sha_model(model):
    return model.model_id


def load_calibration(output_dir, *, expected_model_identity=None, expected_training_hashes=None, require_names=None):
    """Load only this benchmark's provenance-bound artifacts and verify bytes."""
    output = Path(output_dir).resolve(); report_path = output / "calibration_manifest.json"
    report = json.loads(report_path.read_text())
    if report.get("format") != _FORMAT or report.get("calibration_namespace") != _NAMESPACE or not report.get("training_only"):
        raise ValueError("not a Qwen training-only calibration artifact")
    if len(report.get("training_cases", [])) != 3 or any(case.get("split") != "train" for case in report["training_cases"]):
        raise ValueError("calibration provenance contains non-training inputs")
    if expected_model_identity is not None and _json_copy(expected_model_identity) != report["model_identity"]:
        raise ValueError("calibration source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes) != report["training_hashes"]:
        raise ValueError("calibration training hashes mismatch")
    if require_names is not None and set(require_names) - set(report["models"]):
        raise ValueError("required calibrated models are missing")
    models, paths, charges = {}, {}, {}
    for name, entry in report["models"].items():
        path = output / entry["file"]
        if path.parent != output or path.stat().st_size != entry["serialized_bytes"] or _sha(path) != entry["sha256"]:
            raise ValueError("calibration model file size/hash mismatch")
        model = learned_lower.load_model(path) if entry["family"] == "lower" else learned_codec.load_model(path)
        if model.model_id != entry["model_id"] or model.config.get("calibration_namespace") != _NAMESPACE:
            raise ValueError("calibration model identity/namespace mismatch")
        if model.config.get("source_model_identity") != report["model_identity"]:
            raise ValueError("calibration model is bound to another source model")
        if entry["trained"] and model.config.get("source_training_hashes") != report["training_hashes"]:
            raise ValueError("calibration model is bound to other training inputs")
        models[name], paths[name], charges[name] = model, str(path), path.stat().st_size
    return CalibrationArtifacts(models, paths, charges, report, str(report_path))


fit_calibration = train_calibration

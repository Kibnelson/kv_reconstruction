"""Pinned BF16 model capture and explicit CPU/CUDA/XPU correctness helpers.

No automatic model offload. A full GPU cache is used for capture and generation;
this module does not claim compact-resident model attention or a capacity win.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from contextlib import nullcontext
from pathlib import Path
import platform
import time
from typing import Callable

import numpy as np

try:
    import torch
except ImportError:  # Doctor remains useful before installing PyTorch.
    torch = None


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is not installed; install the backend build for this machine")
    return torch


def select_device(requested="auto"):
    """Auto means CUDA, then XPU. CPU must be explicitly requested."""
    t = _require_torch()
    if str(requested) == "auto":
        for kind in ("cuda", "xpu"):
            backend = getattr(t, kind, None)
            if backend is not None and backend.is_available():
                return t.device(kind, backend.current_device())
        raise RuntimeError("No CUDA or XPU device is available; use --device cpu explicitly for a CPU-only correctness run")
    device = t.device(requested)
    if device.type == "cpu":
        return t.device("cpu")
    if device.type not in ("cuda", "xpu"):
        raise ValueError("Only explicit cpu, CUDA, and XPU backends are supported")
    backend = getattr(t, device.type, None)
    if backend is None or not backend.is_available():
        raise RuntimeError(f"Requested {device.type} backend is unavailable in this PyTorch installation")
    index = backend.current_device() if device.index is None else device.index
    if not 0 <= index < backend.device_count():
        raise ValueError("Requested accelerator index does not exist")
    return t.device(device.type, index)


def synchronize(device):
    device = _require_torch().device(device)
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def memory_snapshot(device):
    """Allocator counters and device free bytes when the backend exposes them."""
    device = _require_torch().device(device)
    if device.type == "cpu":
        return {"device": "cpu", "allocated_bytes": None, "reserved_bytes": None,
                "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                "free_device_bytes": None, "total_device_bytes": None,
                "note": "CPU allocator peaks are not measured by this helper"}
    backend = getattr(torch, device.type)
    result = {"device": str(device)}
    for field, name in (("allocated_bytes", "memory_allocated"),
                        ("reserved_bytes", "memory_reserved"),
                        ("peak_allocated_bytes", "max_memory_allocated"),
                        ("peak_reserved_bytes", "max_memory_reserved")):
        function = getattr(backend, name, None)
        try:
            result[field] = int(function(device)) if function is not None else None
        except (RuntimeError, TypeError, NotImplementedError):
            result[field] = None
    try:
        free, total = backend.mem_get_info(device)
        result.update(free_device_bytes=int(free), total_device_bytes=int(total))
    except (AttributeError, RuntimeError, TypeError, NotImplementedError):
        result.update(free_device_bytes=None, total_device_bytes=None)
    return result


def reset_peak_memory(device):
    device = _require_torch().device(device)
    synchronize(device)
    if device.type != "cpu":
        function = getattr(getattr(torch, device.type), "reset_peak_memory_stats", None)
        if function is not None:
            function(device)
    return memory_snapshot(device)


def timed_call(function: Callable, device):
    """Return (value, timings); host synchronized wall and optional GPU events."""
    device = _require_torch().device(device)
    backend = getattr(torch, device.type, None)
    context = backend.device(device) if device.type != "cpu" else nullcontext()
    with context:
        synchronize(device)
        events = None
        if device.type != "cpu":
            try:
                events = (backend.Event(enable_timing=True), backend.Event(enable_timing=True))
            except (AttributeError, TypeError, RuntimeError, NotImplementedError):
                events = None
        begin = time.perf_counter()
        if events:
            events[0].record()
        value = function()
        if events:
            events[1].record()
        synchronize(device)
        wall = time.perf_counter() - begin
        try:
            elapsed = float(events[0].elapsed_time(events[1])) / 1000 if events else None
        except (AttributeError, RuntimeError, NotImplementedError):
            elapsed = None
    return value, {"wall_seconds": wall, "device_event_seconds": elapsed,
                   "device": str(device), "synchronized": True}


def _bf16_probe(device):
    device = _require_torch().device(device)
    try:
        x = torch.ones((8, 8), dtype=torch.bfloat16, device=device)
        y = x @ x
        ok = y.dtype == torch.bfloat16 and bool(torch.all(y == 8).item())
        synchronize(device)
        del x, y
        return {"success": ok, "error": None,
                "scope": "BF16 allocation and matmul functionality, not proof of native hardware execution or SDPA support"}
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}",
                "scope": "BF16 allocation and matmul functionality"}


def operation_probes(device):
    """Small independent native-codec checks; failures do not veto BF16 use.

    Success establishes only the listed operations on this backend, not every
    codec kernel or hardware-native execution. No model or network is accessed.
    """
    device = _require_torch().device(device)

    def int64_bitwise():
        values = torch.tensor([0, 1, 255, 32767], dtype=torch.int64, device=device)
        result = ((values >> 3) ^ (values & 7)) << 1
        expected = torch.tensor([0, 2, 48, 8176], dtype=torch.int64, device=device)
        return bool(torch.equal(result, expected))

    def uint16_view_copy():
        values = torch.tensor([0, 1, 0x8000, 0x7F80, 0xFFFF], dtype=torch.uint16, device=device)
        copied = values.view(torch.bfloat16).contiguous().view(torch.uint16).clone().to("cpu")
        expected = torch.tensor([0, 1, 0x8000, 0x7F80, 0xFFFF], dtype=torch.uint16)
        return bool(torch.equal(copied, expected))

    def float64_pointwise():
        values = torch.tensor([-3.0, 0.5, 2.0], dtype=torch.float64, device=device)
        result = torch.round(values * 0.5 + 1.0)
        expected = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device=device)
        return bool(torch.equal(result, expected))

    probes = {}
    for name, function, scope in (
        ("int64_bitwise", int64_bitwise, "int64 device allocation, shifts, XOR, AND and equality"),
        ("uint16_view_copy", uint16_view_copy, "uint16/BF16 bit reinterpretation, device clone and copy to CPU"),
        ("float64_pointwise", float64_pointwise, "float64 device allocation, multiply/add, rounding and equality"),
    ):
        try:
            success = function()
            synchronize(device)
            probes[name] = {"success": success, "error": None if success else "Unexpected probe result", "scope": scope}
        except Exception as error:
            probes[name] = {"success": False, "error": f"{type(error).__name__}: {error}", "scope": scope}
    return probes


def doctor(requested="auto", probe=True):
    """Read local capability information only; no model/tokenizer download."""
    versions = {}
    for name in ("torch", "transformers", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "versions": versions,
              "requested_device": str(requested), "accelerators": [],
              "model_download_performed": False}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                  for line in meminfo.read_text().splitlines()}
        result["host_available_memory_bytes"] = int(fields["MemAvailable"].split()[0]) * 1024 if "MemAvailable" in fields else None
    if torch is not None:
        result.update(torch_cuda_build=torch.version.cuda,
                      torch_hip_build=getattr(torch.version, "hip", None))
        for kind in ("cuda", "xpu"):
            backend = getattr(torch, kind, None)
            available = backend is not None and backend.is_available()
            entry = {"backend": kind, "available": bool(available), "devices": []}
            if available:
                for index in range(backend.device_count()):
                    dev = torch.device(kind, index)
                    info = {"index": index, "name": backend.get_device_name(index),
                            "memory": memory_snapshot(dev)}
                    bf16 = getattr(backend, "is_bf16_supported", None)
                    try:
                        with backend.device(index):
                            info["reported_bf16_support"] = bool(bf16()) if bf16 else None
                    except (AttributeError, RuntimeError, TypeError, NotImplementedError):
                        info["reported_bf16_support"] = None
                    entry["devices"].append(info)
            result["accelerators"].append(entry)
    try:
        device = select_device(requested)
        result.update(selected_device=str(device), selection_error=None,
                      memory=memory_snapshot(device))
        if probe:
            result["bf16_probe"] = _bf16_probe(device)
            result["operation_probes"] = operation_probes(device)
            result["native_codec_capability_note"] = (
                "Native E8 and binary predictors currently use float64. Some Intel/XPU "
                "devices may support BF16 while rejecting those float64 operations. "
                "An auxiliary probe failure does not veto other BF16 model/native "
                "format runs; report unsupported methods separately. These compact "
                "probes do not establish complete codec support or performance."
            )
    except (RuntimeError, ValueError) as error:
        result.update(selected_device=None, selection_error=str(error))
    return result


def model_storage_bytes(model):
    """Count unique actual parameter/buffer storage, including tied weights once."""
    seen = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            total += int(storage.nbytes())
            seen.add(key)
    return total


def load_model(model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, device="auto",
               local_files_only=False, *, tokenizer_validator=None):
    """Return (BF16 model, tokenizer, provenance); capability check comes first.

    Optional tokenizer-only validation runs after resolving the pinned tokenizer
    and before downloading/loading model weights or allocating model tensors.
    """
    device = select_device(device)
    probe = _bf16_probe(device)
    if not probe["success"]:
        raise RuntimeError(f"BF16 device probe failed before model download: {probe['error']}")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.hub import cached_file

    local = Path(str(model_id)).expanduser().is_dir()
    kwargs = {"local_files_only": bool(local_files_only), "trust_remote_code": False}
    if revision is not None and not local:
        kwargs["revision"] = revision
    config_path = cached_file(str(model_id), "config.json",
                             **{key: value for key, value in kwargs.items() if key != "trust_remote_code"})
    raw_config = Path(config_path).read_bytes()
    config = AutoConfig.from_pretrained(str(model_id), **kwargs)
    resolved = getattr(config, "_commit_hash", None) if not local else None
    if not local and revision and len(revision) == 40 and resolved != revision:
        raise RuntimeError("Resolved model configuration commit differs from requested pinned revision")
    # Once the config resolved a commit, bind tokenizer and weights to the same
    # commit even if a caller supplied a mutable branch name.
    if resolved:
        kwargs["revision"] = resolved
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_id), **kwargs)
    if tokenizer_validator is not None:
        tokenizer_validator(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(str(model_id), config=config,
                dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs).eval()
    model.to(device)
    synchronize(device)
    bad = [str(parameter.device) for parameter in model.parameters()
           if parameter.device != device]
    if bad:
        raise RuntimeError("Model was not placed completely on the selected explicit device")
    runtime_config = json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()
    identity = {"model_id": str(model_id), "requested_revision": revision,
                "resolved_commit": resolved, "config_sha256": hashlib.sha256(raw_config).hexdigest(),
                "runtime_config_sha256": hashlib.sha256(runtime_config).hexdigest(),
                "local_model": local, "local_weights_content_hashed": False,
                "device": str(device), "dtype": "bfloat16", "attention_backend": "sdpa",
                "model_storage_bytes": model_storage_bytes(model),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "load_seconds": time.perf_counter() - start,
                "placement": "Explicit single device; no device_map, automatic offload, or remote code",
                "bf16_probe": probe}
    return model, tokenizer, identity


def _pairs(cache):
    """Read populated tensor KV from both modern and legacy HF caches.

    Transformers 5 removed to_legacy_cache. Its supported layer storage is
    cache.layers[i].keys/values. Prefer that interface, including on 4.57,
    without invoking deprecated accessors or silently dropping empty layers.
    """
    t = _require_torch()
    cache_type = type(cache).__module__ + "." + type(cache).__qualname__
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    context = f"{cache_type} (transformers={version})"
    if hasattr(cache, "layers"):
        layers = cache.layers
        if layers is None:
            raise TypeError(f"Unsupported cache {context}: layers is None")
        pairs = tuple((getattr(layer, "keys", None), getattr(layer, "values", None)) for layer in layers)
    elif callable(getattr(cache, "to_legacy_cache", None)):
        pairs = tuple(cache.to_legacy_cache())
    elif isinstance(cache, (tuple, list)):
        pairs = tuple(cache)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) != len(cache.value_cache):
            raise ValueError(f"Invalid cache {context}: key/value layer counts differ")
        pairs = tuple(zip(cache.key_cache, cache.value_cache))
    else:
        raise TypeError(f"Unsupported cache {context}: expected layers with keys/values, "
                        "to_legacy_cache(), key_cache/value_cache, or legacy KV pairs")
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"Invalid cache {context}: layer {index} is not a K/V pair")
        if not all(t.is_tensor(value) for value in pair):
            raise TypeError(f"Invalid or uninitialized cache {context}: layer {index} "
                            "must contain key and value tensors; no layers were skipped")
    return pairs


def cache_geometry(cache):
    pairs = _pairs(cache)
    if not pairs:
        raise ValueError("Empty KV cache")
    shape = tuple(pairs[0][0].shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("Only batch-one [1,KVheads,tokens,head_dim] caches are supported")
    for k, v in pairs:
        if tuple(k.shape) != shape or tuple(v.shape) != shape:
            raise ValueError("All layers must expose matching full-context K/V shapes")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError("Actual model KV tensors must be BF16")
    return {"layers": len(pairs), "kv_heads": shape[1], "tokens": shape[2],
            "head_dim": shape[3], "columns": shape[1] * shape[3],
            "source": "Actual returned cache tensor dimensions, not hidden_size/query_heads"}


def cache_to_matrices(cache):
    """Copy actual stored BF16 words to independent CPU uint16[T,KVheads*D]."""
    cache_geometry(cache)
    matrices = []
    for k, v in _pairs(cache):
        pair = []
        for tensor in (k, v):
            words = tensor[0].permute(1, 0, 2).contiguous().view(torch.uint16)
            pair.append(words.reshape(tensor.shape[2], -1).cpu().numpy().copy())
        matrices.append(tuple(pair))
    return matrices


def cache_from_matrices(matrices, config, device="cpu"):
    """Restore full KV using actual columns and explicit head_dim when present."""
    _require_torch()
    from transformers import DynamicCache
    device = torch.device(device)
    heads = int(getattr(config, "num_key_value_heads", None) or config.num_attention_heads)
    explicit_dim = getattr(config, "head_dim", None)
    if heads < 1 or not matrices:
        raise ValueError("Invalid KV head count or empty matrices")
    expected_layers = getattr(config, "num_hidden_layers", None)
    if expected_layers is not None and len(matrices) != int(expected_layers):
        raise ValueError("Cache layer count disagrees with model configuration")
    expected_shape = np.asarray(matrices[0][0]).shape
    if len(expected_shape) != 2 or expected_shape[1] % heads:
        raise ValueError("Cache columns do not match configured KV heads")
    dim = expected_shape[1] // heads
    if explicit_dim is not None and int(explicit_dim) != dim:
        raise ValueError("Actual cache head dimension disagrees with explicit config.head_dim")
    cache = DynamicCache(config=config)
    for layer, pair in enumerate(matrices):
        if len(pair) != 2:
            raise ValueError("Each cache layer requires K and V")
        restored = []
        for source in pair:
            array = np.asarray(source)
            if array.dtype != np.uint16 or array.shape != expected_shape:
                raise ValueError("All CPU cache matrices must have matching uint16[T,C] shapes")
            # Always private storage, including the explicit CPU path.
            words = torch.from_numpy(np.ascontiguousarray(array)).to(device).clone()
            tensor = words.view(torch.bfloat16).reshape(array.shape[0], heads, dim)
            restored.append(tensor.permute(1, 0, 2).unsqueeze(0).contiguous())
        cache.update(restored[0], restored[1], layer)
    return cache


def _step(model, token, cache, device):
    count = int(cache.get_seq_length())
    return model(input_ids=torch.tensor([[int(token)]], dtype=torch.long, device=device),
                 attention_mask=torch.ones((1, count + 1), dtype=torch.long, device=device),
                 past_key_values=cache, use_cache=True, logits_to_keep=1)


def capture(model, ids, device, chunk_tokens=512, prefill_observer=None):
    """Chunk prefill and export CPU words; complete cache still occupies device.

    An optional synchronous observer receives ``(cache, start, end, total)``
    after each chunk, while the returned KV is still on the selected device.
    It may read the complete causal prefix, including the newly added rows.
    Observer work is included in the reported prefill timing. This hook does
    not evict raw cache tensors or establish compact-resident model attention.
    """
    _require_torch()
    device = torch.device(device)
    if not isinstance(chunk_tokens, int) or chunk_tokens < 1 or not len(ids):
        raise ValueError("Positive chunk_tokens and nonempty token IDs are required")
    if prefill_observer is not None and not callable(prefill_observer):
        raise TypeError("prefill_observer must be callable")
    tokens = torch.as_tensor(ids, dtype=torch.long, device=device).reshape(1, -1)
    initial = reset_peak_memory(device)
    with torch.inference_mode():
        def run_prefill():
            cache = None
            output = None
            for start in range(0, tokens.shape[1], chunk_tokens):
                end = min(start + chunk_tokens, tokens.shape[1])
                output = model(input_ids=tokens[:, start:end],
                    attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                if prefill_observer is not None:
                    prefill_observer(cache, start, end, tokens.shape[1])
            return output
        output, prefill_timing = timed_call(run_prefill, device)
        prefill_timing["observer_included"] = prefill_observer is not None
        cache_class = type(output.past_key_values).__module__ + "." + type(output.past_key_values).__qualname__
        geometry = cache_geometry(output.past_key_values)
        expected_layers = getattr(model.config, "num_hidden_layers", None)
        if expected_layers is not None and geometry["layers"] != int(expected_layers):
            raise ValueError("Returned cache layer count differs from model configuration")
        if geometry["tokens"] != tokens.shape[1]:
            raise ValueError("Returned cache does not retain the complete prompt")
        if not bool(torch.all(torch.isfinite(output.logits)).item()):
            raise ValueError("Model prefill produced nonfinite logits on the selected backend")
        first_token = int(output.logits[0, -1].argmax().item())
        prefill_memory = memory_snapshot(device)
        reset_peak_memory(device)
        matrices, copy_timing = timed_call(lambda: cache_to_matrices(output.past_key_values), device)
        copy_memory = memory_snapshot(device)
    del output, tokens
    synchronize(device)
    result = {"matrices": matrices, "first_token": first_token, "geometry": geometry,
            "cache_class": cache_class,
            "raw_payload_bytes": sum(array.nbytes for pair in matrices for array in pair),
            "timings": {"prefill": prefill_timing, "device_to_cpu_words": copy_timing},
            "memory": {"initial": initial, "prefill": prefill_memory,
                       "device_to_cpu_words": copy_memory, "after_cache_release": memory_snapshot(device)},
            "chunk_tokens": chunk_tokens,
            "scope": "Full device KV capture with CPU export; no encoded-cache attention or offload"}
    if prefill_observer is not None:
        result["prefill_observer_scope"] = (
            "Synchronous observer after every prefill chunk, before the next "
            "chunk and final CPU export; observer work is included in prefill "
            "timing. Complete raw device KV remains resident."
        )
        report = getattr(prefill_observer, "report", None)
        if report is not None:
            if not callable(report):
                raise TypeError("prefill_observer.report must be callable")
            recipes = report()
            if not isinstance(recipes, dict):
                raise TypeError("prefill_observer.report() must return a dict")
            result["prefill_recipes"] = recipes
    return result


def _matrices_equal(left, right):
    return len(left) == len(right) and all(np.array_equal(a, b)
        for pa, pb in zip(left, right) for a, b in zip(pa, pb))


def _cache_bits_equal(left, right):
    a, b = _pairs(left), _pairs(right)
    return len(a) == len(b) and all(torch.equal(x.contiguous().view(torch.uint16),
                                               y.contiguous().view(torch.uint16))
        for pa, pb in zip(a, b) for x, y in zip(pa, pb))


def generation_check(model, matrices, decode_callback, device, *, first_token, steps=2):
    """Compare reference and restored full caches; CPU callback runs each step.

    Two full caches coexist for comparison. This is a correctness check, not a
    compact-residency/memory-capacity experiment. Recent rows remain raw on CPU.
    """
    _require_torch()
    device = torch.device(device)
    if not isinstance(steps, int) or not 1 <= steps <= 64:
        raise ValueError("Generation correctness steps must be between 1 and 64")
    initial = reset_peak_memory(device)
    reference = cache_from_matrices(matrices, model.config, device)
    recent = [(np.empty((0, k.shape[1]), np.uint16), np.empty((0, v.shape[1]), np.uint16))
              for k, v in matrices]
    token = int(first_token)
    rows = []
    with torch.inference_mode():
        for index in range(steps):
            reference_out, ref_timing = timed_call(lambda: _step(model, token, reference, device), device)
            reference = reference_out.past_key_values
            start = time.perf_counter()
            decoded = decode_callback()
            if not _matrices_equal(matrices, decoded):
                raise ValueError("Decoder callback changed original prompt KV words")
            rebuilt = [(np.concatenate((k, rk)), np.concatenate((v, rv)))
                       for (k, v), (rk, rv) in zip(decoded, recent)]
            cpu_seconds = time.perf_counter() - start
            restored, transfer_timing = timed_call(lambda: cache_from_matrices(rebuilt, model.config, device), device)
            observed, model_timing = timed_call(lambda: _step(model, token, restored, device), device)
            ref_logits, test_logits = reference_out.logits, observed.logits
            bitwise = bool(torch.equal(ref_logits.contiguous().view(torch.uint8), test_logits.contiguous().view(torch.uint8)))
            finite = bool(torch.all(torch.isfinite(ref_logits)).item() and torch.all(torch.isfinite(test_logits)).item())
            error = float((ref_logits.float() - test_logits.float()).abs().max().item()) if finite else None
            ref_token = int(ref_logits[0, -1].argmax().item())
            test_token = int(test_logits[0, -1].argmax().item())
            cache_equal = bool(_cache_bits_equal(reference, observed.past_key_values))
            appended = cache_to_matrices(observed.past_key_values)
            recent = [(np.concatenate((rk, k[-1:])), np.concatenate((rv, v[-1:])))
                      for (rk, rv), (k, v) in zip(recent, appended)]
            rows.append({"step": index, "input_token": token, "reference_next_token": ref_token,
                         "restored_next_token": test_token, "tokens_equal": ref_token == test_token,
                         "logits_bitwise_equal": bitwise, "logits_finite": finite,
                         "max_abs_logit_difference": error, "appended_full_cache_bitwise_equal": cache_equal,
                         "cpu_decode_and_rebuild_seconds": cpu_seconds, "restore_transfer": transfer_timing,
                         "reference_model": ref_timing, "restored_model": model_timing})
            token = ref_token
            del observed, restored, reference_out, ref_logits, test_logits, appended, rebuilt, decoded
    peak = memory_snapshot(device)
    del reference
    synchronize(device)
    return {"steps": steps, "first_token": int(first_token), "per_step": rows,
            "all_logits_bitwise_equal": all(row["logits_bitwise_equal"] for row in rows),
            "all_tokens_equal": all(row["tokens_equal"] for row in rows),
            "all_appended_KV_bitwise_equal": all(row["appended_full_cache_bitwise_equal"] for row in rows),
            "max_abs_logit_difference": max((row["max_abs_logit_difference"] for row in rows
                                              if row["max_abs_logit_difference"] is not None), default=None),
            "recent_raw_cpu_bytes": sum(array.nbytes for pair in recent for array in pair),
            "memory": {"initial": initial, "comparison_peak": peak,
                       "after_cache_release": memory_snapshot(device)},
            "scope": "Full reference and fully restored caches; per-step CPU decoding and H2D reconstruction; no compact-resident model attention"}


# Explicit aliases for callers using the legacy smoke-test terminology.
tensor_cache = cache_from_matrices
cache_matrices = cache_to_matrices

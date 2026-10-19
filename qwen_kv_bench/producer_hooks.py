"""Bounded diagnostic capture of actual Qwen3 KV producer stages.

Hooks observe the original forward without returning replacement tensors. Only
selected layer/token-tile intersections are copied to CPU. Hidden features are
temporary diagnostic inputs, not free decoder state or a compact cache format.
RoPE replay uses the model implementation and original device during forward.
"""
from __future__ import annotations

from collections import Counter
import importlib
import math
from numbers import Integral
import time

import numpy as np
import torch

from .model_runtime import _pairs


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bits(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise ValueError("Producer capture requires actual BF16 intermediate tensors")
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


class ProducerCapture:
    """Observe selected paired K/V tiles inside a BF16 Qwen3 forward.

    ``max_tiles`` counts layer/token pairs, each holding both K and V. Selection
    is fixed before prefill. Pass ``None`` to capture all tiles (small-model/full
    DEV diagnostics only); no hidden feature is serialized by this class.
    An optional ``on_tile(record, context)`` runs synchronously inside attention
    forward, immediately after the actual cache layer is updated and the tile
    is complete. Its CPU diagnostic work is included in model prefill time.
    """

    _STAGES = ("input_layernorm", "attention_pre", "k_proj", "k_norm", "v_proj", "attention_post")
    _FIELDS = ("hidden", "k_proj", "k_pre_rope", "cos", "sin", "k", "v", "_v_hook", "_k_replay")

    def __init__(self, model, total_tokens, tile_tokens=64, max_tiles=72, on_tile=None):
        self.model = model
        if on_tile is not None and not callable(on_tile):
            raise TypeError("on_tile must be callable or None")
        self.on_tile = on_tile
        self.total_tokens = _positive_integer(total_tokens, "total_tokens")
        self.tile_tokens = _positive_integer(tile_tokens, "tile_tokens")
        if max_tiles is not None:
            max_tiles = _positive_integer(max_tiles, "max_tiles")
        config = getattr(model, "config", None)
        core = getattr(model, "model", None)
        if getattr(config, "model_type", None) != "qwen3" or core is None or \
                not hasattr(core, "layers") or not hasattr(core, "rotary_emb"):
            raise ValueError("ProducerCapture supports the standard Qwen3ForCausalLM architecture only")
        self.layers = list(core.layers)
        if not self.layers or len(self.layers) != int(config.num_hidden_layers):
            raise ValueError("Unsupported Qwen3 decoder layer geometry")
        self.layer_count = len(self.layers)
        if max_tiles is not None and max_tiles < self.layer_count:
            raise ValueError("max_tiles must cover at least one tile per layer, or be None")
        self.max_tiles = max_tiles
        self.rotary = core.rotary_emb
        self.hidden_size = int(config.hidden_size)
        self.heads = int(config.num_key_value_heads)
        self.head_dim = None
        self._rotate = {}
        for index, layer in enumerate(self.layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(layer, "input_layernorm") or any(
                    not hasattr(attention, name) for name in ("k_proj", "v_proj", "k_norm", "head_dim")):
                raise ValueError("Unsupported Qwen3 projection/normalization modules")
            dimension = int(attention.head_dim)
            if dimension < 2 or dimension % 2 or (self.head_dim is not None and dimension != self.head_dim):
                raise ValueError("Unsupported or inconsistent Qwen3 explicit head_dim")
            self.head_dim = dimension
            if getattr(config, "head_dim", dimension) != dimension or \
                    attention.k_proj.out_features != self.heads * dimension or \
                    attention.v_proj.out_features != self.heads * dimension or \
                    attention.k_proj.in_features != self.hidden_size or \
                    attention.k_norm.weight.numel() != dimension:
                raise ValueError("Qwen3 projection dimensions disagree with explicit head_dim")
            module = importlib.import_module(type(attention).__module__)
            rotate = getattr(module, "apply_rotary_pos_emb", None)
            if rotate is None:
                raise ValueError("Unsupported Qwen3 rotary implementation")
            self._rotate[index] = rotate
        self.channels = self.heads * self.head_dim
        self._handles = []
        self._entered = False
        self._closed = False
        self._failed = False
        self._finalized = False
        self._frame = None
        self._end = 0
        self._chunks = 0
        self._counts = Counter()
        self._stage_counts = {i: Counter() for i in range(self.layer_count)}
        self._copy_bytes = 0
        self._ready_count = 0
        self._callback_count = 0
        self._callback_seconds = 0.0
        self._norm_weights = {}
        self._norm_eps = {}
        self._inv_freq = None
        self._attention_scaling = None
        self._rope_stable = True
        tiles = math.ceil(self.total_tokens / self.tile_tokens)
        total_pairs = self.layer_count * tiles
        count = total_pairs if max_tiles is None else min(max_tiles, total_pairs)
        ordinals = np.linspace(0, total_pairs - 1, count, dtype=np.int64)
        self._records = []
        self._by_layer = {i: [] for i in range(self.layer_count)}
        for ordinal in ordinals:
            layer = int(ordinal) // tiles
            start = (int(ordinal) % tiles) * self.tile_tokens
            record = dict(layer=layer, start=start, tokens=min(self.tile_tokens, self.total_tokens - start),
                head_dim=self.head_dim, _buffers={}, _written={}, _emitted=False)
            self._records.append(record)
            self._by_layer[layer].append(record)
        if any(not records for records in self._by_layer.values()):
            raise ValueError("Selected tile plan does not cover every layer")
        self._total_pairs = total_pairs

    def _guard(self, function):
        def observed(*args, **kwargs):
            try:
                function(*args, **kwargs)
            except BaseException:
                self._failed = True
                self.close()
                raise
            return None  # Observation never replaces model inputs or outputs.
        return observed

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("ProducerCapture instances cannot be entered twice")
        self._entered = True
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._guard(self._root_pre), with_kwargs=True))
            self._handles.append(self.model.register_forward_hook(self._guard(self._root_post),
                with_kwargs=True, always_call=True))
            self._handles.append(self.rotary.register_forward_hook(self._guard(self._rotary_post), with_kwargs=True))
            for index, layer in enumerate(self.layers):
                attention = layer.self_attn
                self._norm_weights[index] = _bits(attention.k_norm.weight)
                self._copy_bytes += self._norm_weights[index].nbytes
                self._norm_eps[index] = float(attention.k_norm.variance_epsilon)
                self._handles.append(layer.input_layernorm.register_forward_hook(
                    self._guard(lambda m, a, o, i=index: self._feature(i, "input_layernorm", "hidden", o))))
                self._handles.append(attention.register_forward_pre_hook(
                    self._guard(lambda m, a, k, i=index: self._attention_pre(i, a, k)), with_kwargs=True))
                self._handles.append(attention.register_forward_hook(
                    self._guard(lambda m, a, k, o, i=index: self._attention_post(i, a, k, o)), with_kwargs=True))
                self._handles.append(attention.k_proj.register_forward_hook(
                    self._guard(lambda m, a, o, i=index: self._feature(i, "k_proj", "k_proj", o))))
                self._handles.append(attention.k_norm.register_forward_hook(
                    self._guard(lambda m, a, o, i=index: self._key_norm(i, o))))
                self._handles.append(attention.v_proj.register_forward_hook(
                    self._guard(lambda m, a, o, i=index: self._feature(i, "v_proj", "_v_hook", o))))
        except BaseException:
            self._failed = True
            self.close()
            raise
        return self

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._frame = None
        self._closed = True

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        if exception_type is None and (self._failed or self._end != self.total_tokens):
            raise ValueError("Producer forward capture failed or has incomplete token coverage")
        return False

    def _require_frame(self):
        if self._frame is None:
            raise ValueError("Producer stage executed outside a tracked model forward")
        return self._frame

    def _root_pre(self, module, args, kwargs):
        if self._frame is not None or self._finalized:
            raise ValueError("Nested or finalized producer forwards are unsupported")
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Producer capture requires nonempty input_ids with batch size one")
        end = self._end + int(ids.shape[1])
        if end > self.total_tokens:
            raise ValueError("Producer forward exceeds declared total_tokens")
        self._frame = dict(start=self._end, end=end, tokens=int(ids.shape[1]), embeddings={}, caches={})
        self._counts["root_pre"] += 1

    def _rotary_post(self, module, args, kwargs, output):
        frame = self._require_frame()
        position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != (1, frame["tokens"]):
            raise ValueError("Unsupported Qwen3 rotary position_ids shape")
        expected = torch.arange(frame["start"], frame["end"], device=position_ids.device, dtype=position_ids.dtype)
        if not torch.equal(position_ids[0], expected):
            raise ValueError("Producer decoder context requires contiguous zero-based absolute positions")
        if not isinstance(output, (tuple, list)) or len(output) != 2 or any(
                tuple(t.shape) != (1, frame["tokens"], self.head_dim) for t in output):
            raise ValueError("Unsupported Qwen3 rotary output geometry")
        inv_freq = module.inv_freq.detach().to(dtype=torch.float32).cpu().numpy().copy()
        if inv_freq.shape != (self.head_dim // 2,):
            raise ValueError("Unsupported Qwen3 rotary inverse-frequency geometry")
        scaling = module.attention_scaling
        if isinstance(scaling, torch.Tensor):
            if scaling.numel() != 1:
                raise ValueError("Non-scalar Qwen3 rotary attention scaling is unsupported")
            scaling = float(scaling.detach().cpu().item())
        else:
            scaling = float(scaling)
        self._copy_bytes += inv_freq.nbytes
        if self._inv_freq is None:
            self._inv_freq, self._attention_scaling = inv_freq, scaling
        elif not np.array_equal(self._inv_freq, inv_freq) or self._attention_scaling != scaling:
            self._rope_stable = False
        self._counts["rotary"] += 1
        frame["rotary_seen"] = True

    def _pieces(self, layer):
        frame = self._require_frame()
        for record in self._by_layer[layer]:
            first = max(record["start"], frame["start"])
            last = min(record["start"] + record["tokens"], frame["end"])
            if first < last:
                yield record, first - frame["start"], last - frame["start"], first - record["start"]

    def _put(self, record, name, rows, offset):
        data = _bits(rows)
        if data.ndim != 2 or offset < 0 or offset + len(data) > record["tokens"]:
            raise ValueError("Invalid selected producer tile intersection")
        if name not in record["_buffers"]:
            record["_buffers"][name] = np.empty((record["tokens"], data.shape[1]), dtype=np.uint16)
            record["_written"][name] = np.zeros(record["tokens"], dtype=bool)
        buffer = record["_buffers"][name]
        written = record["_written"][name]
        if buffer.shape[1] != data.shape[1] or written[offset:offset + len(data)].any():
            raise ValueError("Producer stage shape changed or selected token rows were observed twice")
        buffer[offset:offset + len(data)] = data
        written[offset:offset + len(data)] = True
        self._copy_bytes += data.nbytes

    def _write(self, layer, name, matrix):
        frame = self._require_frame()
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.shape[0] != frame["tokens"]:
            raise ValueError("Unsupported Qwen3 intermediate token geometry")
        expected = self.hidden_size if name == "hidden" else self.channels
        if matrix.shape[1] != expected:
            raise ValueError("Producer intermediate columns disagree with explicit head_dim")
        for record, first, last, offset in self._pieces(layer):
            self._put(record, name, matrix[first:last], offset)

    def _feature(self, layer, stage, name, output):
        frame = self._require_frame()
        if not isinstance(output, torch.Tensor) or tuple(output.shape[:2]) != (1, frame["tokens"]):
            raise ValueError("Unsupported Qwen3 intermediate batch or token geometry")
        self._stage_counts[layer][stage] += 1
        self._write(layer, name, output[0].reshape(frame["tokens"], -1))

    def _attention_pre(self, layer, args, kwargs):
        frame = self._require_frame()
        embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        if not frame.get("rotary_seen") or not isinstance(embeddings, (tuple, list)) or len(embeddings) != 2:
            raise ValueError("Actual attention position embeddings were not observed")
        cos, sin = embeddings
        if any(not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) !=
                (1, frame["tokens"], self.head_dim) for tensor in embeddings):
            raise ValueError("Unsupported attention cosine/sine geometry")
        frame["embeddings"][layer] = (cos, sin)
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value", args[3] if len(args) > 3 else None))
        if cache is None:
            raise ValueError("Producer attention must receive its actual mutable KV cache")
        frame["caches"][layer] = cache
        self._stage_counts[layer]["attention_pre"] += 1
        for record, first, last, offset in self._pieces(layer):
            self._put(record, "cos", cos[0, first:last], offset)
            self._put(record, "sin", sin[0, first:last], offset)

    def _key_norm(self, layer, output):
        frame = self._require_frame()
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != \
                (1, frame["tokens"], self.heads, self.head_dim):
            raise ValueError("Unsupported Qwen3 key normalization geometry")
        self._stage_counts[layer]["k_norm"] += 1
        self._write(layer, "k_pre_rope", output[0].reshape(frame["tokens"], self.channels))
        if layer not in frame["embeddings"]:
            raise ValueError("Key normalization executed before attention position embedding hook")
        cos, sin = frame["embeddings"][layer]
        for record, first, last, offset in self._pieces(layer):
            key = output[:, first:last].transpose(1, 2)
            # Apply exactly the loaded model's function on the original device.
            # Q only supplies a same-shape dummy argument; K's result is
            # independent of Q. No global monkeypatch or tensor mutation.
            _, rotated = self._rotate[layer](key, key, cos[:, first:last], sin[:, first:last])
            matrix = rotated[0].transpose(0, 1).reshape(last - first, self.channels)
            self._put(record, "_k_replay", matrix, offset)

    def _context(self):
        if self._inv_freq is None:
            raise ValueError("Actual rotary decoder context has not been observed")
        self._inv_freq.flags.writeable = False
        for weight in self._norm_weights.values():
            weight.flags.writeable = False
        return dict(rope_inv_freq=self._inv_freq, rope_attention_scaling=self._attention_scaling,
            rope_stable=self._rope_stable, norm_weights=dict(self._norm_weights),
            norm_eps=dict(self._norm_eps), head_dim=self.head_dim,
            position_rule="contiguous absolute positions starting at zero")

    def _public_record(self, record):
        public = {key: record[key] for key in ("layer", "start", "tokens", "head_dim")}
        for name, value in record["_buffers"].items():
            if not name.startswith("_"):
                value.flags.writeable = False
                public[name] = value
        public["k_norm_weight"] = self._norm_weights[record["layer"]]
        public["k_norm_eps"] = self._norm_eps[record["layer"]]
        return public

    def _attention_post(self, layer, args, kwargs, output):
        frame = self._require_frame()
        cache = frame["caches"][layer]
        # On the first chunk later layers may still be uninitialized.  Inspect
        # only this layer, which the actual attention forward just updated.
        if hasattr(cache, "layers"):
            cached_layer = cache.layers[layer]
            pair = (cached_layer.keys, cached_layer.values)
        else:
            pair = cache[layer]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Unsupported current-layer producer cache")
        for kind, tensor in zip(("k", "v"), pair):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != \
                    (1, self.heads, frame["end"], self.head_dim):
                raise ValueError("Current producer cache layer must retain the full prefix")
            for record, first, last, offset in self._pieces(layer):
                begin, stop = frame["start"] + first, frame["start"] + last
                matrix = tensor[0, :, begin:stop].transpose(0, 1).reshape(stop - begin, self.channels)
                self._put(record, kind, matrix, offset)
        self._stage_counts[layer]["attention_post"] += 1
        for record in self._by_layer[layer]:
            if record["_emitted"] or set(record["_buffers"]) != set(self._FIELDS) or any(
                    not mask.all() for mask in record["_written"].values()):
                continue
            record["_emitted"] = True
            self._ready_count += 1
            if self.on_tile is not None:
                before = time.perf_counter()
                self.on_tile(self._public_record(record), self._context())
                self._callback_seconds += time.perf_counter() - before
                self._callback_count += 1

    def _root_post(self, module, args, kwargs, output):
        if output is None:
            self._failed = True
            self.close()
            return
        frame = self._require_frame()
        pairs = _pairs(getattr(output, "past_key_values", None))
        if len(pairs) != self.layer_count:
            raise ValueError("Producer output cache layer count changed")
        for layer, pair in enumerate(pairs):
            for tensor in pair:
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != \
                        (1, self.heads, frame["end"], self.head_dim):
                    raise ValueError("Producer output cache must retain the full declared prefix")
        self._end = frame["end"]
        self._chunks += 1
        self._counts["root_post"] += 1
        self._frame = None

    def finalize(self, matrices):
        """Return paired CPU records, numerical audit, and small decoder context."""
        self.close()
        if self._failed or self._end != self.total_tokens or self._finalized:
            raise ValueError("Cannot finalize failed, incomplete, or previously finalized producer capture")
        if self._inv_freq is None or self._counts["root_pre"] != self._chunks or \
                self._counts["root_post"] != self._chunks or self._counts["rotary"] != self._chunks:
            raise ValueError("Producer root/rotary hook coverage is incomplete")
        if self._ready_count != len(self._records) or (self.on_tile is not None and self._callback_count != len(self._records)):
            raise ValueError("Selected producer tiles were not completed inside attention forward")
        if len(matrices) != self.layer_count:
            raise ValueError("Exported cache has the wrong layer count")
        for layer in range(self.layer_count):
            if any(self._stage_counts[layer][stage] != self._chunks for stage in self._STAGES):
                raise ValueError("Producer inner-stage hook coverage is incomplete")
            if len(matrices[layer]) != 2 or any(not isinstance(matrix, np.ndarray) or matrix.dtype != np.uint16 or
                    matrix.shape != (self.total_tokens, self.channels) for matrix in matrices[layer]):
                raise ValueError("Exported cache has the wrong dtype or explicit head geometry")
        records, checks = [], []
        for record in self._records:
            buffers = record["_buffers"]
            if set(buffers) != set(self._FIELDS) or any(not mask.all() for mask in record["_written"].values()):
                raise ValueError("Selected producer tile has missing stage or token rows")
            layer, start, tokens = record["layer"], record["start"], record["tokens"]
            k_export, v_export = (matrix[start:start + tokens] for matrix in matrices[layer])
            differences = dict(
                v_hook_cache_mismatched_words=int(np.count_nonzero(buffers["_v_hook"] != buffers["v"])),
                k_rope_replay_cache_mismatched_words=int(np.count_nonzero(buffers["_k_replay"] != buffers["k"])),
                k_cache_export_mismatched_words=int(np.count_nonzero(buffers["k"] != k_export)),
                v_cache_export_mismatched_words=int(np.count_nonzero(buffers["v"] != v_export)))
            check = dict(layer=layer, start=start, tokens=tokens, **differences,
                all_stage_targets_exact=not any(differences.values()))
            check["selected_shapes"] = {name: list(value.shape) for name, value in buffers.items()}
            checks.append(check)
            records.append(self._public_record(record))
        context = self._context()
        audit = dict(format="qwen3_inside_forward_producer_capture_v1",
            selected_tile_pairs=len(records), logical_KV_records=2 * len(records),
            total_tile_pairs=self._total_pairs, coverage="full" if len(records) == self._total_pairs else "sampled",
            total_tokens=self.total_tokens, tile_tokens=self.tile_tokens,
            layers=self.layer_count, head_dim=self.head_dim, kv_heads=self.heads,
            channels=self.channels, hidden_size=self.hidden_size, chunks_observed=self._chunks,
            hook_counts=dict(self._counts),
            inner_stage_counts={str(layer): dict(counts) for layer, counts in self._stage_counts.items()},
            all_stage_targets_exact=all(check["all_stage_targets_exact"] for check in checks),
            v_hook_equals_cache=all(check["v_hook_cache_mismatched_words"] == 0 for check in checks),
            k_rope_replay_equals_cache=all(check["k_rope_replay_cache_mismatched_words"] == 0 for check in checks),
            exported_cache_equals_hook_cache=all(check["k_cache_export_mismatched_words"] == 0 and
                check["v_cache_export_mismatched_words"] == 0 for check in checks),
            rope_stable=self._rope_stable, rope_position_rule=context["position_rule"],
            replay_device="original producer device; actual loaded Qwen3 apply_rotary_pos_emb",
            copied_selected_and_context_bytes=self._copy_bytes,
            retained_public_selected_bytes=sum(value.nbytes for record in records
                for name, value in record.items() if isinstance(value, np.ndarray) and name != "k_norm_weight"),
            decoder_context_array_bytes=self._inv_freq.nbytes + sum(w.nbytes for w in self._norm_weights.values()),
            selected_tiles_ready_inside_attention=self._ready_count,
            on_tile_enabled=self.on_tile is not None, on_tile_callback_count=self._callback_count,
            on_tile_callback_wall_seconds=self._callback_seconds,
            on_tile_stage="attention_forward_post_hook; before decoder layer resumes and root model returns",
            raw_hidden_features_persisted=False, hooks_removed=not self._handles,
            selected_records=checks,
            scope="Read-only inside-forward hooks; only fixed selected tile intersections exported. Hidden and projection tensors are temporary diagnostics, not free reconstruction state. Model retains full raw cache. No speed, capacity, compressed attention or serving result is implied.")
        self._finalized = True
        # Release duplicate audit/replay arrays; returned records retain only
        # public selected producer features and true K/V targets.
        for record in self._records:
            record["_buffers"] = {}
            record["_written"] = {}
        return records, audit, context

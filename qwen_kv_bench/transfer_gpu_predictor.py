"""Device-tensor parity component for the exact transfer predictor.

Only fixed-point feature preparation and predictor replay are implemented here.
The original constriction entropy coder remains CPU-only. ``prediction_sweep``
uses supplied known KV to advance causal context; it does not decode KV from a
compressed stream, establish a serving speedup, or implement resident attention.

INT64 products are reduced in bounded chunks using ordinary tensor operations.
No INT64 GEMM availability is assumed. A conservative sum-of-absolute-products
bound is checked before multiplying; therefore reduction order cannot introduce
integer overflow. The validation guards synchronize device scalars and their
cost belongs in component timings. This reference is not an optimized CUDA kernel.
"""
from __future__ import annotations

import copy

import numpy as np

try:
    import torch
except ImportError:
    torch = None


_MAX_FIXED = 1 << 40
_ACCUMULATION_BOUND = 1 << 62


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for transfer predictor component tests")
    return torch


def _copy_tensor(array, device, dtype=None):
    t = _require_torch()
    # Own a writable CPU copy before torch conversion; frozen NumPy tables may
    # be read-only, and no device tensor aliases the caller's reference state.
    value = t.from_numpy(np.array(array, copy=True))
    return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)


def _maximum_integer(value):
    if not value.numel():
        return 0
    # Python abs also handles INT64_MIN safely, unlike integer-tensor abs.
    return max(abs(int(value.amin().item())), abs(int(value.amax().item())))


def numpy_prediction_sweep(model, known_words, start=0, metadata=None):
    """CPU oracle using the original NumPy fixed-point predictor operations.

    This is fixture-driven prediction replay, not decompression. The exact same
    known-word prefixes feed the NumPy oracle and device-tensor component.
    """
    from . import transfer_reference as reference
    source = np.asarray(known_words)
    if source.dtype != np.uint16:
        raise ValueError("known words must be uint16 BF16 patterns")
    layers, tokens, width = reference._shape(model, source.shape, start)
    metadata = reference._meta(metadata, layers, tokens, model.metadata_rank)
    positions = np.arange(start, start + tokens)
    previous = np.empty((tokens, 0), dtype=np.int64)
    output = np.empty(source.shape, dtype=np.float64)
    for layer in range(layers):
        fixed = reference._all_fixed(source[layer], positions, model)
        meta = metadata[layer].astype(np.int64) * 512
        features = np.concatenate((previous, meta, np.zeros((tokens, width), dtype=np.int64)), axis=1)
        offset = previous.shape[1] + model.metadata_rank
        for index, block in enumerate(model.predictors[layer]):
            ch0 = index * model.block_size
            output[layer, :, ch0:ch0 + model.block_size] = reference._predict(features, block, ch0, positions, model)
            features[:, offset + ch0:offset + ch0 + model.block_size] = fixed[:, ch0:ch0 + model.block_size]
        previous = features[:, offset:]
    return output


class TorchTransferPredictor:
    """Frozen reference state copied to one explicit CPU or CUDA device."""

    @classmethod
    def from_reference(cls, model, device="cpu", coefficient_dtype="preserve", input_chunk=64, token_chunk=32):
        t = _require_torch()
        if coefficient_dtype not in ("preserve", "int32"):
            raise ValueError("coefficient_dtype must be preserve or int32")
        for name, size in (("input_chunk", input_chunk), ("token_chunk", token_chunk)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(name + " must be a positive integer")
        selected_device = t.device(device)
        if selected_device.type not in ("cpu", "cuda"):
            raise ValueError("this component supports explicit CPU or CUDA devices")
        if selected_device.type == "cuda" and not t.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        result = cls()
        result.device = selected_device
        result.geometry = copy.deepcopy(model.geometry)
        result.block_size = model.block_size
        result.metadata_rank = model.metadata_rank
        result.state_sha256 = model.state_sha256
        result.coefficient_dtype_policy = coefficient_dtype
        result.input_chunk = input_chunk
        result.token_chunk = token_chunk
        result.cosine = _copy_tensor(model.cosine, selected_device, t.int64)
        result.sine = _copy_tensor(model.sine, selected_device, t.int64)
        result.device = result.cosine.device
        result.predictors = []
        for blocks in model.predictors:
            converted = []
            for block in blocks:
                matrix = np.asarray(block["A"])
                if matrix.dtype.kind != "i":
                    raise ValueError("frozen coefficients must have a signed integer dtype")
                bias = np.asarray(block["b"])
                if bias.dtype.kind != "i" or np.any(bias < -_MAX_FIXED) or np.any(bias > _MAX_FIXED):
                    raise ValueError("bias exceeds finite fixed-point domain")
                if coefficient_dtype == "int32":
                    if np.any(matrix < -(1 << 31)) or np.any(matrix >= 1 << 31):
                        raise ValueError("frozen coefficients do not fit INT32 exactly")
                    matrix = matrix.astype(np.int32)
                maximum_l1 = max((sum(abs(int(x)) for x in row) for row in matrix), default=0)
                converted.append(dict(
                    A=_copy_tensor(matrix, selected_device),
                    columns=_copy_tensor(block["columns"], selected_device, t.int64),
                    b=_copy_tensor(block["b"], selected_device, t.int64),
                    inv_s=_copy_tensor(block["inv_s"], selected_device, t.float64),
                    max_row_l1=maximum_l1))
            result.predictors.append(converted)
        return result

    def _on_device(self, tensor, dtype, name):
        t = _require_torch()
        if not isinstance(tensor, t.Tensor) or tensor.dtype != dtype or tensor.device != self.cosine.device:
            raise ValueError(name + " must have the required dtype on the predictor device")
        return tensor

    def positions(self, start, tokens):
        t = _require_torch()
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(tokens, bool) or not isinstance(tokens, int):
            raise ValueError("start and tokens must be integers")
        if tokens < 1 or start < 0 or start + tokens > self.geometry["max_positions"]:
            raise ValueError("positions outside frozen RoPE tables")
        return t.arange(start, start + tokens, dtype=t.int64, device=self.device)

    def _positions(self, positions, tokens):
        t = _require_torch()
        self._on_device(positions, t.int64, "positions")
        if positions.shape != (tokens,) or (tokens and (int(positions.amin().item()) < 0 or int(positions.amax().item()) >= self.geometry["max_positions"])):
            raise ValueError("invalid frozen-table positions")
        return positions

    def fixed_words(self, words):
        """Expand exact BF16 patterns and reproduce NumPy round-to-even features."""
        t = _require_torch()
        self._on_device(words, t.uint16, "words")
        expanded = (words.to(t.int32) << 16).contiguous().view(t.float32).to(t.float64)
        scaled = t.round(expanded * 65536.0)
        if not bool(t.isfinite(scaled).all().item()) or bool((scaled.abs() > _MAX_FIXED).any().item()):
            raise ValueError("words exceed the finite fixed-point reference domain")
        return scaled.to(t.int64)

    def rope_int(self, value, ch0, positions, inverse=False):
        t = _require_torch()
        self._on_device(value, t.int64, "RoPE input")
        if value.ndim != 2 or value.shape[1] % 2 or ch0 < 0 or ch0 % 2:
            raise ValueError("RoPE input must contain complete adjacent pairs")
        self._positions(positions, len(value))
        if _maximum_integer(value) > _MAX_FIXED:
            raise ValueError("RoPE fixed-point input overflow")
        dimension = self.geometry["head_dim"]
        index = ((ch0 + t.arange(0, value.shape[1], 2, dtype=t.int64, device=self.device)) % dimension) // 2
        cosine = self.cosine.index_select(0, positions).index_select(1, index)
        sine = self.sine.index_select(0, positions).index_select(1, index)
        if inverse:
            sine = -sine
        a, b = value[:, 0::2], value[:, 1::2]
        output = t.empty_like(value)
        output[:, 0::2] = (a * cosine - b * sine + (1 << 14)) >> 15
        output[:, 1::2] = (a * sine + b * cosine + (1 << 14)) >> 15
        return output

    def canonical_features(self, words, positions):
        t = _require_torch()
        channels = self.geometry["channels"]
        if words.ndim != 2 or words.shape[1] != 2 * channels:
            raise ValueError("canonical words must have [tokens,2*channels] shape")
        value = self.fixed_words(words)
        return t.cat((self.rope_int(value[:, :channels], 0, positions, True), value[:, channels:]), dim=1)

    def predict_block(self, features, layer, block_index, positions):
        """Return float64 means from exact integer accumulation and saved RoPE."""
        t = _require_torch()
        self._on_device(features, t.int64, "features")
        if features.ndim != 2 or not 0 <= layer < len(self.predictors) or not 0 <= block_index < len(self.predictors[layer]):
            raise ValueError("invalid layer, block, or feature shape")
        self._positions(positions, len(features))
        block = self.predictors[layer][block_index]
        expected_features = (2 * self.geometry["channels"] if layer else 0) + self.metadata_rank + 2 * self.geometry["channels"]
        if features.shape[1] != expected_features:
            raise ValueError("feature width differs from frozen state")
        selected = features.index_select(1, block["columns"])
        maximum = _maximum_integer(selected)
        if maximum * block["max_row_l1"] + (1 << 15) >= _ACCUMULATION_BOUND:
            raise ValueError("predictor INT64 accumulation would overflow")
        matrix = block["A"]
        accumulation = t.zeros((len(features), len(block["b"])), dtype=t.int64, device=self.device)
        for begin in range(0, len(features), self.token_chunk):
            end = min(begin + self.token_chunk, len(features))
            for channel in range(0, selected.shape[1], self.input_chunk):
                finish = min(channel + self.input_chunk, selected.shape[1])
                # TensorIterator promotes INT32 coefficient storage and INT64
                # features to INT64 products; sum explicitly retains INT64.
                products = selected[begin:end, None, channel:finish] * matrix[None, :, channel:finish]
                accumulation[begin:end] += products.sum(dim=2, dtype=t.int64)
        pre = ((accumulation + (1 << 15)) >> 16) + block["b"]
        if _maximum_integer(pre) > _MAX_FIXED:
            raise ValueError("predictor fixed-point output overflow")
        ch0 = block_index * self.block_size
        if ch0 < self.geometry["channels"]:
            pre = self.rope_int(pre, ch0, positions)
        return pre.to(t.float64) / 65536.0

    def inverse_scales(self, layer, block_index, tokens):
        return self.predictors[layer][block_index]["inv_s"].unsqueeze(0).expand(tokens, -1)

    def prediction_sweep(self, known_words, start=0, metadata=None):
        """Replay means using known KV as causal references, not entropy decoding."""
        t = _require_torch()
        self._on_device(known_words, t.uint16, "known words")
        layers = len(self.predictors)
        width = 2 * self.geometry["channels"]
        if known_words.ndim != 3 or known_words.shape[0] != layers or known_words.shape[2] != width:
            raise ValueError("known words must have [layers,tokens,2*channels] shape")
        tokens = known_words.shape[1]
        positions = self.positions(start, tokens)
        if self.metadata_rank:
            self._on_device(metadata, t.int8, "metadata")
            if metadata.shape != (layers, tokens, self.metadata_rank):
                raise ValueError("metadata shape differs from frozen state")
        elif metadata is not None and metadata.numel():
            raise ValueError("metadata supplied to metadata-free predictor")
        previous = t.empty((tokens, 0), dtype=t.int64, device=self.device)
        means = t.empty(known_words.shape, dtype=t.float64, device=self.device)
        for layer in range(layers):
            fixed = self.canonical_features(known_words[layer], positions)
            meta = metadata[layer].to(t.int64) * 512 if self.metadata_rank else t.empty((tokens, 0), dtype=t.int64, device=self.device)
            features = t.cat((previous, meta, t.zeros((tokens, width), dtype=t.int64, device=self.device)), dim=1)
            offset = previous.shape[1] + self.metadata_rank
            for index in range(len(self.predictors[layer])):
                ch0 = index * self.block_size
                means[layer, :, ch0:ch0 + self.block_size] = self.predict_block(features, layer, index, positions)
                features[:, offset + ch0:offset + ch0 + self.block_size] = fixed[:, ch0:ch0 + self.block_size]
            previous = features[:, offset:]
        return means

    def storage_statistics(self):
        tensors = [self.cosine, self.sine]
        coefficient_bytes = 0
        for blocks in self.predictors:
            for block in blocks:
                tensors.extend(block[key] for key in ("A", "columns", "b", "inv_s"))
                coefficient_bytes += block["A"].untyped_storage().nbytes()
        seen = set(); array_bytes = 0
        for value in tensors:
            store = value.untyped_storage()
            identity = (str(value.device), store.data_ptr(), store.nbytes())
            if identity not in seen:
                array_bytes += store.nbytes(); seen.add(identity)
        return dict(device=str(self.cosine.device), source_state_sha256=self.state_sha256,
                    component="fixed-point predictor and feature replay only; entropy codec is CPU",
                    persistent_tensor_bytes=int(array_bytes), coefficient_tensor_bytes=int(coefficient_bytes),
                    coefficient_dtype_policy=self.coefficient_dtype_policy,
                    coefficient_dtypes=sorted({str(block["A"].dtype) for blocks in self.predictors for block in blocks}),
                    input_chunk=self.input_chunk, token_chunk=self.token_chunk,
                    product_scratch_upper_bound_bytes=self.token_chunk * self.block_size * self.input_chunk * 8,
                    scratch_scope="product tensor only; excludes selected features, accumulation, outputs and allocator overhead",
                    guards="device scalar synchronizations and conservative integer overflow checks are enabled")

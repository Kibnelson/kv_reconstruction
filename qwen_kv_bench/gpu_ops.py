"""New tensor-native KV packets, separate from the CPU entropy wire formats.

Only PyTorch/stdlib are required.  The same functions run on CPU for functional
testing; CUDA/XPU support is a runtime capability, never inferred from a NUC.
Huffman, Rice, Deflate and published quantizer kernels are NOT implemented here.
Native variants use grouped fixed-width packing of exact integer residuals.
All packet headers, predictor bytes and actual state tensor storage are counted.
Allocator accounting must additionally be measured by the caller on its device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
from typing import Any

import torch


NATIVE_METHODS = ("raw", "exponent", "learned", "e8_symbol", "learned_e8", "binary")
CPU_ENTROPY_ONLY = ("huffman", "deflate", "rice", "original_entropy_formats")


def _storage_bytes(tensors):
    seen = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            total += storage.nbytes()
            seen.add(key)
    return int(total)


def allocator_bytes(device):
    """Actual live allocator counter when this PyTorch device exposes one."""
    device = torch.device(device)
    backend = getattr(torch, device.type, None)
    fn = getattr(backend, "memory_allocated", None)
    if fn is None or device.type == "cpu":
        return None
    return int(fn(device))


def _wire(tensor):
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def _zigzag(values):
    values = values.to(torch.int64)
    return (values << 1) ^ (values >> 63)


def _unzigzag(values):
    values = values.to(torch.int64)
    return (values >> 1) ^ -(values & 1)


def _width(maximum, limit=32):
    powers = torch.bitwise_left_shift(torch.ones(limit, device=maximum.device, dtype=torch.int64),
                                     torch.arange(limit, device=maximum.device, dtype=torch.int64))
    return (maximum.to(torch.int64).reshape(-1, 1) >= powers).sum(dim=1).to(torch.uint8)


def _pack_variable(values, widths, group_size, max_width):
    """Device-side variable group widths; no per-value CPU transfers."""
    values = values.reshape(-1).to(torch.int64)
    expanded = widths.to(torch.int64).repeat_interleave(group_size)[:values.numel()]
    shifts = torch.arange(max_width, device=values.device, dtype=torch.int64)
    bits = ((values[:, None] >> shifts) & 1).to(torch.uint8)
    valid = shifts[None, :] < expanded[:, None]
    stream = bits[valid]
    nbits = stream.numel()
    if nbits % 8:
        stream = torch.cat((stream, torch.zeros(8 - nbits % 8, device=values.device, dtype=torch.uint8)))
    weights = (1 << torch.arange(8, device=values.device, dtype=torch.int64))
    packed = (stream.reshape(-1, 8).to(torch.int64) * weights).sum(dim=1).to(torch.uint8)
    return packed, int(nbits)


def _unpack_variable(payload, widths, count, group_size, max_width, nbits):
    shifts8 = torch.arange(8, device=payload.device, dtype=torch.int64)
    stream = ((payload.to(torch.int64)[:, None] >> shifts8) & 1).reshape(-1)[:nbits]
    expanded = widths.to(torch.int64).repeat_interleave(group_size)[:count]
    shifts = torch.arange(max_width, device=payload.device, dtype=torch.int64)
    valid = shifts[None, :] < expanded[:, None]
    bits = torch.zeros((count, max_width), device=payload.device, dtype=torch.int64)
    bits[valid] = stream
    return (bits << shifts).sum(dim=1)


def _pack_fixed(values, width):
    values = values.reshape(-1)
    widths = torch.full((1,), width, dtype=torch.uint8, device=values.device)
    return _pack_variable(values, widths, max(1, values.numel()), max(1, width))


def _unpack_fixed(payload, count, width):
    widths = torch.full((1,), width, dtype=torch.uint8, device=payload.device)
    return _unpack_variable(payload, widths, count, max(1, count), max(1, width), count * width)


def _adaptive(values, fields, descriptions, name, group_size=64, signed=False, max_width=16):
    values = values.reshape(-1).to(torch.int64)
    coded = _zigzag(values) if signed else values
    padding = (-coded.numel()) % group_size
    grouped = torch.cat((coded, torch.zeros(padding, device=coded.device, dtype=torch.int64))).reshape(-1, group_size)
    widths = _width(grouped.amax(dim=1), max_width)
    packed, nbits = _pack_variable(coded, widths, group_size, max_width)
    fields[name + "_widths"] = widths.contiguous()
    fields[name] = packed
    descriptions[name] = dict(count=values.numel(), group_size=group_size, max_width=max_width,
                              signed=signed, nbits=nbits)


def _restore(fields, descriptions, name):
    d = descriptions[name]
    result = _unpack_variable(fields[name], fields[name + "_widths"], d["count"], d["group_size"],
                              d["max_width"], d["nbits"])
    return _unzigzag(result) if d["signed"] else result


def canonical_matmul(left, matrix):
    left, matrix = left.to(torch.float64), matrix.to(torch.float64)
    result = torch.zeros((left.shape[0], matrix.shape[1]), device=left.device, dtype=torch.float64)
    # Eager separate mul/add operations; no matmul/compile/FMA reduction tree.
    for i in range(left.shape[1]):
        product = left[:, i:i + 1] * matrix[i:i + 1, :]
        result = result + product
    return result


def _row_dot(left, right):
    result = torch.zeros(left.shape[0], device=left.device, dtype=torch.float64)
    for i in range(left.shape[1]):
        product = left[:, i].to(torch.float64) * right[:, i].to(torch.float64)
        result = result + product
    return result


def _bf16_amplitude(values):
    # BF16 conversion and expansion are the actual stored-amplitude contract.
    words = values.to(torch.float32).to(torch.bfloat16).view(torch.uint16)
    return words, words.view(torch.bfloat16).to(torch.float32)


@dataclass
class NativeState:
    codebooks: dict[str, torch.Tensor] = field(default_factory=dict)
    rotations: dict[tuple, torch.Tensor] = field(default_factory=dict)
    state_id: str = "no_learned_state"
    rotation_ids: dict[tuple, str] = field(default_factory=dict)

    @property
    def physical_nbytes(self):
        return _storage_bytes(list(self.codebooks.values()) + list(self.rotations.values()))

    def rotation(self, dimension, transform="identity", seed=104729, device=None):
        key = (dimension, transform, seed)
        if key not in self.rotations:
            device = torch.device(device or next(iter(self.codebooks.values())).device)
            if transform == "identity":
                r = torch.eye(dimension, dtype=torch.float64)
            elif transform == "hadamard":
                r = torch.ones((1, 1), dtype=torch.float64)
                while len(r) < dimension:
                    r = torch.cat((torch.cat((r, r), dim=1), torch.cat((r, -r), dim=1)), dim=0)
                r = r / math.sqrt(dimension)
            elif transform == "dense":
                generator = torch.Generator(device="cpu").manual_seed(seed)
                r, upper = torch.linalg.qr(torch.randn((dimension, dimension), generator=generator, dtype=torch.float64))
                r = r * torch.where(upper.diag() < 0, -1.0, 1.0)[None, :]
            else:
                raise ValueError("unsupported transform")
            self.rotation_ids[key] = hashlib.sha256(r.contiguous().numpy().tobytes()).hexdigest()
            self.rotations[key] = r.to(device=device).contiguous()
        return self.rotations[key]


def make_state(codebooks=None, device="cpu", state_id=None, rotations=None):
    """State preparation may copy CPU dictionaries once; records stay on device."""
    books = {}
    digest = hashlib.sha256()
    for key, source in sorted((codebooks or {}).items()):
        value = torch.as_tensor(source, dtype=torch.int32, device=device).contiguous()
        if value.ndim != 2 or value.shape[1] != 8:
            raise ValueError("expected 8D dictionary rows")
        if not 1 <= len(value) <= 65536 or bool(torch.any((value < 0) | (value > 32767)).item()):
            raise ValueError("invalid magnitude dictionary")
        books[str(key)] = value
        digest.update(str(key).encode())
        digest.update(value.detach().cpu().numpy().tobytes())  # one-time state preparation
    result = NativeState(books, {}, state_id or digest.hexdigest())
    for key, source in (rotations or {}).items():
        dimension, transform, seed = key
        value = torch.as_tensor(source, dtype=torch.float64, device="cpu").contiguous()
        if value.shape != (dimension, dimension):
            raise ValueError("invalid explicit transform dimensions")
        result.rotation_ids[key] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        result.rotations[key] = value.to(device=device).contiguous()
    return result


@dataclass
class Packet:
    fields: dict[str, torch.Tensor]
    header: torch.Tensor
    metadata: dict[str, Any]
    _descriptor_bytes: bytes = field(default=b"", repr=False)
    _header_verified: bool = field(default=False, repr=False)

    @property
    def physical_nbytes(self):
        return _storage_bytes([self.header, *self.fields.values()])

    @property
    def stats(self):
        return dict(format="new_native_group_bitpack_v1", method=self.metadata["method"],
            serialized_bytes=self.physical_nbytes, raw_payload_bytes=2 * math.prod(self.metadata["shape"]),
            component_bytes=dict(header=self.header.numel(), **{key: value.untyped_storage().nbytes() for key, value in self.fields.items()}),
            native_entropy_backend="grouped_fixed_width_bitpacking", original_cpu_format=False,
            runtime_diagnostics={"dynamic_boolean_indexing": True,
                "allocation_shape_may_synchronize_device": True,
                "per_word_cpu_transfers": False, "eager_reference_not_fused": True,
                "prefetched_validated_host_launch_descriptor": True,
                "device_header_d2h_on_each_decode": False,
                "host_descriptor_guard_cost_in_decode": True},
            allocator_accounting="caller must measure memory_allocated delta/peak separately")


def _descriptor_bytes(metadata):
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()


def validate_packet(packet, check_header=True):
    """Validate authoritative wire metadata at restore, not every GPU launch.

    check_header=True bulk-copies the small header to CPU if necessary, parses
    that actual header, and compares it with the host launch descriptor.  New
    encode packets are valid by construction.  Isolated device decode uses the
    prefetched validated descriptor and cheap CPU consistency guards; it never
    transfers KV words or the header D2H on each launch.  No payload checksum is
    supplied by this eager benchmark format; callers verify recovered KV bits.
    """
    if packet.header.dtype != torch.uint8 or packet.header.ndim != 1 or not packet.header.is_contiguous():
        raise ValueError("invalid native packet header tensor")
    canonical = _descriptor_bytes(packet.metadata)
    if packet._descriptor_bytes and canonical != packet._descriptor_bytes:
        raise ValueError("native host launch descriptor changed after validation")
    if check_header:
        serialized = packet.header.detach().cpu().numpy().tobytes()
        try:
            parsed = json.loads(serialized)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("invalid native packet JSON header") from error
        if parsed != packet.metadata or serialized != canonical:
            raise ValueError("native wire header and host descriptor mismatch")
    elif not packet._header_verified:
        raise ValueError("native packet header has not been validated")
    meta = packet.metadata
    if (meta.get("version") != 1 or meta.get("method") not in NATIVE_METHODS or
            not isinstance(meta.get("shape"), list) or len(meta["shape"]) != 2 or
            any(not isinstance(n, int) or n < 0 for n in meta["shape"])):
        raise ValueError("invalid native packet descriptor")
    if set(meta.get("field_bytes", {})) != set(packet.fields):
        raise ValueError("native packet field set mismatch")
    for key, value in packet.fields.items():
        if (value.dtype != torch.uint8 or value.ndim != 1 or not value.is_contiguous()
                or value.device != packet.header.device or value.numel() != meta["field_bytes"][key]):
            raise ValueError("native packet field metadata mismatch: " + key)
    packet._descriptor_bytes = canonical
    packet._header_verified = True
    return packet.metadata


def packet_from_cpu_parts(header, fields):
    """Restore from actual CPU wire header and fields, with no external metadata."""
    if header.device.type != "cpu":
        raise ValueError("CPU wire parts required")
    try:
        metadata = json.loads(header.detach().numpy().tobytes())
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid native packet JSON header") from error
    packet = Packet(fields, header, metadata)
    validate_packet(packet, check_header=True)
    return packet


def packet_to_device(packet, device, non_blocking=False):
    """Explicit bulk packet transfer; caller times it separately from decode."""
    validate_packet(packet, check_header=True)
    return Packet({key: value.to(device=device, non_blocking=non_blocking, copy=True)
                   for key, value in packet.fields.items()},
                  packet.header.to(device=device, non_blocking=non_blocking, copy=True),
                  json.loads(json.dumps(packet.metadata)), packet._descriptor_bytes, True)


def packet_to_cpu(packet, pin_memory=False):
    result = packet_to_device(packet, "cpu")
    if pin_memory:
        result.fields = {key: value.pin_memory() for key, value in result.fields.items()}
        result.header = result.header.pin_memory()
    return result


def native_options(preset="main"):
    """Names intentionally distinguish these formats from CPU entropy formats."""
    if preset not in ("main", "all"):
        raise ValueError("native preset must be main or all")
    methods = [dict(name="native_raw_words", method="raw", options={}),
        dict(name="native_exponent_group_bitpack", method="exponent", options={}),
        dict(name="native_learned_residual_bitpack", method="learned", options={"assignment": "radius"}),
        dict(name="native_e8_symbol_bitpack", method="e8_symbol", options={"step_log2": 6}),
        dict(name="native_learned_e8_residual_bitpack", method="learned_e8", options={"step_log2": 10})]
    dimensions, precisions = ((64,), ("bf16",)) if preset == "main" else ((8, 32, 64), ("f32", "bf16"))
    for dimension in dimensions:
        for transform in ("identity", "dense", "hadamard"):
            for precision in precisions:
                methods.append(dict(name=f"native_binary_{transform}_d{dimension}_{precision}_bitpack",
                    method="binary", options=dict(dimension=dimension, transform=transform,
                    amplitude_precision=precision, assignment="radius")))
    return methods


def _finish(fields, meta, device):
    meta["field_bytes"] = {key: value.numel() * value.element_size() for key, value in fields.items()}
    serialized = _descriptor_bytes(meta)
    header = torch.tensor(list(serialized), device=device, dtype=torch.uint8)
    result = Packet(fields, header, meta, serialized, True)
    validate_packet(result, check_header=False)
    return result


def _assign(magnitude, codebook, assignment="radius"):
    blocks = magnitude.reshape(-1, 8).to(torch.int32)
    ids = []
    for start in range(0, len(blocks), 512):
        error = (blocks[start:start + 512, None, :] - codebook[None, :, :]).abs()
        score = error.amax(dim=2) if assignment == "radius" else error.sum(dim=2)
        ids.append(score.argmin(dim=1))
    return torch.cat(ids) if ids else torch.empty(0, device=magnitude.device, dtype=torch.int64)


def _d8(values):
    rounded = torch.round(values).to(torch.int64)
    odd = (rounded.sum(dim=1) & 1) != 0
    error = values - rounded.to(torch.float64)
    index = error.abs().argmax(dim=1)
    rows = torch.arange(len(values), device=values.device)
    direction = torch.where(error[rows, index] >= 0, 1, -1)
    rounded[rows, index] += direction * odd.to(torch.int64)
    return rounded


def nearest_e8_twice(values):
    values = values.to(torch.float64).reshape(-1, 8)
    integer = _d8(values) * 2
    half = _d8(values - 0.5) * 2 + 1
    di = ((values - integer.to(torch.float64) / 2) ** 2).sum(dim=1)
    dh = ((values - half.to(torch.float64) / 2) ** 2).sum(dim=1)
    return torch.where((dh < di)[:, None], half, integer)


def _store_e8(points2, fields, descriptions, group_size):
    integers = torch.div(points2, 2, rounding_mode="floor")
    flags = points2[:, 0] & 1
    fields["e8_coset"], _ = _pack_fixed(flags, 1)
    _adaptive(integers[:, :7].reshape(-1), fields, descriptions, "e8_first7", group_size, True, 17)
    high = torch.div(integers[:, 7], 2, rounding_mode="floor")
    _adaptive(high, fields, descriptions, "e8_last_high", group_size, True, 17)


def _restore_e8(fields, descriptions, count):
    first = _restore(fields, descriptions, "e8_first7").reshape(count, 7)
    high = _restore(fields, descriptions, "e8_last_high")
    last = 2 * high + (first.sum(dim=1) & 1)
    integers = torch.cat((first, last[:, None]), dim=1)
    flags = _unpack_fixed(fields["e8_coset"], count, 1)
    return integers * 2 + flags[:, None]


def _binary_prediction(base, bits, amplitudes, rotation):
    signs = bits.to(torch.float64) * 2 - 1
    u = canonical_matmul(signs, rotation.T) / math.sqrt(rotation.shape[0])
    return torch.round(base.to(torch.float64) + amplitudes.to(torch.float64)[:, None] * u).clamp(0, 32767).to(torch.int64)


def encode(words, state=None, method="raw", key="global", group_size=64, assignment="radius",
           dimension=64, transform="identity", seed=104729, amplitude_precision="bf16", step_log2=10):
    """Encode GPU uint16 words without copying record values to CPU.

    Data-dependent bitstream allocation uses tensor indexing and can synchronize
    the device allocator.  This is a correctness-oriented eager torch codec,
    not a fused fast compression kernel.  Time it on the target hardware.
    """
    if words.dtype != torch.uint16 or words.ndim != 2:
        raise ValueError("expected torch.uint16 [tokens,channels] BF16 words")
    if method not in NATIVE_METHODS:
        raise NotImplementedError("no native decoder for CPU entropy formats")
    if group_size < 1 or assignment not in ("radius", "l1") or amplitude_precision not in ("f32", "bf16"):
        raise ValueError("unsupported native parameters")
    state = state or NativeState()
    fields, descriptions = {}, {}
    meta = dict(version=1, method=method, shape=list(words.shape), key=key, state_id=state.state_id,
        group_size=group_size, assignment=assignment, descriptions=descriptions,
        dimension=dimension, transform=transform, seed=seed, amplitude_precision=amplitude_precision,
        step_log2=step_log2, original_cpu_format=False)
    flat = words.contiguous().reshape(-1).to(torch.int64)
    count = flat.numel()
    if method == "raw" or count == 0:
        fields["raw"] = _wire(words.clone())
        meta["method"] = "raw"
        return _finish(fields, meta, words.device)
    if method == "exponent":
        exponent = (flat >> 7) & 255
        lower = (((flat >> 8) & 128) | (flat & 127)).to(torch.uint8)
        padding = (-count) % group_size
        padded = torch.cat((exponent, torch.full((padding,), 255, device=words.device, dtype=torch.int64)))
        minimum = padded.reshape(-1, group_size).amin(dim=1).to(torch.uint8)
        delta = exponent - minimum.to(torch.int64).repeat_interleave(group_size)[:count]
        fields["sign_mantissa"], fields["exponent_minimum"] = lower.contiguous(), minimum
        _adaptive(delta, fields, descriptions, "exponent_delta", group_size, False, 8)
        return _finish(fields, meta, words.device)
    if method in ("learned", "learned_e8", "binary") and key not in state.codebooks:
        raise ValueError("learned native method requires a dictionary")
    if method == "binary" and dimension not in (8, 32, 64):
        raise ValueError("binary dimension must be 8,32,64")
    vector_dim = dimension if method == "binary" else 8
    padded_count = ((count + vector_dim - 1) // vector_dim) * vector_dim
    padded = torch.cat((flat, torch.zeros(padded_count - count, device=words.device, dtype=torch.int64)))
    magnitude = padded & 32767
    fields["original_signs"], _ = _pack_fixed(padded >> 15, 1)
    meta["padded_count"] = padded_count
    if method in ("learned", "learned_e8", "binary"):
        book = state.codebooks[key]
        if book.device != words.device:
            raise ValueError("native dictionary and words must be on the same device")
        ids = _assign(magnitude, book, assignment)
        qbits = (len(book) - 1).bit_length()
        fields["learned_ids"], _ = _pack_fixed(ids, qbits)
        meta["learned_index_bits"] = qbits
        base = book[ids].reshape(-1).to(torch.int64)
    else:
        base = torch.zeros_like(magnitude)
        if method == "e8_symbol":
            if group_size % 8:
                raise ValueError("E8 symbol groups must be multiples of eight")
            padding = (-padded_count) % group_size
            grouped = torch.cat((magnitude, torch.full((padding,), 32767, device=words.device,
                                                       dtype=torch.int64))).reshape(-1, group_size)
            minimum = grouped.amin(dim=1)
            fields["e8_symbol_minimum"] = _wire(minimum.to(torch.uint16))
            base = minimum.repeat_interleave(group_size)[:padded_count]
    if method in ("e8_symbol", "learned_e8"):
        if not 1 <= step_log2 <= 14:
            raise ValueError("invalid E8 step")
        residual = magnitude - base
        points2 = nearest_e8_twice(residual.reshape(-1, 8).to(torch.float64) / (1 << step_log2))
        _store_e8(points2, fields, descriptions, group_size)
        prediction = (base + points2.reshape(-1) * (1 << (step_log2 - 1))).clamp(0, 32767)
    elif method == "binary":
        rotation = state.rotation(dimension, transform, seed, words.device)
        meta["rotation_id"] = state.rotation_ids[(dimension, transform, seed)]
        residual = (magnitude - base).reshape(-1, dimension)
        projected = canonical_matmul(residual, rotation)
        bits = (projected >= 0).to(torch.int64)
        direction = canonical_matmul(bits.to(torch.float64) * 2 - 1, rotation.T) / math.sqrt(dimension)
        amplitudes = _row_dot(residual, direction).clamp_min(0).to(torch.float32)
        if amplitude_precision == "bf16":
            amp_words, amplitudes = _bf16_amplitude(amplitudes)
            fields["amplitudes"] = _wire(amp_words)
        else:
            fields["amplitudes"] = _wire(amplitudes)
        fields["binary_signs"], _ = _pack_fixed(bits.reshape(-1), 1)
        prediction = _binary_prediction(base.reshape(-1, dimension), bits, amplitudes, rotation).reshape(-1)
    else:
        prediction = base
    remaining = magnitude - prediction
    _adaptive(remaining, fields, descriptions, "exact_residual", group_size, True, 16)
    return _finish(fields, meta, words.device)


def decode(packet, state=None):
    """Tensor materialization with a prefetched validated CPU launch descriptor."""
    state = state or NativeState()
    validate_packet(packet, check_header=not packet._header_verified)
    meta, fields = packet.metadata, packet.fields
    if state.state_id != meta["state_id"]:
        raise ValueError("native state identity mismatch")
    method = meta["method"]
    count = math.prod(meta["shape"])
    if method == "raw":
        return fields["raw"].view(torch.uint16).reshape(meta["shape"]).clone()
    if method == "exponent":
        delta = _restore(fields, meta["descriptions"], "exponent_delta")
        minimum = fields["exponent_minimum"].to(torch.int64).repeat_interleave(meta["group_size"])[:count]
        exponent = delta + minimum
        lower = fields["sign_mantissa"].to(torch.int64)
        return (((lower & 128) << 8) | (lower & 127) | (exponent << 7)).to(torch.uint16).reshape(meta["shape"])
    padded = meta["padded_count"]
    signs = _unpack_fixed(fields["original_signs"], padded, 1)
    if method in ("learned", "learned_e8", "binary"):
        ids = _unpack_fixed(fields["learned_ids"], padded // 8, meta["learned_index_bits"])
        base = state.codebooks[meta["key"]][ids].reshape(-1).to(torch.int64)
    else:
        base = torch.zeros(padded, device=signs.device, dtype=torch.int64)
        if method == "e8_symbol":
            minimum = fields["e8_symbol_minimum"].view(torch.uint16).to(torch.int64)
            base = minimum.repeat_interleave(meta["group_size"])[:padded]
    if method in ("e8_symbol", "learned_e8"):
        points2 = _restore_e8(fields, meta["descriptions"], padded // 8)
        prediction = (base + points2.reshape(-1) * (1 << (meta["step_log2"] - 1))).clamp(0, 32767)
    elif method == "binary":
        dimension = meta["dimension"]
        bits = _unpack_fixed(fields["binary_signs"], padded, 1).reshape(-1, dimension)
        if meta["amplitude_precision"] == "bf16":
            amplitudes = fields["amplitudes"].view(torch.bfloat16).to(torch.float32)
        else:
            amplitudes = fields["amplitudes"].view(torch.float32)
        rotation = state.rotation(dimension, meta["transform"], meta["seed"], signs.device)
        if state.rotation_ids[(dimension, meta["transform"], meta["seed"])] != meta["rotation_id"]:
            raise ValueError("native rotation state identity mismatch")
        prediction = _binary_prediction(base.reshape(-1, dimension), bits, amplitudes, rotation).reshape(-1)
    else:
        prediction = base
    magnitude = prediction + _restore(fields, meta["descriptions"], "exact_residual")
    return (magnitude | (signs << 15))[:count].to(torch.uint16).reshape(meta["shape"])


def words_from_bf16(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError("cache tensor must actually be BF16")
    return tensor.contiguous().view(torch.uint16)


def bf16_from_words(words):
    if words.dtype != torch.uint16:
        raise ValueError("expected uint16 word bits")
    return words.contiguous().view(torch.bfloat16)


def self_test(device="cpu"):
    """Functional test on the requested device; unsupported ops raise honestly."""
    words = torch.arange(65536, device=device, dtype=torch.int64).to(torch.uint16).reshape(256, 256)
    books = {"self_test": torch.arange(16 * 8, dtype=torch.int32).reshape(16, 8) + 15900}
    state = make_state(books, device=device)
    results = {}
    for method in NATIVE_METHODS:
        packet = encode(words, state, method=method, key="self_test", transform="dense", dimension=32)
        restored = decode(packet, state)
        exact = bool(torch.all(words.to(torch.int32) == restored.to(torch.int32)).item())
        if not exact:
            raise AssertionError("native packet failed exact BF16 word recovery: " + method)
        if packet.physical_nbytes != sum(packet.stats["component_bytes"].values()):
            raise AssertionError("native packet storage accounting mismatch")
        results[method] = dict(exact=True, bf16_patterns_checked=65536,
            tensor_storage_bytes=packet.physical_nbytes)
    return dict(device=str(torch.device(device)), methods=results,
        scope="Functional tensor-op check on the named device; no GPU claim for CPU execution.")

"""Device-native implementation of the supplied team's exact packed pool records.

The SAME tensor operations run on CPU or CUDA. No codec operation copies KV to
CPU or calls the NumPy reference. The caller chooses the device by placing its
input/state there; unavailable CUDA raises rather than falling back to CPU.

This is a correctness/reference implementation, NOT an attention kernel. Python
orchestrates batches of records. Variable escape allocation uses one scalar
``sum().item()`` per batch and therefore synchronizes CUDA; validation and stats
also synchronize. Vectorized temporary tensors can exceed packed payload size.
State initialization may copy calibration widths to CPU once to cache shapes.

Each record is byte-identical to original combined.py: little-endian u16 escape
count, raw sign/mantissa bytes, MSB-first exponent codes, ordered raw escapes.
The optional block-offset directory is stored separately and charged separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch


def _physical_bytes(tensors):
    seen = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            seen.add(identity)
            total += storage.nbytes()
    return int(total)


@dataclass(frozen=True)
class PoolState:
    references: torch.Tensor
    widths: torch.Tensor
    width_sums: tuple[int, ...]

    @property
    def physical_nbytes(self):
        return _physical_bytes((self.references, self.widths))

    @property
    def device(self):
        return self.references.device


@dataclass(frozen=True)
class PoolPacket:
    payload: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, int, int]
    tile_tokens: int
    format: str = "team_pool_original_v1"

    @property
    def payload_nbytes(self):
        return self.payload.numel() * self.payload.element_size()

    @property
    def index_nbytes(self):
        return self.offsets.numel() * self.offsets.element_size()

    @property
    def physical_nbytes(self):
        return _physical_bytes((self.payload, self.offsets))

    @property
    def escape_counts(self):
        start = self.offsets[:-1]
        return self.payload[start].to(torch.int64) + (self.payload[start + 1].to(torch.int64) << 8)

    @property
    def tensor_fields(self):
        """Every tensor required for reconstruction, for complete transfer accounting."""
        return {"payload": self.payload, "offsets": self.offsets}


def packet_to_device(packet, device, non_blocking=False):
    """Copy ALL packet tensors; geometry/format remain shared host metadata.

    For an asynchronous pinned-host benchmark, pin both tensor_fields before
    calling this function. Calibration state is separate and must be measured
    as either warm resident state or an explicitly charged cold transfer.
    """
    return PoolPacket(packet.payload.to(device, non_blocking=non_blocking),
                      packet.offsets.to(device, non_blocking=non_blocking),
                      packet.shape, packet.tile_tokens, packet.format)


def make_state(references, widths, device="cpu"):
    """Create compact persistent uint8 calibration tensors on the chosen device.

    References/widths are fitted offline. Validation and width-sum caching are
    initialization work, not device encode/decode work. No model or KV is read.
    """
    refs = torch.as_tensor(references).detach().to(device="cpu", dtype=torch.int64)
    bits = torch.as_tensor(widths).detach().to(device="cpu", dtype=torch.int64)
    if refs.ndim != 2 or refs.shape != bits.shape or min(refs.shape) <= 0:
        raise ValueError("references and widths must have matching nonempty [layers,channels] shapes")
    if bool(((refs < 0) | (refs > 255)).any()) or bool(((bits < 1) | (bits > 8)).any()):
        raise ValueError("references must fit uint8 and widths must be in [1,8]")
    width_sums = tuple(int(value) for value in bits.sum(dim=1).tolist())
    return PoolState(refs.to(device=device, dtype=torch.uint8).contiguous().clone(),
                     bits.to(device=device, dtype=torch.uint8).contiguous().clone(), width_sums)


def _state_input_check(words, state, tile_tokens, tiles_per_batch):
    if words.dtype != torch.uint16 or words.ndim != 3 or min(words.shape) <= 0:
        raise ValueError("words must be nonempty original uint16 BF16 patterns [layers,tokens,channels]")
    if tuple(state.references.shape) != (words.shape[0], words.shape[2]):
        raise ValueError("input and calibration geometries differ")
    if words.device != state.device or state.widths.device != state.device:
        raise ValueError("input and calibration must already be on the same device")
    if tile_tokens <= 0 or words.shape[1] % tile_tokens:
        raise ValueError("token count must be a positive multiple of tile_tokens")
    if tile_tokens * words.shape[2] > 65535:
        raise ValueError("a tile must fit the original 16-bit escape counter")
    if tiles_per_batch <= 0:
        raise ValueError("tiles_per_batch must be positive")


def _encode_batch(words, references, widths, width_sum):
    # words: records x tile_tokens x channels; all tensors stay on words.device.
    records, tile_tokens, channels = words.shape
    count = tile_tokens * channels
    device = words.device
    integer = words.to(torch.int32)
    exponent = (integer >> 7) & 255
    sign_mantissa = (((integer >> 8) & 128) | (integer & 127)).to(torch.uint8).reshape(records, count)
    delta = exponent - references.to(torch.int32)[None, None, :]
    zigzag = torch.where(delta >= 0, 2 * delta, -2 * delta - 1)
    width = widths.to(torch.int64)
    sentinel = (1 << width) - 1
    escaped = zigzag > (sentinel[None, None, :] - 1)
    code = torch.where(escaped, sentinel[None, None, :], zigzag).to(torch.int32)

    bit_number = torch.arange(8, device=device, dtype=torch.int64)
    shift = (width[None, :, None] - 1 - bit_number[None, None, :]).clamp_min(0)
    expanded_bits = ((code[..., None] >> shift[None, ...]) & 1).to(torch.uint8)
    valid = (bit_number[None, None, :] < width[None, :, None]).expand(tile_tokens, -1, -1)
    stream = expanded_bits.reshape(records, -1)[:, valid.reshape(-1)]
    nbits = tile_tokens * width_sum
    padding = (-nbits) % 8
    if padding:
        stream = torch.cat((stream, torch.zeros((records, padding), dtype=torch.uint8, device=device)), dim=1)
    weights = 1 << torch.arange(7, -1, -1, device=device, dtype=torch.int64)
    dense = (stream.reshape(records, -1, 8).to(torch.int64) * weights).sum(dim=2).to(torch.uint8)
    dense_bytes = (nbits + 7) // 8
    fixed_bytes = 2 + count + dense_bytes

    escape_mask = escaped.reshape(records, count)
    escape_counts = escape_mask.sum(dim=1, dtype=torch.int64)
    sizes = fixed_bytes + escape_counts
    starts = sizes.cumsum(dim=0) - sizes
    # Explicit allocation synchronization; no per-value CPU copies or CPU codec.
    payload = torch.empty(int(sizes.sum().item()), dtype=torch.uint8, device=device)
    payload[starts] = (escape_counts & 255).to(torch.uint8)
    payload[starts + 1] = (escape_counts >> 8).to(torch.uint8)
    payload[starts[:, None] + 2 + torch.arange(count, device=device)[None, :]] = sign_mantissa
    payload[starts[:, None] + 2 + count + torch.arange(dense_bytes, device=device)[None, :]] = dense
    escape_rank = escape_mask.cumsum(dim=1) - 1
    escape_positions = starts[:, None] + fixed_bytes + escape_rank
    payload[escape_positions[escape_mask]] = exponent.reshape(records, count)[escape_mask].to(torch.uint8)
    return payload, starts


def encode_pool(words, state, tile_tokens=16, tiles_per_batch=16):
    """Pack exact BF16 words on their actual device, retaining ordered escapes.

    Batch size bounds temporary expanded-bit tensors. The offset directory is a
    separate, paid int64 tensor; it is not included in original payload bytes.
    """
    _state_input_check(words, state, tile_tokens, tiles_per_batch)
    layers, tokens, channels = words.shape
    tiles = tokens // tile_tokens
    payload_parts = []
    offset_parts = []
    byte_offset = 0
    for layer in range(layers):
        blocks = words[layer].reshape(tiles, tile_tokens, channels)
        for first in range(0, tiles, tiles_per_batch):
            payload, starts = _encode_batch(blocks[first:first + tiles_per_batch],
                                            state.references[layer], state.widths[layer], state.width_sums[layer])
            payload_parts.append(payload)
            offset_parts.append(starts + byte_offset)
            byte_offset += payload.numel()
    offset_parts.append(torch.tensor([byte_offset], dtype=torch.int64, device=words.device))
    return PoolPacket(torch.cat(payload_parts), torch.cat(offset_parts), tuple(words.shape), tile_tokens)


def _decode_batch(packet, record_offsets, references, widths, width_sum, validate):
    tile_tokens = packet.tile_tokens
    channels = widths.numel()
    count = tile_tokens * channels
    nbits = tile_tokens * width_sum
    dense_bytes = (nbits + 7) // 8
    fixed_bytes = 2 + count + dense_bytes
    payload = packet.payload
    device = payload.device
    starts = record_offsets[:-1]
    records = starts.numel()
    escape_count = payload[starts].to(torch.int64) + (payload[starts + 1].to(torch.int64) << 8)
    if validate and not torch.equal(record_offsets[1:] - starts, fixed_bytes + escape_count):
        raise ValueError("record length does not match original header/width layout")
    sm = payload[starts[:, None] + 2 + torch.arange(count, device=device)[None, :]].to(torch.int64)
    dense = payload[starts[:, None] + 2 + count + torch.arange(dense_bytes, device=device)[None, :]]
    stream = ((dense.to(torch.int64)[..., None] >> torch.arange(7, -1, -1, device=device)) & 1).reshape(records, -1)[:, :nbits]
    width = widths.to(torch.int64).repeat(tile_tokens)
    code_starts = width.cumsum(dim=0) - width
    bit_number = torch.arange(8, device=device, dtype=torch.int64)
    valid = bit_number[None, :] < width[:, None]
    indices = (code_starts[:, None] + bit_number[None, :]).clamp_max(nbits - 1)
    shifts = (width[:, None] - 1 - bit_number[None, :]).clamp_min(0)
    code = (stream[:, indices] * torch.where(valid, 1 << shifts, 0)[None, ...]).sum(dim=2)
    escaped = code == ((1 << width[None, :]) - 1)
    if validate and not torch.equal(escaped.sum(dim=1), escape_count):
        raise ValueError("header escape count differs from sentinels")
    delta = torch.where((code & 1) != 0, -((code + 1) >> 1), code >> 1)
    exponent = references.to(torch.int64).repeat(tile_tokens)[None, :] + delta
    rank = escaped.cumsum(dim=1) - 1
    escape_indices = (starts[:, None] + fixed_bytes + rank).clamp(0, payload.numel() - 1)
    raw_exponents = payload[escape_indices].to(torch.int64)
    exponent = torch.where(escaped, raw_exponents, exponent)
    return (((sm & 128) << 8) | (exponent << 7) | (sm & 127)).to(torch.uint16).reshape(records, tile_tokens, channels)


def decode_pool(packet, state, tiles_per_batch=16, validate=True):
    """Reconstruct exact uint16 BF16 words on packet.payload.device.

    validate=False is ONLY for trusted packets produced by encode_pool with
    matching unmodified calibration. It skips directory/header bounds checks;
    untrusted or damaged packets must use the default validate=True.
    """
    if packet.format != "team_pool_original_v1" or packet.payload.dtype != torch.uint8 or packet.offsets.dtype != torch.int64:
        raise ValueError("unsupported pool record representation")
    if packet.payload.ndim != 1 or packet.offsets.ndim != 1 or not packet.payload.numel():
        raise ValueError("pool payload/offsets must be nonempty flat tensors")
    if packet.payload.device != state.device or packet.offsets.device != state.device or state.widths.device != state.device:
        raise ValueError("packet and calibration must already be on the same device")
    if (not isinstance(packet.shape, (tuple, list)) or len(packet.shape) != 3
            or any(not isinstance(value, Integral) or isinstance(value, bool) or value <= 0 for value in packet.shape)):
        raise ValueError("pool shape must contain positive layers, tokens and channels")
    if not isinstance(packet.tile_tokens, Integral) or isinstance(packet.tile_tokens, bool) or packet.tile_tokens <= 0:
        raise ValueError("tile_tokens must be positive")
    layers, tokens, channels = packet.shape
    if packet.tile_tokens * channels > 65535:
        raise ValueError("a tile must fit the original 16-bit escape counter")
    if (state.references.dtype != torch.uint8 or state.widths.dtype != torch.uint8
            or tuple(state.references.shape) != (layers, channels) or state.widths.shape != state.references.shape
            or len(state.width_sums) != layers or any(value <= 0 or value > 8 * channels for value in state.width_sums)
            or tokens % packet.tile_tokens or tiles_per_batch <= 0):
        raise ValueError("invalid pool geometry or batch size")
    tiles = tokens // packet.tile_tokens
    if packet.offsets.numel() != layers * tiles + 1:
        raise ValueError("incorrect block offset directory size")
    if validate:
        if int(packet.offsets[0].item()) != 0 or int(packet.offsets[-1].item()) != packet.payload.numel():
            raise ValueError("invalid pool directory endpoints")
        if bool((packet.offsets[1:] <= packet.offsets[:-1]).any().item()):
            raise ValueError("pool offsets must be strictly increasing")
        if bool(((state.widths < 1) | (state.widths > 8)).any().item()):
            raise ValueError("invalid calibration widths")
        cached_sums = torch.tensor(state.width_sums, device=state.device, dtype=torch.int64)
        if not torch.equal(state.widths.sum(dim=1, dtype=torch.int64), cached_sums):
            raise ValueError("cached calibration width sums are stale")
        # Validate every fixed portion before reading even the two-byte header.
        # Together with endpoints and monotonic offsets this prevents malformed
        # records from causing an out-of-bounds CUDA gather/device-side assert.
        minimum_sizes = (2 + packet.tile_tokens * channels
                         + (packet.tile_tokens * cached_sums + 7) // 8).repeat_interleave(tiles)
        if bool(((packet.offsets[1:] - packet.offsets[:-1]) < minimum_sizes).any().item()):
            raise ValueError("record is shorter than its minimum header/raw/dense layout")
    layer_outputs = []
    for layer in range(layers):
        parts = []
        for first in range(0, tiles, tiles_per_batch):
            last = min(tiles, first + tiles_per_batch)
            record_offsets = packet.offsets[layer * tiles + first:layer * tiles + last + 1]
            parts.append(_decode_batch(packet, record_offsets, state.references[layer], state.widths[layer],
                                       state.width_sums[layer], validate))
        layer_outputs.append(torch.cat(parts).reshape(tokens, channels))
    return torch.stack(layer_outputs)


def packet_stats(packet, state=None, include_offsets=False):
    """Host diagnostics; outside timed kernels unless explicitly requested."""
    words = math.prod(packet.shape)
    counts = packet.escape_counts
    result = dict(format=packet.format, device=str(packet.payload.device), shape=list(packet.shape),
                  tile_tokens=packet.tile_tokens, blocks=packet.offsets.numel() - 1,
                  raw_bf16_bytes=2 * words, payload_bytes=packet.payload_nbytes,
                  block_index_bytes=packet.index_nbytes, physical_packet_tensor_bytes=packet.physical_nbytes,
                  payload_bits_per_value=8 * packet.payload_nbytes / words,
                  indexed_bits_per_value=8 * packet.physical_nbytes / words,
                  escape_header_bytes=2 * (packet.offsets.numel() - 1),
                  escape_count=int(counts.sum().item()), escape_counts=counts.cpu().tolist(),
                  geometry_metadata="shape/tile/format supplied out of band; not serialized in payload",
                  gpu_attention_kernel=False,
                  host_synchronization="one allocation-size item per batch; validation/stats also synchronize")
    if state is not None:
        result["physical_state_tensor_bytes"] = state.physical_nbytes
        result["state_width_sums_host_integers"] = len(state.width_sums)
    if include_offsets:
        result["offsets"] = packet.offsets.cpu().tolist()
    return result

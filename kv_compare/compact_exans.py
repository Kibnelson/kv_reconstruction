"""Complete compact carrier around the unchanged pinned ExANS C ABI.

The native codec still uses fixed 770,048-byte records. Only each header's
actual_bytes prefix travels; an int32 directory locates those prefixes. GPU
gather/scatter performs packing and restoration to the native record arena.
Raw overflow blocks and the padded physical tail are charged explicitly.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import math
import struct

import torch

from .causal_packet import json_bytes
from .framed_baselines import frame, unframe
from .official_exans import COMMIT, RAW, RECORD, Exans
from .official_exans import METHOD as FIXED_METHOD, STATE as FIXED_STATE

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

METHOD = 'exans_official_compact_framed'
HEADER_BYTES = 64
SYMBOLS = RAW // 2
PHASE_BYTES = SYMBOLS // 128 // 8 + ((122 * 2 + 31) // 32) * 4
FRAME_OFFSET = (HEADER_BYTES + SYMBOLS + PHASE_BYTES + 15) // 16 * 16
MAX_OFFSET = (1 << 31) - 1
HEADER = struct.Struct('<IHHQ12I')
MAGIC = 0x31505845
BLOCK = 8192


def layout_id():
    value = 1469598103934665603
    for item in (RAW, 1, RAW):  # Native creation uses one RAW-byte segment.
        for byte in int(item).to_bytes(4, 'little'):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


STATE = hashlib.sha256(json_bytes(dict(commit=COMMIT, raw=RAW, record=RECORD,
    native_version=2, carrier_version=1, header=HEADER_BYTES, layout=layout_id()))).hexdigest()


def validate_header(data, expected_bytes=None):
    """Validate the reviewed native v2 header before native pointer use."""
    if len(data) != HEADER_BYTES:
        raise ValueError('native header must have 64 bytes')
    magic, version, hb, layout, *v = HEADER.unpack(bytes(data))
    raw, actual, so, sb, po, pb, fo, fb, flags, r0, r1, r2 = v
    if (magic != MAGIC or version != 2 or hb != HEADER_BYTES or layout != layout_id()
        or raw != RAW or not FRAME_OFFSET < actual <= RECORD
        or so != HEADER_BYTES or sb != SYMBOLS or po != HEADER_BYTES + SYMBOLS
        or pb != PHASE_BYTES or fo != FRAME_OFFSET or fb != actual - FRAME_OFFSET
        or flags or r0 or r1 or r2 or (expected_bytes is not None and actual != expected_bytes)):
        raise ValueError('invalid native ExANS record header/length')
    return actual


def validate_directory(shape, good, bad, offsets, payload_bytes, raw_bytes, capacity):
    if (not shape or len(shape) > 4 or any(type(x) is not int or x < 1 for x in shape)
        or type(capacity) is not int or capacity < 1):
        raise ValueError('invalid compact ExANS geometry/capacity')
    n = math.prod(shape) * 2
    count = (n + RAW - 1) // RAW
    if n > MAX_OFFSET or count > capacity:
        raise ValueError('compact ExANS request exceeds indexed capacity')
    if (any(type(x) is not int for x in good + bad + offsets)
        or good != sorted(set(good)) or bad != sorted(set(bad))
        or sorted(good + bad) != list(range(count))):
        raise ValueError('fit/raw directories must partition physical blocks')
    if (len(offsets) != len(good) + 1 or not offsets or offsets[0] != 0
        or type(payload_bytes) is not int or not 0 <= payload_bytes <= MAX_OFFSET
        or offsets[-1] != payload_bytes or any(not 0 <= x <= MAX_OFFSET for x in offsets)
        or any(not FRAME_OFFSET < b - a <= RECORD for a, b in zip(offsets, offsets[1:]))
        or type(raw_bytes) is not int or raw_bytes != len(bad) * RAW or payload_bytes + raw_bytes > MAX_OFFSET):
        raise ValueError('invalid compact record offsets or raw extent')
    return dict(nbytes=n, blocks=count, input_padding_bytes=count * RAW - n)


def packet_bytes(packet):
    """Exact serialized envelope size without exporting the GPU body."""
    return 24 + len(json_bytes(packet.header)) + 8*packet.starts.numel() + 2*packet.payload.numel() + 32


def _bytes(tensor):
    return bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _headers_fixed(records):
    n = records.numel() // RECORD
    return records.view(n, RECORD)[:, :HEADER_BYTES].contiguous()


def _check_headers(headers, expected=None):
    blob = _bytes(headers)
    return [validate_header(blob[i:i + HEADER_BYTES], None if expected is None else expected[j])
            for j, i in enumerate(range(0, len(blob), HEADER_BYTES))]


if triton is not None:
    @triton.jit
    def _pack(src, offsets, dst, STRIDE: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        i = tl.program_id(1) * B + tl.arange(0, B)
        a, b = tl.load(offsets + row), tl.load(offsets + row + 1)
        value = tl.load(src + row * STRIDE + i, mask=i < b - a, other=0)
        tl.store(dst + a + i, value, mask=i < b - a)

    @triton.jit
    def _scatter(src, offsets, indexes, dst, STRIDE: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        i = tl.program_id(1) * B + tl.arange(0, B)
        a, b = tl.load(offsets + row), tl.load(offsets + row + 1)
        index = tl.load(indexes + row)
        value = tl.load(src + a + i, mask=i < b - a, other=0)
        tl.store(dst + index * STRIDE + i, value, mask=i < STRIDE)

    @triton.jit
    def _headers(src, offsets, dst, B: tl.constexpr):
        row = tl.program_id(0)
        i = tl.arange(0, B)
        a = tl.load(offsets + row)
        tl.store(dst + row * B + i, tl.load(src + a + i))


def compact_records(records):
    """Return int32 offsets and actual record prefixes; CPU is a software oracle."""
    if records.dtype != torch.uint8 or records.ndim != 1 or not records.is_contiguous() or records.numel() % RECORD:
        raise ValueError('flat native record bytes required')
    if records.numel() > MAX_OFFSET:
        raise ValueError('native record extent exceeds int32 directory')
    n = records.numel() // RECORD
    headers = _headers_fixed(records)
    lengths = _check_headers(headers)
    if records.device.type == 'cuda':
        if triton is None:
            raise RuntimeError('Triton is required for GPU compaction')
        counts = headers.view(torch.int32)[:, 5].contiguous()
        offsets64 = torch.cat((torch.zeros(1, dtype=torch.int64, device=records.device),
                               torch.cumsum(counts, 0, dtype=torch.int64)))
        total = int(offsets64[-1].item())
        if total != sum(lengths) or total > MAX_OFFSET:
            raise ValueError('invalid GPU record scan')
        offsets = offsets64.to(torch.int32)
        payload = torch.empty(total, dtype=torch.uint8, device=records.device)
        if n:
            _pack[(n, triton.cdiv(RECORD, BLOCK))](records, offsets, payload, STRIDE=RECORD, B=BLOCK, num_warps=4)
    elif records.device.type == 'cpu':
        starts = [0]
        for size in lengths: starts.append(starts[-1] + size)
        offsets = torch.tensor(starts, dtype=torch.int32)
        payload = torch.cat([records[i*RECORD:i*RECORD+size] for i, size in enumerate(lengths)]) if n else records.clone()
    else:
        raise ValueError('only CPU oracle and actual CUDA are supported')
    return offsets, payload


def expand_records(payload, offsets, good, target):
    """Scatter compact prefixes and zero native slot tails; never call the codec."""
    if (payload.dtype != torch.uint8 or payload.ndim != 1 or not payload.is_contiguous()
        or offsets.dtype != torch.int32 or offsets.ndim != 1 or not offsets.is_contiguous()
        or good.dtype != torch.int32 or good.ndim != 1 or not good.is_contiguous()
        or target.dtype != torch.uint8 or target.ndim != 1 or not target.is_contiguous()
        or target.numel() % RECORD or any(t.device != target.device for t in (payload, offsets, good))):
        raise ValueError('incompatible compact scatter tensors')
    starts, indexes = offsets.cpu().tolist(), good.cpu().tolist()
    if (len(starts) != len(indexes) + 1 or not starts or starts[0] != 0 or starts[-1] != payload.numel()
        or indexes != sorted(set(indexes)) or any(i < 0 or i >= target.numel() // RECORD for i in indexes)
        or any(not FRAME_OFFSET < b-a <= RECORD for a,b in zip(starts, starts[1:]))):
        raise ValueError('invalid compact scatter directory')
    if target.device.type == 'cuda':
        if triton is None: raise RuntimeError('Triton is required for GPU scatter')
        if indexes:
            _scatter[(len(indexes), triton.cdiv(RECORD, BLOCK))](payload, offsets, good, target, STRIDE=RECORD, B=BLOCK, num_warps=4)
    elif target.device.type == 'cpu':
        for j, index in enumerate(indexes):
            slot = target[index*RECORD:(index+1)*RECORD]; slot.zero_()
            slot[:starts[j+1]-starts[j]].copy_(payload[starts[j]:starts[j+1]])
    else:
        raise ValueError('only CPU oracle and actual CUDA are supported')
    return target


def _compact_headers(payload, offsets):
    n = offsets.numel() - 1
    if payload.device.type == 'cuda':
        if triton is None: raise RuntimeError('Triton is required for GPU header gather')
        headers = torch.empty((n, HEADER_BYTES), dtype=torch.uint8, device=payload.device)
        if n: _headers[(n,)](payload, offsets, headers, B=HEADER_BYTES, num_warps=4)
        return headers
    starts = offsets.tolist()
    return torch.stack([payload[a:a+HEADER_BYTES] for a in starts[:-1]]) if n else torch.empty((0, HEADER_BYTES), dtype=torch.uint8)


def _parts(packet, capacity):
    shape, parts, extra = unframe(packet, METHOD, STATE)
    expected = {'fit_indices':torch.int32, 'raw_indices':torch.int32, 'record_offsets':torch.int32,
                'records':torch.uint8, 'raw':torch.uint8}
    if set(parts) != set(expected) or any(parts[k].dtype != v for k,v in expected.items()):
        raise ValueError('invalid compact ExANS fields')
    good, bad, offsets = (parts[k].cpu().tolist() for k in ('fit_indices','raw_indices','record_offsets'))
    layout = validate_directory(shape, good, bad, offsets, parts['records'].numel(), parts['raw'].numel(), capacity)
    wanted = dict(layout, raw_block_bytes=RAW, record_bytes=RECORD, native_header_bytes=HEADER_BYTES)
    if extra != wanted:
        raise ValueError('compact ExANS geometry/header differs')
    _check_headers(_compact_headers(parts['records'], parts['record_offsets']),
                   [b-a for a,b in zip(offsets, offsets[1:])])
    return shape, parts, good, bad, layout


def compact_packet(fixed, capacity):
    """Software-testable conversion of the existing complete fixed-record packet."""
    shape, parts, extra = unframe(fixed, FIXED_METHOD, FIXED_STATE)
    if set(parts) != {'fit_indices','raw_indices','records','raw'} or parts['records'].dtype != torch.uint8 or parts['raw'].dtype != torch.uint8:
        raise ValueError('invalid source fixed-record fields')
    if parts['fit_indices'].dtype != torch.int32 or parts['raw_indices'].dtype != torch.int32:
        raise ValueError('invalid source fixed-record directory')
    good, bad = parts['fit_indices'].cpu().tolist(), parts['raw_indices'].cpu().tolist()
    layout = validate_directory(shape, good, bad, [i*RECORD for i in range(len(good)+1)],
        len(good)*RECORD, parts['raw'].numel(), capacity)
    if (parts['records'].numel() != len(good)*RECORD or extra != dict(blocks=layout['blocks'],
        raw_block_bytes=RAW, record_bytes=RECORD, input_padding_bytes=layout['input_padding_bytes'])):
        raise ValueError('source fixed-record extent/geometry differs')
    offsets, packed = compact_records(parts['records'])
    result = frame(METHOD, shape, STATE, [('fit_indices',parts['fit_indices']), ('raw_indices',parts['raw_indices']),
        ('record_offsets',offsets), ('records',packed), ('raw',parts['raw'])],
        dict(layout, raw_block_bytes=RAW, record_bytes=RECORD, native_header_bytes=HEADER_BYTES))
    if packet_bytes(result) > MAX_OFFSET:
        raise ValueError('complete compact carrier exceeds int32 indexing')
    return result


class CompactExans(Exans):
    method = METHOD
    state_sha256 = STATE

    def compact_fixed(self, packet):
        if packet.payload.device != self.device: raise ValueError('packet must reside on the codec device')
        return compact_packet(packet, self.blocks)

    def encode(self, words):
        if (words.dtype != torch.int16 or not words.is_contiguous() or words.device != self.device
            or not 1 <= words.ndim <= 4 or words.numel() == 0 or words.numel()*2 > MAX_OFFSET):
            raise ValueError('original contiguous BF16 word view required')
        return self.compact_fixed(super().encode(words))

    def expand(self, packet):
        if packet.payload.device != self.device: raise ValueError('packet must reside on the decoder device')
        shape, parts, good, bad, layout = _parts(packet, self.blocks)
        expand_records(parts['records'], parts['record_offsets'], parts['fit_indices'], self.records)
        return shape, parts, good, bad, layout

    def decode(self, packet):
        shape, parts, good, bad, layout = self.expand(packet)
        out = torch.empty(layout['blocks'], RAW, dtype=torch.uint8, device=self.device)
        if good:
            self.check(self.lib.openlake_expans_decode(self.handle, self.records.data_ptr(),
                self.arr(C.c_uint32,good), len(good), self.arr(C.c_uint64,[out.data_ptr()+i*RAW for i in good]),
                self.arr(C.c_uint32,[RAW]*len(good)), self.arr(C.c_uint32,range(len(good)+1)),
                torch.cuda.current_stream(self.device).cuda_stream))
        if bad:
            out.index_copy_(0,parts['raw_indices'].to(torch.int64),parts['raw'].view(-1,RAW))
        return out.flatten()[:layout['nbytes']].view(torch.int16).reshape(shape)

    def accounting(self, packet):
        fields = {v['name']:v for v in packet.header['fields']}; extra=packet.header['extra']
        good, bad = fields['fit_indices']['count'], fields['raw_indices']['count']
        payload = sum(v['bytes'] for v in fields.values())
        total=packet_bytes(packet)
        return dict(packet_bytes=total,input_padding_bytes=extra['input_padding_bytes'], fitting_blocks=good, raw_fallback_blocks=bad,
            native_used_record_bytes=fields['records']['bytes'], charged_raw_fallback_bytes=fields['raw']['bytes'],
            record_directory_bytes=fields['record_offsets']['bytes'], block_directory_bytes=4*(good+bad),
            charged_payload_field_bytes=payload, fixed_records_for_fitting_blocks=good*RECORD,
            native_record_arena_bytes_at_request_size=extra['blocks']*RECORD,
            native_record_arena_bytes_at_capacity=self.records.numel(),
            padded_bf16_output_bytes=extra['blocks']*RAW, framing_and_alignment_bytes=total-payload,
            raw_block_bytes=RAW,record_bytes=RECORD,native_allocation_peaks='unmeasured',
            scope='Complete compact carrier with actual native prefixes, directories, framing and raw fallback; native fixed-slot arena and BF16 output are additional.')

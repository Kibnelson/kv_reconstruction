"""Layer-mapped lossless SplitZip I/O, keeping the bundled wire unchanged.

Kernel arithmetic is derived from Intelligent-Microsystems-Lab/SplitZip
codec_gpu.py at 42c743118a29db3270ae6a921a4a4eecb20cad04,
SHA256 e28fe3263e8906b26d59789cbc5bccf14edc8391b0dae986aafaf34adb2d392c.
Copyright (c) 2026 Yipin Guo and Siddharth Joshi, MIT License;
see mapped_splitzip_LICENSE.txt. Only source/destination addressing changes.

This is a separately labelled integration implementation, not an unchanged
author baseline. CPU execution requires explicit TRITON_INTERPRET=1 and is a
software semantic check, never a performance fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import triton
import triton.language as tl

SOURCE_COMMIT = '42c743118a29db3270ae6a921a4a4eecb20cad04'
SOURCE_SHA256 = 'e28fe3263e8906b26d59789cbc5bccf14edc8391b0dae986aafaf34adb2d392c'
CHUNK = 1024


@dataclass(frozen=True)
class LayerMap:
    """Owned mapping prepared once, with a version guard against mutation."""
    tensor: torch.Tensor
    layers: tuple
    physical_layers: int
    version: int


def prepare_mapping(index, layers):
    if type(layers) is not int or layers < 1 or not isinstance(index, torch.Tensor):
        raise ValueError('positive physical layer count and INT64 index tensor required')
    if index.dtype != torch.int64 or index.ndim != 1 or not index.is_contiguous():
        raise ValueError('flat contiguous INT64 layer map required')
    values = index.cpu().tolist()
    if not values or values != sorted(set(values)) or any(i < 0 or i >= layers for i in values):
        raise ValueError('fallback layers must be increasing, unique and in bounds')
    owned = index.clone()
    return LayerMap(owned, tuple(values), layers, owned._version)


def _geometry(words, mapping):
    if (words.dtype != torch.int16 or words.ndim != 3 or not words.is_contiguous()
        or any(n <= 0 for n in words.shape)):
        raise ValueError('contiguous original INT16 BF16 words [L,T,W] required')
    if not isinstance(mapping, LayerMap):
        mapping = prepare_mapping(mapping, words.shape[0])
    if (mapping.physical_layers != words.shape[0] or mapping.tensor.device != words.device
        or mapping.tensor.dtype != torch.int64 or mapping.tensor._version != mapping.version):
        raise ValueError('owned layer map changed or has incompatible geometry/device')
    per_layer = words.shape[1] * words.shape[2]
    if per_layer % CHUNK or words.shape[2] % 4 or words.data_ptr() % 8 or words.numel() >= 1 << 31:
        raise ValueError('mapped I/O requires complete 1024-word layers and INT32 extents')
    if words.device.type != 'cuda' and not (words.device.type == 'cpu' and os.environ.get('TRITON_INTERPRET') == '1'):
        raise RuntimeError('actual CUDA required; CPU only with explicit Triton software interpreter')
    return mapping, per_layer, len(mapping.layers) * per_layer


def _lut(codec, name, words, count):
    tensor = getattr(codec, name, None)
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8
        or tensor.device != words.device or tensor.ndim != 1 or tensor.numel() != count
        or not tensor.is_contiguous()):
        raise ValueError('invalid SplitZip LUT: ' + name)
    return tensor


@triton.jit
def _mapped_encode_count(inp64, mapping, marked_lut, pk16, sm32, counts,
                                   n: tl.constexpr, n_chunks: tl.constexpr, LAYER_QUADS: tl.constexpr, BLOCK_QUADS: tl.constexpr):
    pid = tl.program_id(0)
    q_base = pid * BLOCK_QUADS
    po = tl.arange(0, BLOCK_QUADS)
    qoffs = q_base + po
    elem_base = qoffs * 4
    mask0 = elem_base < n
    mask1 = elem_base + 1 < n
    mask2 = elem_base + 2 < n
    mask3 = elem_base + 3 < n

    layer = tl.load(mapping + qoffs // LAYER_QUADS, mask=mask0, other=0).to(tl.int64)
    physical = layer * LAYER_QUADS + qoffs % LAYER_QUADS
    raw_pack = tl.load(inp64 + physical, mask=mask0, other=0).to(tl.uint64)
    raw0 = (raw_pack & 0xFFFF).to(tl.int32)
    raw1 = ((raw_pack >> 16) & 0xFFFF).to(tl.int32)
    raw2 = ((raw_pack >> 32) & 0xFFFF).to(tl.int32)
    raw3 = ((raw_pack >> 48) & 0xFFFF).to(tl.int32)

    exp0 = (raw0 >> 7) & 0xFF
    exp1 = (raw1 >> 7) & 0xFF
    exp2 = (raw2 >> 7) & 0xFF
    exp3 = (raw3 >> 7) & 0xFF
    code0 = tl.load(marked_lut + exp0, mask=mask0, other=16).to(tl.int32)
    code1 = tl.load(marked_lut + exp1, mask=mask1, other=16).to(tl.int32)
    code2 = tl.load(marked_lut + exp2, mask=mask2, other=16).to(tl.int32)
    code3 = tl.load(marked_lut + exp3, mask=mask3, other=16).to(tl.int32)

    pk0 = (((code0 & 0x0F) << 4) | (code1 & 0x0F)) & 0xFF
    pk1 = (((code2 & 0x0F) << 4) | (code3 & 0x0F)) & 0xFF
    tl.store(pk16 + qoffs, (pk0 | (pk1 << 8)).to(tl.uint16), mask=mask0)

    sm0 = ((raw0 >> 8) & 0x80) | (raw0 & 0x7F)
    sm1 = ((raw1 >> 8) & 0x80) | (raw1 & 0x7F)
    sm2 = ((raw2 >> 8) & 0x80) | (raw2 & 0x7F)
    sm3 = ((raw3 >> 8) & 0x80) | (raw3 & 0x7F)
    sm_pack = sm0 | (sm1 << 8) | (sm2 << 16) | (sm3 << 24)
    tl.store(sm32 + qoffs, sm_pack, mask=mask0)

    esc = (
        ((code0 > 15) & mask0).to(tl.int32)
        + ((code1 > 15) & mask1).to(tl.int32)
        + ((code2 > 15) & mask2).to(tl.int32)
        + ((code3 > 15) & mask3).to(tl.int32)
    )
    first = po < (BLOCK_QUADS // 2)
    count0 = tl.sum(tl.where(first, esc, 0), axis=0)
    count1 = tl.sum(tl.where(first, 0, esc), axis=0)
    chunk0 = pid * 2
    tl.store(counts + chunk0, count0, mask=chunk0 < n_chunks)
    tl.store(counts + chunk0 + 1, count1, mask=chunk0 + 1 < n_chunks)


@triton.jit
def _mapped_write_escapes(inp, mapping, common_lut, starts, chunk_id, local_pos, esc_val,
                         n: tl.constexpr, LAYER_WORDS: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    offs = base + tl.arange(0, CHUNK)
    mask = offs < n
    layer = tl.load(mapping + offs // LAYER_WORDS, mask=mask, other=0).to(tl.int64)
    physical = layer * LAYER_WORDS + offs % LAYER_WORDS
    raw = tl.load(inp + physical, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    rank = tl.cumsum(esc.to(tl.int32), 0) - 1
    start = tl.load(starts + pid)
    out = start + rank
    tl.store(chunk_id + out, pid, mask=esc)
    tl.store(local_pos + out, (offs - base).to(tl.uint16), mask=esc)
    tl.store(esc_val + out, exp.to(tl.uint8), mask=esc)


@triton.jit
def _mapped_decode_quads(pk16, sm32, dlut, mapping, out64, n_quads, LAYER_QUADS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    packed = tl.load(pk16 + offs, mask=mask, other=0).to(tl.int32)
    p0 = packed & 0xFF
    p1 = (packed >> 8) & 0xFF

    sm_pack = tl.load(sm32 + offs, mask=mask, other=0).to(tl.int32)
    sm0 = sm_pack & 0xFF
    sm1 = (sm_pack >> 8) & 0xFF
    sm2 = (sm_pack >> 16) & 0xFF
    sm3 = (sm_pack >> 24) & 0xFF

    exp0 = tl.load(dlut + ((p0 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (p0 & 0x0F), mask=mask, other=0).to(tl.int32)
    exp2 = tl.load(dlut + ((p1 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp3 = tl.load(dlut + (p1 & 0x0F), mask=mask, other=0).to(tl.int32)

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    raw2 = ((sm2 & 0x80) << 8) | (exp2 << 7) | (sm2 & 0x7F)
    raw3 = ((sm3 & 0x80) << 8) | (exp3 << 7) | (sm3 & 0x7F)

    quad = (
        raw0.to(tl.uint64)
        | (raw1.to(tl.uint64) << 16)
        | (raw2.to(tl.uint64) << 32)
        | (raw3.to(tl.uint64) << 48)
    )
    layer = tl.load(mapping + offs // LAYER_QUADS, mask=mask, other=0).to(tl.int64)
    physical = layer * LAYER_QUADS + offs % LAYER_QUADS
    tl.store(out64 + physical, quad, mask=mask)


@triton.jit
def _mapped_fix_escapes(chunk_id, local_pos, esc_val, sm, mapping, out,
                              n_esc, LAYER_WORDS: tl.constexpr, CHUNK: tl.constexpr, BLOCK_ESC: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_ESC + tl.arange(0, BLOCK_ESC)
    mask = offs < n_esc
    chunk = tl.load(chunk_id + offs, mask=mask, other=0).to(tl.int32)
    local = tl.load(local_pos + offs, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + offs, mask=mask, other=0).to(tl.int32)
    pos = chunk * CHUNK + local
    s = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)
    layer = tl.load(mapping + pos // LAYER_WORDS, mask=mask, other=0).to(tl.int64)
    physical = layer * LAYER_WORDS + pos % LAYER_WORDS
    tl.store(out + physical, fixed.to(tl.int16), mask=mask)


def encode_layers(codec, module, words, fallback_index):
    """Encode increasing selected layers without a gathered BF16 temporary."""
    mapping, layer_words, n = _geometry(words, fallback_index)
    if codec.chunk_size != CHUNK:
        raise ValueError('mapped codec uses the frozen 1024-word chunk geometry')
    marked = _lut(codec, 'enc_lut_marked', words, 256)
    common = _lut(codec, 'common_lut', words, 256)
    pk = torch.empty(n // 2, dtype=torch.uint8, device=words.device)
    sm = torch.empty(n, dtype=torch.uint8, device=words.device)
    n_chunks = n // CHUNK
    counts = torch.empty(n_chunks, dtype=torch.int32, device=words.device)
    _mapped_encode_count[(triton.cdiv(n_chunks, 2),)](
        words.view(torch.int64), mapping.tensor, marked, pk.view(torch.int16), sm.view(torch.int32),
        counts, n, n_chunks, LAYER_QUADS=layer_words // 4, BLOCK_QUADS=512, num_warps=4)
    offsets = torch.cumsum(counts, dim=0)
    starts = offsets - counts
    n_esc = int(offsets[-1].item()) if offsets.numel() else 0
    chunk_id = torch.empty(n_esc, dtype=torch.int32, device=words.device)
    local_pos = torch.empty(n_esc, dtype=torch.uint16, device=words.device)
    esc_val = torch.empty(n_esc, dtype=torch.uint8, device=words.device)
    if n_esc:
        _mapped_write_escapes[(n_chunks,)](
            words.flatten(), mapping.tensor, common, starts, chunk_id, local_pos, esc_val,
            n, LAYER_WORDS=layer_words, CHUNK=CHUNK, num_warps=4)
    return module.ChunkLocalGPUEncoded(pk, sm, counts, starts, chunk_id, local_pos, esc_val, n, n_esc, CHUNK)


def decode_layers_into(codec, encoded, fallback_index, out, *, validated=False):
    """Decode into final layer positions; no decoded fallback temporary.

    The integration may pass validated=True only after its checked packet
    parser has validated the same escape fields. Structural checks remain.
    """
    if type(validated) is not bool:
        raise ValueError('explicit packet-validation flag required')
    mapping, layer_words, n = _geometry(out, fallback_index)
    lut = _lut(codec, 'dec_lut', out, 16)
    ne = encoded.n_esc
    if (encoded.n != n or type(ne) is not int or not 0 <= ne <= n
        or encoded.chunk_size != CHUNK or codec.chunk_size != CHUNK):
        raise ValueError('invalid mapped bundle geometry')
    expected = dict(pk=(torch.uint8, n // 2), sm=(torch.uint8, n), chunk_id=(torch.int32, ne),
                    local_pos=(torch.uint16, ne), esc_val=(torch.uint8, ne))
    for name, (dtype, count) in expected.items():
        value = getattr(encoded, name)
        if (value.dtype != dtype or value.device != out.device or value.ndim != 1
            or value.numel() != count or not value.is_contiguous()):
            raise ValueError('invalid mapped bundle field: ' + name)
    if encoded.pk.data_ptr() % 2 or encoded.sm.data_ptr() % 4:
        raise ValueError('mapped packed fields must retain native view alignment')
    if ne and not validated:
        chunk, pos = encoded.chunk_id.long(), encoded.local_pos.long()
        if bool(((chunk < 0) | (pos >= CHUNK) | (chunk * CHUNK + pos >= n)).any()):
            raise ValueError('mapped escape position out of bounds')
    _mapped_decode_quads[(triton.cdiv(n // 4, 512),)](
        encoded.pk.view(torch.int16), encoded.sm.view(torch.int32), lut, mapping.tensor,
        out.view(torch.int64), n // 4, LAYER_QUADS=layer_words // 4, BLOCK=512, num_warps=4)
    if ne:
        _mapped_fix_escapes[(triton.cdiv(ne, 128),)](
            encoded.chunk_id, encoded.local_pos, encoded.esc_val, encoded.sm, mapping.tensor,
            out.flatten(), ne, LAYER_WORDS=layer_words, CHUNK=CHUNK, BLOCK_ESC=128, num_warps=4)
    return out

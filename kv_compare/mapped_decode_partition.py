"""Partitioned mapped SplitZip decoding with the original bundled wire.

Dense reconstruction and escape arithmetic derive from SplitZip codec_gpu.py
at 42c743118a29db3270ae6a921a4a4eecb20cad04 (SHA256
 e28fe3263e8906b26d59789cbc5bccf14edc8391b0dae986aafaf34adb2d392c).
Copyright (c) 2026 Yipin Guo and Siddharth Joshi, MIT License;
see mapped_splitzip_LICENSE.txt. These are local integration kernels, not the
unchanged author baseline. Only layer selection/addressing and scheduling vary.

The critical partition contains every fallback layer consumed as the immediate
predecessor of a coded layer. Its complement can run concurrently with rANS:
the two partitions write disjoint physical layers. Both preserve every BF16 bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .mapped_splitzip import CHUNK, LayerMap, _geometry, _lut


@dataclass(frozen=True)
class DecodePartition:
    """Owned, version-guarded logical ordinals prepared outside timed calls."""
    mapping: LayerMap
    mask: tuple
    critical_layers: tuple
    bulk_layers: tuple
    critical_ordinals: torch.Tensor
    bulk_ordinals: torch.Tensor
    critical_flags: torch.Tensor
    versions: tuple

    @property
    def tensors(self):
        return (self.mapping.tensor, self.critical_ordinals, self.bulk_ordinals,
                self.critical_flags)

    @property
    def shared_bytes(self):
        # The mapping already belongs to the mapped codec and is charged there.
        return sum(t.numel() * t.element_size() for t in self.tensors[1:])


def prepare_partition(mapping, mask):
    """Bind increasing fallback ordinals to the fixed layer execution mask."""
    if not isinstance(mapping, LayerMap):
        raise ValueError('a prepared owned LayerMap is required')
    if (not isinstance(mask, (list, tuple)) or len(mask) != mapping.physical_layers
        or not mask or any(type(x) is not bool for x in mask)):
        raise ValueError('one explicit boolean per physical layer is required')
    if (mapping.tensor._version != mapping.version or mapping.tensor.dtype != torch.int64
        or mapping.tensor.ndim != 1 or not mapping.tensor.is_contiguous()
        or mapping.tensor.numel() != len(mapping.layers)):
        raise ValueError('owned fallback map changed')
    fallback = tuple(i for i, ours in enumerate(mask) if not ours)
    if not fallback or fallback != mapping.layers:
        raise ValueError('fallback mapping must equal the fixed mask complement')
    critical = tuple(i for i in fallback if i + 1 < len(mask) and mask[i + 1])
    bulk = tuple(i for i in fallback if i not in critical)
    critical_ord = tuple(i for i, layer in enumerate(fallback) if layer in critical)
    bulk_ord = tuple(i for i, layer in enumerate(fallback) if layer in bulk)
    device = mapping.tensor.device
    ct = torch.tensor(critical_ord, dtype=torch.int64, device=device)
    bt = torch.tensor(bulk_ord, dtype=torch.int64, device=device)
    flags = torch.tensor([int(layer in critical) for layer in fallback], dtype=torch.uint8, device=device)
    return DecodePartition(mapping, tuple(mask), critical, bulk, ct, bt, flags,
                           (ct._version, bt._version, flags._version))


@triton.jit
def _partition_decode_quads(pk16, sm32, dlut, mapping, ordinals, out64,
                            n_quads, LAYER_QUADS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    ordinal = tl.load(ordinals + offs // LAYER_QUADS, mask=mask, other=0).to(tl.int64)
    logical = ordinal * LAYER_QUADS + offs % LAYER_QUADS
    packed = tl.load(pk16 + logical, mask=mask, other=0).to(tl.int32)
    p0 = packed & 0xFF
    p1 = (packed >> 8) & 0xFF
    sm_pack = tl.load(sm32 + logical, mask=mask, other=0).to(tl.int32)
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
    quad = (raw0.to(tl.uint64) | (raw1.to(tl.uint64) << 16)
            | (raw2.to(tl.uint64) << 32) | (raw3.to(tl.uint64) << 48))
    layer = tl.load(mapping + ordinal, mask=mask, other=0).to(tl.int64)
    physical = layer * LAYER_QUADS + offs % LAYER_QUADS
    tl.store(out64 + physical, quad, mask=mask)


@triton.jit
def _partition_fix_escapes(chunk_id, local_pos, esc_val, sm, mapping, critical_flags, out,
                           n_esc, LAYER_WORDS: tl.constexpr, CHUNK: tl.constexpr,
                           SELECT_CRITICAL: tl.constexpr, BLOCK_ESC: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_ESC + tl.arange(0, BLOCK_ESC)
    valid = offs < n_esc
    chunk = tl.load(chunk_id + offs, mask=valid, other=0).to(tl.int32)
    local = tl.load(local_pos + offs, mask=valid, other=0).to(tl.int32)
    pos = chunk * CHUNK + local
    ordinal = pos // LAYER_WORDS
    critical = tl.load(critical_flags + ordinal, mask=valid, other=0) != 0
    if SELECT_CRITICAL:
        selected = valid & critical
    else:
        selected = valid & ~critical
    exp = tl.load(esc_val + offs, mask=selected, other=0).to(tl.int32)
    s = tl.load(sm + pos, mask=selected, other=0).to(tl.int32)
    fixed = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)
    layer = tl.load(mapping + ordinal, mask=selected, other=0).to(tl.int64)
    physical = layer * LAYER_WORDS + pos % LAYER_WORDS
    tl.store(out + physical, fixed.to(tl.int16), mask=selected)


def _validate_plan(plan, out):
    if not isinstance(plan, DecodePartition):
        raise ValueError('a prepared DecodePartition is required')
    mapping, layer_words, n = _geometry(out, plan.mapping)
    expected = ((plan.critical_ordinals, torch.int64, len(plan.critical_layers)),
                (plan.bulk_ordinals, torch.int64, len(plan.bulk_layers)),
                (plan.critical_flags, torch.uint8, len(mapping.layers)))
    for (tensor, dtype, count), version in zip(expected, plan.versions):
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype
            or tensor.device != out.device or tensor.ndim != 1
            or not tensor.is_contiguous() or tensor.numel() != count
            or tensor._version != version):
            raise ValueError('owned partition tensors changed or have incompatible structure')
    if len(plan.versions) != 3:
        raise ValueError('invalid partition version proof')
    return mapping, layer_words, n


def decode_subset_into(codec, encoded, plan, out, partition, *, validated=False):
    """Decode exactly one partition into final physical layers.

    ``validated=True`` requires the existing checked packet parser to have
    validated escape positions. Structural checks and owned-plan guards always
    run, without copying tensors to the host. ``validated=False`` also checks
    escape bounds; that optional safety check may synchronize CUDA.
    """
    if partition not in ('critical', 'bulk') or type(validated) is not bool:
        raise ValueError('explicit critical/bulk partition and validation flag required')
    mapping, layer_words, n = _validate_plan(plan, out)
    lut = _lut(codec, 'dec_lut', out, 16)
    ne = encoded.n_esc
    if (encoded.n != n or type(ne) is not int or not 0 <= ne <= n
        or encoded.chunk_size != CHUNK or codec.chunk_size != CHUNK):
        raise ValueError('invalid mapped bundle geometry')
    expected = dict(pk=(torch.uint8, n // 2), sm=(torch.uint8, n), chunk_id=(torch.int32, ne),
                    local_pos=(torch.uint16, ne), esc_val=(torch.uint8, ne))
    for name, (dtype, count) in expected.items():
        value = getattr(encoded, name, None)
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device != out.device
            or value.ndim != 1 or value.numel() != count or not value.is_contiguous()):
            raise ValueError('invalid mapped bundle field: ' + name)
    if encoded.pk.data_ptr() % 2 or encoded.sm.data_ptr() % 4:
        raise ValueError('mapped packed fields must retain native view alignment')
    if ne and not validated:
        chunk, pos = encoded.chunk_id.long(), encoded.local_pos.long()
        if bool(((chunk < 0) | (pos >= CHUNK) | (chunk * CHUNK + pos >= n)).any()):
            raise ValueError('mapped escape position out of bounds')
    ordinals = plan.critical_ordinals if partition == 'critical' else plan.bulk_ordinals
    if not ordinals.numel():
        return out
    n_quads = ordinals.numel() * (layer_words // 4)
    _partition_decode_quads[(triton.cdiv(n_quads, 512),)](
        encoded.pk.view(torch.int16), encoded.sm.view(torch.int32), lut, mapping.tensor,
        ordinals, out.view(torch.int64), n_quads, LAYER_QUADS=layer_words // 4,
        BLOCK=512, num_warps=4)
    if ne:
        _partition_fix_escapes[(triton.cdiv(ne, 128),)](
            encoded.chunk_id, encoded.local_pos, encoded.esc_val, encoded.sm, mapping.tensor,
            plan.critical_flags, out.flatten(), ne, LAYER_WORDS=layer_words, CHUNK=CHUNK,
            SELECT_CRITICAL=partition == 'critical', BLOCK_ESC=128, num_warps=4)
    return out

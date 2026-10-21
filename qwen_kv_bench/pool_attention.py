"""One-query GQA attention over P1 sealed pages and a raw tail.

This is an independent attention component, not a Hugging Face cache backend.
The P1 loader reconstructs words inside the attention program; no full-cache
BF16 output tensor exists. Dense and packed controls use the same arithmetic.
"""
import math
import numpy as np
from . import pool_p1 as p


def reference(query, words, layout, tail=None, scale=None, layer_start=0):
    """Float64 independent dense attention oracle for finite model inputs."""
    q = np.asarray(query, np.float64)
    if q.ndim != 3 or q.shape[0]<1 or layer_start<0 or layer_start+q.shape[0]>layout.layers or q.shape[2] != layout.head_dim:
        raise ValueError('query must be [layer,query head,head dimension]')
    if q.shape[1]<1 or q.shape[1] % layout.heads:
        raise ValueError('query heads must be divisible by KV heads')
    w = np.asarray(words)
    if w.dtype != np.uint16 or w.shape != layout.shape:
        raise ValueError('wrong original word geometry')
    if tail is not None:
        t = np.asarray(tail)
        if t.dtype != np.uint16 or t.ndim != 3 or t.shape[::2] != w.shape[::2]:
            raise ValueError('wrong raw tail geometry')
        w = np.concatenate((w, t), axis=1)
    values = (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if not np.isfinite(values).all() or not np.isfinite(q).all():
        raise ValueError('attention oracle requires finite values')
    factor = 1 / math.sqrt(layout.head_dim) if scale is None else float(scale)
    k, v = np.split(values, 2, axis=2)
    k = k.reshape(layout.layers, -1, layout.heads, layout.head_dim)
    v = v.reshape(k.shape)
    out = np.empty_like(q)
    for l in range(q.shape[0]):
        for h in range(q.shape[1]):
            kh = h // (q.shape[1] // layout.heads)
            score = k[l+layer_start, :, kh] @ q[l, h] * factor
            prob = np.exp(score-score.max()); prob /= prob.sum()
            out[l, h] = prob @ v[l+layer_start, :, kh]
    return out


def _attention(query, layout, *, words=None, arena=None, state=None, tail=None, scale=None, layer_start=0):
    torch, device = p._cuda_torch(query.device)
    if query.dtype != torch.float32 or not query.is_contiguous() or query.ndim != 3:
        raise ValueError('contiguous FLOAT32 query required')
    if isinstance(layer_start,bool) or not isinstance(layer_start,int) or layer_start<0 or query.shape[0]<1 or layer_start+query.shape[0]>layout.layers or query.shape[2] != layout.head_dim or query.shape[1]<1 or query.shape[1] % layout.heads:
        raise ValueError('wrong query geometry or GQA mapping')
    if tail is None:
        tail = torch.empty((layout.layers, 0, layout.channels), dtype=torch.uint16, device=device)
    if tail.device != device or tail.dtype != torch.uint16 or not tail.is_contiguous() or tail.ndim != 3 or tuple(tail.shape[::2]) != (layout.layers, layout.channels):
        raise ValueError('contiguous UINT16 raw tail on the query device required')
    if tail.shape[1] >= layout.tile_tokens:
        raise ValueError('raw tail must be shorter than one token tile')
    packed = arena is not None
    if packed:
        p._check_device_metadata(torch, state, arena)
        if arena.payload.device != device or arena.layout != layout or arena.calibration_sha256 != state.sha256:
            raise ValueError('wrong pool identity/device')
        a = arena.payload; desc = arena.descriptors; refs = state.references; bits = state.widths
        words = tail  # Eliminated by the packed compile-time branch.
        capacity = a.numel()
    else:
        if words is None or words.device != device or words.dtype != torch.uint16 or not words.is_contiguous() or tuple(words.shape) != layout.shape:
            raise ValueError('contiguous original UINT16 words on the query device required')
        a = words.view(torch.uint8).reshape(-1); desc = refs = bits = a; capacity = 0
    factor = 1 / math.sqrt(layout.head_dim) if scale is None else float(scale)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('positive finite scale required')
    out = torch.empty_like(query)
    status = torch.empty(query.shape[:2], dtype=torch.int32, device=device)
    from .pool_attention_kernels import attention
    with torch.cuda.device(device):
        attention[(query.shape[0], query.shape[1])](query, words, tail, a, a.view(torch.uint16), a.view(torch.int32),
            desc, refs, bits, out, status, PACKED=packed, CAPACITY=capacity,
            T=layout.tokens, C=layout.channels, HEADS=layout.heads, QHEADS=query.shape[1],
            D=layout.head_dim, TILE=layout.tile_tokens, PAGES=layout.pages,
            N=layout.sector_words, SECTORS=layout.sectors, TAIL=tail.shape[1], SCALE=factor,LSTART=layer_start,
            num_warps=4, enable_fp_fusion=False)
    return out, status


def packed_attention(query, arena, state, tail=None, scale=None, layer_start=0):
    return _attention(query, arena.layout, arena=arena, state=state, tail=tail, scale=scale,layer_start=layer_start)


def dense_attention(query, words, layout, tail=None, scale=None, layer_start=0):
    return _attention(query, layout, words=words, tail=tail, scale=scale,layer_start=layer_start)

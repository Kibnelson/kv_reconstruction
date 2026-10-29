"""Checked exact products for the standalone lossless decoder.

This dispatch uses the existing frozen layerwise._igemm_kernel. It introduces
no kernel, predictor, probability or packet change. CPU execution requires the
explicit Triton interpreter and is a software semantic check, never a native
GPU fallback or performance estimate.
"""
from __future__ import annotations

import os

import torch

from .team_onestep import layerwise as lw


ALLOWED_TILES = ((128, 128, 64, 4), (64, 128, 64, 4))
INT32_MAX = (1 << 31) - 1


def validate_tiles(tiles):
    """Return one explicitly supported (BM, BN, BK, warps) configuration."""
    if (not isinstance(tiles, (list, tuple)) or len(tiles) != 4
            or any(type(value) is not int for value in tiles)
            or tuple(tiles) not in ALLOWED_TILES):
        raise ValueError('tiles must be (128,128,64,4) or (64,128,64,4)')
    return tuple(tiles)


def _span(value):
    start = value.data_ptr()
    return start, start + value.numel() * value.element_size()


def exact_product(x, A, out, tiles):
    """Write x[M,K] @ A[N,K].T exactly into caller-owned INT32 out[M,N].

    All operands must be contiguous matrices on one CUDA device. Complete
    tile geometry is required because the original kernel has no boundary
    masks; unsupported shapes are rejected rather than routed to cuBLAS.
    Nonzero storage offsets are valid for contiguous matrix views.
    """
    BM, BN, BK, warps = validate_tiles(tiles)
    for name, value, dtype in (('x', x, torch.int8), ('A', A, torch.int8), ('out', out, torch.int32)):
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.ndim != 2 or not value.is_contiguous()):
            raise ValueError(name + ' must be a contiguous ' + str(dtype) + ' matrix')
    if x.device != A.device or x.device != out.device:
        raise ValueError('product operands must share one device')
    M, K = x.shape
    N, AK = A.shape
    if min(M, N, K) < 1 or AK != K or tuple(out.shape) != (M, N):
        raise ValueError('product geometry must be x[M,K], A[N,K], out[M,N]')
    if M % BM or N % BN or K % BK:
        raise ValueError('product geometry must divide the selected tiles exactly')
    # Every input byte can be -128. Bounding every partial sum as well as the
    # final sum makes the result independent of integer accumulation order.
    if K * 128 * 128 > INT32_MAX:
        raise ValueError('product reduction exceeds the conservative INT32 accumulation bound')
    if max(x.numel(), A.numel(), out.numel()) > INT32_MAX:
        raise ValueError('product extents exceed signed INT32 kernel indexing')
    if x.device.type != 'cuda' and not (x.device.type == 'cpu' and os.environ.get('TRITON_INTERPRET') == '1'):
        raise RuntimeError('actual CUDA required; CPU is allowed only with explicit TRITON_INTERPRET=1')
    begin, end = _span(out)
    if any(begin < other_end and other_begin < end for other_begin, other_end in (_span(x), _span(A))):
        raise ValueError('product output must not overlap either input')
    grid = ((M // BM) * (N // BN),)
    lw._igemm_kernel[grid](x, A, out, M, N, K, BM=BM, BN=BN, BK=BK,
                          GROUP=8, num_warps=warps, num_stages=3)
    return out

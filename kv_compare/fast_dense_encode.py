"""Dense forward statistics through cuBLAS; original T coder and decoder.

Only model() changes. Means, feature quantization, integer surprise arithmetic,
global prior, stream layout, entropy coding and decoder remain original T.
"""
from __future__ import annotations

import os
import time

import torch

from . import layerwise as lw
from . import fused as fu


def _require_global_dense(consts, kwargs):
    if (int(consts.get("groups", 1)) != 1 or kwargs.get("rank") is not None
            or kwargs.get("fuse", False) or kwargs.get("split_streams", False)
            or kwargs.get("chunks", 1) != 1 or kwargs.get("product", "cublas") != "cublas"):
        raise ValueError("dense v1 requires the fixed dense, unfused, single-stream predictor")


class FastDenseEncode(lw.LayerwiseOneStep):
    def __init__(self, consts, C, hd, **kwargs):
        _require_global_dense(consts, kwargs)
        super().__init__(consts, C, hd, **kwargs)

    def model(self, words):
        L, T, W = words.shape
        self.layout(T)
        if (L, W) != (self.L, self.W) or words.device != self.device or words.dtype != torch.int16:
            raise ValueError("words differ from the frozen codec geometry/device")
        k, dev = self.k, self.device
        xall = torch.zeros(L, T, W, dtype=torch.int8, device=dev)
        prior = torch.ones(L, T, dtype=torch.float32, device=dev)
        for layer in range(L - 1):
            self._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),), words[layer], xall[layer],
                k["isx"][layer], k["cos"], k["sin"], 0, T * W, W, self.C,
                HD=self.hd, BLOCK=1024, warps=4)
        acc = torch.empty(T, W, dtype=torch.int32, device=dev)
        sums = torch.empty(self.G, T, dtype=torch.int32, device=dev)
        for layer in range(L - 1):
            if layer == 0:
                acc.zero_()
            else:
                self._product(xall[layer - 1], layer, acc)
            self._launch(lw._lsurp_kernel, ((T // self.TB) * self.G,), words[layer], acc,
                k["sa"][layer], k["b"][layer], k["inv"][layer], k["cos"], k["sin"], sums,
                T, W, self.G, self.HG, TB=self.TB, H2=self.hd // 2)
            self._launch(lw._lprior_kernel, (lw.triton.cdiv(T, 256),), sums, prior[layer + 1],
                T, self.G, self.prior_scale, BLOCK=256, warps=4)
        return xall, prior


class CachedFusedDense(lw.LayerwiseOneStep):
    """Original fused forward computation, initialized once outside capture.

    LayerwiseOneStep.model constructs FusedOneStep every call. That constructor
    builds static probability tables on the CPU and copies them to the GPU,
    even though its forward model does not use those tables. Keeping the same
    object alive removes repeated setup and its non-capturable CPU transfers.
    No original codec file or kernel is modified.
    """
    def __init__(self, consts, C, hd, **kwargs):
        _require_global_dense(consts, kwargs)
        super().__init__(consts, C, hd, **kwargs)
        self.forward = fu.FusedOneStep(self.k, self.C, self.hd,
            tokens_per_program=32, rows=4, warps=4)

    def model(self, words):
        return self.forward.model(words)


class CudaReplay:
    """One fixed-shape operation, warmed/captured on a side stream.

    Outputs and the callable retain all static tensor references. No pool sharing
    between independent graphs. Setup time and resident memory are recorded.
    A caller must test replay on changed input, not just the capture's data.
    """
    _capture_streams = {}

    def __init__(self, operation, device, warmup=3):
        device = torch.device(device)
        if (device.type != "cuda" or os.environ.get("TRITON_INTERPRET", "0") != "0"
                or not torch.cuda.is_available()):
            raise RuntimeError("CUDA graph replay requires actual CUDA; no interpreter/CPU fallback")
        if warmup < 1:
            raise ValueError("capture requires side-stream warmup")
        self.operation, self.output = operation, None
        self.graph = torch.cuda.CUDAGraph()
        key = device.index if device.index is not None else torch.cuda.current_device()
        if key not in self._capture_streams:
            self._capture_streams[key] = torch.cuda.Stream(device)
        self.stream = self._capture_streams[key]
        original_stream = torch.cuda.current_stream(device)
        start = time.perf_counter()
        try:
            torch.cuda.synchronize(device)
            self.stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(self.stream):
                for _ in range(warmup):
                    self.output = operation()
            torch.cuda.current_stream(device).wait_stream(self.stream)
            torch.cuda.synchronize(device)
            self.output = None
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.output = operation()
            self.graph.replay()
            torch.cuda.synchronize(device)
            self.setup = {"wall_ms": 1000 * (time.perf_counter() - start),
                "side_stream_warmup": warmup,
                "peak_incremental_allocated_bytes": max(0, torch.cuda.max_memory_allocated(device) - before),
                "resident_allocated_bytes_after_capture": torch.cuda.memory_allocated(device),
                "resident_reserved_bytes_after_capture": torch.cuda.memory_reserved(device),
                "private_pool_shared": False}
        except Exception:
            torch.cuda.set_stream(original_stream)
            self.close()
            raise

    def __call__(self):
        self.graph.replay()
        return self.output

    def close(self):
        self.output = None
        self.operation = None
        if self.graph is not None:
            try:
                self.graph.reset()
            finally:
                self.graph = None

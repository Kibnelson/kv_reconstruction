"""A byte-compatible causal engine with host-keyed request RoPE tables.

The measured reference engine and fitted-state identity stay unchanged. Only
table lookup changes: _run already knows the contiguous position range on the
host, so looking up its tables does not need CUDA scalar readbacks.
"""
from contextlib import contextmanager

import torch

from .causal_packet import Engine
from .core import Rope


class RequestRope(Rope):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_key = None

    @contextmanager
    def request(self, start, tokens, device):
        if self._request_key is not None:
            raise RuntimeError("a causal engine cannot run overlapping requests")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._request_key = (str(device), int(start), int(start) + tokens - 1, tokens)
        try:
            yield
        finally:
            self._request_key = None

    def tables(self, positions):
        key = self._request_key
        if key is None:
            # Preserve the ordinary helper outside the bound _run call.
            return super().tables(positions)
        if (str(positions.device) != key[0] or positions.ndim != 1 or
                positions.numel() != key[3] or positions.dtype != torch.int64):
            raise ValueError("position metadata differs from the bound request")
        if key not in self._tables:
            # Identical arithmetic and single-entry cache policy to Rope.tables.
            angle = positions.to(torch.float64)[:, None] * self.inv_freq.to(positions.device)[None, :]
            self._tables = {key: (torch.cos(angle) * self.scale, torch.sin(angle) * self.scale)}
        return self._tables[key]


class FastEngine(Engine):
    """Inherited predictor, entropy coder, validation and packet representation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope = RequestRope(self.meta["inv_freq"], self.meta["kv_heads"], self.meta["head_dim"],
            self.meta.get("rope_layout", "half"), self.meta.get("attention_scaling", 1.))

    def _run(self, shape, start, kappa, source=None, packet=None, calibrate=False):
        # Engine._run constructs exactly arange(start, start + shape[1]). The
        # key belongs to this call, is cleared on errors, and changes on every
        # start/length/device change. No arbitrary-position cache inference.
        with self.rope.request(start, shape[1], self.device):
            return super()._run(shape, start, kappa, source, packet, calibrate)

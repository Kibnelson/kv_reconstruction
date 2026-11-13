"""Isolated experiment: preserve head difficulty when adapting T's spread.

The existing predictor, nominal per-channel spread and rANS format are fixed.
Only the decoder-known prior that multiplies spread changes. No token-specific
hint is sent. Calibration fits a small map on TRAIN; DEV selects the blending
strength using actual complete packets. TEST never selects a configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from . import fused as fu
from . import layerwise as lw
from .head_spread_kernels import _head_prior_kernel, _head_ldec_kernel, _head_lenc_kernel


FORMAT = "t_head_spread_policy_v1"


@dataclass
class Policy:
    kind: int = 0                 # 0: global; 1: same-head; 2: TRAIN-fitted head map
    alpha: float = 0.0
    head_shift: int = 0           # fixed negative control; no token shuffling
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None

    def validate(self, layers, heads):
        if self.kind not in (0, 1, 2) or not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("invalid head-spread policy")
        if not isinstance(self.head_shift, int) or not 0 <= self.head_shift < heads:
            raise ValueError("head shift is outside the fitted geometry")
        if self.kind == 2:
            if self.weights is None or self.bias is None:
                raise ValueError("learned policy requires frozen weights and bias")
            if self.weights.shape != (layers, heads, heads) or self.bias.shape != (layers, heads):
                raise ValueError("learned policy geometry mismatch")
            if self.weights.dtype != np.float32 or self.bias.dtype != np.float32:
                raise ValueError("learned policy must be frozen as float32")
            if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
                raise ValueError("nonfinite learned policy")
        elif self.weights is not None or self.bias is not None:
            raise ValueError("unlearned policy must not carry unused weights")
        return self

    @property
    def array_bytes(self):
        return sum(x.nbytes for x in (self.weights, self.bias) if x is not None)

    def description(self):
        return {"format": FORMAT, "kind": self.kind, "alpha": self.alpha,
                "head_shift": self.head_shift, "paid_token_metadata_bytes": 0}

    def digest(self):
        h = hashlib.sha256(json.dumps(self.description(), sort_keys=True, separators=(",", ":")).encode())
        for x in (self.weights, self.bias):
            if x is not None:
                h.update(np.asarray(x, dtype="<f4").tobytes())
        return h.hexdigest()


def fit_head_map(records, head_dim, ridge=1e-2):
    """Ridge-map previous-head surprises to current-head surprises on TRAIN only.

    Records are integer surprise sums produced by the fixed codec predictor,
    [layers,heads,tokens]. The nominal spread and predictor do not depend on the
    experimental prior, so targets are common to all variants. The map is merely
    a proposal: actual DEV packet bytes select its blend with the global prior.
    """
    if not records or not math.isfinite(ridge) or ridge <= 0 or head_dim < 1:
        raise ValueError("nonempty TRAIN records and positive ridge/head_dim required")
    layers, heads, _ = records[0].shape
    if any(x.ndim != 3 or x.shape[:2] != (layers, heads) or x.shape[2] < 2 or x.dtype != np.int32 for x in records):
        raise ValueError("TRAIN records must share integer surprise geometry")
    weights = np.zeros((layers, heads, heads), np.float32)
    bias = np.ones((layers, heads), np.float32)
    normalizer = 1.0 / (head_dim * fu.QS)
    for layer in range(1, layers):
        x = np.concatenate([r[layer - 1].T for r in records]).astype(np.float64) * normalizer
        y = np.concatenate([r[layer].T for r in records]).astype(np.float64) * normalizer
        xm, ym = x.mean(0), y.mean(0)
        xc, yc = x - xm, y - ym
        gram = xc.T @ xc / len(x)
        penalty = ridge * max(float(np.trace(gram)) / heads, 1e-8)
        gram.flat[::heads + 1] += penalty
        w = np.linalg.solve(gram, xc.T @ yc / len(x)).T
        # Bound exceptionally ill-conditioned fits before freezing. The same
        # bounded matrix sets the intercept; both are charged shared state.
        w = np.clip(w, -8.0, 8.0)
        weights[layer] = w.astype(np.float32)
        bias[layer] = (ym - w @ xm).astype(np.float32)
    return Policy(2, 1.0, 0, weights, bias).validate(layers, heads)


def prior_reference(sums, policy, head_dim):
    """NumPy float32 oracle, same sequential order as the shared Triton helper."""
    heads, tokens = sums.shape
    if sums.dtype != np.int32:
        raise ValueError("surprise sums must be int32")
    scale = np.float32(1.0 / (heads * head_dim * fu.QS))
    total = np.zeros(tokens, np.int32)
    for g in range(heads):
        total += sums[g]
    global_prior = total.astype(np.float32) * scale
    result = np.empty((heads, tokens), np.float32)
    for g in range(heads):
        if policy.kind == 0 or policy.alpha == 0:
            result[g] = global_prior
            continue
        head_scale = np.float32(scale * heads)
        if policy.kind == 1:
            local = sums[(g + policy.head_shift) % heads].astype(np.float32) * head_scale
        else:
            # Reference is a single layer: its matrix is [heads,heads].
            local = np.full(tokens, policy.bias[g], np.float32)
            for gg in range(heads):
                value = sums[(gg + policy.head_shift) % heads].astype(np.float32) * head_scale
                local = local + policy.weights[g, gg] * value
        result[g] = global_prior + np.float32(policy.alpha) * (local - global_prior)
    return result


class HeadSpreadCodec(lw.LayerwiseOneStep):
    """Original T with experiment-only coding kernels and head-aware priors.

    No baseline file is changed. The dispatch only replaces the two coding
    kernels. All product, feature, RoPE, framing-size and buffer logic stays in
    LayerwiseOneStep. This first experiment excludes reduced rank/split streams
    to keep the change in probability widths isolated.
    """
    def __init__(self, consts, C, hd, policy, **kwargs):
        if kwargs.get("rank") is not None or kwargs.get("split_streams", False):
            raise ValueError("head-spread v1 tests fixed dense/grouped predictors only")
        if kwargs.get("fuse", False):
            raise ValueError("head-spread v1 requires the baseline's FP-fusion-disabled arithmetic")
        super().__init__(consts, C, hd, **kwargs)
        if self.W * 524288 >= 2**31:
            raise ValueError("surprise sum exceeds the baseline int32 domain")
        self.policy = policy.validate(self.L, self.G)
        self.mix = torch.from_numpy(policy.weights.copy()).to(self.device) if policy.kind == 2 else None
        self.bias = torch.from_numpy(policy.bias.copy()).to(self.device) if policy.kind == 2 else None

    def _parameters(self, layer):
        # Dummy pointers are never loaded by the unlearned specialization.
        if self.mix is None:
            return self.k["b"][layer], self.k["b"][layer]
        return self.mix[layer], self.bias[layer]

    def _launch(self, kernel, grid, *args, **kwargs):
        if kernel is lw._lenc_kernel:
            kernel = _head_lenc_kernel
        elif kernel is lw._ldec_kernel:
            # The original call passes the current layer's sa row at index 7.
            # Storage offsets are host metadata (no device read/sync). Validate
            # the view rather than maintaining mutable per-call layer state.
            row = args[7]
            base = self.k["sa"]
            delta = row.storage_offset() - base.storage_offset()
            if (row.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
                    or delta % self.W or not 0 <= delta // self.W < self.L):
                raise ValueError("unexpected baseline decoder layer view")
            layer = delta // self.W
            mix, bias = self._parameters(layer)
            kwargs.update(mix_ptr=mix, bias_ptr=bias, POLICY_KIND=self.policy.kind,
                          ALPHA=self.policy.alpha, HEAD_SHIFT=self.policy.head_shift)
            kernel = _head_ldec_kernel
        return super()._launch(kernel, grid, *args, **kwargs)

    def statistics(self, words, include_last=True):
        """Exact-word forward pass; used only by the encoder and TRAIN fitting."""
        L, T, W = words.shape
        self.layout(T)
        if L != self.L or W != self.W or words.device != self.device or words.dtype != torch.int16:
            raise ValueError("words differ from the frozen codec geometry/device")
        xall = torch.zeros(L, T, W, dtype=torch.int8, device=self.device)
        for layer in range(L if self.ng > 1 else L - 1):
            self._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),), words[layer], xall[layer],
                self.k["isx"][layer], self.k["cos"], self.k["sin"], 0, T * W, W, self.C,
                HD=self.hd, BLOCK=1024, warps=4)
        sums = torch.zeros(L, self.G, T, dtype=torch.int32, device=self.device)
        acc = torch.empty(T, W, dtype=torch.int32, device=self.device)
        for layer in range(L if include_last else L - 1):
            self._acc_full(xall, layer, acc)
            self._launch(lw._lsurp_kernel, ((T // self.TB) * self.G,), words[layer], acc,
                self.k["sa"][layer], self.k["b"][layer], self.k["inv"][layer],
                self.k["cos"], self.k["sin"], sums[layer], T, W, self.G, self.HG,
                TB=self.TB, H2=self.hd // 2)
        return xall, sums

    def priors(self, sums):
        L, G, T = sums.shape
        if (L, G) != (self.L, self.G) or sums.dtype != torch.int32 or sums.device != self.device or T % self.TB:
            raise ValueError("surprise tensor differs from codec geometry/device")
        prior = torch.ones(L, G, T, dtype=torch.float32, device=self.device)
        for layer in range(1, L):
            mix, bias = self._parameters(layer)
            self._launch(_head_prior_kernel, (G, T // self.TB), sums[layer - 1], prior[layer], mix, bias,
                T, G, self.prior_scale, KIND=self.policy.kind, ALPHA=self.policy.alpha,
                SHIFT=self.policy.head_shift, BLOCK=self.TB)
        return prior

    def model(self, words):
        xall, sums = self.statistics(words, include_last=False)
        return xall, self.priors(sums)

    @property
    def policy_tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.mix, self.bias) if x is not None)

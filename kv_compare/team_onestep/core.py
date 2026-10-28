"""BF16 words, RoPE geometry and exact per-word coding costs (PyTorch, any device).

A BF16 word is held as an int32 tensor in 0..65535. ``ordered`` maps words to a monotone integer order (negative
words reversed), so neighbouring ordered indices are neighbouring values. A word's coding cost under a logistic
prediction is the logistic mass over its rounding interval, mixed with a 2^-16 uniform share so that every one of
the 65,536 words stays codable (at most about 32 bits).
"""
from __future__ import annotations

import math

import numpy as np
import torch

LN2X2 = 2 * math.log(2)
EPS = 2.0 ** -16


def as_words(array, device=None) -> torch.Tensor:
    """numpy uint16 / torch uint16 / int16 / int32 words -> int32 tensor in 0..65535."""
    t = torch.from_numpy(np.array(array, copy=True)) if isinstance(array, np.ndarray) else array
    if t.dtype == torch.uint16:
        t = t.view(torch.int16)
    if t.dtype == torch.int16:
        t = t.to(torch.int32) & 0xFFFF
    else:
        t = t.to(torch.int32)
    return t if device is None else t.to(device)


def words_to_float(words: torch.Tensor) -> torch.Tensor:
    """int32 BF16 words -> float64 values (exact)."""
    return (words.to(torch.int32) << 16).view(torch.float32).to(torch.float64)


def float_to_words(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 (nearest, ties to even) and return int32 words."""
    return x.to(torch.float32).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF


def to_bf16(x: torch.Tensor) -> torch.Tensor:
    return words_to_float(float_to_words(x))


def words_to_uint16(words: torch.Tensor) -> np.ndarray:
    return words.to(torch.int32).cpu().numpy().astype(np.uint16)


def ordered(words: torch.Tensor) -> torch.Tensor:
    w = words.to(torch.int64) & 0xFFFF
    return torch.where((w & 0x8000) != 0, (~w) & 0xFFFF, w ^ 0x8000)


def word_of(o: torch.Tensor) -> torch.Tensor:
    o = o.to(torch.int64)
    return torch.where((o & 0x8000) != 0, o ^ 0x8000, (~o) & 0xFFFF)


_EDGES = {}


def interval_edges(device) -> torch.Tensor:
    """BD[o] = midpoint between ordered words o-1 and o; +-inf for the Inf/NaN groups and the ends."""
    key = str(device)
    if key not in _EDGES:
        values = words_to_float(word_of(torch.arange(65536, device=device)))
        edges = torch.empty(65537, dtype=torch.float64, device=device)
        edges[1:65536] = (values[:-1] + values[1:]) / 2
        edges[0:129] = -math.inf
        edges[65408:] = math.inf
        _EDGES[key] = edges
    return _EDGES[key]


def logistic_cdf(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def logistic_mass(lo, hi, mu, s):
    """Mass of Logistic(mu, s) on [lo, hi), evaluated on the tail side for accuracy."""
    upper = torch.sigmoid(-(lo - mu) / s) - torch.sigmoid(-(hi - mu) / s)      # used when lo >= mu
    lower = torch.sigmoid((hi - mu) / s) - torch.sigmoid((lo - mu) / s)
    mass = torch.where(lo >= mu, upper, lower)
    return torch.nan_to_num(mass.clamp_min(0), nan=0.0)


def exact_bits(words: torch.Tensor, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """-log2 P(word) for exact (lossless) coding of BF16 words against Logistic(mu, s)."""
    edges = interval_edges(words.device)
    o = ordered(words)
    p = logistic_mass(edges[o], edges[o + 1], mu, s)
    return -torch.log2((1 - EPS) * p + EPS / 65536)


def bin_bits(k: torch.Tensor, step: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """-log2 P(bin k) for a residual quantized with step `step` against Logistic(0, s)."""
    p = torch.sigmoid((k + 0.5) * step / s) - torch.sigmoid((k - 0.5) * step / s)
    return -torch.log2((1 - EPS) * p.clamp_min(0) + EPS / 65536)


class Rope:
    """RoPE geometry of the cached keys: row layout [keys (Hk*hd) | values (Hk*hd)].

    layout "half": HuggingFace rotate_half, channel d pairs with d + hd/2 (Llama, Qwen in Transformers).
    layout "adjacent": channel 2i pairs with 2i+1 (GGUF / llama.cpp NORM mode).
    inv_freq: the model's own inverse frequencies (hd/2 values), including any scaling (Llama 3 uses
    frequency-dependent scaling), times the model's attention scaling when it multiplies cos/sin.
    """

    def __init__(self, inv_freq, heads, head_dim, layout="half", attention_scaling=1.0):
        self.inv_freq = torch.as_tensor(np.asarray(inv_freq, dtype=np.float64))
        if self.inv_freq.numel() != head_dim // 2:
            raise ValueError("inv_freq must have head_dim/2 entries")
        if layout not in ("half", "adjacent"):
            raise ValueError("layout must be 'half' or 'adjacent'")
        self.heads, self.hd, self.layout, self.scale = heads, head_dim, layout, float(attention_scaling)
        self.columns = heads * head_dim
        self._tables = {}

    def tables(self, positions: torch.Tensor):
        key = (str(positions.device), int(positions[0]), int(positions[-1]), positions.numel())
        if key not in self._tables:
            angle = positions.to(torch.float64)[:, None] * self.inv_freq.to(positions.device)[None, :]
            self._tables = {key: (torch.cos(angle) * self.scale, torch.sin(angle) * self.scale)}
        return self._tables[key]

    def pair_partner(self) -> np.ndarray:
        """Partner channel of every key channel (key block only)."""
        c = np.arange(self.columns)
        head, d = c // self.hd, c % self.hd
        if self.layout == "half":
            return head * self.hd + (d + self.hd // 2) % self.hd
        return c ^ 1

    def _split(self, keys):
        n = keys.shape[0]
        k = keys.reshape(n, -1, self.hd)
        if self.layout == "half":
            return k[..., : self.hd // 2], k[..., self.hd // 2:]
        return k[..., 0::2], k[..., 1::2]

    def _join(self, a, b, n):
        if self.layout == "half":
            return torch.cat([a, b], -1).reshape(n, -1)
        out = torch.empty(a.shape[:-1] + (self.hd,), dtype=a.dtype, device=a.device)
        out[..., 0::2], out[..., 1::2] = a, b
        return out.reshape(n, -1)

    def rotate_heads(self, x: torch.Tensor, positions: torch.Tensor, inverse=False) -> torch.Tensor:
        """Rotate (or undo the rotation of) rows of whole heads, x [n, heads*hd]."""
        cos, sin = self.tables(positions)
        cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
        if inverse:
            sin = -sin
            if self.scale != 1.0:
                cos, sin = cos / self.scale ** 2, sin / self.scale ** 2
        a, b = self._split(x)
        return self._join(a * cos - b * sin, b * cos + a * sin, x.shape[0])

    def derotate_row(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """KV rows [n, 2*columns] with RoPE-rotated keys -> keys de-rotated, values unchanged."""
        out = x.clone()
        out[:, : self.columns] = self.rotate_heads(x[:, : self.columns], positions, inverse=True)
        return out

    def rotate_row(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, : self.columns] = self.rotate_heads(x[:, : self.columns], positions)
        return out

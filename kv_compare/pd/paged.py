"""A request's KV between vLLM's paged cache and our [layers, tokens, channels] words.

vLLM 0.32 hands connectors one standardized view per attention layer, [blocks, kv heads, block size, 2 x head dim]
(RFC #42082: the logical shape is fixed, whatever the physical layout), keys in the first head_dim of the last
axis and values in the second, keys already rotated (RoPE). Our words are int16 [L, T, 2 x heads x head dim]: every
key channel (head-major) then every value channel, the order of our captures and codec constants.

Token t of a request lives in block block_ids[t // block_size] at offset t % block_size (the scheduler's block size;
a kernel block size that divides it is mapped too).
"""
from __future__ import annotations

import re

import torch


def layer_order(names):
    """Attention layer names sorted by their layer index (model.layers.<i>.self_attn.attn)."""
    def idx(n):
        m = re.search(r"layers\.(\d+)\.", n)
        if m is None:
            raise ValueError(f"no layer index in {n!r}")
        return int(m.group(1))
    return sorted(names, key=idx)


class Paged:
    """Gathers and scatters a request's tokens for one set of per-layer cache views."""

    def __init__(self, caches, block_size, heads, head_dim):
        """caches: [view per layer, in layer order]; block_size: the scheduler's block size."""
        self.caches = list(caches)
        if not self.caches:
            raise ValueError("no KV cache layers")
        v = self.caches[0]
        if v.dim() != 4 or v.shape[1] != heads or v.shape[3] != 2 * head_dim:
            raise ValueError(f"cache view {tuple(v.shape)} is not [blocks, {heads}, block, {2 * head_dim}]")
        if v.dtype != torch.bfloat16:
            raise ValueError(f"cache dtype {v.dtype}: the codecs code BF16 words")
        self.kb = int(v.shape[2])                    # kernel block size (tokens per block of the view)
        self.bs = int(block_size)
        if self.bs % self.kb:
            raise ValueError(f"scheduler block size {self.bs} is not a multiple of the view's {self.kb}")
        self.heads, self.hd = int(heads), int(head_dim)
        self.device = v.device

    def slots(self, block_ids, t0, t1):
        """(view block index, offset) of tokens t0..t1-1."""
        t = torch.arange(t0, t1, dtype=torch.int64)
        b = torch.as_tensor(block_ids, dtype=torch.int64)[t // self.bs]
        r = self.bs // self.kb
        within = t % self.bs
        blk = b * r + within // self.kb
        return blk.to(self.device), (within % self.kb).to(self.device)

    def gather(self, block_ids, t0, t1):
        """int16 words [L, t1 - t0, 2 x heads x head dim] of tokens t0..t1-1."""
        blk, off = self.slots(block_ids, t0, t1)
        T, C = t1 - t0, self.heads * self.hd
        out = torch.empty(len(self.caches), T, 2 * C, dtype=torch.int16, device=self.device)
        for l, kv in enumerate(self.caches):
            x = kv[blk, :, off]                      # [T, heads, 2 x hd]
            out[l, :, :C] = x[..., :self.hd].reshape(T, C).view(torch.int16)
            out[l, :, C:] = x[..., self.hd:].reshape(T, C).view(torch.int16)
        return out

    def scatter_layer(self, l, words_l, blk, off):
        """Words of one layer [T, 2 x heads x head dim] into the cache at (blk, off)."""
        T, C = words_l.shape[0], self.heads * self.hd
        w = words_l.view(torch.bfloat16)
        x = torch.empty(T, self.heads, 2 * self.hd, dtype=torch.bfloat16, device=self.device)
        x[..., :self.hd] = w[:, :C].reshape(T, self.heads, self.hd)
        x[..., self.hd:] = w[:, C:].reshape(T, self.heads, self.hd)
        self.caches[l][blk, :, off] = x

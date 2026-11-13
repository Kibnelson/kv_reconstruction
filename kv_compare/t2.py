"""T2: T with layer 0 recomputed by the receiver and a constant set chosen per request.

Kept apart from T: no module of T changes. This file adds a codec class that runs T's own kernels
(kv_compare.layerwise) over layers 1..L-1, and a wrapper for several constant sets.

  Layer 0 does not travel. A token's layer-0 K and V depend only on its id and position, and the receiver has the
  token ids and the model, so it computes layer 0 itself (kv_compare.t_ideas.layer0_kv: embedding, the first
  layer's norm, the K and V projections, RoPE). From those words it makes T's 8-bit features of layer 0 (layer 1's
  input) and T's surprise sums of layer 0 (layer 1's spread prior) with T's kernels. It then decodes layers 1..L-1
  exactly as T does. The stream is T's stream without layer 0: the encoder runs T's coding kernel over layers
  L-1..1 and writes the final states after layer 1, and the decoder starts there (the kernels' FIRST flag).

  The receiver's recompute must give the sender's layer 0 bit for bit. The sender checks it (the same recompute,
  with the prefill's own pass shapes) and otherwise sends the request as T, layer 0 coded; a header flag says which.

  Constant sets: T's constants fitted on several calibration texts (kv_compare.t_ideas.cal_sets). The sender
  ranks the sets by T's cost on a sample of the request (pick_fast: layers 1-2 of 1,024 tokens, T's int8 product in
  plain torch; or pick: coding layers 1-4 with each set) and sends the request with the cheapest; the header
  carries its name.
"""
from __future__ import annotations

import math

import torch

from .layerwise import (LayerwiseOneStep, _lenc_kernel, _ldec_kernel, _lfeat_kernel, _lsurp_kernel, _null, gc,
                        int_mm, triton)


class Layer0Recomputed(LayerwiseOneStep):
    """T (LayerwiseOneStep, any mode) whose streams carry layers 1..L-1; the receiver supplies layer 0."""

    def encode(self, words, xall=None, prior=None, out=None, layer_words=None):
        """As LayerwiseOneStep.encode, for layers L-1..1 (words still has every layer: layer 1 is predicted from
        layer 0's features, and its prior comes from layer 0's surprise). layer_words[0] is 0; layer 1 gets the
        final states and alignment."""
        L, T, W = words.shape
        if L < 2:
            raise ValueError("Layer0Recomputed needs at least two layers")
        S, cap = self.layout(T)
        if xall is None:
            xall, prior = self.model(words)
        dev, k, t = self.device, self.k, self.tables
        if out is None:
            out = torch.empty(S * cap, dtype=torch.int16, device=dev)
        starts = torch.empty(S, dtype=torch.int32, device=dev)
        state = torch.full((S * self.hd,), 65536, dtype=torch.int32, device=dev)
        pos = (torch.arange(S, dtype=torch.int32, device=dev) + 1) * cap
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        smult = torch.empty(S * self.TB, dtype=torch.float32, device=dev)
        for layer in reversed(range(1, L)):
            self._acc_full(xall, layer, acc)
            before = pos.sum(dtype=torch.int64) if layer_words is not None and layer > 1 else None
            self._launch(_lenc_kernel, (S,), words[layer], prior[layer], state, pos, out, starts, acc,
                         k["sa"][layer], k["b"][layer], k["sp"][layer], k["cos"], k["sin"], smult, t["cumf"], t["ko"],
                         T, W, self.G, self.HG, FLUSH=layer == 1, TB=self.TB, H2=self.hd // 2,
                         UNIFORM=self.mode == "uniform")
            if before is not None:
                layer_words[layer] = before - pos.sum(dtype=torch.int64)
        if layer_words is not None:
            layer_words[0] = 0
            layer_words[1] = (self.bits(starts, T) - 32 * S) // 16 - layer_words[2:].sum()
        return out, starts

    def layer0_state(self, l0, bufs):
        """Layer 0's features into bufs["xbuf"][0] and its surprise sums into bufs["ssum"][0], from the receiver's
        layer-0 words l0 [T, W] (int16): T's kernels, so they equal what T's decoder would have made."""
        T, W = l0.shape
        k = self.k
        self._launch(_lfeat_kernel, (triton.cdiv(T * W, 1024),), l0, bufs["xbuf"][0], k["isx"][0], k["cos"], k["sin"],
                     0, T * W, W, self.C, HD=self.hd, BLOCK=1024, warps=4)
        acc0 = torch.zeros(T, W, dtype=torch.int32, device=self.device)
        if self.ng > 1:                              # head groups: layer 0's own block-triangular sums
            acc0 += int_mm(bufs["xbuf"][0], k["Acq"][0])
        self._launch(_lsurp_kernel, ((T // self.TB) * self.G,), l0, acc0, k["sa"][0], k["b"][0], k["inv"][0],
                     k["cos"], k["sin"], bufs["ssum"][0], T, W, self.G, self.HG, TB=self.TB, H2=self.hd // 2)

    def decode(self, buf, starts, T, l0, out=None, bufs=None, sink=None):
        """As LayerwiseOneStep.decode with the receiver's layer 0 l0 [T, W] (int16) given: layers 1..L-1 from the
        stream. Without a sink the result has every layer (layer 0 = l0); with one, sink(0, l0) runs first."""
        L, W = self.L, self.W
        S, _ = self.layout(T)
        nl = 2 if sink is not None else L
        words = torch.empty(nl, T, W, dtype=torch.int16, device=self.device) if out is None else out
        bufs = self.buffers(T) if bufs is None else bufs
        k, t = self.k, self.tables
        l0 = l0.contiguous()
        self.layer0_state(l0, bufs)
        if sink is not None:
            sink(0, l0)
        else:
            words[0].copy_(l0)
        Tc = T // self.chunks
        P = (Tc // self.TB) * self.G
        cur = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        streams, gstreams = bufs["streams"], bufs.get("gemm_streams")
        for s in (streams or []) + (gstreams or []):
            s.wait_stream(cur)
        for layer in range(1, L):
            wl = words[layer % 2] if sink is not None else words[layer]
            if sink is not None and layer >= 2:
                for s in (streams or []) + (gstreams or []):
                    s.wait_stream(cur)
            for c in range(self.chunks):
                rows = slice(c * Tc, (c + 1) * Tc)
                acc = bufs["acc"][c]
                gctx = torch.cuda.stream(gstreams[c]) if gstreams else (
                    torch.cuda.stream(streams[c]) if streams else _null())
                with gctx:
                    if gstreams:
                        gstreams[c].wait_stream(streams[c])
                    self._product(bufs["xbuf"][(layer + 1) % 2][rows], layer, acc, bufs["scratch"][c])
                if gstreams:
                    streams[c].wait_stream(gstreams[c])
                ctx = torch.cuda.stream(streams[c]) if streams else _null()
                with ctx:
                    gw = self.GN * self.hd if self.ng > 1 else W
                    for j in range(self.ng):
                        if j > 0:
                            xin = bufs["xbuf"][layer % 2][rows, :j * gw].contiguous()
                            acc[:, j * gw:(j + 1) * gw] += int_mm(xin, self.Ag[j][layer])
                        GN = self.GN if self.ng > 1 else self.G
                        self._launch(_ldec_kernel, ((Tc // self.TB) * GN,), buf, starts, bufs["state"], bufs["pos"],
                                     wl, acc, bufs["xbuf"][layer % 2], k["sa"][layer], k["b"][layer],
                                     k["sp"][layer], k["inv"][layer], k["isx"][layer], k["cos"], k["sin"],
                                     bufs["ssum"][(layer + 1) % 2], bufs["ssum"][layer % 2],
                                     bufs["smult"][c * P * self.TB:], t["cumf"], t["lut"], t["tail"], c * Tc, T, W,
                                     self.G, self.HG, j * GN if self.ng > 1 else 0, GN, self.prior_scale,
                                     FIRST=layer == 1, LAST=layer == L - 1, TB=self.TB, H2=self.hd // 2,
                                     LUT_SHIFT=16 - gc.LUT_BITS, LUT_W=gc.LUT_W, REFINE=gc.REFINE,
                                     UNIFORM=self.mode == "uniform", SEARCH=self.search,
                                     FEAT=self.features == "inline", LEAN=self.lean, ONES=False)
                    if self.features == "pass" and layer < L - 1:
                        nv = Tc * W
                        self._launch(_lfeat_kernel, (triton.cdiv(nv, 1024),), wl, bufs["xbuf"][layer % 2],
                                     k["isx"][layer], k["cos"], k["sin"], c * Tc, nv, W, self.C, HD=self.hd,
                                     BLOCK=1024, warps=4)
            if sink is not None:
                for s in (streams or []) + (gstreams or []):
                    cur.wait_stream(s)
                sink(layer, wl)
        for s in (streams or []) + (gstreams or []):
            cur.wait_stream(s)
        return words


def slice_layers(d, n, L):
    """A constants dict cut to its first n layers (every tensor whose first dimension is the layer count, except
    the RoPE tables)."""
    out = {}
    for key, v in d.items():
        if torch.is_tensor(v) and key not in ("cos", "sin") and v.dim() >= 1 and v.shape[0] == L:
            out[key] = v[:n]
        else:
            out[key] = v
    return out


def sample_rows(T, tokens=1024, blocks=8):
    """Row indices of `blocks` evenly spaced blocks of tokens // blocks tokens each (the whole request, sampled)."""
    if tokens >= T:
        return torch.arange(T)
    size = max(1, tokens // blocks)
    starts = torch.linspace(0, T - size, blocks).round().long()
    return torch.cat([torch.arange(int(a), int(a) + size) for a in starts])


def _turn(x, cs, sn, C, hd, inverse):
    """Keys of rows x [n, W] rotated (T's prediction) or de-rotated (T's features) with the RoPE tables of the rows."""
    n = x.shape[0]
    h2 = hd // 2
    k = x[:, :C].reshape(n, -1, hd)
    a, b = k[..., :h2], k[..., h2:]
    c, s = cs[:, None, :], sn[:, None, :]
    if inverse:
        k = torch.cat([a * c + b * s, b * c - a * s], -1)
    else:
        k = torch.cat([a * c - b * s, b * c + a * s], -1)
    return torch.cat([k.reshape(n, C), x[:, C:]], 1)


def _features(prev, consts_l1, cs, sn, C, hd):
    """T's 8-bit features of a layer's rows (float32 values) with that layer's scales."""
    q = torch.floor(_turn(prev, cs, sn, C, hd, True) * consts_l1[None, :] + 0.5)
    return torch.where(q == q, q, torch.zeros_like(q)).clamp(-127, 127).to(torch.int8)


def proxy_bits(consts, words, C, hd, layers=(1, 2), rows=None):
    """T's exact model bits of `layers` on the token rows `rows`, from one set's constants in plain torch: T's 8-bit
    features of the previous layer (de-rotated keys, the set's scales), T's int8 product and its scale, bias and
    RoPE, the set's spreads (no spread prior). The reference for proxy_costs. words int16 [L, T, W]."""
    from .core import exact_bits, words_to_float
    dev = words.device
    rows = torch.arange(words.shape[1], device=dev) if rows is None else rows.to(dev)
    cs, sn = consts["cos"][rows], consts["sin"][rows]
    total = 0.0
    for l in layers:
        prev = words_to_float(words[l - 1, rows].to(torch.int32) & 0xFFFF).to(torch.float32)
        acc = int_mm(_features(prev, consts["isx"][l - 1], cs, sn, C, hd), consts["Aq"][l])
        mu = _turn(acc.to(torch.float32) * consts["sa"][l][None, :] + consts["b"][l][None, :], cs, sn, C, hd, False)
        w = words[l, rows].to(torch.int32) & 0xFFFF
        total += float(exact_bits(w, mu.double(), consts["sp"][l].double()[None, :].expand_as(mu)).sum())
    return total


def proxy_costs(sets, words, C, hd, layers=(1, 2), rows=None):
    """For several constant sets at once: the part of T's cost that differs between sets, in bits - the logistic
    density -log2 f(x; mu, s) of every value (T's prediction and spreads as in proxy_bits). A BF16 word's
    probability is its density times its bin's width, and the width depends on the word only, so the ranking of
    the sets is that of T's bits wherever the spread is wider than the bins. float32, batched over the sets: a few
    dozen kernels whatever the number of sets. sets: list of constants; returns a list of floats."""
    from .core import words_to_float
    dev = words.device
    rows = torch.arange(words.shape[1], device=dev) if rows is None else rows.to(dev)
    cs, sn = sets[0]["cos"][rows], sets[0]["sin"][rows]
    S, n = len(sets), rows.numel()
    total = torch.zeros(S, dtype=torch.float64, device=dev)
    for l in layers:
        prev = words_to_float(words[l - 1, rows].to(torch.int32) & 0xFFFF).to(torch.float32)
        acc = torch.stack([int_mm(_features(prev, c["isx"][l - 1], cs, sn, C, hd), c["Aq"][l]) for c in sets])
        sa = torch.stack([c["sa"][l] for c in sets])[:, None, :]
        b = torch.stack([c["b"][l] for c in sets])[:, None, :]
        sp = torch.stack([c["sp"][l] for c in sets]).to(torch.float32)[:, None, :]
        mud = (acc.to(torch.float32) * sa + b).reshape(S * n, -1)
        mu = _turn(mud, cs.repeat(S, 1), sn.repeat(S, 1), C, hd, False).reshape(S, n, -1)
        x = words_to_float(words[l, rows].to(torch.int32) & 0xFFFF).to(torch.float32)
        z = (x[None] - mu).abs() / sp
        nll = z + 2 * torch.log1p(torch.exp(-z)) + torch.log(sp)
        total += nll.sum((1, 2), dtype=torch.float64)
    return (total / math.log(2)).tolist()


class T2:
    """The T2 codec over several constant sets in one mode.

    builders {set name: callable(n_layers or None, recomputed, tokens=None) -> codec}: the mode's codec for that
    set (kv_compare.t_ideas.mode_codec arguments), on all layers (None) or cut to the first n layers, for the
    request's token count or `tokens` (the pick may look at the first tokens only). Codecs are built once and kept.
    """

    def __init__(self, builders, pick_layers=4):
        self.builders = dict(builders)
        self.names = list(self.builders)
        self.pick_layers = int(pick_layers)
        self._full = {}
        self._part = {}

    def codec(self, name, recomputed=True):
        key = (name, recomputed)
        if key not in self._full:
            self._full[key] = self.builders[name](None, recomputed)
        return self._full[key]

    def pick(self, words, layers=None, tokens=None):
        """The set whose codec codes layers 1..layers (default pick_layers) of the first `tokens` tokens (default
        all) in the fewest bits ({set: bits} too)."""
        n = min((layers or self.pick_layers) + 1, words.shape[0])
        part = words[:n] if tokens is None else words[:n, :tokens]
        part = part.contiguous()
        bits = {}
        for name in self.names:
            key = (name, n, tokens)
            if key not in self._part:
                self._part[key] = self.builders[name](n, True, tokens)
            c = self._part[key]
            xall, prior = c.model(part)
            _, starts = c.encode(part, xall, prior)
            bits[name] = c.bits(starts, part.shape[1])
        return min(bits, key=bits.get), bits

    def pick_fast(self, words, consts, C, hd, layers=(1, 2), tokens=1024):
        """The set with the fewest proxy bits (proxy_bits) on layers `layers` of `tokens` tokens sampled across the
        request; consts {set: T's dense constants}. A few small products instead of coding with every set."""
        rows = sample_rows(words.shape[1], tokens).to(words.device)
        cost = proxy_costs([consts[n] for n in self.names], words, C, hd, layers, rows)
        bits = dict(zip(self.names, cost))
        return min(bits, key=bits.get), bits

    def encode(self, words, l0_ok, name=None):
        """{"words", "starts"} and the header {"set", "layer0"}: layer0 "recomputed" (l0_ok: the receiver's
        recompute equals words[0]) or "coded" (T as it is, with the chosen set)."""
        name = name or self.pick(words)[0]
        mode = "recomputed" if l0_ok else "coded"
        c = self.codec(name, l0_ok)
        xall, prior = c.model(words)
        buf, starts = c.encode(words, xall, prior)
        return {"words": buf, "starts": starts}, {"set": name, "layer0": mode, "T": int(words.shape[1])}

    def decode(self, parts, header, l0=None, sink=None):
        c = self.codec(header["set"], header["layer0"] == "recomputed")
        if header["layer0"] == "recomputed":
            if l0 is None:
                raise ValueError("the header says layer 0 is recomputed: pass the receiver's layer 0")
            return c.decode(parts["words"], parts["starts"], header["T"], l0, sink=sink)
        return c.decode(parts["words"], parts["starts"], header["T"], sink=sink)

    def bits(self, parts, header):
        return self.codec(header["set"], header["layer0"] == "recomputed").bits(parts["starts"], header["T"])

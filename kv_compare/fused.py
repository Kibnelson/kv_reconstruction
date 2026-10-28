"""Fused GPU codec for the one-step predictor: prediction, RoPE, spread, value-slice coding and next-layer features
in one Triton kernel per direction, layer by layer over blocks of tokens.

Predictor (per layer l >= 1): the token's own layer l-1, de-rotated, as 8-bit features x_q = round(x / sx) (per
channel scale from calibration, clamped to +-127), times 8-bit coefficients A_q (per output channel scale sa):

    mu_derot[j] = sa[j] * sum_i A_q[j, i] x_q[i] + b[j]        (exact integer sum, then float32)

then keys are rotated to their position with float32 cos/sin tables. Layer 0 is predicted by its mean. Spread: per
channel mean absolute residual on the scales document / (2 ln 2), times the token's multiplier
clip((64 * prior + 16) / 80, 0.25, 4), where prior is the token's mean surprise |x - mu| / (2 ln 2 * spread) over
the previous layer, summed in fixed point (exact, so encoder and decoder agree in any order).

A program owns TB tokens. Its lanes are R rows x HD channels (one head): lane (r, c) carries one rANS stream through
every layer and head, coding channel c of tokens r, r + R, r + 2R, ... of the program, and each row is its own
stream segment, so the prefix sums that place the 16-bit words stay within a row (with R warps and four
consecutive lanes per thread, a row of 128 lanes is one warp). Per layer and head the program
computes the [TB x HD] prediction tile with integer tensor-core products into scratch memory, decodes it in TB / R
steps of R tokens, then turns the decoded head into the next layer's features. A token's layers only depend on that
token, so programs never wait on each other, and the streams run through the whole document, so their 32-bit
states cost almost nothing (the encoder must see every layer before it can finish a stream).

Encoding needs the exact features and priors first (a forward model pass over the exact words), then runs the same
tiles in reverse order, as rANS requires. Both directions share the prediction and coding code (gpu_coder).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import gpu_coder as gc
from .core import LN2X2, words_to_float
from .predict import Config

triton, tl = gc.triton, gc.tl
QS = 4096                                      # surprise fixed point: units of 1/4096
ONESTEP = Config("onestep", ("P",), 1, "natural", True)


def _feat_ref(x, rope, positions, isx):
    """Reference features (statistics only): de-rotate, times 1/scale, round, clamp to int8 range."""
    f = rope.derotate_row(x.to(torch.float64), positions)
    return torch.clamp(torch.floor(f.to(torch.float32) * isx + 0.5), -127, 127)


def prepare(study, log=print):
    """Fit the one-step predictor per layer and build the kernel constants (all float32 / int8 on study.device)."""
    if abs(study.rope.scale - 1.0) > 0:
        raise ValueError("the fused codec assumes RoPE attention scaling 1")
    if study.rope.layout != "half":
        raise ValueError("the fused codec assumes the 'half' RoPE layout")
    L, W, hd, dev = study.L, study.W, study.hd, study.device
    f32 = torch.float32
    # feature scales: amax of each layer's de-rotated features over the calibration documents
    sx = torch.ones(L, W, dtype=f32, device=dev)
    for l in range(L):
        m = torch.zeros(W, dtype=f32, device=dev)
        for d in study.train:
            m = torch.maximum(m, study.feats(d, l, cache=False).abs().amax(0))
        sx[l] = (m / 127).clamp_min(1e-30)
    Aq = torch.zeros(L, W, W, dtype=torch.int8, device=dev)
    sa = torch.ones(L, W, dtype=f32, device=dev)
    b = torch.zeros(L, W, dtype=f32, device=dev)
    for l in range(L):
        if l == 0:
            n = sum(d.tokens for d in study.train)
            b[0] = sum(study.feats(d, 0, cache=False).double().sum(0) for d in study.train).to(f32) / n
            continue
        mu, S = study.covariance(l, ["P"])
        F = study.solve(mu, [S], ["P"], 1, "natural", keep_mu=True)
        A = F.A[:, :W].double()                                   # rows: output channels (natural order)
        fold = A * sx[l - 1].double()[None, :]                    # coefficients on the quantized features
        s_row = (fold.abs().amax(1) / 127).clamp_min(1e-30)
        q = torch.clamp(torch.round(fold / s_row[:, None]), -127, 127)
        Aq[l] = q.to(torch.int8)
        sa[l] = s_row.to(f32)
        mu_x, mu_y = F.mu[:W].double(), F.mu[W:].double()
        b[l] = (mu_y - s_row * (q @ (mu_x / sx[l - 1].double()))).to(f32)
        study.forget(l - 1)
    cos, sin = study.rope.tables(torch.arange(study.tables.tokens, device=dev))
    cos, sin = cos.to(f32).contiguous(), sin.to(f32).contiguous()
    consts = {"Aq": Aq, "sa": sa, "b": b, "sx": sx, "isx": (1.0 / sx.double()).to(f32), "cos": cos, "sin": sin}
    # spreads: mean absolute residual of the quantized predictor on the scales document
    sp = torch.zeros(L, W, dtype=f32, device=dev)
    pos = torch.arange(study.tables.tokens, device=dev)
    prev = None
    for l in range(L):
        x = words_to_float(study.tables.words(l, dev))
        m = reference_mu(consts, l, prev, pos, study.C, hd)
        sp[l] = ((x - m.double()).abs().mean(0) / LN2X2 + 1e-12).to(f32)
        prev = _feat_ref(x, study.rope, pos, consts["isx"][l])
    consts["sp"] = sp
    consts["inv"] = (QS / (LN2X2 * sp.double())).to(f32)
    log(f"  fused one-step predictor prepared for {L} layers")
    return consts


def reference_mu(consts, l, prev_feat, positions, C, hd):
    """Torch reference of the kernel's prediction for one layer (float32 elementwise operations as in the kernel)."""
    W = consts["Aq"].shape[1]
    n = positions.numel()
    if l == 0 or prev_feat is None:
        acc = torch.zeros(n, W, dtype=torch.float32, device=positions.device)
    else:
        acc = (prev_feat.double() @ consts["Aq"][l].double().T).to(torch.float32)   # exact integers
    mud = acc * consts["sa"][l][None, :] + consts["b"][l][None, :]
    h2 = hd // 2
    k = mud[:, :C].reshape(n, -1, hd)
    a, bb = k[..., :h2], k[..., h2:]
    cs, sn = consts["cos"][positions][:, None, :], consts["sin"][positions][:, None, :]
    rot = torch.cat([a * cs - bb * sn, bb * cs + a * sn], -1).reshape(n, C)
    return torch.cat([rot, mud[:, C:]], 1)


if triton is not None:

    @triton.jit
    def _mu_tile(x_ptr, a_ptr, sa_ptr, b_ptr, cos_ptr, sin_ptr, t0, c0, key, W,
                 TB: tl.constexpr, H2: tl.constexpr, KCH: tl.constexpr):
        """Prediction tile [TB tokens x 2*H2 channels] (one head), as two halves; keys rotated if key."""
        rt = t0 + tl.arange(0, TB)
        ca = c0 + tl.arange(0, H2)
        cb = ca + H2
        acc_a = tl.zeros([TB, H2], dtype=tl.int32)
        acc_b = tl.zeros([TB, H2], dtype=tl.int32)
        for kk in range(0, W, KCH):
            rk = kk + tl.arange(0, KCH)
            xt = tl.load(x_ptr + rt[:, None] * W + rk[None, :])
            wa = tl.load(a_ptr + ca[:, None] * W + rk[None, :])
            wb = tl.load(a_ptr + cb[:, None] * W + rk[None, :])
            acc_a = tl.dot(xt, tl.trans(wa), acc_a, out_dtype=tl.int32)
            acc_b = tl.dot(xt, tl.trans(wb), acc_b, out_dtype=tl.int32)
        mua = acc_a.to(tl.float32) * tl.load(sa_ptr + ca)[None, :] + tl.load(b_ptr + ca)[None, :]
        mub = acc_b.to(tl.float32) * tl.load(sa_ptr + cb)[None, :] + tl.load(b_ptr + cb)[None, :]
        rh = tl.arange(0, H2)
        cs = tl.load(cos_ptr + rt[:, None] * H2 + rh[None, :])
        sn = tl.load(sin_ptr + rt[:, None] * H2 + rh[None, :])
        ra = mua * cs - mub * sn
        rb = mub * cs + mua * sn
        return tl.where(key, ra, mua), tl.where(key, rb, mub)

    @triton.jit
    def _sq(x, mu, inv):
        """Fixed-point surprise of one value, floor(|x - mu| * QS / (2 ln 2 * spread)); NaN and Inf give the cap
        (W * 2^19 < 2^31, so a token's integer sum cannot overflow)."""
        q = tl.floor(tl.abs(x - mu) * inv)
        return tl.minimum(tl.where(q == q, q, 524288.0), 524288.0).to(tl.int32)

    @triton.jit
    def _tile_features(w_ptr, f_ptr, isx_ptr, cos_ptr, sin_ptr, t0, c0, key, W, TB: tl.constexpr, H2: tl.constexpr):
        """Words of one head [TB tokens] (layer base w_ptr) -> int8 features of the next layer at f_ptr: keys
        de-rotated, times 1/scale, rounded, clamped; NaN words give 0."""
        rt = t0 + tl.arange(0, TB)
        rh = tl.arange(0, H2)
        ca = c0 + rh
        xa = gc._word_val(tl.load(w_ptr + rt[:, None] * W + ca[None, :]).to(tl.int32) & 0xFFFF)
        xb = gc._word_val(tl.load(w_ptr + rt[:, None] * W + (ca + H2)[None, :]).to(tl.int32) & 0xFFFF)
        cs = tl.load(cos_ptr + rt[:, None] * H2 + rh[None, :])
        sn = tl.load(sin_ptr + rt[:, None] * H2 + rh[None, :])
        da = tl.where(key, xa * cs + xb * sn, xa)
        db = tl.where(key, xb * cs - xa * sn, xb)
        qa = tl.floor(da * tl.load(isx_ptr + ca)[None, :] + 0.5)
        qb = tl.floor(db * tl.load(isx_ptr + ca + H2)[None, :] + 0.5)
        qa = tl.where(qa == qa, qa, 0.0)                  # NaN first: min/max with NaN differ between the GPU and
        qb = tl.where(qb == qb, qb, 0.0)                  # the interpreter
        qa = tl.minimum(tl.maximum(qa, -127.0), 127.0)
        qb = tl.minimum(tl.maximum(qb, -127.0), 127.0)
        tl.store(f_ptr + rt[:, None] * W + ca[None, :], qa.to(tl.int8))
        tl.store(f_ptr + rt[:, None] * W + (ca + H2)[None, :], qb.to(tl.int8))

    @triton.jit
    def _mult(prior):
        return tl.minimum(tl.maximum((prior * 64.0 + 16.0) * 0.0125, 0.25), 4.0)

    @triton.jit
    def _model_kernel(words_ptr, xall_ptr, prior_ptr, a_ptr, sa_ptr, b_ptr, inv_ptr, isx_ptr, cos_ptr, sin_ptr,
                      L, T, W, G, HG, PRIOR_SCALE,
                      TB: tl.constexpr, H2: tl.constexpr, KCH: tl.constexpr):
        """Forward pass over the exact words: every layer's features and every token's prior (for the encoder)."""
        pid = tl.program_id(0)
        t0 = pid * TB
        rt = t0 + tl.arange(0, TB)
        rh = tl.arange(0, H2)
        tl.store(prior_ptr + rt, tl.full([TB], 1.0, tl.float32))
        for layer in range(0, L):
            xin = xall_ptr + tl.maximum(layer - 1, 0) * T * W
            wl = words_ptr + layer * T * W
            sacc = tl.zeros([TB], dtype=tl.int32)
            for g in range(0, G):
                c0 = g * 2 * H2
                key = g < HG
                mua, mub = _mu_tile(xin, a_ptr + layer * W * W, sa_ptr + layer * W, b_ptr + layer * W, cos_ptr,
                                    sin_ptr, t0, c0, key, W, TB, H2, KCH)
                ca = c0 + rh
                xa = gc._word_val(tl.load(wl + rt[:, None] * W + ca[None, :]).to(tl.int32) & 0xFFFF)
                xb = gc._word_val(tl.load(wl + rt[:, None] * W + (ca + H2)[None, :]).to(tl.int32) & 0xFFFF)
                sacc += tl.sum(_sq(xa, mua, tl.load(inv_ptr + layer * W + ca)[None, :]), 1)
                sacc += tl.sum(_sq(xb, mub, tl.load(inv_ptr + layer * W + ca + H2)[None, :]), 1)
                _tile_features(wl, xall_ptr + layer * T * W, isx_ptr + layer * W, cos_ptr, sin_ptr, t0, c0, key, W,
                               TB, H2)
            tl.debug_barrier()                           # the next layer's products read these features
            if layer + 1 < L:
                tl.store(prior_ptr + (layer + 1) * T + rt, sacc.to(tl.float32) * PRIOR_SCALE)

    @triton.jit
    def _store_tile(ptr, mua, mub, TB: tl.constexpr, H2: tl.constexpr):
        rl = tl.arange(0, TB)
        rh = tl.arange(0, H2)
        tl.store(ptr + rl[:, None] * (2 * H2) + rh[None, :], mua)
        tl.store(ptr + rl[:, None] * (2 * H2) + (rh + H2)[None, :], mub)

    @triton.jit
    def _fenc_kernel(words_ptr, xall_ptr, prior_ptr, a_ptr, sa_ptr, b_ptr, sp_ptr, cos_ptr, sin_ptr,
                     smu_ptr, smult_ptr, out_ptr, start_ptr, cumf_ptr, ko_ptr,
                     L, T, W, G, HG, CAP,
                     TB: tl.constexpr, H2: tl.constexpr, KCH: tl.constexpr, R: tl.constexpr, UNIFORM: tl.constexpr):
        """Encoder: the decoder's order reversed (layers, heads, token steps); one stream segment per row."""
        pid = tl.program_id(0)
        t0 = pid * TB
        HD: tl.constexpr = 2 * H2
        NS: tl.constexpr = TB // R
        rr = tl.arange(0, R)
        rc = tl.arange(0, HD)
        j = tl.arange(0, R * HD)                     # lane: row j // HD (a token of the step), channel j % HD
        jr = j // HD
        jc = j % HD
        seg = pid * R + rr
        smu = smu_ptr + pid * TB * HD
        smult = smult_ptr + pid * TB
        x = tl.full([R * HD], 65536, tl.uint32)
        wp = (seg + 1) * CAP
        valid = j >= 0
        for li in range(0, L):
            layer = L - 1 - li
            xin = xall_ptr + tl.maximum(layer - 1, 0) * T * W
            tl.debug_barrier()
            tl.store(smult + tl.arange(0, TB), _mult(tl.load(prior_ptr + layer * T + t0 + tl.arange(0, TB))))
            for gi in range(0, G):
                g = G - 1 - gi
                c0 = g * HD
                key = g < HG
                mua, mub = _mu_tile(xin, a_ptr + layer * W * W, sa_ptr + layer * W, b_ptr + layer * W, cos_ptr,
                                    sin_ptr, t0, c0, key, W, TB, H2, KCH)
                _store_tile(smu, mua, mub, TB, H2)
                tl.debug_barrier()
                spl = tl.load(sp_ptr + layer * W + c0 + jc)
                for ii in range(0, NS):
                    i = NS - 1 - ii
                    mu = tl.load(smu + i * R * HD + j)
                    s = spl * tl.load(smult + i * R + jr)
                    w = tl.load(words_ptr + layer * T * W + (t0 + i * R + jr) * W + c0 + jc).to(tl.int32) & 0xFFFF
                    x, wp = gc._enc_value(x, wp, w, mu, s, valid, out_ptr, cumf_ptr, ko_ptr, UNIFORM, R)
                tl.debug_barrier()
        # states: one 32-bit word per lane at an 8-aligned position just below the data (the decoder loads them as
        # contiguous aligned int32); start = start of the data
        base = tl.multiple_of(((wp - 2 * HD) // 8) * 8, 8)
        x2 = tl.reshape(x, [R, HD]).to(tl.int32, bitcast=True)
        tl.store(out_ptr.to(tl.pointer_type(tl.int32)) + (base // 2)[:, None] + rc[None, :], x2)
        tl.store(start_ptr + seg, wp)

    @triton.jit
    def _fdec_kernel(buf_ptr, start_ptr, out_ptr, xbuf_ptr, a_ptr, sa_ptr, b_ptr, sp_ptr, inv_ptr, isx_ptr,
                     cos_ptr, sin_ptr, smu_ptr, ssum_ptr, smult_ptr, cumf_ptr, lut_ptr, tail_ptr,
                     L, T, W, G, HG, PRIOR_SCALE,
                     TB: tl.constexpr, H2: tl.constexpr, KCH: tl.constexpr, R: tl.constexpr, LUT_SHIFT: tl.constexpr,
                     LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr, SEARCH: tl.constexpr):
        """Decoder: per layer and head, the prediction tile (into scratch), then TB / R steps of R tokens x one head
        (lanes: R rows of the head's channels, one stream segment per row), then the head's next-layer features."""
        pid = tl.program_id(0)
        t0 = pid * TB
        HD: tl.constexpr = 2 * H2
        NS: tl.constexpr = TB // R
        rr = tl.arange(0, R)
        rc = tl.arange(0, HD)
        rb = tl.arange(0, TB)
        j = tl.arange(0, R * HD)
        jr = j // HD
        jc = j % HD
        seg = pid * R + rr
        smu = smu_ptr + pid * TB * HD
        ssum = ssum_ptr + pid * G * TB
        smult = smult_ptr + pid * TB
        valid = j >= 0
        rp = tl.load(start_ptr + seg)
        base = tl.multiple_of(((rp - 2 * HD) // 8) * 8, 8)
        x2 = tl.load(buf_ptr.to(tl.pointer_type(tl.int32)) + (base // 2)[:, None] + rc[None, :])
        x = tl.reshape(x2, [R * HD]).to(tl.uint32, bitcast=True)
        tl.store(smult + rb, _mult(tl.full([TB], 1.0, tl.float32)))
        for layer in range(0, L):
            xin = xbuf_ptr + ((layer + 1) % 2) * T * W
            xout = xbuf_ptr + (layer % 2) * T * W
            wl = out_ptr + layer * T * W
            for g in range(0, G):
                c0 = g * HD
                key = g < HG
                mua, mub = _mu_tile(xin, a_ptr + layer * W * W, sa_ptr + layer * W, b_ptr + layer * W, cos_ptr,
                                    sin_ptr, t0, c0, key, W, TB, H2, KCH)
                _store_tile(smu, mua, mub, TB, H2)
                tl.debug_barrier()
                spl = tl.load(sp_ptr + layer * W + c0 + jc)
                invl = tl.load(inv_ptr + layer * W + c0 + jc)
                for i in range(0, NS):
                    mu = tl.load(smu + i * R * HD + j)
                    s = spl * tl.load(smult + i * R + jr)
                    x, rp, w = gc._dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr,
                                             LUT_SHIFT, LUT_W, REFINE, UNIFORM, R, SEARCH)
                    tl.store(wl + (t0 + i * R + jr) * W + c0 + jc, w.to(tl.int16))
                    q = tl.reshape(_sq(gc._word_val(w), mu, invl), [R, HD])
                    tl.store(ssum + g * TB + i * R + rr, tl.sum(q, 1))
                tl.debug_barrier()                       # this head's words, before its features read them
                _tile_features(wl, xout, isx_ptr + layer * W, cos_ptr, sin_ptr, t0, c0, key, W, TB, H2)
            tl.debug_barrier()                           # features and surprise sums of the whole layer
            sacc = tl.zeros([TB], dtype=tl.int32)
            for g in range(0, G):
                sacc += tl.load(ssum + g * TB + rb)
            tl.store(smult + rb, _mult(sacc.to(tl.float32) * PRIOR_SCALE))
            tl.debug_barrier()


class FusedOneStep:
    """Encode / decode a document's KV (all layers) with the fused one-step codec.

    A program owns tokens_per_program tokens; its lanes are rows x head channels (rows = tokens decoded at once,
    each row its own stream segment, so the prefix sums stay within a row)."""

    def __init__(self, consts, C, hd, tokens_per_program=64, rows=8, kchunk=64, warps=8, mode="uniform", fuse=False,
                 search=False):
        if triton is None:
            raise RuntimeError("Triton is required for the fused codec")
        self.k = consts
        self.L, self.W = consts["Aq"].shape[0], consts["Aq"].shape[1]
        self.C, self.hd = C, hd
        self.TB, self.R, self.KCH, self.warps, self.mode = int(tokens_per_program), int(rows), int(kchunk), int(warps), mode
        if self.TB % self.R:
            raise ValueError("rows must divide tokens_per_program")
        if mode not in ("exact", "uniform"):
            raise ValueError("mode is 'exact' or 'uniform'")
        self.search = bool(search)
        self.device = consts["Aq"].device
        self.tables = gc.slice_tables(self.device)
        self.opts = {} if fuse else {"enable_fp_fusion": False}
        self.G, self.HG = self.W // hd, C // hd
        self.prior_scale = 1.0 / (self.W * QS)

    def _launch(self, kernel, grid, *args, **kw):
        try:
            return kernel[grid](*args, **kw, num_warps=self.warps, **self.opts)
        except TypeError:                        # launch options unsupported (e.g. interpreter)
            return kernel[grid](*args, **kw)

    def layout(self, T):
        """(programs, capacity of one row segment in 16-bit words)."""
        if T % self.TB:
            raise ValueError("tokens must be a multiple of tokens_per_program")
        if T > self.k["cos"].shape[0]:
            raise ValueError("more tokens than the RoPE tables cover")
        P = T // self.TB
        values_per_lane = self.L * self.G * (self.TB // self.R)
        return P, self.hd * (2 * values_per_lane + 2) + 8

    def _scratch(self, P):
        f32 = dict(dtype=torch.float32, device=self.device)
        return (torch.empty(P * self.TB * self.hd, **f32), torch.empty(P * self.TB, **f32),
                torch.empty(P * self.G * self.TB, dtype=torch.int32, device=self.device))

    def model(self, words):
        """Features [L, T, W] int8 and priors [L, T] of the exact words [L, T, W] int16."""
        L, T, W = words.shape
        P, _ = self.layout(T)
        xall = torch.zeros(L, T, W, dtype=torch.int8, device=self.device)
        prior = torch.empty(L, T, dtype=torch.float32, device=self.device)
        k = self.k
        self._launch(_model_kernel, (P,), words, xall, prior, k["Aq"], k["sa"], k["b"], k["inv"], k["isx"], k["cos"],
                     k["sin"], L, T, W, self.G, self.HG, self.prior_scale, TB=self.TB, H2=self.hd // 2, KCH=self.KCH)
        return xall, prior

    def encode(self, words, xall=None, prior=None, out=None):
        """words [L, T, W] int16 -> (buffer int16, starts int32 [programs x rows])."""
        L, T, W = words.shape
        P, cap = self.layout(T)
        if xall is None:
            xall, prior = self.model(words)
        if out is None:
            out = torch.empty(P * self.R * cap, dtype=torch.int16, device=self.device)
        starts = torch.empty(P * self.R, dtype=torch.int32, device=self.device)
        smu, smult, _ = self._scratch(P)
        k, t = self.k, self.tables
        self._launch(_fenc_kernel, (P,), words, xall, prior, k["Aq"], k["sa"], k["b"], k["sp"], k["cos"], k["sin"],
                     smu, smult, out, starts, t["cumf"], t["ko"], L, T, W, self.G, self.HG, cap,
                     TB=self.TB, H2=self.hd // 2, KCH=self.KCH, R=self.R, UNIFORM=self.mode == "uniform")
        return out, starts

    def decode(self, buf, starts, T, out=None, scratch=None):
        L, W = self.L, self.W
        P, cap = self.layout(T)
        words = torch.empty(L, T, W, dtype=torch.int16, device=self.device) if out is None else out
        if scratch is None:
            scratch = (torch.zeros(2, T, W, dtype=torch.int8, device=self.device),) + self._scratch(P)
        xbuf, smu, smult, ssum = scratch
        k, t = self.k, self.tables
        self._launch(_fdec_kernel, (P,), buf, starts, words, xbuf, k["Aq"], k["sa"], k["b"], k["sp"], k["inv"],
                     k["isx"], k["cos"], k["sin"], smu, ssum, smult, t["cumf"], t["lut"], t["tail"], L, T, W, self.G,
                     self.HG, self.prior_scale, TB=self.TB, H2=self.hd // 2, KCH=self.KCH, R=self.R,
                     LUT_SHIFT=16 - gc.LUT_BITS, LUT_W=gc.LUT_W, REFINE=gc.REFINE, UNIFORM=self.mode == "uniform",
                     SEARCH=self.search)
        return words

    def bits(self, starts, T):
        """Coded size: every row segment's used words (data, states and their alignment padding) plus a 32-bit
        start per segment."""
        P, cap = self.layout(T)
        ends = (torch.arange(P * self.R, device=starts.device, dtype=torch.int64) + 1) * cap
        base = torch.div(starts.to(torch.int64) - 2 * self.hd, 8, rounding_mode="floor") * 8
        return 16 * int((ends - base).sum()) + 32 * P * self.R

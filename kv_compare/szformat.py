"""Our one-step prediction inside SplitZip's kind of format: fixed-length codes, no entropy coder (bits only).

SplitZip is fast because nothing is variable-length: every value has a raw sign+mantissa byte and a 4-bit code for
its exponent (the 15 most common exponents; the 16th code is an escape whose exponent goes to a sparse side stream
with its 16-bit position). A decoder places every value independently. These variants keep that property but let
our one-step prediction (the GPU codec's int8 product, previous layer to this layer, with its spread) choose the
code book of each value. The decoder knows the prediction before it reads the value, so a value's code length may
depend on its context (a prefix sum over a block gives every value's bit offset); within a context every code has
the same length.

  SplitZip-style        one top-16 exponent code book for everything (the lossless stage's baseline)
  exponent, context     raw sign+mantissa byte; the exponent relative to max(exponent of the prediction, of the
                        spread), coded with k bits per context (2^k - 1 symbols + escape: 16-bit position + 8 bits);
                        context = exponent of the prediction minus exponent of the spread
  sign + exponent       the same with the sign's agreement with the prediction in the symbol (raw: 7 mantissa bits)
  ordered residual      r = ordered index of the word - ordered index of the rounded prediction; per context the
                        low b bits raw and floor(r / 2^b) coded with k bits (2^k - 1 symbols + escape: 16-bit
                        position + 16 bits); context = log2(spread / ulp of the prediction) in half steps. (Poor
                        whenever the sign is uncertain: the ordered index jumps by ~2^15 across zero.)
  sign + magnitude      the symbol is the sign's agreement with the prediction and the magnitude residual
                        r = |word| - |rounded prediction| (BF16 magnitude bits are ordered), or - |rounded spread|
                        when the signs differ; then as the ordered residual (b raw bits, k-bit code per context)

Code books (and b, k per context) are chosen on calibration documents, then scored on the test documents. The
spread is the codec's: per-channel spread x the adaptive multiplier from the previous layer's surprise ("static":
without the multiplier). Prediction from the exact previous layer (a lossless codec decodes it exactly)."""
from __future__ import annotations

import time

import torch

from . import fused as fu
from . import gpu_coder as gc
from . import layerwise as lw
from .core import LN2X2, words_to_float

NB = 1 << 17                     # ordered residual: floor(r / 2^b) kept in [-2^16, 2^16)
B_MAX = 12
K_MAX = 6
TOP = (1 << K_MAX) - 1
NC_D = 37                        # contexts: floor(2 log2(spread / ulp)) in [-8, 28]
NC_B = 13                        # contexts: e_mu - e_s in [-3, 9]


def _exp_field(w):
    return (w >> 7) & 255


def _predictions(study, consts, rank, doc, log_every=None):
    """Per layer: (words int64 [T, W], prediction float32, spread with and without the adaptive multiplier)."""
    dev = study.device
    T, W, C, hd = doc.tokens, study.W, study.C, study.hd
    pos = study.positions(T)
    prior = torch.ones(T, dtype=torch.float32, device=dev)
    xprev = None
    k = consts if rank is None else dict(consts, sa=rank["sa"], b=rank["b"], sp=rank["sp"])
    for l in range(study.L):
        w = doc.words(l, dev)
        x = words_to_float(w).to(torch.float32)
        if l == 0:
            acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        elif rank is None:
            acc = lw.int_mm(xprev, consts["Aq"][l])
        else:
            z = lw.int_mm(xprev, rank["Bq"][l])
            zq = lw._requant_ref(z, rank["m1"][l], rank["m0"][l]).to(torch.int8)
            acc = lw.int_mm(zq, rank["Uq"][l])
        mu = lw._mu_from_acc(acc, k["sa"][l], k["b"][l], k["cos"], k["sin"], C, hd)
        sp = k["sp"][l][None, :]
        mult = torch.clamp((prior * 64.0 + 16.0) * 0.0125, 0.25, 4.0)[:, None]
        q = ((x - mu).abs() / (LN2X2 * sp))
        q = torch.where(torch.isfinite(q), q, torch.full_like(q, 128.0)).clamp(max=128.0)
        prior = q.mean(1)
        f = fu._feat_ref(x, study.rope, pos, consts["isx"][l])
        xprev = torch.where(torch.isnan(f), torch.zeros_like(f), f).to(torch.int8)
        yield l, w.to(torch.int64), mu, sp * mult, sp.expand_as(mu)
        del acc, x, q


def _symbols_D(w, mu, s):
    """Ordered residual and its context per value."""
    wm = gc._bf16_word_t(mu)
    r = gc._ord_t(w) - gc._ord_t(wm)
    e_mu = torch.clamp(_exp_field(wm), min=1)
    lam = torch.log2(s.clamp_min(1e-30)) - (e_mu - 134).to(torch.float32)
    c = torch.clamp(torch.floor(2 * lam), -8, 28).to(torch.int64) + 8
    return c.flatten(), r.flatten()


def _symbols_F(w, mu, s):
    """Sign agreement + magnitude residual (against |prediction|, or |spread| when the signs differ)."""
    wm = gc._bf16_word_t(mu)
    ws = gc._bf16_word_t(s)
    agree = ((w >> 15) & 1) == ((wm >> 15) & 1)
    r = (w & 0x7FFF) - torch.where(agree, wm & 0x7FFF, ws & 0x7FFF)
    e_mu = torch.clamp(_exp_field(wm), min=1)
    lam = torch.log2(s.clamp_min(1e-30)) - (e_mu - 134).to(torch.float32)
    c = torch.clamp(torch.floor(2 * lam), -8, 28).to(torch.int64) + 8
    return c.flatten(), r.flatten(), agree.flatten().to(torch.int64)


def _symbols_B(w, mu, s, with_sign):
    wm = gc._bf16_word_t(mu)
    ws = gc._bf16_word_t(s)
    e, e_mu, e_s = _exp_field(w), _exp_field(wm), _exp_field(ws)
    ref = torch.maximum(e_mu, e_s)
    d = torch.clamp(e - ref + 128, 0, 255)
    if with_sign:
        agree = (((w >> 15) & 1) == (mu < 0).to(torch.int64)).to(torch.int64)
        d = d + 256 * agree
    c = torch.clamp(e_mu - e_s, -3, 9) + 3
    return c.flatten(), d.flatten()


class _Book:
    """Per-context fixed-length code book chosen on calibration counts (one layer)."""

    def __init__(self, nc, nsym, raw_bits, escape_bits, shifts):
        self.nc, self.nsym, self.raw, self.esc, self.shifts = nc, nsym, raw_bits, escape_bits, shifts
        self.counts = None

    def add(self, c, sym_of_shift):
        """sym_of_shift(b) -> symbols in [0, nsym) for raw-bit count b (one count table per b)."""
        tabs = []
        for b in self.shifts:
            key = c * self.nsym + sym_of_shift(b)
            tabs.append(torch.bincount(key, minlength=self.nc * self.nsym).reshape(self.nc, self.nsym))
        t = torch.stack(tabs)
        self.counts = t if self.counts is None else self.counts + t

    def choose(self):
        cnt = self.counts                                            # [nb, nc, nsym]
        top_v, top_i = torch.topk(cnt, TOP, dim=2)
        cov = torch.cumsum(top_v, 2)                                 # coverage of the top m symbols
        n = cnt.sum(2)[0].to(torch.float64)                          # values per context (same for every b)
        best = torch.full((self.nc,), float("inf"), dtype=torch.float64, device=cnt.device)
        bsel = torch.zeros(self.nc, dtype=torch.int64, device=cnt.device)
        ksel = torch.ones(self.nc, dtype=torch.int64, device=cnt.device)
        for bi, b in enumerate(self.shifts):
            for k in range(1, K_MAX + 1):
                m = (1 << k) - 1
                inside = cov[bi, :, m - 1].to(torch.float64)
                cost = n * (self.raw(b) + k) + self.esc * (n - inside)
                better = cost < best
                best = torch.where(better, cost, best)
                bsel = torch.where(better, torch.full_like(bsel, bi), bsel)
                ksel = torch.where(better, torch.full_like(ksel, k), ksel)
        member = torch.zeros(self.nc, self.nsym, dtype=torch.bool, device=cnt.device)
        for c in range(self.nc):
            m = (1 << int(ksel[c])) - 1
            member[c, top_i[int(bsel[c]), c, :m]] = True
        self.bsel, self.ksel, self.member = bsel, ksel, member
        self.counts = None
        return self

    def bits(self, c, sym_of_shift):
        """Total bits of the values (contexts c) with the chosen books."""
        total = 0.0
        for bi, b in enumerate(self.shifts):
            sel = self.bsel[c] == bi
            if not bool(sel.any()):
                continue
            cs = c[sel]
            sym = sym_of_shift(b)[sel]
            inside = self.member[cs, sym]
            k = self.ksel[cs].to(torch.float64)
            total += float((self.raw(b) + k).sum()) + self.esc * float((~inside).sum())
        return total


def probe(study, consts, rank=None, calib=None, log=print):
    """Bits per value of every variant on study.tests (per layer and mean). calib: calibration documents
    (default: the first two of study.train). rank: prepare_rank constants (the fast-link product) or None."""
    dev = study.device
    calib = study.train[:2] if calib is None else calib
    L = study.L
    names = ["SplitZip-style", "exponent, context", "sign + exponent, context", "ordered residual",
             "ordered residual, static spread", "sign + magnitude", "sign + magnitude, static spread"]
    out = {n: {d.name: [] for d in study.tests} for n in names}
    # SplitZip-style code book: the 16 most common exponents over the calibration documents and layers
    hist = torch.zeros(256, dtype=torch.int64, device=dev)
    for d in calib:
        for l in range(L):
            hist += torch.bincount(_exp_field(d.words(l, dev).to(torch.int64)).flatten(), minlength=256)
    sz_in = torch.zeros(256, dtype=torch.bool, device=dev)
    sz_in[torch.topk(hist, 16).indices] = True
    t0 = time.time()
    gens = {d.name: _predictions(study, consts, rank, d) for d in calib + study.tests}
    for l in range(L):
        cur = {name: next(g) for name, g in gens.items()}
        books = {
            "exponent, context": _Book(NC_B, 256, lambda b: 8, 24, [0]),
            "sign + exponent, context": _Book(NC_B, 512, lambda b: 7, 25, [0]),
            "ordered residual": _Book(NC_D, NB, lambda b: b, 32, list(range(B_MAX + 1))),
            "ordered residual, static spread": _Book(NC_D, NB, lambda b: b, 32, list(range(B_MAX + 1))),
            "sign + magnitude": _Book(NC_D, NB, lambda b: b, 32, list(range(B_MAX + 1))),
            "sign + magnitude, static spread": _Book(NC_D, NB, lambda b: b, 32, list(range(B_MAX + 1))),
        }

        def feed(name, w, mu, s, ss):
            if name == "exponent, context":
                c, d = _symbols_B(w, mu, s, False)
                return c, (lambda b: d)
            if name == "sign + exponent, context":
                c, d = _symbols_B(w, mu, s, True)
                return c, (lambda b: d)
            if name.startswith("sign + magnitude"):
                c, r, a = _symbols_F(w, mu, s if name == "sign + magnitude" else ss)
                h = NB // 4                           # half the symbols per sign case: r_hi in [-2^15, 2^15)
                return c, (lambda b: a * (NB // 2) + torch.clamp(torch.div(r, 1 << b, rounding_mode="floor"),
                                                                 -h, h - 1) + h)
            c, r = _symbols_D(w, mu, s if name == "ordered residual" else ss)
            return c, (lambda b: torch.clamp(torch.div(r, 1 << b, rounding_mode="floor"), -NB // 2, NB // 2 - 1) + NB // 2)

        for d in calib:
            _, w, mu, s, ss = cur[d.name]
            for name, bk in books.items():
                c, f = feed(name, w, mu, s, ss)
                bk.add(c, f)
        for bk in books.values():
            bk.choose()
        for d in study.tests:
            _, w, mu, s, ss = cur[d.name]
            n = w.numel()
            out["SplitZip-style"][d.name].append(12 + 24 * float((~sz_in[_exp_field(w)]).sum()) / n)
            for name, bk in books.items():
                c, f = feed(name, w, mu, s, ss)
                out[name][d.name].append(bk.bits(c, f) / n)
        del cur, books
        if log:
            log(f"  SplitZip-format probe{' (rank ' + str(rank['rank']) + ')' if rank else ''}: layer {l} done "
                f"({time.time() - t0:.0f} s)")
    res = {"layers": L, "rank": rank["rank"] if rank else None, "calibration": [d.name for d in calib],
           "variants": {}}
    for name in names:
        per = {dn: sum(v) / len(v) for dn, v in out[name].items()}
        res["variants"][name] = {"bits": per, "mean": sum(per.values()) / len(per), "per_layer": out[name]}
    return res

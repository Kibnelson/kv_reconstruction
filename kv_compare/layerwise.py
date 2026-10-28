"""One-step codec decoded layer by layer with the product in cuBLAS: per layer, an exact int8 matrix product
(torch._int_mm: the previous layer's 8-bit features times the 8-bit coefficients, int32 result), then one Triton
kernel that does everything else for that layer: the prediction's scale, bias and RoPE, the spread, the
value-slice coding, the surprise sums for the next layer's spread, and the next layer's 8-bit features.

Same predictor, spread and features as the fused codec (fused.prepare builds the constants, and the per-lane
arithmetic below repeats the fused tile arithmetic operation for operation, so the encoder can use the fused
forward pass for its features and priors).

A program is one warp: 128 lanes = the channels of one head, for a block of TB tokens; lane c carries one rANS
stream for channel c of that head and block through every layer (the state and position are kept in memory between
the per-layer kernels), so a layer has (T / TB) x heads programs while the stream states cost one 32-bit word per lane
for the whole document. Tokens can be split into chunks decoded on separate CUDA streams: a chunk's layer only
depends on the same chunk's previous layer, so one chunk's product runs while another chunk decodes."""
from __future__ import annotations

import torch

from . import fused as fu
from . import gpu_coder as gc

triton, tl = gc.triton, gc.tl


def int_mm(x, A, out=None):
    """Exact x [n, K] int8 times A [N, K] int8 transposed -> int32 [n, N] (cuBLAS on the GPU)."""
    if x.is_cuda:
        if out is not None:
            try:
                return torch._int_mm(x, A.t(), out=out)
            except TypeError:                    # no out= variant
                pass
        y = torch._int_mm(x, A.t())
        return y if out is None else out.copy_(y)
    y = (x.double() @ A.double().t()).to(torch.int32)    # exact: |sum| < 2^31 and float64 holds it
    return y if out is None else out.copy_(y)


if triton is not None:

    @triton.jit
    def _lane_mu(acc_row, own, part, sao, bo, sap, bp, cs, sn, low, key):
        """The fused tile's prediction for one token, per lane: scale and bias, then RoPE for keys (lane in the
        first half: own * cos - partner * sin; second half: own * cos + partner * sin)."""
        ao = tl.load(acc_row + own).to(tl.float32) * sao + bo
        ap = tl.load(acc_row + part).to(tl.float32) * sap + bp
        rot = tl.where(low, ao * cs - ap * sn, ao * cs + ap * sn)
        return tl.where(key, rot, ao)

    @triton.jit
    def _feature(v, vp, cs, sn, low, key, isx):
        """Next-layer 8-bit feature from a decoded value and its RoPE partner (the fused tile's arithmetic)."""
        d = tl.where(key, tl.where(low, v * cs + vp * sn, v * cs - vp * sn), v)
        q = tl.floor(d * isx + 0.5)
        q = tl.where(q == q, q, 0.0)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        return q.to(tl.int8)

    @triton.jit
    def _lfeat_kernel(words_ptr, xout_ptr, isx_ptr, cos_ptr, sin_ptr, t_base, NV, W, C, HD: tl.constexpr,
                      BLOCK: tl.constexpr):
        """Features of a chunk of one decoded layer (a separate pass), element-wise over token x channel."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < NV
        idx = tl.where(m, idx, 0)
        t = t_base + idx // W
        c = idx % W
        cc = c % HD
        H2: tl.constexpr = HD // 2
        low = cc < H2
        part = tl.where(low, c + H2, c - H2)
        hh = tl.where(low, cc, cc - H2)
        v = gc._word_val(tl.load(words_ptr + t * W + c).to(tl.int32) & 0xFFFF)
        vp = gc._word_val(tl.load(words_ptr + t * W + part).to(tl.int32) & 0xFFFF)
        cs = tl.load(cos_ptr + t * H2 + hh)
        sn = tl.load(sin_ptr + t * H2 + hh)
        tl.store(xout_ptr + t * W + c, _feature(v, vp, cs, sn, low, c < C, tl.load(isx_ptr + c)), mask=m)

    @triton.jit
    def _ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, t_base, T, W, G, HG, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, FEAT: tl.constexpr, LEAN: tl.constexpr):
        """Decode one layer of one token block x head (pointers are the layer's: out_ptr, sa_ptr, ...).
        FEAT: make the next layer's features here (else a separate pass does); LEAN: reload the per-channel
        constants every token instead of keeping them in registers."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
        t0 = t_base + blk * TB
        HD: tl.constexpr = 2 * H2
        cc = tl.arange(0, HD)
        c0 = g * HD
        key = g < HG
        low = cc < H2
        own = c0 + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        seg = (t0 // TB) * G + g
        valid = cc >= 0
        if FIRST:                                    # states from the stream: 8-aligned, just below the data
            rp = tl.load(start_ptr + seg)
            base = tl.multiple_of(((rp - 2 * HD) // 8) * 8, 8)
            x = tl.load(buf_ptr.to(tl.pointer_type(tl.int32)) + base // 2 + cc).to(tl.uint32, bitcast=True)
        else:
            rp = tl.load(pos_ptr + seg)
            x = tl.load(state_ptr + seg * HD + cc).to(tl.uint32, bitcast=True)
        # spread multipliers of the block's tokens, from the previous layer's surprise sums (all heads)
        rb = tl.arange(0, TB)
        smult = smult_ptr + pid * TB
        if FIRST:
            tl.store(smult + rb, fu._mult(tl.full([TB], 1.0, tl.float32)))
        else:
            sacc = tl.zeros([TB], dtype=tl.int32)
            for gg in range(0, G):
                sacc += tl.load(ssum_in_ptr + gg * T + t0 + rb)
            tl.store(smult + rb, fu._mult(sacc.to(tl.float32) * PRIOR_SCALE))
        tl.debug_barrier()
        if not LEAN:
            sao = tl.load(sa_ptr + own)
            bo = tl.load(b_ptr + own)
            sap = tl.load(sa_ptr + part)
            bp = tl.load(b_ptr + part)
            spl = tl.load(sp_ptr + own)
            invl = tl.load(inv_ptr + own)
            isxl = tl.load(isx_ptr + own)
        for i in range(0, TB):
            t = t0 + i
            if LEAN:
                sao = tl.load(sa_ptr + own)
                bo = tl.load(b_ptr + own)
                sap = tl.load(sa_ptr + part)
                bp = tl.load(b_ptr + part)
                spl = tl.load(sp_ptr + own)
                invl = tl.load(inv_ptr + own)
                isxl = tl.load(isx_ptr + own)
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = _lane_mu(acc_ptr + (t - t_base) * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            s = spl * tl.load(smult + i)
            x, rp, w = gc._dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, LUT_SHIFT, LUT_W,
                                     REFINE, UNIFORM, 0, SEARCH)
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            v = gc._word_val(w)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            if FEAT:
                tl.debug_barrier()                   # the partner lane's word, for the de-rotated features
                vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
                tl.store(xout_ptr + t * W + own, _feature(v, vp, cs, sn, low, key, isxl))
        if not LAST:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, rp)

    @triton.jit
    def _lenc_kernel(words_ptr, prior_ptr, state_ptr, pos_ptr, out_ptr, start_ptr, acc_ptr, sa_ptr, b_ptr, sp_ptr,
                     cos_ptr, sin_ptr, smult_ptr, cumf_ptr, ko_ptr, T, W, G, HG,
                     FLUSH: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr, UNIFORM: tl.constexpr):
        """Encode one layer of one token block x head, tokens in reverse (layers are launched last first)."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
        t0 = blk * TB
        HD: tl.constexpr = 2 * H2
        cc = tl.arange(0, HD)
        c0 = g * HD
        key = g < HG
        low = cc < H2
        own = c0 + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        seg = pid
        valid = cc >= 0
        wp = tl.load(pos_ptr + seg)
        x = tl.load(state_ptr + seg * HD + cc).to(tl.uint32, bitcast=True)
        rb = tl.arange(0, TB)
        smult = smult_ptr + pid * TB
        tl.store(smult + rb, fu._mult(tl.load(prior_ptr + t0 + rb)))
        tl.debug_barrier()
        sao = tl.load(sa_ptr + own)
        bo = tl.load(b_ptr + own)
        sap = tl.load(sa_ptr + part)
        bp = tl.load(b_ptr + part)
        spl = tl.load(sp_ptr + own)
        for ii in range(0, TB):
            i = TB - 1 - ii
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = _lane_mu(acc_ptr + t * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            s = spl * tl.load(smult + i)
            w = tl.load(words_ptr + t * W + own).to(tl.int32) & 0xFFFF
            x, wp = gc._enc_value(x, wp, w, mu, s, valid, out_ptr, cumf_ptr, ko_ptr, UNIFORM, 0)
        if FLUSH:                                    # states 8-aligned just below the data; start = data start
            base = tl.multiple_of(((wp - 2 * HD) // 8) * 8, 8)
            tl.store(out_ptr.to(tl.pointer_type(tl.int32)) + base // 2 + cc, x.to(tl.int32, bitcast=True))
            tl.store(start_ptr + seg, wp)
        else:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, wp)


class LayerwiseOneStep:
    """Encode / decode a document's KV (all layers) with cuBLAS products and one coding kernel per layer."""

    def __init__(self, consts, C, hd, tokens_per_program=128, mode="uniform", search=True, chunks=1, fuse=False,
                 features="inline", lean=False, split_streams=False):
        """features: "inline" (the coding kernel makes the next layer's features) or "pass" (a separate kernel);
        lean: reload per-channel constants every token (fewer registers); split_streams: per chunk, the products on
        a high-priority CUDA stream and the coding on another, so pending products are scheduled first."""
        if triton is None:
            raise RuntimeError("Triton is required")
        if mode not in ("exact", "uniform"):
            raise ValueError("mode is 'exact' or 'uniform'")
        self.k = consts
        self.L, self.W = consts["Aq"].shape[0], consts["Aq"].shape[1]
        self.C, self.hd, self.TB = C, hd, int(tokens_per_program)
        self.mode, self.search, self.chunks = mode, bool(search), int(chunks)
        if features not in ("inline", "pass"):
            raise ValueError("features is 'inline' or 'pass'")
        self.features, self.lean, self.split = features, bool(lean), bool(split_streams)
        self.device = consts["Aq"].device
        self.tables = gc.slice_tables(self.device)
        self.opts = {} if fuse else {"enable_fp_fusion": False}
        self.G, self.HG = self.W // hd, C // hd
        self.prior_scale = 1.0 / (self.W * fu.QS)

    def _launch(self, kernel, grid, *args, warps=1, **kw):
        try:
            return kernel[grid](*args, **kw, num_warps=warps, **self.opts)
        except TypeError:                            # interpreter: no launch options
            return kernel[grid](*args, **kw)

    def layout(self, T):
        """(segments, capacity of a segment in 16-bit words); one segment per token block x head."""
        if T % (self.TB * self.chunks):
            raise ValueError("tokens must be a multiple of tokens_per_program x chunks")
        if T > self.k["cos"].shape[0]:
            raise ValueError("more tokens than the RoPE tables cover")
        return (T // self.TB) * self.G, self.hd * (2 * self.L * self.TB + 2) + 8

    def model(self, words):
        """Features and priors of the exact words (the fused forward pass: the same arithmetic)."""
        return fu.FusedOneStep(self.k, self.C, self.hd, tokens_per_program=32, rows=4, warps=4).model(words)

    def encode(self, words, xall=None, prior=None, out=None):
        L, T, W = words.shape
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
        for layer in reversed(range(L)):
            if layer == 0:
                acc.zero_()
            else:
                int_mm(xall[layer - 1], k["Aq"][layer], out=acc)
            self._launch(_lenc_kernel, (S,), words[layer], prior[layer], state, pos, out, starts, acc,
                         k["sa"][layer], k["b"][layer], k["sp"][layer], k["cos"], k["sin"], smult, t["cumf"], t["ko"],
                         T, W, self.G, self.HG, FLUSH=layer == 0, TB=self.TB, H2=self.hd // 2,
                         UNIFORM=self.mode == "uniform")
        return out, starts

    def buffers(self, T):
        """Decoder working memory, reusable across calls."""
        S, _ = self.layout(T)
        dev = self.device
        Tc = T // self.chunks
        return {"xbuf": torch.zeros(2, T, self.W, dtype=torch.int8, device=dev),
                "ssum": torch.zeros(2, self.G, T, dtype=torch.int32, device=dev),
                "state": torch.empty(S * self.hd, dtype=torch.int32, device=dev),
                "pos": torch.empty(S, dtype=torch.int32, device=dev),
                "smult": torch.empty(S * self.TB, dtype=torch.float32, device=dev),
                "acc": [torch.zeros(Tc, self.W, dtype=torch.int32, device=dev) for _ in range(self.chunks)],
                "streams": ([torch.cuda.Stream(dev) for _ in range(self.chunks)]
                            if dev.type == "cuda" and (self.chunks > 1 or self.split) else None),
                "gemm_streams": ([torch.cuda.Stream(dev, priority=-1) for _ in range(self.chunks)]
                                 if dev.type == "cuda" and self.split else None)}

    def decode(self, buf, starts, T, out=None, bufs=None, accs=None, products_only=False):
        """accs: precomputed int32 products per layer (list of [T, W]; layer 0 ignored) - times the coding
        alone; products_only: run only the products (on features already in bufs["xbuf"]) - times them alone."""
        L, W = self.L, self.W
        S, _ = self.layout(T)
        words = torch.empty(L, T, W, dtype=torch.int16, device=self.device) if out is None else out
        bufs = self.buffers(T) if bufs is None else bufs
        k, t = self.k, self.tables
        Tc = T // self.chunks
        P = (Tc // self.TB) * self.G                  # programs per chunk and layer
        cur = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        streams, gstreams = bufs["streams"], bufs.get("gemm_streams")
        for s in (streams or []) + (gstreams or []):
            s.wait_stream(cur)
        for layer in range(L):                       # issued layer by layer, so the chunks' streams interleave
            for c in range(self.chunks):
                rows = slice(c * Tc, (c + 1) * Tc)
                acc = bufs["acc"][c] if accs is None else accs[layer][rows]
                if accs is None:
                    gctx = torch.cuda.stream(gstreams[c]) if gstreams else (
                        torch.cuda.stream(streams[c]) if streams else _null())
                    with gctx:
                        if gstreams:
                            gstreams[c].wait_stream(streams[c])      # features of the previous layer, acc free
                        if layer == 0:
                            acc.zero_()
                        else:
                            int_mm(bufs["xbuf"][(layer + 1) % 2][rows], k["Aq"][layer], out=acc)
                    if gstreams:
                        streams[c].wait_stream(gstreams[c])
                if products_only:
                    continue
                ctx = torch.cuda.stream(streams[c]) if streams else _null()
                with ctx:
                    self._launch(_ldec_kernel, (P,), buf, starts, bufs["state"], bufs["pos"], words[layer], acc,
                                 bufs["xbuf"][layer % 2], k["sa"][layer], k["b"][layer], k["sp"][layer],
                                 k["inv"][layer], k["isx"][layer], k["cos"], k["sin"], bufs["ssum"][(layer + 1) % 2],
                                 bufs["ssum"][layer % 2], bufs["smult"][c * P * self.TB:], t["cumf"], t["lut"],
                                 t["tail"], c * Tc, T, W, self.G, self.HG, self.prior_scale,
                                 FIRST=layer == 0, LAST=layer == L - 1, TB=self.TB, H2=self.hd // 2,
                                 LUT_SHIFT=16 - gc.LUT_BITS, LUT_W=gc.LUT_W, REFINE=gc.REFINE,
                                 UNIFORM=self.mode == "uniform", SEARCH=self.search,
                                 FEAT=self.features == "inline", LEAN=self.lean)
                    if self.features == "pass" and layer < L - 1:
                        nv = Tc * W
                        self._launch(_lfeat_kernel, (triton.cdiv(nv, 1024),), words[layer], bufs["xbuf"][layer % 2],
                                     k["isx"][layer], k["cos"], k["sin"], c * Tc, nv, W, self.C, HD=self.hd, BLOCK=1024,
                                     warps=4)
        for s in (streams or []) + (gstreams or []):
            cur.wait_stream(s)
        return words

    def products(self, xall):
        """The exact int32 products of every layer from features xall (for timing the coding alone)."""
        out = [None]
        for layer in range(1, self.L):
            out.append(int_mm(xall[layer - 1], self.k["Aq"][layer]))
        out[0] = torch.zeros_like(out[1])
        return out

    def bits(self, starts, T):
        """Coded size: every segment's used words (data, states, alignment) plus a 32-bit start per segment."""
        S, cap = self.layout(T)
        ends = (torch.arange(S, device=starts.device, dtype=torch.int64) + 1) * cap
        base = torch.div(starts.to(torch.int64) - 2 * self.hd, 8, rounding_mode="floor") * 8
        return 16 * int((ends - base).sum()) + 32 * S


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

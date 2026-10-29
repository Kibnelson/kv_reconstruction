"""Our per-layer codec mixed with SplitZip: each layer coded either by ours or by SplitZip's own codec.

Code the layers where ours saves the most with ours and the rest with SplitZip, so that a fast link (where our
decode is the bottleneck) still gets most of our saving where it is largest. With no layer of ours the mix is
SplitZip itself, so the best mix for a link is never slower than SplitZip.

A layer of ours whose previous layer is SplitZip-coded predicts from that layer's 8-bit features (a features pass
over its decoded words) and starts its adaptive spread from prior 1 (there is no previous prediction to take the
surprise from); encoder and decoder do the same. SplitZip's layers are decoded first (they depend on nothing),
then our layers in order. The SplitZip codec is the authors' Triton codec (codec_gpu.ChunkLocalSplitZipGPU of
Intelligent-Microsystems-Lab/SplitZip), calibrated as in their benchmark (top-16 exponents of a calibration
tensor); any object with calibrate / encode (-> .compressed_bytes) / decode works.
"""
from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import torch

from . import layerwise as lw

triton = lw.triton
SPLITZIP_REPO = "https://github.com/Intelligent-Microsystems-Lab/SplitZip.git"
SPLITZIP_COMMIT = "42c743118a29db3270ae6a921a4a4eecb20cad04"


def load_splitzip(source, device, log=print):
    """The authors' SplitZip codec from a checkout (cloned at SPLITZIP_COMMIT into `source` if missing)."""
    source = Path(source)
    if not (source / "codec_gpu.py").is_file():
        log(f"  cloning {SPLITZIP_REPO} into {source}")
        subprocess.run(["git", "clone", "-q", SPLITZIP_REPO, str(source)], check=True)
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != SPLITZIP_COMMIT:
        subprocess.run(["git", "-C", str(source), "fetch", "-q", "origin", SPLITZIP_COMMIT], check=False)
        subprocess.run(["git", "-C", str(source), "checkout", "-q", SPLITZIP_COMMIT], check=True)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from codec_gpu import ChunkLocalSplitZipGPU
    return ChunkLocalSplitZipGPU(device=str(device), chunk_size=1024), SPLITZIP_COMMIT


class TorchSplitZipStyle:
    """A plain-PyTorch stand-in with SplitZip's interface (tests on CPU): top-16 exponent code book, 4-bit codes,
    raw sign+mantissa bytes, escapes with 7-byte records like the official codec."""

    def __init__(self, device="cpu"):
        self.device = device
        self.book = None

    def calibrate(self, sample):
        e = ((sample.contiguous().view(torch.int16).to(torch.int32) >> 7) & 0xFF).flatten()
        cnt = torch.bincount(e, minlength=256)
        self.book = torch.zeros(256, dtype=torch.bool, device=e.device)
        self.book[torch.topk(cnt, 16).indices] = True
        return float(cnt[self.book].sum() / cnt.sum())

    def encode(self, tensor):
        w = tensor.contiguous().view(torch.int16).clone()
        e = (w.to(torch.int32) >> 7) & 0xFF
        nesc = int((~self.book.to(e.device)[e]).sum())

        class Enc:
            pass
        enc = Enc()
        enc.words, enc.compressed_bytes = w, (w.numel() + 1) // 2 + w.numel() + 7 * nesc
        return enc

    def decode(self, enc):
        return enc.words.view(torch.bfloat16)


class MixedOneStep(lw.LayerwiseOneStep):
    """ours: per layer True (our codec) or False (SplitZip); other: a calibrated SplitZip codec."""

    def __init__(self, consts, C, hd, ours, other, **kw):
        super().__init__(consts, C, hd, **kw)
        self.ours = [bool(v) for v in ours]
        if len(self.ours) != self.L:
            raise ValueError("one ours/SplitZip flag per layer")
        self.other = other
        self.mine = [l for l in range(self.L) if self.ours[l]]

    def encode(self, words, xall=None, prior=None, out=None, layer_words=None):
        """Returns a dict: our stream and starts (layers of ours), SplitZip's encodings (its layers)."""
        L, T, W = words.shape
        S, cap = self.layout(T)
        enc = {"sz": {l: self.other.encode(words[l].view(torch.bfloat16)) for l in range(L) if not self.ours[l]}}
        if not self.mine:
            return enc
        if xall is None:
            xall, prior = self.model(words)
        prior = prior.clone()
        for l in self.mine:
            if l == 0 or not self.ours[l - 1]:
                prior[l] = 1.0
        dev, k, t = self.device, self.k, self.tables
        if out is None:
            out = torch.empty(S * cap, dtype=torch.int16, device=dev)
        starts = torch.empty(S, dtype=torch.int32, device=dev)
        state = torch.full((S * self.hd,), 65536, dtype=torch.int32, device=dev)
        pos = (torch.arange(S, dtype=torch.int32, device=dev) + 1) * cap
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        smult = torch.empty(S * self.TB, dtype=torch.float32, device=dev)
        first = self.mine[0]
        for layer in reversed(self.mine):
            if layer == 0:
                acc.zero_()
            else:
                self._product(xall[layer - 1], layer, acc)
            self._launch(lw._lenc_kernel, (S,), words[layer], prior[layer], state, pos, out, starts, acc,
                         k["sa"][layer], k["b"][layer], k["sp"][layer], k["cos"], k["sin"], smult, t["cumf"], t["ko"],
                         T, W, self.G, self.HG, FLUSH=layer == first, TB=self.TB, H2=self.hd // 2,
                         UNIFORM=self.mode == "uniform")
        enc.update(stream=out, starts=starts)
        return enc

    def total_bits(self, enc, T):
        b = sum(8 * e.compressed_bytes for e in enc["sz"].values())
        if self.mine:
            b += self.bits(enc["starts"], T)
        return b

    def decode_mixed(self, enc, T, out=None, bufs=None):
        L, W = self.L, self.W
        words = torch.empty(L, T, W, dtype=torch.int16, device=self.device) if out is None else out
        for l, e in enc["sz"].items():                # SplitZip's layers first: they depend on nothing
            words[l].copy_(self.other.decode(e).view(torch.int16).view(T, W))
        if not self.mine:
            return words
        bufs = self.buffers(T) if bufs is None else bufs
        k, t = self.k, self.tables
        buf, starts = enc["stream"], enc["starts"]
        Tc = T // self.chunks
        P = (Tc // self.TB) * self.G
        cur = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        streams = bufs["streams"]
        for s in streams or []:
            s.wait_stream(cur)
        first, last = self.mine[0], self.mine[-1]
        for layer in self.mine:
            for c in range(self.chunks):
                rows = slice(c * Tc, (c + 1) * Tc)
                acc = bufs["acc"][c]
                ctx = torch.cuda.stream(streams[c]) if streams else lw._null()
                with ctx:
                    if layer == 0:
                        acc.zero_()
                    else:
                        if not self.ours[layer - 1]:      # features of a SplitZip layer, for this chunk's rows
                            nv = Tc * W
                            self._launch(lw._lfeat_kernel, (triton.cdiv(nv, 1024),), words[layer - 1],
                                         bufs["xbuf"][(layer + 1) % 2], k["isx"][layer - 1], k["cos"], k["sin"],
                                         c * Tc, nv, W, self.C, HD=self.hd, BLOCK=1024, warps=4)
                        self._product(bufs["xbuf"][(layer + 1) % 2][rows], layer, acc, bufs["scratch"][c])
                    self._launch(lw._ldec_kernel, (P,), buf, starts, bufs["state"], bufs["pos"], words[layer], acc,
                                 bufs["xbuf"][layer % 2], k["sa"][layer], k["b"][layer], k["sp"][layer],
                                 k["inv"][layer], k["isx"][layer], k["cos"], k["sin"], bufs["ssum"][(layer + 1) % 2],
                                 bufs["ssum"][layer % 2], bufs["smult"][c * P * self.TB:], t["cumf"], t["lut"],
                                 t["tail"], c * Tc, T, W, self.G, self.HG, self.prior_scale,
                                 FIRST=layer == first, LAST=layer == last, TB=self.TB, H2=self.hd // 2,
                                 LUT_SHIFT=16 - lw.gc.LUT_BITS, LUT_W=lw.gc.LUT_W, REFINE=lw.gc.REFINE,
                                 UNIFORM=self.mode == "uniform", SEARCH=self.search, FEAT=True, LEAN=False,
                                 ONES=layer == 0 or not self.ours[layer - 1])
        for s in streams or []:
            cur.wait_stream(s)
        return words


def mix_orders(ours_layer_bits, sz_layer_bits):
    """Layers sorted by what ours saves over SplitZip (largest first)."""
    sav = [z - b for b, z in zip(ours_layer_bits, sz_layer_bits)]
    return sorted(range(len(sav)), key=lambda l: -sav[l])

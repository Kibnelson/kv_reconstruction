"""AQUA-KV's perplexity protocol (aquakv/evaluate_perplexity.py, paper Section 4) for every method in one harness.

The protocol: the whole WikiText-2 raw test split joined with blank lines and tokenized once (BOS first),
cut into consecutive windows of 8,192 tokens (the incomplete last one dropped); each window is fed 32 tokens per
forward pass to a cache that keeps the first 4 tokens (attention sinks) and the most recent tokens in BF16 and
compresses the rest: once at least 128 tokens have accumulated past the sinks, the whole buffer is compressed
layer by layer and later forward passes read its reconstruction (the pass that triggers it still reads the
exact values, as AQUA-KV's cache returns them). Perplexity over all tokens of all windows (GPTQ style), from BF16
logits as the authors' script stores them, with their BF16 loss (comparable to the paper; it resolves only about
0.8% of perplexity per window) and, from the same logits, a float32 loss for comparing methods.

StreamingCache implements that buffering for any compressor: a callable (start position, X [L, m, W] float32
BF16 values, keys after RoPE then values, as our captures hold them) -> (reconstruction [L, m, W], bits). The
compressors: HIGGS without predictors, AQUA-KV (our faithful re-implementation and its predictors), KIVI (its
quantizer, keys per channel over token groups, values per token over channel groups), and our closed-loop GPU
codec (a real stream per compressed block). Bits are counted two ways: per compressed value (what the papers
report as bits per value) and for the whole cache at the end of each window (sinks and buffer in BF16)."""
from __future__ import annotations

import math
import time

import torch
from transformers.cache_utils import DynamicCache

from . import baselines as bl
from .core import to_bf16

SEQLEN, STEP, PREFIX, BUFFER = 8192, 32, 4, 128


class StreamingCache(DynamicCache):
    """DynamicCache whose tokens past the sinks are compressed in blocks once the buffer holds >= `buffer`
    tokens (all of it, rounded down to a multiple of `granule`); the compression of a block triggered in one
    forward pass is applied before the next one reads the cache."""

    def __init__(self, compressor, num_layers, prefix=PREFIX, buffer=BUFFER, granule=1):
        super().__init__()
        self.comp, self.L = compressor, num_layers
        self.prefix, self.buffer, self.granule = prefix, buffer, granule
        self.done = 0                                # tokens [0, done) are sinks or compressed
        self.pending = 0                             # tokens to compress before the next pass reads
        self.bits = 0.0
        self.compressed = 0                          # compressed tokens
        self.seconds = 0.0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0 and self.pending:
            self._flush()
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx == self.L - 1:
            n = k.shape[-2]
            if self.done < self.prefix:
                self.done = min(self.prefix, n)
            m = n - self.done
            if m >= self.buffer:
                self.pending = m - m % self.granule
        return k, v

    def _flush(self):
        a, m = self.done, self.pending
        t0 = time.perf_counter()
        X = torch.stack([torch.cat([self._rows(self.layers[l].keys, a, m), self._rows(self.layers[l].values, a, m)], 1)
                         for l in range(self.L)])
        rec, bits = self.comp(a, X)
        for l in range(self.L):
            C = rec.shape[2] // 2
            kl, vl = self.layers[l].keys, self.layers[l].values
            kl[..., a:a + m, :] = self._heads(rec[l, :, :C], kl)
            vl[..., a:a + m, :] = self._heads(rec[l, :, C:], vl)
        self.bits += bits
        self.compressed += m
        self.done += m
        self.pending = 0
        self.seconds += time.perf_counter() - t0

    @staticmethod
    def _rows(t, a, m):                              # [1, H, n, hd] -> [m, H*hd] float32
        return t[0, :, a:a + m, :].transpose(0, 1).reshape(m, -1).to(torch.float32)

    @staticmethod
    def _heads(x, like):                             # [m, H*hd] -> [1, H, m, hd] in the cache's dtype
        H, hd = like.shape[1], like.shape[3]
        return x.reshape(-1, H, hd).transpose(0, 1)[None].to(like.dtype)

    def finish(self):
        """Compress a block still pending at the end of the window (bits as the cache would hold them)."""
        if self.pending:
            self._flush()


@torch.no_grad()
def window_ids(tokenizer, text, seqlen=SEQLEN):
    """The authors' windows: the whole text tokenized at once, consecutive windows, the last incomplete one
    dropped."""
    data = tokenizer(text, return_tensors="pt")["input_ids"]
    return [data[:, s:s + seqlen] for s in range(0, data.shape[1], seqlen) if s + seqlen < data.shape[1]]


@torch.no_grad()
def perplexity(model, windows, device, make_cache=None, step=STEP, log=None):
    """evaluate_perplexity of aquakv: per window a fresh cache, `step` tokens per forward pass, BF16 logits,
    cross-entropy over the window's shifted tokens; returns (perplexity, per-window NLL sums, token counts,
    caches' bit statistics)."""
    total_nll, total_tok, total_f32 = 0.0, 0, 0.0
    stats = {"compressed_bits": 0.0, "compressed_values": 0, "cache_bits": 0.0, "cache_values": 0,
             "compress_seconds": 0.0}
    per, per32 = [], []
    W = None
    for wi, ids in enumerate(windows):
        ids = ids.to(device)
        cache = DynamicCache() if make_cache is None else make_cache()
        dtype = next(model.parameters()).dtype
        logits = torch.zeros((1, ids.shape[1], model.get_output_embeddings().out_features), device=device, dtype=dtype)
        for i in range(0, ids.shape[1], step):
            out = model(ids[:, i:i + step], use_cache=True, past_key_values=cache)
            logits[:, i:i + step] = out.logits
        flat, lab = logits[0, :-1], ids[0, 1:]
        loss = torch.nn.CrossEntropyLoss()(flat, lab)          # the authors': BF16 logits, a BF16 mean
        n = ids.shape[1] - 1
        total_nll += float(loss.float()) * n
        total_tok += n
        per.append(float(loss.float()) * n)
        f32 = sum(float(torch.nn.functional.cross_entropy(flat[j:j + 1024].float(), lab[j:j + 1024], reduction="sum"))
                  for j in range(0, n, 1024))                  # the same logits, float32 sums (resolves small gaps)
        total_f32 += f32
        per32.append(f32)
        if isinstance(cache, StreamingCache):
            cache.finish()
            L = cache.L
            W = cache.layers[0].keys.shape[1] * cache.layers[0].keys.shape[3] * 2
            T = ids.shape[1]
            stats["compressed_bits"] += cache.bits
            stats["compressed_values"] += cache.compressed * L * W
            stats["cache_bits"] += cache.bits + (T - cache.compressed) * L * W * 16
            stats["cache_values"] += T * L * W
            stats["compress_seconds"] += cache.seconds
        del cache, logits, out
        if log:
            log(f"    window {wi + 1}/{len(windows)}: running perplexity {math.exp(total_nll / total_tok):.4f} "
                f"(float32 loss {math.exp(total_f32 / total_tok):.4f})")
    res = {"perplexity": math.exp(total_nll / total_tok), "perplexity_f32": math.exp(total_f32 / total_tok),
           "nll_sums": per, "nll_sums_f32": per32, "tokens": total_tok, "windows": len(windows),
           "tokens_per_window": total_tok // max(len(windows), 1)}
    if stats["compressed_values"]:
        res.update(bits_per_compressed_value=stats["compressed_bits"] / stats["compressed_values"],
                   bits_per_value_whole_cache=stats["cache_bits"] / stats["cache_values"],
                   compress_seconds=stats["compress_seconds"])
    return res


# ---------------------------------------------------------------------------------------------- compressors
class HiggsNoPredictor:
    """HIGGS alone (the paper's HIGGS-b-gs1024 baseline): keys without RoPE and values quantized per token with
    the n-point grid in every layer; bits b per value plus one 16-bit scale per token, K/V and 1024 channels."""

    def __init__(self, study, bits):
        self.study, self.bits, self.n = study, bits, bl.AQUA_GRID[bits]
        self.granule = 1

    def __call__(self, start, X):
        st = self.study
        L, m, W = X.shape
        C = W // 2
        pos = torch.arange(start, start + m, device=st.device)
        out = torch.empty_like(X)
        for l in range(L):
            x = X[l].to(st.device, torch.float64)
            xd = to_bf16(st.rope.derotate_row(x, pos))
            k = bl._aqua_recon(torch.zeros_like(xd[:, :C]), xd[:, :C], self.n)
            v = bl._aqua_recon(torch.zeros_like(xd[:, C:]), xd[:, C:], self.n)
            out[l] = to_bf16(st.rope.rotate_row(torch.cat([k, v], 1), pos)).to(X.dtype)
        groups = -(-C // bl.AQUA_GROUP)
        return out, float(L * m * (2 * C * self.bits + 2 * groups * 16))


class AquaKV:
    """AQUA-KV (SingleChunkQuantizedCacheWithPredictors, our faithful re-implementation, checked against the
    authors' functions): layer 0 HIGGS 4-bit without a predictor; layer l keys predicted from the block's
    reconstructed keys of layer l-1, values from [reconstructed keys of l, reconstructed values of l-1], BF16
    linear predictors fitted on their own closed-loop reconstructions (baselines.aqua_official_fit); residuals
    HIGGS b-bit."""

    def __init__(self, study, fits, bits):
        self.study, self.fits, self.bits = study, fits, bits
        self.n, self.n0 = bl.AQUA_GRID[bits], bl.AQUA_GRID[4]
        self.granule = 1

    def __call__(self, start, X):
        st = self.study
        L, m, W = X.shape
        C = W // 2
        pos = torch.arange(start, start + m, device=st.device)
        out = torch.empty_like(X)
        prev = None
        for l in range(L):
            xd = to_bf16(st.rope.derotate_row(X[l].to(st.device, torch.float64), pos))
            if l == 0:
                kr = bl._aqua_recon(torch.zeros_like(xd[:, :C]), xd[:, :C], self.n0)
                vr = bl._aqua_recon(torch.zeros_like(xd[:, C:]), xd[:, C:], self.n0)
            else:
                lk, lv = self.fits[l]
                kr = bl._aqua_recon(bl._aqua_apply(lk, prev[0]), xd[:, :C], self.n)
                vr = bl._aqua_recon(bl._aqua_apply(lv, torch.cat([kr, prev[1]], 1)), xd[:, C:], self.n)
            prev = (kr, vr)
            out[l] = to_bf16(st.rope.rotate_row(torch.cat([kr, vr], 1), pos)).to(X.dtype)
        groups = -(-C // bl.AQUA_GROUP)
        per = lambda b: 2 * C * b + 2 * groups * 16
        return out, float(m * (per(4) + (L - 1) * per(self.bits)))


class Kivi:
    """KIVI's quantizer (baselines._kivi_groups, checked against the authors' kernels): post-RoPE keys per
    channel over groups of `group` tokens, values per token over groups of `group` channels; blocks are whole
    groups of tokens (the rest waits in the buffer). Bits b per value plus 2 x 16 per group."""

    def __init__(self, bits, group):
        self.bits, self.group = bits, group
        self.granule = group

    def __call__(self, start, X):
        L, m, W = X.shape
        C, G = W // 2, self.group
        out = torch.empty_like(X)
        for l in range(L):
            x = X[l].to(torch.float64)
            rec = x.clone()
            rec[:, :C] = bl._kivi_groups(x[:, :C].T.reshape(C, m // G, G), self.bits).reshape(C, m).T
            rec[:, C:] = bl._kivi_groups(x[:, C:].reshape(m, C // G, G), self.bits).reshape(m, C)
            out[l] = to_bf16(rec).to(X.dtype)
        return out, float(L * 2 * (m * C * self.bits + (m * C // G) * 32))


class OursGpu:
    """Our closed-loop GPU codec (kv_compare.lossy_gpu, one-step predictor) on each block: the block's words
    quantized layer by layer (prediction from the block's reconstructed previous layer), coded to a real rANS
    stream; the RoPE tables start at the block's position. Bits = the stream's."""

    def __init__(self, consts, C, hd, kappa, token_step=False):
        self.consts, self.C, self.hd, self.kappa, self.token_step = consts, C, hd, kappa, token_step
        self.granule = 1

    def __call__(self, start, X):
        from .lossy_gpu import LossyLayerwise
        L, m, W = X.shape
        dev = self.consts["Aq"].device
        tb = 64
        while m % tb:
            tb //= 2
        k = dict(self.consts, cos=self.consts["cos"][start:start + m].contiguous(),
                 sin=self.consts["sin"][start:start + m].contiguous())
        codec = LossyLayerwise(k, self.C, self.hd, self.kappa, tokens_per_program=tb, chunks=1,
                               token_step=self.token_step)
        words = to_bf16(X.to(dev, torch.float64)).to(torch.bfloat16).view(torch.int16)
        rec, xall, prior = codec.quantize(words)
        _, starts, _ = codec.encode(words, rec, xall, prior)
        bits = float(codec.bits(starts, m))
        out = rec.view(torch.bfloat16).to(torch.float32).to(X.device)
        return out, bits


class HalfOnly:
    """Diagnostic: another compressor applied to the keys only ("k") or the values only ("v"); the other half
    goes back exact. Which half costs the perplexity. Bits: the wrapped compressor's, for its half only
    (half of its bits, as both halves are coded alike)."""

    def __init__(self, comp, which):
        if which not in ("k", "v"):
            raise ValueError("which is 'k' or 'v'")
        self.comp, self.which = comp, which
        self.granule = comp.granule

    def __call__(self, start, X):
        rec, bits = self.comp(start, X)
        C = X.shape[2] // 2
        out = X.clone()
        if self.which == "k":
            out[:, :, :C] = rec[:, :, :C]
        else:
            out[:, :, C:] = rec[:, :, C:]
        return out, bits / 2

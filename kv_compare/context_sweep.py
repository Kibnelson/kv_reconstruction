"""Lossless codecs across context lengths and batch sizes, as SplitZip's paper sweeps them (512 to 131K tokens,
batch 1 and 16).

One case is the KV of `batch` requests of `tokens` tokens each, coded together as one stream (positions restart
at 0 for every request, so the RoPE tables are the requests' own). Every method is a real stream, decoded on the
GPU and checked bit for bit, and timed with CUDA events after warm-up:

  ours, per layer, dense           the rANS codec with the dense one-step product (9.0 bits on 8K documents)
  ours, per layer, rank 256        the reduced-rank product, requantized in the product's epilogue (9.4 bits)
  ours, SplitZip format, rank 256  our prediction in fixed-length codes, the row kernel (10.2 bits)
  SplitZip                         the authors' codec
  ExANS                            the authors' codec (blocks of 499,712 values; a remainder is left out)

Encode times are the sender's whole work from the words (our forward pass for features and priors included).
With graph=True our decoders are also timed as one CUDA graph replay per stream (decode_ms_graph), which removes
Python's per-launch cost (about 65 us a launch); SplitZip and ExANS are timed as their authors' code launches them.
Our codecs code at most `max_stream_tokens` tokens per stream (a long case is several streams, coded and decoded
one after the other, times added): one stream's worst-case buffer must stay below 2^31 16-bit words.
"""
from __future__ import annotations

import gc

import torch

from .gpu_speed import timer as _timer


def rope_consts(study, consts, positions):
    """The codec constants with RoPE tables for these token positions (one row per token)."""
    cos, sin = study.rope.tables(positions.to(consts["cos"].device))
    return dict(consts, cos=cos.to(torch.float32).contiguous(), sin=sin.to(torch.float32).contiguous())


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


GROUP_NAMES = {2: "keys then values", 4: "4 head groups", 8: "8 head groups", 16: "head by head"}


def measure_case(study, consts, rc, words, positions, device, log=print, books=None, sz=None, exans=None,
                 rank_tiles=(128, 128, 64, 4), reps=5, max_stream_tokens=8192, grouped=None, only=None, graph=False,
                 baseline_stream_tokens=0):
    """Rows for one case: words int16 [L, T, W] on the device, positions [T]. grouped: {groups: constants of
    kv_compare.grouped.prepare_groups} for our head-group predictors; only: measure just these methods; graph:
    time our decoders as CUDA graphs too; baseline_stream_tokens: code SplitZip and ExANS in streams of this many
    tokens too, like ours (bits and times added; 0: the whole case in one call). Both codecs code independent
    chunks or blocks, so the streams change their bits by at most the remainder ExANS leaves out; they keep a 32K
    case of an 8B model within 24 GB."""
    from .fused import FusedOneStep
    from .layerwise import LayerwiseOneStep
    from .szcodec import SplitZipFormatCodec
    L, T, W = words.shape
    n = L * T * W
    k = rope_consts(study, consts, positions)
    gpu = device.type == "cuda"

    def timer(fn, device, warmup, reps):              # on the CPU (tests only): one run, no warm-up
        return _timer(fn, device, warmup=warmup if gpu else 0, reps=reps if gpu else 1)
    r = rc["rank"] if rc is not None else rank_tiles[1]

    def tiles_for(Tp):
        return rank_tiles if (Tp % (2 * rank_tiles[0]) == 0 and r % rank_tiles[1] == 0 and W % rank_tiles[2] == 0) \
            else (16, 16, 32, 4)                     # (short cases and small test models)
    spans = [(a, min(a + max_stream_tokens, T)) for a in range(0, T, max_stream_tokens)]

    def piece(a, b, base=None):
        kk = k if base is None else dict(base, cos=k["cos"], sin=k["sin"])
        return words[:, a:b].contiguous(), dict(kk, cos=kk["cos"][a:b].contiguous(), sin=kk["sin"][a:b].contiguous())
    rows = []

    def record(method, fn):
        if only is not None and method not in only:
            return
        row = {"method": method, "values": n, "tokens": T}
        try:
            row.update(fn())
            log(f"    {method}: exact={row.get('bit_exact')}, {row['bits']:.3f} bits/value, encode "
                f"{row['encode_ms']:.1f} ms, decode {row['decode_ms']:.2f} ms")
        except Exception as exc:                     # out of memory at long contexts, a shape a kernel refuses
            row["error"] = repr(exc)[:300]
            log(f"    {method}: not measured ({row['error']})")
        rows.append(row)
        _free(device)

    def ours_layerwise(rank, base=None):
        ok, bits, te_all, td_all = True, 0.0, 0.0, 0.0
        tg_all, okg, gerr = 0.0, True, None
        for a, b in spans:
            wp, kp = piece(a, b, base)
            Tp = b - a
            ng = int(kp.get("groups", 1))
            # many head groups: each launch holds one group's heads, so smaller token blocks and more chunks keep
            # enough programs in flight (a smaller block costs 32 / (layers x block) bits per value of states)
            TBc, chc = (32, 2) if ng <= 4 else (16, 4)
            while Tp % (TBc * chc):
                chc //= 2
            codec = LayerwiseOneStep(kp, study.C, study.hd, tokens_per_program=TBc, mode="uniform", search=True,
                                     chunks=chc, rank=rank, requant="epilogue" if rank else "kernel",
                                     rank_tiles=tiles_for(Tp))
            if rank is None and base is None:
                fwd = FusedOneStep(kp, study.C, study.hd, tokens_per_program=32, rows=4, warps=4)
                model = lambda: fwd.model(wp)
            else:
                model = lambda: codec.model(wp)
            xall, prior = model()
            buf, starts = codec.encode(wp, xall, prior)
            bufs = codec.buffers(Tp)
            dec = codec.decode(buf, starts, Tp, bufs=bufs)
            ok = ok and bool(torch.equal(dec, wp))
            bits += codec.bits(starts, Tp)
            tm = timer(model, device, warmup=1, reps=3)
            te = timer(lambda: codec.encode(wp, xall, prior, out=buf), device, warmup=1, reps=3)
            td = timer(lambda: codec.decode(buf, starts, Tp, out=dec, bufs=bufs), device, warmup=2, reps=reps)
            te_all, td_all = te_all + tm + te, td_all + td
            if graph and gpu and gerr is None:
                try:
                    okp, tgp = graph_decode_ms(codec, buf, starts, Tp, dec, bufs, wp, device, reps)
                    okg, tg_all = okg and okp, tg_all + tgp
                except Exception as exc:             # a capture the driver refuses: keep the plain timing
                    gerr = repr(exc)[:300]
            del codec, xall, prior, buf, starts, bufs, dec, wp, kp
            _free(device)
        out = {"bit_exact": ok, "bits": bits / n, "encode_ms": 1e3 * te_all, "decode_ms": 1e3 * td_all,
               "streams": len(spans)}
        if graph and gpu:
            out.update({"graph_error": gerr} if gerr else {"decode_ms_graph": tg_all, "bit_exact_graph": okg})
        return out

    def ours_format():
        ok, bits, te_all, td_all = True, 0.0, 0.0, 0.0
        for a, b in spans:
            wp, kp = piece(a, b)
            Tp = b - a
            codec = SplitZipFormatCodec(kp, study.C, study.hd, rank=rc, product="acc", tokens_per_program=16, warps=1,
                                        rank_tiles=tiles_for(Tp), layout="row", chunks=2)
            codec.share_books(books)
            stream, starts = codec.encode(wp)
            bufs = codec.buffers(Tp)
            dec = codec.decode(stream, starts, Tp, bufs=bufs)
            ok = ok and bool(torch.equal(dec, wp))
            bits += codec.bits(stream, starts)
            te = timer(lambda: codec.encode(wp), device, warmup=0, reps=1)
            td = timer(lambda: codec.decode(stream, starts, Tp, out=dec, bufs=bufs), device, warmup=2, reps=reps)
            te_all, td_all = te_all + te, td_all + td
            del codec, stream, starts, bufs, dec, wp, kp
            _free(device)
        return {"bit_exact": ok, "bits": bits / n, "encode_ms": 1e3 * te_all, "decode_ms": 1e3 * td_all,
                "streams": len(spans), "encoder": "PyTorch packer (not optimised)"}

    bspans = ([(a, min(a + baseline_stream_tokens, T)) for a in range(0, T, baseline_stream_tokens)]
              if baseline_stream_tokens else [(0, T)])

    def splitzip():
        ok, nbytes, te_all, td_all = True, 0, 0.0, 0.0
        for a, b in bspans:
            wp = words if len(bspans) == 1 else words[:, a:b].contiguous()
            enc = sz.encode(wp.view(torch.bfloat16))
            dec = sz.decode(enc)
            ok = ok and bool(torch.equal(dec.reshape(-1).view(torch.int16), wp.reshape(-1)))
            nbytes += enc.compressed_bytes
            te_all += timer(lambda: sz.encode(wp.view(torch.bfloat16)), device, warmup=3, reps=10)
            td_all += timer(lambda: sz.decode(enc), device, warmup=3, reps=10)
            del enc, dec, wp
            _free(device)
        return {"bit_exact": ok, "bits": 8 * nbytes / n, "encode_ms": 1e3 * te_all, "decode_ms": 1e3 * td_all,
                "streams": len(bspans)}

    def exans_run():
        from . import exans_official as ex
        ok, bits, vals, te_all, td_all = True, 0.0, 0, 0.0, 0.0
        for a, b in bspans:
            wp = words if len(bspans) == 1 else words[:, a:b].contiguous()
            r = ex.measure(exans, wp, log=lambda m: None)
            ok = ok and r["bit_exact"]
            bits += r["bits_per_value"] * r["values"]
            vals += r["values"]
            te_all += r["encode_ms_events"]
            td_all += r["decode_ms_events"]
            del wp
            _free(device)
        return {"bit_exact": ok, "bits": bits / vals, "encode_ms": te_all, "decode_ms": td_all, "values_coded": vals,
                "streams": len(bspans)}

    record("ours, per layer, dense", lambda: ours_layerwise(None))
    for g, cg in sorted((grouped or {}).items()):
        record(f"ours, per layer, {GROUP_NAMES.get(g, f'{g} head groups')}", lambda cg=cg: ours_layerwise(None, cg))
    if rc is not None:
        record("ours, per layer, rank 256", lambda: ours_layerwise(rc))
        if books is not None:
            record("ours, SplitZip format, rank 256", ours_format)
    if sz is not None:
        record("SplitZip (authors' codec)", splitzip)
    if exans is not None:
        record("ExANS (authors' codec)", exans_run)
    return rows


def best_decode_ms(row):
    """The faster exact decode of a row: plain launches, or the CUDA graph replay when that was exact too."""
    t = row["decode_ms"]
    if row.get("bit_exact_graph") and row.get("decode_ms_graph") is not None:
        t = min(t, row["decode_ms_graph"])
    return t


def exponent_hist(words):
    """Counts of the 256 BF16 exponents of int16 words (any shape), float64, on the words' device."""
    h = torch.zeros(256, dtype=torch.float64, device=words.device)
    flat = words.reshape(-1)
    for a in range(0, flat.numel(), 1 << 27):        # in pieces: no int32 copy of a whole case
        e = (flat[a:a + (1 << 27)].to(torch.int32) >> 7) & 0xFF
        h += torch.bincount(e, minlength=256).to(torch.float64)
    return h


def exponent_stats(words, book_words=None, top=(8, 16), book_hist=None):
    """SplitZip's code-book statistics of BF16 words (int16, any shape): the share of values whose 8-bit exponent
    is among the `k` most frequent exponents of book_words, or of the counts book_hist (A->B; SplitZip calibrates
    on WikiText-2 train), and of the words themselves (B->B, the oracle), and the exponent entropy in bits."""
    hb = exponent_hist(words)
    ha = book_hist if book_hist is not None else (exponent_hist(book_words) if book_words is not None else None)
    if ha is not None:
        ha = ha.to(hb.device)
    total = hb.sum()
    p = hb / total
    out = {"values": int(total.item()),
           "exponent_entropy_bits": float(-(p[p > 0] * torch.log2(p[p > 0])).sum().item()),
           "distinct_exponents": int((hb > 0).sum().item())}
    for k in top:
        out[f"top{k}_coverage_self"] = float((hb[torch.topk(hb, k).indices].sum() / total).item())
        if ha is not None:
            out[f"top{k}_coverage_calibrated"] = float((hb[torch.topk(ha, k).indices].sum() / total).item())
    return out


def link_ms(bits, values, decode_ms, gbps):
    """Transfer time on an ideal link with decoding overlapped with arrival: the longer of the two."""
    wire = bits * values / (gbps * 1e9) * 1e3
    return max(wire, decode_ms)


def graph_decode_ms(codec, buf, starts, T, dec, bufs, words, device, reps=5):
    """The whole decode captured once as a CUDA graph and replayed (no per-kernel launch cost from Python):
    (exact after replay, median milliseconds)."""
    g = torch.cuda.CUDAGraph()
    cur = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    side.wait_stream(cur)
    with torch.cuda.stream(side):                    # warm-up outside the capture
        codec.decode(buf, starts, T, out=dec, bufs=bufs)
    cur.wait_stream(side)
    with torch.cuda.graph(g):
        codec.decode(buf, starts, T, out=dec, bufs=bufs)
    dec.zero_()
    g.replay()
    torch.cuda.synchronize(device)
    ok = bool(torch.equal(dec, words))
    td = _timer(g.replay, device, warmup=2, reps=reps)
    del g
    return ok, 1e3 * td


def group_config_rows(study, consts, grouped, words, positions, device, configs, log=print, reps=5, graph=True):
    """Our per-layer codec with the one-step predictor (groups 1) and each head-group predictor, for every
    (tokens per program, token chunks) in configs: bits and decode time of a real stream, decoded bit-exact. Many
    chunks decode on separate CUDA streams side by side, which keeps the GPU busy while a head group waits for
    the groups before it."""
    from .fused import FusedOneStep
    from .layerwise import LayerwiseOneStep
    L, T, W = words.shape
    n = L * T * W
    gpu = device.type == "cuda"
    rows = []
    for g, base in [(1, consts)] + sorted(grouped.items()):
        k = rope_consts(study, base, positions)
        for TB, ch in configs:
            row = {"groups": g, "tokens_per_program": TB, "chunks": ch, "tokens": T, "values": n}
            if T % (TB * ch):
                continue
            try:
                codec = LayerwiseOneStep(k, study.C, study.hd, tokens_per_program=TB, mode="uniform", search=True,
                                         chunks=ch)
                if g == 1:
                    fwd = FusedOneStep(k, study.C, study.hd, tokens_per_program=32, rows=4, warps=4)
                    xall, prior = fwd.model(words)
                else:
                    xall, prior = codec.model(words)
                buf, starts = codec.encode(words, xall, prior)
                bufs = codec.buffers(T)
                dec = codec.decode(buf, starts, T, bufs=bufs)
                ok = bool(torch.equal(dec, words))
                td = _timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs), device,
                            warmup=2 if gpu else 0, reps=reps if gpu else 1)
                row.update(bit_exact=ok, bits=codec.bits(starts, T) / n, decode_ms=1e3 * td)
                if graph and gpu:
                    try:
                        okg, tg = graph_decode_ms(codec, buf, starts, T, dec, bufs, words, device, reps)
                        row.update(bit_exact_graph=okg, decode_ms_graph=tg)
                    except Exception as exc:         # a capture the driver refuses: keep the plain timing
                        row["graph_error"] = repr(exc)[:300]
                        log(f"    groups {g}, {TB}x{ch}: CUDA graph not measured ({row['graph_error']})")
                log(f"    groups {g}, {TB} tokens per program, {ch} chunks: exact={ok}, {row['bits']:.3f} bits/value, "
                    f"decode {1e3 * td:.2f} ms" + (f", as a CUDA graph {row['decode_ms_graph']:.2f} ms "
                                                   f"(exact={row['bit_exact_graph']})" if "decode_ms_graph" in row else ""))
                del codec, xall, prior, buf, starts, bufs, dec
            except Exception as exc:
                row["error"] = repr(exc)[:300]
                log(f"    groups {g}, {TB}x{ch}: not measured ({row['error']})")
            rows.append(row)
            _free(device)
    return rows

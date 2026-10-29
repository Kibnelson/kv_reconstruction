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


def measure_case(study, consts, rc, words, positions, device, log=print, books=None, sz=None, exans=None,
                 rank_tiles=(128, 128, 64, 4), reps=5, max_stream_tokens=8192):
    """Rows for one case: words int16 [L, T, W] on the device, positions [T]."""
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

    def piece(a, b):
        return words[:, a:b].contiguous(), dict(k, cos=k["cos"][a:b].contiguous(), sin=k["sin"][a:b].contiguous())
    rows = []

    def record(method, fn):
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

    def ours_layerwise(rank):
        ok, bits, te_all, td_all = True, 0.0, 0.0, 0.0
        for a, b in spans:
            wp, kp = piece(a, b)
            Tp = b - a
            codec = LayerwiseOneStep(kp, study.C, study.hd, tokens_per_program=32, mode="uniform", search=True,
                                     chunks=2, rank=rank, requant="epilogue" if rank else "kernel",
                                     rank_tiles=tiles_for(Tp))
            if rank is None:
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
            del codec, xall, prior, buf, starts, bufs, dec, wp, kp
            _free(device)
        return {"bit_exact": ok, "bits": bits / n, "encode_ms": 1e3 * te_all, "decode_ms": 1e3 * td_all,
                "streams": len(spans)}

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

    def splitzip():
        enc = sz.encode(words.view(torch.bfloat16))
        dec = sz.decode(enc)
        ok = bool(torch.equal(dec.reshape(-1).view(torch.int16), words.reshape(-1)))
        te = timer(lambda: sz.encode(words.view(torch.bfloat16)), device, warmup=3, reps=10)
        td = timer(lambda: sz.decode(enc), device, warmup=3, reps=10)
        return {"bit_exact": ok, "bits": 8 * enc.compressed_bytes / n, "encode_ms": 1e3 * te, "decode_ms": 1e3 * td}

    def exans_run():
        from . import exans_official as ex
        r = ex.measure(exans, words, log=lambda m: None)
        return {"bit_exact": r["bit_exact"], "bits": r["bits_per_value"], "encode_ms": r["encode_ms_events"],
                "decode_ms": r["decode_ms_events"], "values_coded": r["values"]}

    record("ours, per layer, dense", lambda: ours_layerwise(None))
    if rc is not None:
        record("ours, per layer, rank 256", lambda: ours_layerwise(rc))
        if books is not None:
            record("ours, SplitZip format, rank 256", ours_format)
    if sz is not None:
        record("SplitZip (authors' codec)", splitzip)
    if exans is not None:
        record("ExANS (authors' codec)", exans_run)
    return rows


def link_ms(bits, values, decode_ms, gbps):
    """Transfer time on an ideal link with decoding overlapped with arrival: the longer of the two."""
    wire = bits * values / (gbps * 1e9) * 1e3
    return max(wire, decode_ms)

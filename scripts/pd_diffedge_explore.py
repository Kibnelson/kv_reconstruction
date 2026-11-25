#!/usr/bin/env python3
"""What else predicts a moved page's residual: numpy over diffedge's sample (diffedge_sample.npz), on a CPU.

  python3 scripts/pd_diffedge_explore.py SAMPLE.npz [--out explore.md]

The sample holds pages with their latest, previous and first versions (and twins with D's copy): a few layers, head
0's keys and values. Keys are turned by RoPE to the page's place, as the stage does. The residual r = target - turned
latest. Fractions are of the residual's energy (sum of squares) left after each composite step, on held-out pages
(the sample's pages split by conversation, even ones to fit):

  second order   r against l (latest - previous), l per layer and K/V (pages with two versions or more)
  cross-layer    r at layer l+1 from r at layer l by a linear map (head_dim x head_dim, ridge), the pairs sampled
  token lag      r at token t from r at token t-1, per channel
  low rank       r projected on the top k directions of the training pages' residuals (what a rank-k delta leaves)
  by agreement   |r| against |target| by agreement bucket
  twins          words equal to D's, and the residual's energy against the page's
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

BUCKETS = (0, 16, 64, 256, 1024)
BLABEL = ("0-15", "16-63", "64-255", "256-1023", "1024+")


def f32(words):
    """BF16 words (int16) -> float32."""
    return (np.asarray(words).astype(np.int32).astype(np.uint32) << 16).view(np.float32)


def turn(keys, shift, inv, hd):
    """keys [..., hd] float32 turned on by shift positions (rotate-half RoPE), float32 like the stage."""
    ang = (float(shift) * np.asarray(inv, dtype=np.float64)).astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = keys[..., :hd // 2], keys[..., hd // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def load(path):
    z = np.load(path)
    hd = int(z["head_dim"])
    inv = z["rope_inv_freq"]
    layers = [int(x) for x in z["layers"]]
    out = {"hd": hd, "layers": layers, "moved": None, "twin": None}
    if "moved_target" in z:
        meta = json.loads(str(z["moved_meta"]))
        T = f32(z["moved_target"])                                 # [n, layers, 16, 2 hd]
        V = {k: f32(z[f"moved_{k}"]) for k in ("latest", "previous", "first")}
        sh = z["moved_shift"]                                      # [n, 3]: latest, previous, first
        for j, k in enumerate(("latest", "previous", "first")):
            if len(inv):
                for i in range(len(T)):
                    V[k][i, :, :, :hd] = turn(V[k][i, :, :, :hd], sh[i, j], inv, hd)
        out["moved"] = {"T": T, "V": V, "meta": meta, "agree": z["moved_agree"][:, 0]}   # agreement with latest
    if "twin_target" in z:
        out["twin"] = {"T": f32(z["twin_target"]), "D": f32(z["twin_d"]), "meta": json.loads(str(z["twin_meta"]))}
    return out


def split(meta):
    """Pages to fit (even conversations of the sample) and to test (the rest; all if only one conversation)."""
    convs = sorted({m["conv"] for m in meta})
    fit = {c for k, c in enumerate(convs) if k % 2 == 0}
    a = np.array([m["conv"] in fit for m in meta])
    if a.all() or not a.any():
        a = np.arange(len(meta)) % 2 == 0
    return a, ~a


def energy(x):
    return float(np.sum(np.square(x, dtype=np.float64)))


def left(r, pred):
    """Energy of r left after removing pred, as a fraction."""
    e = energy(r)
    return energy(r - pred) / e if e > 0 else None


def analyse(s):
    hd, layers = s["hd"], s["layers"]
    res = {"layers": layers, "head_dim": hd}
    m = s["moved"]
    if m is not None:
        T, V, meta = m["T"], m["V"], m["meta"]
        r = T - V["latest"]                                        # [n, Lsel, 16, 2 hd]
        d = V["latest"] - V["previous"]
        tr, te = split(meta)
        two = np.array([x["versions"] >= 2 for x in meta])
        res["pages"] = {"all": len(meta), "fit": int(tr.sum()), "test": int(te.sum()), "two_versions": int(two.sum())}
        kv = {"K": slice(0, hd), "V": slice(hd, 2 * hd)}
        rows = []
        for li, l in enumerate(layers):
            for k, sl in kv.items():
                x = r[:, li, :, sl]
                row = {"layer": l, "kv": k, "rms_target": float(np.sqrt(np.mean(T[:, li, :, sl] ** 2))),
                       "rms_residual": float(np.sqrt(np.mean(x ** 2))),
                       "rms_first": float(np.sqrt(np.mean((T - V["first"])[:, li, :, sl] ** 2)))}
                # second order: l fitted on the fit pages with two versions, left on the test ones
                a, b = tr & two, te & two
                if a.any() and b.any():
                    dd = d[a][:, li, :, sl]
                    lam = float(np.sum(x[a] * dd) / max(np.sum(dd * dd), 1e-30))
                    row["second_order_l"] = lam
                    row["second_order_left"] = left(x[b], lam * d[b][:, li, :, sl])
                # token lag: r_t from r_{t-1}, per channel
                if te.any():
                    p0, p1 = x[tr][:, :-1].reshape(-1, x.shape[-1]), x[tr][:, 1:].reshape(-1, x.shape[-1])
                    beta = np.sum(p0 * p1, 0) / np.maximum(np.sum(p0 * p0, 0), 1e-30)
                    q0, q1 = x[te][:, :-1], x[te][:, 1:]
                    row["token_lag_left"] = left(q1, beta * q0)
                # low rank: top k directions of the fit pages' residuals
                if tr.sum() >= 2 and te.any():
                    A = x[tr].reshape(-1, x.shape[-1]).astype(np.float64)
                    _u, _s, vt = np.linalg.svd(A, full_matrices=False)
                    B = x[te].reshape(-1, x.shape[-1]).astype(np.float64)
                    for kk in (1, 4, 16):
                        P = vt[:kk]
                        row[f"rank{kk}_left"] = left(B, B @ P.T @ P)
                rows.append(row)
        res["by_layer"] = rows
        # cross-layer: r at l+1 from r at l (a linear map per K/V), for the sampled pairs
        pairs = [(i, j) for i, j in zip(range(len(layers)), range(1, len(layers))) if layers[j] == layers[i] + 1]
        cl = []
        for i, j in pairs:
            for k, sl in kv.items():
                X = r[tr][:, i, :, sl].reshape(-1, hd).astype(np.float64)
                Y = r[tr][:, j, :, sl].reshape(-1, hd).astype(np.float64)
                if len(X) < hd or not te.any():
                    continue
                lamr = 1e-3 * np.trace(X.T @ X) / hd + 1e-12        # a residual can be all zero (layer 0's V)
                Wm = np.linalg.lstsq(X.T @ X + lamr * np.eye(hd), X.T @ Y, rcond=None)[0]
                Xt = r[te][:, i, :, sl].reshape(-1, hd)
                Yt = r[te][:, j, :, sl].reshape(-1, hd)
                cl.append({"layers": f"{layers[i]}->{layers[j]}", "kv": k, "left": left(Yt, Xt @ Wm)})
        res["cross_layer"] = cl
        # by agreement bucket: the residual against the page
        ag = np.searchsorted(np.asarray(BUCKETS[1:]), m["agree"], side="right")      # [n, 16]
        byb = []
        for q, lab in enumerate(BLABEL):
            sel = ag == q
            if sel.any():
                rr = np.moveaxis(r, 2, 1)[sel]                         # tokens of the bucket: [n_tok, Lsel, 2 hd]
                tt = np.moveaxis(T, 2, 1)[sel]
                byb.append({"bucket": lab, "tokens": int(sel.sum()),
                            "K": float(np.sqrt(energy(rr[..., :hd]) / max(energy(tt[..., :hd]), 1e-30))),
                            "V": float(np.sqrt(energy(rr[..., hd:]) / max(energy(tt[..., hd:]), 1e-30)))})
        res["by_agreement"] = byb
    t = s["twin"]
    if t is not None:
        res["twins"] = [{"layer": l, "equal_words": float(np.mean(t["T"][:, li] == t["D"][:, li])),
                         "rms_ratio": float(np.sqrt(energy(t["T"][:, li] - t["D"][:, li]) /
                                                    max(energy(t["T"][:, li]), 1e-30)))}
                        for li, l in enumerate(s["layers"])]
        res["twin_pages"] = len(t["meta"])
    return res


def markdown(res):
    f = lambda x: "-" if x is None else f"{x:.3f}"
    out = ["# Diff-edge sample: what else predicts a moved page's residual", ""]
    p = res.get("pages")
    if p:
        out += [f"{p['all']} moved pages ({p['two_versions']} with two versions or more); fitted on {p['fit']}, tested "
                f"on {p['test']}. Residual: target - latest version (keys turned). Fractions: the residual's energy "
                "left on the test pages.", "",
                "| Layer | K/V | rms page | rms residual | rms vs first | second order l | left | token lag left | "
                "rank 1 / 4 / 16 left |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in res["by_layer"]:
            out.append(f"| {r['layer']} | {r['kv']} | {f(r['rms_target'])} | {f(r['rms_residual'])} | {f(r['rms_first'])} "
                       f"| {f(r.get('second_order_l'))} | {f(r.get('second_order_left'))} | {f(r.get('token_lag_left'))} "
                       f"| {f(r.get('rank1_left'))} / {f(r.get('rank4_left'))} / {f(r.get('rank16_left'))} |")
        if res.get("cross_layer"):
            out += ["", "Cross-layer: the residual at a layer from the one below (a linear map), energy left:", "",
                    "| Layers | K/V | left |", "| --- | --- | ---: |"]
            for c in res["cross_layer"]:
                out.append(f"| {c['layers']} | {c['kv']} | {f(c['left'])} |")
        if res.get("by_agreement"):
            out += ["", "By agreement with the latest version: rms residual / rms page", "",
                    "| Bucket | tokens | K | V |", "| --- | ---: | ---: | ---: |"]
            for b in res["by_agreement"]:
                out.append(f"| {b['bucket']} | {b['tokens']:,} | {f(b['K'])} | {f(b['V'])} |")
    if res.get("twins"):
        out += ["", f"Twins ({res['twin_pages']} pages): words equal to D's copy, and rms difference / rms page", "",
                "| Layer | equal | rms ratio |", "| ---: | ---: | ---: |"]
        for t in res["twins"]:
            out.append(f"| {t['layer']} | {100 * t['equal_words']:.1f}% | {f(t['rms_ratio'])} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sample")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = analyse(load(a.sample))
    text = markdown(res)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        with open(a.out.rsplit(".", 1)[0] + ".json", "w") as fh:
            json.dump(res, fh, indent=1)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

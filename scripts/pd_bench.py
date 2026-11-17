#!/usr/bin/env python3
"""The vLLM PD benchmark driver (docs/pd_protocol.md), run in the vLLM environment against running engines and the
proxy (scripts/run_pd_vllm.sh starts them). Stages (--stages, in order):

  check       correctness: the check prompts through every codec with checksums on (each transfer verified
              bit-exact on the decode GPU), greedy outputs compared with the same prompts served without a transfer
  sweep       phase 1, everything sent: for every link speed x codec x request rate, vllm bench serve through the
              proxy (TTFT, TPOT, throughput, goodput), plus the connector's per-transfer records of that run (bits,
              gather / encode / link / install times, the link speed achieved)
  colocated   the same prompts and rates without disaggregation (each decode engine prefills its own requests)
  multiturn   vLLM's multi-turn benchmark through the proxy with conversation routing: the requests' block hashes
              in the connector log feed scripts/pd_opportunity.py (reuse opportunities before any base sending)
  tier        colocated engines with a CPU tier (kv_compare.pd.tier, or LMCache's): for every tier codec x number
              of clients, the tiers emptied (a new generation in --tier-control; vLLM's reset_prefix_cache with
              reset_external on every engine), then vLLM's multi-turn benchmark on fresh conversations (a seed per
              point) with conversation routing; TTFT, TPOT, latency, and the tier's loads and stores
  report      results.json and summary.md

Links: a number is a paced link in Gb/s; "native" leaves the transport unpaced (the machine's own path: the host
ring, or GPU to GPU with the p2p transport). Codec "nixl": vLLM's NixlConnector through the proxy's nixl mode (no
codec, the production reference); "lmcache" in the tier stage: LMCache's CPU offload.

Every run's raw result stays in <out>/runs/; a rerun skips runs already there (--rerun to redo).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def post(url, obj, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


class RunStopped(Exception):
    """An engine stopped answering, or a point's requests never finished: the rest of the run would measure nothing.
    Job 30334: exactkv's install hit an illegal memory access on all three decode GPUs at 10:38 (128-token prompts,
    16 at once); every later point then waited 30 minutes for requests that the dead engines would never finish."""

    def __init__(self, why, engines=None):
        super().__init__(why)
        self.why, self.engines = why, engines or {}


IDLE_TIMEOUT = 600      # s a point's requests may stay in flight after its benchmark ended (--idle-timeout)


def wait_idle(proxy, timeout=None, health=None, every=30):
    """True once the proxy has no request in flight; False after timeout (IDLE_TIMEOUT). health(): {engine: why}
    of engines not answering, asked every `every` s while requests stay in flight; any stops the run then."""
    t = time.time()
    end = t + (IDLE_TIMEOUT if timeout is None else timeout)
    ask = t + every
    while time.time() < end:
        try:
            st = get(proxy + "/t/status")
        except Exception as exc:
            raise RunStopped(f"the proxy does not answer ({exc!r})"[:300], {"proxy": repr(exc)[:200]})
        if sum(st["inflight"]) + sum(st.get("prefill_inflight") or [0]) == 0:
            return True
        if health is not None and time.time() >= ask:
            bad = health()
            if bad:
                raise RunStopped("engines stopped answering while requests were in flight", bad)
            ask = time.time() + every
        time.sleep(1)
    return False


def drained(a):
    return wait_idle(a.proxy, health=lambda: dead_engines(a))


def after_point(a):
    """(drained, why the run stops or None), without raising: the point is recorded first."""
    try:
        return drained(a), None
    except RunStopped as exc:
        return False, exc


def end_point(a, name, ok, stop):
    if stop is not None:
        raise RunStopped(f"{name}: {stop.why}", stop.engines)
    if not ok:
        stuck(a, name)


def engine_urls(a):
    st = get(a.proxy + "/t/status")
    return list(dict.fromkeys(list(st.get("prefills") or []) + [d[0] for d in st["decodes"]]))


def dead_engines(a):
    """{engine url: why} for every engine whose GET /health is not 200 (vLLM answers 503 once its engine core is
    dead, and nothing once its server has exited too)."""
    try:
        urls = engine_urls(a)
    except Exception as exc:
        return {"proxy": repr(exc)[:200]}
    bad = {}
    for u in urls:
        try:
            with urllib.request.urlopen(u + "/health", timeout=30) as r:
                if r.status != 200:
                    bad[u] = f"HTTP {r.status}"
        except Exception as exc:
            bad[u] = repr(exc)[:200]
    return bad


def ensure_alive(a, where):
    bad = dead_engines(a)
    if bad:
        raise RunStopped(f"engines not answering {where}", bad)


def settle(a, name):
    """After a point: its requests finished (True), or the run stops (an engine died, or still in flight after
    IDLE_TIMEOUT s)."""
    if drained(a):
        return True
    stuck(a, name)


def stopped_note(s):
    """One line for the reports: where the run stopped and why."""
    eng = s.get("engines") or {}
    return (f"**The run stopped early** in its {s.get('stage')} stage at "
            f"{time.strftime('%H:%M:%S', time.localtime(s.get('time') or 0))}: {s.get('why')}"
            + (f" (not answering: {', '.join(sorted(eng))})" if eng else "")
            + ". The points before are measured; the rest were not run.")


def stuck(a, name):
    """The run stops: name's requests were still in flight IDLE_TIMEOUT s after its benchmark ended."""
    bad = dead_engines(a)
    raise RunStopped(f"{name}: requests still in flight {IDLE_TIMEOUT} s after its benchmark ended"
                     + ("" if bad else " (every engine answers)"), bad)


def reset_caches(a, tries=30):
    """Every engine's prefix cache emptied (vLLM's POST /reset_prefix_cache; the engines run with
    VLLM_SERVER_DEV_MODE=1), so each codec and rate starts cold: with the same prompts, a later codec's prefill
    otherwise hit its own cache (T2 then has no steps for layer 0) and the decode engines held most prompts already
    (job 30210: t2-rank256 moved 8 or 16 tokens of 2048). A reset fails while an engine still holds blocks (a
    transfer finishing), so it is tried again for a while."""
    st = get(a.proxy + "/t/status")
    urls = list(dict.fromkeys(list(st.get("prefills") or []) + [d[0] for d in st["decodes"]]))
    out = {}
    for u in urls:
        r = None
        for _ in range(tries):
            try:
                r = post(u + "/reset_prefix_cache", {}, timeout=120)
            except Exception as exc:
                r = {"error": repr(exc)[:200]}
            if isinstance(r, dict) and r.get("success", True) is not False and "error" not in r:
                break
            time.sleep(1)
        out[u] = r
    bad = {u: r for u, r in out.items() if not (isinstance(r, dict) and r.get("success", True) is not False
                                                and "error" not in r)}
    if bad:
        log(f"prefix caches not reset: {bad}")
    return {"engines": len(out), "failed": bad}


def warm_pairs(a):
    """One short request over every prefill-decode pair in the current mode, before a sweep point's caches are
    emptied: a pair's first transfer opens its path (CUDA IPC handles, NIXL's handshake; 0.2-0.5 s in job 30210),
    and the check stage, one request at a time, uses the first prefill engine only. No sampling parameters: the
    server's defaults, as vllm bench serve sends them (warm_burst)."""
    st = get(a.proxy + "/t/status")
    n = 0
    for j in range(len(st.get("prefills") or [None])):
        for i in range(len(st["decodes"])):
            try:
                post(a.proxy + "/v1/completions", {"model": a.served_name, "prompt": "warm up " * 40 + str(n),
                                                   "max_tokens": 2, "t_prefill": j, "t_decode": i}, timeout=600)
            except Exception as exc:
                log(f"warm-up of prefill {j} -> decode {i} failed: {exc!r}"[:300])
            n += 1
    drained(a)
    return n


GREEDY = ["--temperature", "0"]


def greedy_args():
    """vllm bench serve's sampling: greedy. vLLM 0.32's bench no longer sends temperature 0 by default ("WARNING: vllm
    bench serve no longer sets temperature==0 (greedy)", in every sweep log of job 30252), so the server's default
    applied: the model's generation config, temperature 0.6 and top-p 0.9. Its top-p kernels compiled at first use, on
    each engine (jobs 30226, 30252), and each codec's run sampled its own text. Greedy needs no top-p kernel, and every
    codec writes the same text."""
    return GREEDY


def warm_burst(a, n=16, greedy=False):
    """n short requests at once in the current mode, sampled as the next point samples: greedy before a sweep
    (greedy_args), the server's default (the model's generation config: top-p) before the tier and multi-turn runs,
    which send no temperature. In job 30226 the first measured point of every run compiled vLLM's
    top-p kernels (Triton, about 2 s: "JIT compilation during inference"): the check stage samples greedily, so
    nothing before had used them; the burst also covers batched sampling."""
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        try:
            body = {"model": a.served_name, "prompt": f"warm-up burst {i} " * 30, "max_tokens": 8}
            if greedy:
                body["temperature"] = 0.0
            post(a.proxy + "/v1/completions", body, timeout=600)
            return None
        except Exception as exc:
            return repr(exc)[:200]
    with ThreadPoolExecutor(n) as ex:
        errs = [e for e in ex.map(one, range(n)) if e]
    if errs:
        log(f"warm-up burst: {len(errs)} of {n} failed: {errs[0]}")
    drained(a)


JIT_RE = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[jit_monitor\.py:\d+\] (.*)")


def log_marks(a):
    """How far each engine log has been written ({log name: bytes}): taken at a point's start and end, it says which
    log lines the point wrote, more precisely than the logs' whole-second times."""
    if not getattr(a, "engine_logs", None):
        return {}
    out = {}
    for f in Path(a.engine_logs).glob("engine_*.log"):
        try:
            out[f.name] = f.stat().st_size
        except OSError:
            pass
    return out


def jit_events(a):
    """vLLM's warnings of a kernel compiled while serving ("JIT compilation during inference"), from the engine logs:
    [(time, engine log, text, byte offset in the log)]."""
    out = []
    if not getattr(a, "engine_logs", None):
        return out
    year = time.localtime().tm_year
    for f in sorted(Path(a.engine_logs).glob("engine_*.log")):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for m in JIT_RE.finditer(text):
            if "JIT compilation during inference" not in m[6]:
                continue
            t = time.mktime((year, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), 0, 0, -1))
            out.append((t, f.name, m[6][:160], len(text[:m.start()].encode())))
    return sorted(out, key=lambda e: (e[0], e[1], e[3]))     # in the order written


def prefix_metrics(urls):
    """The engines' prefix-cache counters (Prometheus /metrics: every vllm:*prefix_cache* series, summed over its
    labels and engines): vLLM's own hits and queries, GPU and external (a connector's: our tier, LMCache)."""
    tot = {}
    for u in urls:
        try:
            with urllib.request.urlopen(u + "/metrics", timeout=30) as r:
                text = r.read().decode(errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            if line.startswith("vllm:") and "prefix_cache" in line and not line.startswith("#"):
                name, _, val = line.rpartition(" ")
                key = name.split("{", 1)[0]
                try:
                    tot[key] = tot.get(key, 0.0) + float(val)
                except ValueError:
                    pass
    return tot


def hit_rates(before, after):
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}

    def rate(kind):
        q = [v for k, v in d.items() if kind in k and "queries" in k]
        h = [v for k, v in d.items() if kind in k and "hits" in k]
        return (100 * sum(h) / sum(q)) if q and sum(q) > 0 else None
    gpu_q = [v for k, v in d.items() if "external" not in k and "queries" in k]
    gpu_h = [v for k, v in d.items() if "external" not in k and "hits" in k]
    return {"counters": d, "gpu_hit_pct": (100 * sum(gpu_h) / sum(gpu_q)) if gpu_q and sum(gpu_q) > 0 else None,
            "external_hit_pct": rate("external")}


def records(log_dir, t0, t1):
    """Connector records (send: prefill, install/timeout: decode) whose time falls in [t0, t1]."""
    out = []
    for p in glob.glob(os.path.join(log_dir, "tkv_*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("finished_at") or r.get("taken_at") or r.get("time")
                if t is not None and t0 <= t <= t1:
                    r.setdefault("log", os.path.basename(p))     # which engine's connector wrote it
                    out.append(r)
    return out


def wire_gbps(sends):
    """The link's throughput while it carried data: each sending engine's bytes over the union of its transfers' times
    (one link per prefill engine), the engines' rates added. A transfer's own rate ("achieved") is lower whenever
    several share the link: one prefill engine feeding three decode engines gave 3.7 Gb/s each on a 10 Gb/s link
    that carried 9.99 (job 30280)."""
    by = {}
    for r in sends:
        if r.get("send_begin") is not None and r.get("send_end") is not None and r["send_end"] > r["send_begin"] \
                and r.get("blob_bytes"):
            by.setdefault(r.get("log"), []).append((r["send_begin"], r["send_end"], r["blob_bytes"]))
    total = 0.0
    for spans in by.values():
        spans.sort()
        busy, (a, b) = 0.0, spans[0][:2]
        for s, e, _ in spans[1:]:
            if s > b:
                busy, a, b = busy + (b - a), s, e
            else:
                b = max(b, e)
        busy += b - a
        total += 8 * sum(n for _, _, n in spans) / busy / 1e9
    return total if by else None


def loaded(log_dir):
    """(codecs every engine loaded, {codec: why one did not}) from the connectors' "ready" records; (None, {}) when
    there are none (vLLM's own connectors)."""
    have, why = None, {}
    for p in glob.glob(os.path.join(log_dir, "tkv_*w_*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") != "ready":
                    continue
                cs = set(r.get("codecs") or [])
                have = cs if have is None else have & cs
                why.update({k: str(v) for k, v in (r.get("codec_errors") or {}).items()})
                why.update({k: str(v) for k, v in (r.get("warm_up") or {}).items() if v is not True})
    return have, why


def usable(a, res, codecs):
    """The codecs to run: those every engine loaded (nixl, lmcache, none always); the rest in res["skipped"]."""
    have, why = loaded(a.log_dir)
    if have is None:
        return list(codecs)
    keep = []
    for c in codecs:
        if c in ("nixl", "lmcache", "none") or c in have:
            keep.append(c)
        else:
            res.setdefault("skipped", {})[c] = why.get(c, "not loaded by every engine")
            log(f"{c}: skipped ({res['skipped'][c][:160]})")
    return keep


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def transfer_stats(recs):
    s = [r for r in recs if r.get("event") == "send"]
    d = [r for r in recs if r.get("event") == "install"]
    tmo = [r for r in recs if r.get("event") == "timeout"]
    ach = [8 * r["blob_bytes"] / (r["send_end"] - r["send_begin"]) / 1e9 for r in s
           if r.get("send_end") and r.get("send_begin") and r["send_end"] > r["send_begin"]]
    l0 = [r for r in s if r.get("layer0")]
    sets = {}
    for r in s:
        if r.get("set"):
            sets[r["set"]] = sets.get(r["set"], 0) + 1
    return {"sends": len(s), "send_errors": sum(1 for r in s if r.get("error")),
            "installs": len(d), "install_errors": sum(1 for r in d if r.get("error")), "timeouts": len(tmo),
            "bits": mean([r.get("bits") for r in s]), "gather_ms": mean([r.get("gather_ms") for r in s]),
            "encode_ms": mean([r.get("encode_ms") for r in s]),
            "link_ms": mean([1e3 * r["link_s"] for r in s if r.get("link_s") is not None]),
            "queue_ms": mean([1e3 * (r["send_begin"] - r["queued"]) for r in s if r.get("send_begin")]),
            "achieved_gbps": mean(ach), "wire_gbps": wire_gbps(s), "rel_err": mean([r.get("rel_err") for r in s]),
            "install_ms": mean([r.get("decode_install_ms") for r in d]),
            "wait_ms": mean([1e3 * (r["done_at"] - r["admitted_at"]) for r in d if r.get("admitted_at")]),
            "exact": sum(1 for r in d if r.get("exact") is True), "inexact": sum(1 for r in d if r.get("exact") is False),
            "spilled_to_host": sum(1 for r in s if r.get("queued_on") == "host"),
            "layer0_recomputed": sum(1 for r in l0 if r["layer0"] == "recomputed") if l0 else None,
            "layer0_coded": sum(1 for r in l0 if r["layer0"] == "coded") if l0 else None,
            "layer0_checks": sorted({str(r.get("layer0_check")) for r in l0})[:5] if l0 else None,
            "sets": sets or None}


def bench_serve(a, dataset, rate, n, out_dir, name, concurrency=None, output_len=None):
    vllm = os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = [vllm if os.path.exists(vllm) else "vllm", "bench", "serve", "--backend", "openai", "--base-url", a.proxy, "--endpoint", "/v1/completions",
           "--model", a.model, "--served-model-name", a.served_name, "--tokenizer", a.model,
           "--dataset-name", "custom", "--dataset-path", dataset,
           "--custom-output-len", str(output_len or a.output_len)] + \
          (["--max-concurrency", str(concurrency)] if concurrency else []) + [
           "--skip-chat-template", "--num-prompts", str(n), "--request-rate", str(rate), "--ignore-eos",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--goodput", f"ttft:{a.ttft_slo_ms}", "--seed", "0", "--disable-tqdm",
           "--save-result", "--result-dir", str(out_dir), "--result-filename", name + ".json"] + greedy_args()
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=getattr(a, "point_timeout", None))
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as exc:        # killed: its requests are aborted with the client
        rc = 124
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = ((exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")) \
            + f"\n[stopped after --point-timeout {a.point_timeout} s]"
        log(f"{name}: the benchmark ran past --point-timeout {a.point_timeout} s; stopped")
    t1 = time.time()
    (out_dir / (name + ".log")).write_text(out[-20000:] + "\n" + err[-20000:])
    res = {}
    p = out_dir / (name + ".json")
    if p.is_file():
        res = json.loads(p.read_text())
    return rc, res, t0, t1


KEEP = ("completed", "duration", "request_throughput", "request_goodput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
        "p99_tpot_ms", "mean_e2el_ms", "p99_e2el_ms", "failed")


def stage_check(a, res):
    rows = [json.loads(l) for l in open(a.prompts / "check.jsonl")]
    out = {}
    ref = None
    post(a.proxy + "/t/config", {"mode": "local"})
    reset_caches(a)
    ref = [completion(a, r["prompt"], r["output_tokens"]) for r in rows]
    for codec in usable(a, res, a.codecs):
        ensure_alive(a, f"before the check of {codec}")
        post(a.proxy + "/t/config", {"mode": "nixl"} if codec == "nixl" else {"mode": "pd", "codec": codec,
                                                                                "verify": True})
        settle(a, f"check {codec}")
        resets = reset_caches(a)
        t0 = time.time()
        got = []
        for r in rows:
            try:
                got.append(completion(a, r["prompt"], r["output_tokens"]))
            except Exception as exc:
                got.append({"error": repr(exc)[:300]})
        idle, stop = after_point(a)
        time.sleep(1)
        st = transfer_stats(records(a.log_dir, t0, time.time()))
        same = sum(1 for x, y in zip(got, ref) if x.get("text") is not None and x.get("text") == y.get("text"))
        if codec == "nixl":                          # vLLM's connector: no checksums; the outputs and no errors
            ok = not any("error" in g for g in got)
        elif is_lossy(codec):                     # AQUA-KV: every transfer installed; its error, not exactness
            ok = st["installs"] == len(rows) and not st["install_errors"] and not any("error" in g for g in got)
        else:
            ok = st["installs"] == len(rows) and st["exact"] == len(rows) and not st["install_errors"] and \
                not any("error" in g for g in got)
        out[codec] = {"prompts": len(rows), "same_text_as_local": same, "transfers": st, "ok": ok,
                      "errors": [g["error"] for g in got if "error" in g], "resets": resets, "lossy": is_lossy(codec)}
        log(f"check {codec}: {'OK' if ok else 'FAILED'} - " +
            (f"{st['exact']}/{len(rows)} transfers bit-exact, " if codec != "nixl" and not is_lossy(codec) else
             f"lossy, KV error {fmt(100 * (st.get('rel_err') or 0), 2)}% (relative L2), " if is_lossy(codec) else
             "transfers n/a (vLLM's connector: no checksums), ") +
            f"{same}/{len(rows)} outputs equal to local prefill, bits {fmt(st['bits'])}")
        res["check"] = out
        end_point(a, f"check {codec}", idle, stop)
    post(a.proxy + "/t/config", {"verify": False})
    res["check"] = out


def is_lossy(codec):
    """AQUA-KV (aquakv-<b>): lossy, a speed reference beside the bit-exact codecs."""
    return str(codec).startswith("aquakv-")


def completion(a, prompt, n):
    r = post(a.proxy + "/v1/completions", {"model": a.served_name, "prompt": prompt, "max_tokens": n,
                                           "temperature": 0.0})
    return {"text": r["choices"][0]["text"]}


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def stage_sweep(a, res, colocated=False):
    key = "colocated" if colocated else "sweep"
    runs = res.setdefault(key, {})
    out_dir = a.out / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    links = [None] if colocated else a.links
    codecs = ["none"] if colocated else usable(a, res, a.codecs)
    for link in links:
        if link is not None:
            Path(a.link_control).write_text(json.dumps({"gbps": None if link == "native" else link}))
            time.sleep(0.5)
        for codec in codecs:
            if colocated:
                post(a.proxy + "/t/config", {"mode": "local"})
            elif codec == "nixl":
                post(a.proxy + "/t/config", {"mode": "nixl"})
            else:
                post(a.proxy + "/t/config", {"mode": "pd", "codec": codec, "verify": False})
            for rate in a.rates:
                name = f"{key}_{codec}_{link}gbps_r{rate}" if not colocated else f"{key}_r{rate}"
                if name in runs and not a.rerun:
                    continue
                settle(a, f"before {name}")
                ensure_alive(a, f"before {name}")
                if rate == a.rates[0]:
                    if not colocated:
                        warm_pairs(a)
                    warm_burst(a, greedy=True)
                resets = reset_caches(a)
                n = a.num_prompts if rate != "inf" else a.num_prompts
                logs0 = log_marks(a)
                rc, r, t0, t1 = bench_serve(a, str(a.dataset), rate, n, out_dir, name)
                logs1 = log_marks(a)
                idle, stop = after_point(a)
                time.sleep(1)
                st = transfer_stats(records(a.log_dir, t0, time.time())) if not colocated and codec != "nixl" else {}
                runs[name] = {"link_gbps": link, "codec": codec, "rate": rate, "exit": rc,
                              "bench": {k: r.get(k) for k in KEEP}, "transfers": st, "resets": resets,
                              "t0": t0, "t1": t1, "logs0": logs0, "logs1": logs1, "idle": idle}
                b = runs[name]["bench"]
                log(f"{name}: exit {rc}, {fmt(b.get('request_throughput'), 2)} req/s, TTFT p50 "
                    f"{fmt(b.get('median_ttft_ms'), 0)} / p99 {fmt(b.get('p99_ttft_ms'), 0)} ms, goodput "
                    f"{fmt(b.get('request_goodput'), 2)}" + (f"; bits {fmt(st.get('bits'))}, the link carried "
                    f"{fmt(st.get('wire_gbps'), 1)} Gb/s while busy" + (f" (set {link})" if link != "native" else "")
                    if st else ""))
                save(a, res)
                end_point(a, name, idle, stop)
    if colocated:
        post(a.proxy + "/t/config", {"mode": "pd"})


# the published baselines' own workloads (docs/pd_baselines.md, "Workloads"): which family a label names
SHAPE_FAMILIES = {
    "sz1": "SplitZip: 1 request at a time, prompt length swept",
    "sz16": "SplitZip: 16 at once, prompt length swept",
    "sz1024": "SplitZip: 1,024-token prompts, requests at once swept",
    "sz32k": "SplitZip: 32,768-token prompts, requests at once swept",
    "ibp": "IBP: 20 at once, 1,920 tokens in, 128 out",
    "exactkv": "exactkv: 4,096 tokens in, 256 out, as many as one decode GPU holds",
}


def parse_shapes(spec):
    """"T:B:O:R:label,..." (prompt tokens, requests at once, output tokens, rounds, family) into one entry per
    distinct (T, B, O), its families listed; R the most any family asks."""
    out = {}
    for item in (x for x in (spec or "").split(",") if x):
        t, b, o, r, *lab = item.split(":")
        k = (int(t), int(b), int(o))
        e = out.setdefault(k, {"tokens": k[0], "batch": k[1], "output": k[2], "rounds": 0, "families": []})
        e["rounds"] = max(e["rounds"], int(r))
        for x in lab:
            if x not in e["families"]:
                e["families"].append(x)
    # the smallest first: a workload too big for the node (memory) fails last, after the rest are measured
    return sorted(out.values(), key=lambda e: (e["tokens"] * e["batch"], e["tokens"], e["output"]))


def stage_shapes(a, res):
    """Every codec on each published workload: B requests at once (vllm bench serve --max-concurrency B), B x R of
    them, T-token prompts (prompts_<T>.jsonl), O tokens out, at each link speed; caches reset before each point."""
    runs = res.setdefault("shapes", {})
    out_dir = a.out / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    codecs = usable(a, res, a.codecs)
    for link in a.links:
        Path(a.link_control).write_text(json.dumps({"gbps": None if link == "native" else link}))
        time.sleep(0.5)
        for codec in codecs:
            if codec == "nixl":
                post(a.proxy + "/t/config", {"mode": "nixl"})
            else:
                post(a.proxy + "/t/config", {"mode": "pd", "codec": codec, "verify": False})
            first = True
            for sh in a.shapes:
                name = f"shape_{codec}_{link}gbps_t{sh['tokens']}_b{sh['batch']}_o{sh['output']}"
                if name in runs and not a.rerun:
                    continue
                data = a.prompts / f"prompts_{sh['tokens']}.jsonl"
                settle(a, f"before {name}")
                ensure_alive(a, f"before {name}")
                if first:
                    warm_pairs(a)
                    warm_burst(a, greedy=True)
                    first = False
                resets = reset_caches(a)
                n = sh["batch"] * sh["rounds"]
                logs0 = log_marks(a)
                rc, r, t0, t1 = bench_serve(a, str(data), "inf", n, out_dir, name, concurrency=sh["batch"],
                                            output_len=sh["output"])
                logs1 = log_marks(a)
                idle, stop = after_point(a)
                time.sleep(1)
                st = transfer_stats(records(a.log_dir, t0, time.time())) if codec != "nixl" else {}
                runs[name] = dict(sh, link_gbps=link, codec=codec, requests=n, exit=rc,
                                  bench={k: r.get(k) for k in KEEP}, transfers=st, resets=resets,
                                  t0=t0, t1=t1, logs0=logs0, logs1=logs1, idle=idle)
                b = runs[name]["bench"]
                log(f"{name} ({'/'.join(sh['families'])}, {n} requests): exit {rc}, TTFT p50 "
                    f"{fmt(b.get('median_ttft_ms'), 0)} / p99 {fmt(b.get('p99_ttft_ms'), 0)} ms, "
                    f"{fmt(b.get('request_throughput'), 2)} req/s, {fmt(b.get('output_throughput'), 0)} tok/s out"
                    + (f"; bits {fmt(st.get('bits'))}, link {fmt(st.get('wire_gbps'), 1)} Gb/s while busy" if st else "")
                    + (f"; {st.get('install_errors', 0) + st.get('send_errors', 0) + st.get('timeouts', 0)} transfers "
                       "failed" if st and (st.get('install_errors') or st.get('send_errors') or st.get('timeouts')) else ""))
                save(a, res)
                end_point(a, name, idle, stop)


def stage_multiturn(a, res):
    src = Path(a.vllm_source) / "benchmarks" / "multi_turn"
    if not (src / "benchmark_serving_multi_turn.py").is_file():
        res["multiturn"] = {"error": f"no {src} (scripts/pd_setup.sh clones the vLLM source)"}
        return
    conf = json.loads((src / "generate_multi_turn.json").read_text())
    conf["text_files"] = [str(p) for p in sorted(Path(a.texts).glob("wikitext2*.txt"))][:1] or conf["text_files"]
    conf["num_conversations"] = a.mt_conversations
    cfile = a.out / "multi_turn_config.json"
    cfile.write_text(json.dumps(conf, indent=1))
    out = {}
    for k, codec in enumerate(a.mt_codecs):
        ensure_alive(a, f"before multiturn {codec}")
        post(a.proxy + "/t/config", {"mode": "pd", "codec": codec, "policy": a.mt_policy, "reset_conversations": True})
        settle(a, f"before multiturn {codec}")
        if k == 0:
            warm_burst(a)
        reset_caches(a)                            # the same conversations (seed 0) for every codec
        name = f"multiturn_{codec}_{a.mt_policy}"
        t0 = time.time()
        cmd = [sys.executable, str(src / "benchmark_serving_multi_turn.py"), "--model", a.model,
               "--served-model-name", a.served_name, "--url", a.proxy, "--input-file", str(cfile),
               "--num-clients", str(a.mt_clients), "--max-active-conversations", str(a.mt_clients * 2),
               "--send-conversation-id", "--seed", "0",
               "--stats-json-output", str(a.out / "runs" / (name + ".json"))]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(src))
        (a.out / "runs" / (name + ".log")).write_text(r.stdout[-30000:] + "\n" + r.stderr[-30000:])
        if r.returncode != 0:
            log(f"{name}: the multi-turn benchmark exited {r.returncode}:\n" + tail_of(r.stderr or r.stdout))
        ok, stop = after_point(a)
        t1 = time.time()
        st = transfer_stats(records(a.log_dir, t0, t1))
        out[name] = {"exit": r.returncode, "t0": t0, "t1": t1, "transfers": st}
        log(f"{name}: exit {r.returncode}, {st['sends']} transfers, bits {fmt(st['bits'])}")
        res["multiturn"] = out
        end_point(a, name, ok, stop)
    post(a.proxy + "/t/config", {"policy": "round_robin"})
    res["multiturn"] = out
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "pd_opportunity.py"), "--log-dir", a.log_dir,
                        "--output", str(a.out / "opportunity.json")] + sum(
                            (["--window", f"{v['t0']},{v['t1']}"] for v in out.values()), []),
                       capture_output=True, text=True)
    log(p.stdout[-3000:] + p.stderr[-2000:])


def tail_of(text, n=12):
    return "\n".join("   " + l for l in text.strip().splitlines()[-n:])


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None and x >= 0)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[k]


def tier_stats(recs):
    loads = [r for r in recs if r.get("event") == "install" and r.get("kind") == "tier"]
    stores = [r for r in recs if r.get("event") == "tier_store"]
    snaps = [r for r in recs if r.get("event") == "tier"]
    ok = [r for r in loads if not r.get("error")]
    raw = sum(r.get("raw_bytes") or 0 for r in ok)
    return {"loads": len(loads), "load_errors": len(loads) - len(ok),
            "exact_chunks": sum(r.get("exact_chunks") or 0 for r in ok),
            "chunks_loaded": sum(r.get("chunks") or 0 for r in ok),
            "loaded_tokens": sum(r.get("tokens") or 0 for r in ok),
            "load_ms": mean([r.get("decode_install_ms") for r in ok]), "h2d_ms": mean([r.get("h2d_ms") for r in ok]),
            "decode_ms": mean([r.get("decode_ms") for r in ok]),
            # bits over the whole chunks moved (chunk_raw_bytes; older records: the tokens needed), and the share of
            # those chunks the requests needed (a hit can start or end inside a chunk)
            "bits": (16 * sum(r.get("loaded_bytes") or 0 for r in ok) / sum(r.get("chunk_raw_bytes") or 0 for r in ok)
                     if ok and all(r.get("chunk_raw_bytes") for r in ok) else
                     16 * sum(r.get("loaded_bytes") or 0 for r in ok) / raw if raw else None),
            # what crossed to the GPU, where it differs from what the tier holds (IBP: in place, the raw size held)
            "moved_bits": (16 * sum(r.get("moved_bytes", r.get("loaded_bytes")) or 0 for r in ok) /
                           sum(r.get("chunk_raw_bytes") or 0 for r in ok)
                           if ok and all(r.get("chunk_raw_bytes") for r in ok) else None),
            "needed_pct": (100 * raw / sum(r.get("chunk_raw_bytes") or 0 for r in ok)
                           if ok and all(r.get("chunk_raw_bytes") for r in ok) else None),
            "stores": len(stores), "store_errors": sum(1 for r in stores if r.get("error")),
            "stored_chunks": sum(r.get("stored") or 0 for r in stores),
            "store_ms": mean([r.get("store_ms") for r in stores if not r.get("error")]),
            "tier_chunks": max((r.get("chunks") or 0 for r in snaps), default=None),
            "evicted": sum(r.get("evicted") or 0 for r in snaps[-4:]) if snaps else None}


def stage_tier(a, res):
    src = Path(a.vllm_source) / "benchmarks" / "multi_turn"
    if not (src / "benchmark_serving_multi_turn.py").is_file():
        res["tier"] = {"error": f"no {src} (scripts/pd_setup.sh clones the vLLM source)"}
        return
    conf = json.loads((src / "generate_multi_turn.json").read_text())
    conf["text_files"] = [str(p) for p in sorted(Path(a.texts).glob("wikitext2*.txt"))][:1] or conf["text_files"]
    conf["num_conversations"] = a.tier_conversations
    pre = conf.get("prompt_input", {}).get("prefix_num_tokens")
    if a.tier_prefix_max and isinstance(pre, dict):    # each conversation's own start, capped (the smoke's tier)
        pre["max"] = min(int(pre.get("max", a.tier_prefix_max)), a.tier_prefix_max)
        if "average" in pre:
            pre["average"] = min(pre["average"], max(1, a.tier_prefix_max * 3 // 10))
    cfile = a.out / "tier_conversations_config.json"
    cfile.write_text(json.dumps(conf, indent=1))
    runs = res.setdefault("tier", {})
    engines = [d[0] for d in get(a.proxy + "/t/status")["decodes"]]
    post(a.proxy + "/t/config", {"mode": "local"})
    warm_burst(a)
    ctl = Path(a.tier_control) if a.tier_control else None
    gen = int(time.time())
    for codec in usable(a, res, a.tier_codecs):
        for clients in a.tier_clients:
            name = f"tier_{codec}_c{clients}"
            if name in runs and not a.rerun:
                continue
            settle(a, f"before {name}")
            ensure_alive(a, f"before {name}")
            gen += 1
            if ctl is not None:
                ctl.write_text(json.dumps({"generation": gen, "codec": codec}))
            time.sleep(1.0)
            resets = []
            for u in engines:
                try:
                    resets.append(post(u + "/reset_prefix_cache?reset_external=true", {}, timeout=120))
                except Exception as exc:
                    resets.append({"error": repr(exc)[:200]})
            post(a.proxy + "/t/config", {"mode": "local", "policy": "conversation", "reset_conversations": True})
            out_json = a.out / "runs" / (name + ".json")
            out_json.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(src / "benchmark_serving_multi_turn.py"), "--model", a.model,
                   "--served-model-name", a.served_name, "--url", a.proxy, "--input-file", str(cfile),
                   "--num-clients", str(clients), "--max-active-conversations", str(2 * clients),
                   "--send-conversation-id", "--seed", str(gen % 100000), "--max-turns", str(a.tier_max_turns),
                   "--stats-json-output", str(out_json)]
            m0 = prefix_metrics(engines)
            logs0 = log_marks(a)
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(src))
            (a.out / "runs" / (name + ".log")).write_text(r.stdout[-30000:] + "\n" + r.stderr[-30000:])
            if r.returncode != 0:
                log(f"{name}: the multi-turn benchmark exited {r.returncode}:\n" + tail_of(r.stderr or r.stdout))
            ok, stop = after_point(a)
            time.sleep(2)
            hits = hit_rates(m0, prefix_metrics(engines))
            t1 = time.time()
            logs1 = log_marks(a)
            stats = json.loads(out_json.read_text()) if out_json.is_file() else []
            ttft = [x.get("ttft_ms") for x in stats]
            starts = [x.get("start_time_ms") for x in stats if x.get("start_time_ms")]
            ends = [x["start_time_ms"] + x["latency_ms"] for x in stats if x.get("start_time_ms") and x.get("latency_ms")]
            dur = (max(ends) - min(starts)) / 1e3 if starts and ends else None
            bench = {"requests": len(stats), "duration_s": dur,
                     "request_throughput": len(stats) / dur if dur else None,
                     "mean_ttft_ms": mean(ttft), "median_ttft_ms": pct(ttft, 50), "p90_ttft_ms": pct(ttft, 90),
                     "p99_ttft_ms": pct(ttft, 99), "median_tpot_ms": pct([x.get("tpot_ms") for x in stats], 50),
                     "mean_latency_ms": mean([x.get("latency_ms") for x in stats]),
                     "input_tokens": mean([x.get("input_num_tokens") for x in stats])}
            st = tier_stats(records(a.log_dir, t0, t1))
            runs[name] = {"codec": codec, "clients": clients, "exit": r.returncode, "bench": bench, "tier": st,
                          "resets": resets, "generation": gen, "prefix_cache": hits, "t0": t0, "t1": t1,
                          "logs0": logs0, "logs1": logs1}
            log(f"{name}: exit {r.returncode}, {bench['requests']} requests, {fmt(bench['request_throughput'], 2)} "
                f"req/s, TTFT p50 {fmt(bench['median_ttft_ms'], 0)} / p99 {fmt(bench['p99_ttft_ms'], 0)} ms; tier "
                f"{st['loads']} loads ({st['loaded_tokens']} tokens, {fmt(st['load_ms'], 1)} ms each, bits "
                f"{fmt(st['bits'])}), {st['stored_chunks']} chunks stored")
            save(a, res)
            end_point(a, name, ok, stop)
    post(a.proxy + "/t/config", {"policy": "round_robin"})


def save(a, res):
    (a.out / "results.json").write_text(json.dumps(res, indent=1))


def jit_report(a, res):
    """vLLM's kernels compiled while serving, each with the measured point it fell in (None: outside them)."""
    spans = [(v, name) for key in ("sweep", "colocated", "tier", "shapes") for name, v in (res.get(key) or {}).items()
             if isinstance(v, dict) and v.get("t0") and v.get("t1")]

    def inside(v, t, f, off):
        m0, m1 = v.get("logs0") or {}, v.get("logs1") or {}
        if f in m0 and f in m1:        # the log's own position: job 30252's compile in the warm-up just before a point
            return m0[f] <= off < m1[f]    # fell within the whole-second slack and was counted inside it
        return v["t0"] - 1 <= t <= v["t1"] + 1
    out = []
    for t, f, text, off in jit_events(a):
        point = next((n for v, n in spans if inside(v, t, f, off)), None)
        out.append({"time": t, "log": f, "text": text, "point": point})
    res["jit"] = out
    save(a, res)
    return out


def stage_report(a, res):
    jit = jit_report(a, res)
    L = ["# vLLM PD with our transfer protocol", ""]
    env = res.get("env", {})
    L.append(f"Model {env.get('model')}; {env.get('gpus')}; vLLM {env.get('vllm')}, torch {env.get('torch')}; "
             f"1 prefill engine, {env.get('decodes')} decode engine(s); prompts {env.get('dataset')}, "
             f"{a.output_len} output tokens; TTFT SLO {a.ttft_slo_ms} ms (goodput).")
    L.append("")
    if res.get("stopped"):
        L += [stopped_note(res["stopped"]), ""]
    if "check" in res:
        L += ["## Correctness (every transfer checked bit-exact on the decode GPU)", "",
              "| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |", "| --- | --- | --- | --- |"]
        for c, v in res["check"].items():
            st = v["transfers"]
            L.append(f"| {c} | {st['exact']}/{v['prompts']}{'' if v['ok'] else ' (FAILED)'} | "
                     f"{v['same_text_as_local']}/{v['prompts']} | {fmt(st['bits'])} |")
        L.append("")
    sw = res.get("sweep", {})
    if sw:
        links = sorted({v["link_gbps"] for v in sw.values()})
        rates = []
        for v in sw.values():
            if v["rate"] not in rates:
                rates.append(v["rate"])
        for link in links:
            L += [f"## Link {link} Gb/s", "",
                  "| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | "
                  "Link time (ms) | Achieved Gb/s |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for c in a.codecs:
                for r in rates:
                    v = sw.get(f"sweep_{c}_{link}gbps_r{r}")
                    if v is None:
                        continue
                    b, st = v["bench"], v["transfers"]
                    L.append(f"| {c} | {r} | {fmt(b.get('request_throughput'), 2)} | {fmt(b.get('request_goodput'), 2)} | "
                             f"{fmt(b.get('median_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)} | "
                             f"{fmt(b.get('median_tpot_ms'), 1)} | {fmt(st.get('bits'))} | "
                             f"{fmt(st.get('encode_ms'), 1)} / {fmt(st.get('install_ms'), 1)} | {fmt(st.get('link_ms'), 1)} | "
                             f"{fmt(st.get('achieved_gbps'), 1)} |")
            L.append("")
    sh = res.get("shapes", {})
    if sh:
        L += ["## The published workloads (TTFT p50 / p99 in ms; requests per second)", "",
              "| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | "
              "Failed transfers |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v in sh.values():
            b, st = v["bench"], v.get("transfers") or {}
            bad = (st.get("install_errors") or 0) + (st.get("send_errors") or 0) + (st.get("timeouts") or 0)
            L.append(f"| {v['link_gbps']} | {v['tokens']} | {v['batch']} | {v['output']} | {v['requests']} | "
                     f"{'/'.join(v['families'])} | {v['codec']} | {fmt(b.get('median_ttft_ms'), 0)} / "
                     f"{fmt(b.get('p99_ttft_ms'), 0)} | {fmt(b.get('request_throughput'), 2)} | {fmt(st.get('bits'))} | "
                     f"{bad if st else '-'} |")
        L.append("")
    co = res.get("colocated", {})
    if co:
        L += ["## Without disaggregation (each decode engine prefills its own requests)", "",
              "| Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) |", "| --- | --- | --- | --- | --- |"]
        for v in co.values():
            b = v["bench"]
            L.append(f"| {v['rate']} | {fmt(b.get('request_throughput'), 2)} | {fmt(b.get('request_goodput'), 2)} | "
                     f"{fmt(b.get('median_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)} | {fmt(b.get('median_tpot_ms'), 1)} |")
        L.append("")
    ti = res.get("tier", {})
    if ti and "error" not in ti:
        L += ["## CPU tier (colocated engines, multi-turn conversations)", "",
              "Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the "
              "tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.", "",
              "| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | "
              "Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v in ti.values():
            b, st, hc = v["bench"], v["tier"], v.get("prefix_cache") or {}
            L.append(f"| {v['codec']} | {v['clients']} | {fmt(b.get('request_throughput'), 2)} | "
                     f"{fmt(b.get('median_ttft_ms'), 0)} / {fmt(b.get('p90_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)} | "
                     f"{fmt(b.get('median_tpot_ms'), 1)} | {fmt(hc.get('gpu_hit_pct'), 1)} / {fmt(hc.get('external_hit_pct'), 1)} | "
                     f"{st.get('loads')} | {st.get('loaded_tokens')} | "
                     f"{fmt(st.get('load_ms'), 1)} ({fmt(st.get('h2d_ms'), 1)} / {fmt(st.get('decode_ms'), 1)}) | "
                     f"{fmt(st.get('bits'))} | {fmt(st.get('needed_pct'), 0)} | {st.get('stored_chunks')} | "
                     f"{st.get('tier_chunks')} |")
        L.append("")
    elif ti:
        L += ["## CPU tier", "", ti["error"], ""]
    if res.get("skipped"):
        L += ["Codecs not run (did not load on every engine): " + "; ".join(f"{c} ({w[:120]})" for c, w in
                                                                         res["skipped"].items()), ""]
    inside = [j for j in jit if j["point"]]
    if jit:
        L += ["## Kernels compiled while serving (vLLM's JIT warnings)", "",
              f"{len(jit)} in the engine logs, {len(inside)} inside a measured point"
              + (" (those points' latencies include the compile):" if inside else "."), ""]
        L += [f"- {time.strftime('%H:%M:%S', time.localtime(j['time']))} {j['log']}: {j['point'] or 'outside the points'}"
              f" - {j['text'][:110]}" for j in jit[:20]]
        L.append("")
    opp = a.out / "opportunity.json"
    if opp.is_file():
        o = json.loads(opp.read_text())
        L += ["## Reuse opportunities (multi-turn, before any base sending)", "", o.get("summary", ""), ""]
    (a.out / "summary.md").write_text("\n".join(L) + "\n")
    log(f"report: {a.out / 'summary.md'}")


def main():
    global IDLE_TIMEOUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="the model folder the engines serve (also the tokenizer)")
    ap.add_argument("--served-name", default="llama")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True, help="folder of scripts/pd_prompts.py")
    ap.add_argument("--dataset", default=None, help="prompt file of the sweep (default prompts_mixed.jsonl)")
    ap.add_argument("--texts", default=None, help="dataset texts (multi-turn conversations are cut from WikiText-2)")
    ap.add_argument("--log-dir", required=True, help="the connectors' log_dir")
    ap.add_argument("--link-control", required=True)
    ap.add_argument("--codecs", default="raw,splitzip,adacode,ours-hbh,ours-kv,ours-dense,ours-rank256")
    ap.add_argument("--links", default="3,5,10,25,40,100,200,400")
    ap.add_argument("--rates", default="0.25,0.5,1,2,inf")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--ttft-slo-ms", type=float, default=2000)
    ap.add_argument("--stages", default="check,sweep,colocated,multiturn,report")
    ap.add_argument("--vllm-source", default=None)
    ap.add_argument("--mt-codecs", default="raw,ours-hbh")
    ap.add_argument("--mt-policy", default="conversation")
    ap.add_argument("--mt-conversations", type=int, default=24)
    ap.add_argument("--mt-clients", type=int, default=4)
    ap.add_argument("--env", default=None, help="JSON with model, gpus, vllm, torch for the report")
    ap.add_argument("--tier-codecs", default="raw,ours-hbh")
    ap.add_argument("--tier-clients", default="8,16")
    ap.add_argument("--tier-conversations", type=int, default=64)
    ap.add_argument("--tier-max-turns", type=int, default=12)
    ap.add_argument("--tier-control", default=None, help="the connectors' tier control file")
    ap.add_argument("--tier-prefix-max", type=int, default=0,
                    help="cap on each conversation's own start in the tier runs (tokens; 0: vLLM's config)")
    ap.add_argument("--engine-logs", default=None, help="folder of the engine logs (engine_*.log), for the JIT report")
    ap.add_argument("--shapes", default="", help="T:B:O:R:label,... the published workloads (stage shapes)")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT,
                    help="s a point's requests may stay in flight after its benchmark ended before the run stops")
    ap.add_argument("--point-timeout", type=float, default=3600,
                    help="s one vllm bench serve may run before it is stopped (0: no limit)")
    a = ap.parse_args()
    IDLE_TIMEOUT = a.idle_timeout
    a.point_timeout = a.point_timeout or None
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    a.prompts = Path(a.prompts)
    a.dataset = Path(a.dataset) if a.dataset else a.prompts / "prompts_mixed.jsonl"
    a.codecs = [c for c in a.codecs.split(",") if c]
    a.mt_codecs = [c for c in a.mt_codecs.split(",") if c]
    a.links = [x if x == "native" else (float(x) if "." in x else int(x)) for x in a.links.split(",")]
    a.tier_codecs = [c for c in a.tier_codecs.split(",") if c]
    a.tier_clients = [int(c) for c in a.tier_clients.split(",") if c]
    a.rates = [x if x == "inf" else float(x) for x in a.rates.split(",")]
    a.shapes = parse_shapes(a.shapes)
    p = a.out / "results.json"
    res = json.loads(p.read_text()) if p.is_file() else {}
    if a.env:
        res["env"] = dict(json.loads(Path(a.env).read_text()), dataset=a.dataset.name)
    res.pop("stopped", None)

    def on_term(signum, frame):              # the run's time limit (RUN_TIMEOUT_S, then 60 s to the kill):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)   # what is measured is saved and reported
        raise RunStopped("stopped by SIGTERM (the run's time limit)")
    signal.signal(signal.SIGTERM, on_term)
    for stage in a.stages.split(","):
        if res.get("stopped") and stage != "report":
            log(f"== {stage}: not run (the run stopped)")
            continue
        log(f"== {stage}")
        try:
            if stage == "check":
                stage_check(a, res)
            elif stage == "sweep":
                stage_sweep(a, res)
            elif stage == "colocated":
                stage_sweep(a, res, colocated=True)
            elif stage == "multiturn":
                stage_multiturn(a, res)
            elif stage == "tier":
                stage_tier(a, res)
            elif stage == "shapes":
                stage_shapes(a, res)
            elif stage == "report":
                stage_report(a, res)
        except RunStopped as exc:
            # what is measured stays; the rest of this run is skipped and the matrix goes on to its next run,
            # with engines of its own
            res["stopped"] = {"stage": stage, "why": exc.why, "engines": exc.engines, "time": time.time()}
            log(f"RUN STOPPED in {stage}: {exc.why}" +
                ("; not answering: " + ", ".join(f"{u} ({w[:80]})" for u, w in exc.engines.items())
                 if exc.engines else ""))
        save(a, res)
    if res.get("stopped"):
        return 3
    if "check" in res and not all(v["ok"] for v in res["check"].values()):
        log("CHECK FAILED for: " + ", ".join(c for c, v in res["check"].items() if not v["ok"]))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

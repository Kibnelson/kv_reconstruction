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
  report      results.json and summary.md

Every run's raw result stays in <out>/runs/; a rerun skips runs already there (--rerun to redo).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
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


def wait_idle(proxy, timeout=1800):
    end = time.time() + timeout
    while time.time() < end:
        if sum(get(proxy + "/t/status")["inflight"]) == 0:
            return True
        time.sleep(1)
    return False


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
                    out.append(r)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def transfer_stats(recs):
    s = [r for r in recs if r.get("event") == "send"]
    d = [r for r in recs if r.get("event") == "install"]
    tmo = [r for r in recs if r.get("event") == "timeout"]
    ach = [8 * r["blob_bytes"] / (r["send_end"] - r["send_begin"]) / 1e9 for r in s
           if r.get("send_end") and r.get("send_begin") and r["send_end"] > r["send_begin"]]
    return {"sends": len(s), "send_errors": sum(1 for r in s if r.get("error")),
            "installs": len(d), "install_errors": sum(1 for r in d if r.get("error")), "timeouts": len(tmo),
            "bits": mean([r.get("bits") for r in s]), "gather_ms": mean([r.get("gather_ms") for r in s]),
            "encode_ms": mean([r.get("encode_ms") for r in s]),
            "link_ms": mean([1e3 * r["link_s"] for r in s if r.get("link_s") is not None]),
            "queue_ms": mean([1e3 * (r["send_begin"] - r["queued"]) for r in s if r.get("send_begin")]),
            "achieved_gbps": mean(ach),
            "install_ms": mean([r.get("decode_install_ms") for r in d]),
            "wait_ms": mean([1e3 * (r["done_at"] - r["admitted_at"]) for r in d if r.get("admitted_at")]),
            "exact": sum(1 for r in d if r.get("exact") is True), "inexact": sum(1 for r in d if r.get("exact") is False),
            "spilled_to_host": sum(1 for r in s if r.get("queued_on") == "host")}


def bench_serve(a, dataset, rate, n, out_dir, name):
    vllm = os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = [vllm if os.path.exists(vllm) else "vllm", "bench", "serve", "--backend", "openai", "--base-url", a.proxy, "--endpoint", "/v1/completions",
           "--model", a.model, "--served-model-name", a.served_name, "--tokenizer", a.model,
           "--dataset-name", "custom", "--dataset-path", dataset, "--custom-output-len", str(a.output_len),
           "--skip-chat-template", "--num-prompts", str(n), "--request-rate", str(rate), "--ignore-eos",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--goodput", f"ttft:{a.ttft_slo_ms}", "--seed", "0", "--disable-tqdm",
           "--save-result", "--result-dir", str(out_dir), "--result-filename", name + ".json"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    (out_dir / (name + ".log")).write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    res = {}
    p = out_dir / (name + ".json")
    if p.is_file():
        res = json.loads(p.read_text())
    return r.returncode, res, t0, t1


KEEP = ("completed", "duration", "request_throughput", "request_goodput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
        "p99_tpot_ms", "mean_e2el_ms", "p99_e2el_ms", "failed")


def stage_check(a, res):
    rows = [json.loads(l) for l in open(a.prompts / "check.jsonl")]
    out = {}
    ref = None
    post(a.proxy + "/t/config", {"mode": "local"})
    ref = [completion(a, r["prompt"], r["output_tokens"]) for r in rows]
    for codec in a.codecs:
        post(a.proxy + "/t/config", {"mode": "pd", "codec": codec, "verify": True})
        t0 = time.time()
        got = []
        for r in rows:
            try:
                got.append(completion(a, r["prompt"], r["output_tokens"]))
            except Exception as exc:
                got.append({"error": repr(exc)[:300]})
        wait_idle(a.proxy)
        time.sleep(1)
        st = transfer_stats(records(a.log_dir, t0, time.time()))
        same = sum(1 for x, y in zip(got, ref) if x.get("text") is not None and x.get("text") == y.get("text"))
        ok = st["installs"] == len(rows) and st["exact"] == len(rows) and not st["install_errors"] and \
            not any("error" in g for g in got)
        out[codec] = {"prompts": len(rows), "same_text_as_local": same, "transfers": st, "ok": ok,
                      "errors": [g["error"] for g in got if "error" in g]}
        log(f"check {codec}: {'OK' if ok else 'FAILED'} - {st['exact']}/{len(rows)} transfers bit-exact, "
            f"{same}/{len(rows)} outputs equal to local prefill, bits {fmt(st['bits'])}")
    post(a.proxy + "/t/config", {"verify": False})
    res["check"] = out


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
    codecs = ["none"] if colocated else a.codecs
    for link in links:
        if link is not None:
            Path(a.link_control).write_text(json.dumps({"gbps": link}))
            time.sleep(0.5)
        for codec in codecs:
            if colocated:
                post(a.proxy + "/t/config", {"mode": "local"})
            else:
                post(a.proxy + "/t/config", {"mode": "pd", "codec": codec, "verify": False})
            for rate in a.rates:
                name = f"{key}_{codec}_{link}gbps_r{rate}" if not colocated else f"{key}_r{rate}"
                if name in runs and not a.rerun:
                    continue
                wait_idle(a.proxy)
                n = a.num_prompts if rate != "inf" else a.num_prompts
                rc, r, t0, t1 = bench_serve(a, str(a.dataset), rate, n, out_dir, name)
                wait_idle(a.proxy)
                time.sleep(1)
                st = transfer_stats(records(a.log_dir, t0, time.time())) if not colocated else {}
                runs[name] = {"link_gbps": link, "codec": codec, "rate": rate, "exit": rc,
                              "bench": {k: r.get(k) for k in KEEP}, "transfers": st}
                b = runs[name]["bench"]
                log(f"{name}: exit {rc}, {fmt(b.get('request_throughput'), 2)} req/s, TTFT p50 "
                    f"{fmt(b.get('median_ttft_ms'), 0)} / p99 {fmt(b.get('p99_ttft_ms'), 0)} ms, goodput "
                    f"{fmt(b.get('request_goodput'), 2)}" + (f"; bits {fmt(st.get('bits'))}, link "
                    f"{fmt(st.get('achieved_gbps'), 1)} Gb/s achieved" if st else ""))
                save(a, res)
    if colocated:
        post(a.proxy + "/t/config", {"mode": "pd"})


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
    for codec in a.mt_codecs:
        post(a.proxy + "/t/config", {"mode": "pd", "codec": codec, "policy": a.mt_policy, "reset_conversations": True})
        name = f"multiturn_{codec}_{a.mt_policy}"
        t0 = time.time()
        cmd = [sys.executable, str(src / "benchmark_serving_multi_turn.py"), "--model", a.model,
               "--served-model-name", a.served_name, "--url", a.proxy, "--input-file", str(cfile),
               "--num-clients", str(a.mt_clients), "--max-active-conversations", str(a.mt_clients * 2),
               "--send-conversation-id", "--seed", "0",
               "--stats-json-output", str(a.out / "runs" / (name + ".json"))]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(src))
        (a.out / "runs" / (name + ".log")).write_text(r.stdout[-30000:] + "\n" + r.stderr[-30000:])
        wait_idle(a.proxy)
        t1 = time.time()
        st = transfer_stats(records(a.log_dir, t0, t1))
        out[name] = {"exit": r.returncode, "t0": t0, "t1": t1, "transfers": st}
        log(f"{name}: exit {r.returncode}, {st['sends']} transfers, bits {fmt(st['bits'])}")
    post(a.proxy + "/t/config", {"policy": "round_robin"})
    res["multiturn"] = out
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "pd_opportunity.py"), "--log-dir", a.log_dir,
                        "--output", str(a.out / "opportunity.json")] + sum(
                            (["--window", f"{v['t0']},{v['t1']}"] for v in out.values()), []),
                       capture_output=True, text=True)
    log(p.stdout[-3000:] + p.stderr[-2000:])


def save(a, res):
    (a.out / "results.json").write_text(json.dumps(res, indent=1))


def stage_report(a, res):
    L = ["# vLLM PD with our transfer protocol", ""]
    env = res.get("env", {})
    L.append(f"Model {env.get('model')}; {env.get('gpus')}; vLLM {env.get('vllm')}, torch {env.get('torch')}; "
             f"1 prefill engine, {env.get('decodes')} decode engine(s); prompts {env.get('dataset')}, "
             f"{a.output_len} output tokens; TTFT SLO {a.ttft_slo_ms} ms (goodput).")
    L.append("")
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
    co = res.get("colocated", {})
    if co:
        L += ["## Without disaggregation (each decode engine prefills its own requests)", "",
              "| Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) |", "| --- | --- | --- | --- | --- |"]
        for v in co.values():
            b = v["bench"]
            L.append(f"| {v['rate']} | {fmt(b.get('request_throughput'), 2)} | {fmt(b.get('request_goodput'), 2)} | "
                     f"{fmt(b.get('median_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)} | {fmt(b.get('median_tpot_ms'), 1)} |")
        L.append("")
    opp = a.out / "opportunity.json"
    if opp.is_file():
        o = json.loads(opp.read_text())
        L += ["## Reuse opportunities (multi-turn, before any base sending)", "", o.get("summary", ""), ""]
    (a.out / "summary.md").write_text("\n".join(L) + "\n")
    log(f"report: {a.out / 'summary.md'}")


def main():
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
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    a.prompts = Path(a.prompts)
    a.dataset = Path(a.dataset) if a.dataset else a.prompts / "prompts_mixed.jsonl"
    a.codecs = [c for c in a.codecs.split(",") if c]
    a.mt_codecs = [c for c in a.mt_codecs.split(",") if c]
    a.links = [float(x) if "." in x else int(x) for x in a.links.split(",")]
    a.rates = [x if x == "inf" else float(x) for x in a.rates.split(",")]
    p = a.out / "results.json"
    res = json.loads(p.read_text()) if p.is_file() else {}
    if a.env:
        res["env"] = dict(json.loads(Path(a.env).read_text()), dataset=a.dataset.name)
    for stage in a.stages.split(","):
        log(f"== {stage}")
        if stage == "check":
            stage_check(a, res)
        elif stage == "sweep":
            stage_sweep(a, res)
        elif stage == "colocated":
            stage_sweep(a, res, colocated=True)
        elif stage == "multiturn":
            stage_multiturn(a, res)
        elif stage == "report":
            stage_report(a, res)
        save(a, res)
    if "check" in res and not all(v["ok"] for v in res["check"].values()):
        log("CHECK FAILED for: " + ", ".join(c for c, v in res["check"].items() if not v["ok"]))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

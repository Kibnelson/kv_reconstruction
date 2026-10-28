#!/usr/bin/env python3
"""Retimed official baselines and complete Lite packet lane-width tuning.

Reuses all existing fitted state. No model download, training, baseline source
change, oracle prediction timing or AQUA page adapter is used.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_llama_causal import device_check, log, timed
from scripts.bench_llama_baselines import confined, verified_inputs
from kv_compare.lite_scoreboard import aggregate, network_model, validate_comparison

FILES = ("scripts/bench_lite_baselines.py", "scripts/package_lite_baselines.py",
    "scripts/run_lite_baselines_a4500.sh", "kv_compare/lite_scoreboard.py", "kv_compare/lite_fast.py",
    "kv_compare/framed_baselines.py", "kv_compare/official_exans.py", "kv_compare/profile_recovery.py",
    "scripts/bench_llama_causal.py", "scripts/bench_llama_baselines.py", "scripts/run_llama_profile_baselines.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); tmp.replace(path)


def launch(argv, path, env=None):
    log("running: " + " ".join(map(str, argv)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        child = subprocess.Popen(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            while True:
                try:
                    code = child.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    log("still running; log: " + str(path))
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise
    if code:
        raise RuntimeError(f"exit {code}; see {path}")


def make_codec(q, device):
    if q["kind"] == "lite":
        from kv_compare.lite_fast import FastEngine
        return FastEngine(q["state"], device, **q["engine_options"])
    if q["kind"] == "exans":
        from scripts.run_llama_profile_baselines import check_source
        from kv_compare.official_exans import Exans
        check_source(q["source"], "exans")
        if digest(q["library"]) != q["library_sha256"]:
            raise ValueError("official ExANS library changed")
        return Exans(q["source"], q["library"], device, q["blocks"])
    from kv_compare.framed_baselines import Baseline, checked_splitzip
    if q["kind"] == "splitzip_official_framed":
        checked_splitzip(q["source"], software_check=q.get("software_check", False))
        if digest(q["state"]) != q["state_sha256"]:
            raise ValueError("original TRAIN-fitted SplitZip LUTs changed")
    return Baseline(q["kind"], device, source=q.get("source"), state=q.get("state"))


def receive(q, torch, device):
    from kv_compare.causal_packet import Packet, word_sha
    codec = make_codec(q["codec"], device)
    checks = []
    try:
        for entry in q["entries"]:
            if digest(entry["packet"]) != entry["packet_sha256"]:
                raise ValueError("receiver packet changed")
            output = codec.decode(Packet.restore(Path(entry["packet"]).read_bytes(), device))
            if word_sha(output) != entry["expected_word_sha256"]:
                raise AssertionError("fresh receiver changed reconstructed words")
            checks.append(dict(id=entry["id"], verified=True, word_sha256=word_sha(output),
                packet_sha256=entry["packet_sha256"], shape=list(output.shape)))
            log("independent " + device.type + " packet replay: " + entry["id"])
    finally:
        if hasattr(codec, "close"):
            codec.close()
    write(q["output"], dict(gpu_measured=device.type == "cuda", no_source_kv_or_predictions=True,
        inputs="fixed codec state + packet bytes; expected digests are assertions", checks=checks))


def worker(q, torch, device):
    import numpy as np
    from kv_compare.causal_packet import Packet, word_sha
    out = Path(q["output"]); out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter(); codec = make_codec(q["codec"], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cold_ms = (time.perf_counter() - begin) * 1000
    rows = []; entries = []
    try:
        for e in q["inputs"]:
            if digest(e["path"]) != e["file_sha256"]:
                raise ValueError("frozen input changed")
            source = torch.from_numpy(np.load(e["path"], allow_pickle=False).view(np.int16).copy()).to(device)
            if word_sha(source) != e["original_word_sha256"]:
                raise ValueError("input is not the matched original BF16 KV")
            for kappa in q["kappas"]:
                ident = f"{q['method']}_k{kappa:g}_{e['document']}"
                expected = q["expected"][f"{e['document']}:{kappa:g}"]
                if q["codec"]["kind"] == "lite":
                    fn = lambda: codec.encode(source, start=expected.get("start", 0), kappa=kappa)
                    if kappa:
                        codec.residual(kappa)
                else:
                    fn = lambda: codec.encode(source)
                packet, enc = timed(fn, torch, device, q["warmup"], q["repeats"])
                if q["codec"]["kind"] == "lite":
                    if word_sha(codec.last_reconstruction) != expected["decoded_word_sha256"]:
                        raise AssertionError("changed Lite encoder reconstruction")
                    codec.last_reconstruction = None
                restored, dec = timed(lambda: codec.decode(packet), torch, device, q["warmup"], q["repeats"])
                decoded_sha = word_sha(restored)
                if decoded_sha != expected["decoded_word_sha256"]:
                    raise AssertionError("receiver differs from previously verified reconstruction")
                exact = bool(torch.equal(restored, source))
                mode = "lossy" if kappa or q["codec"]["kind"] == "fp8_e4m3fn_saturating" else "lossless"
                if mode == "lossless" and not exact:
                    raise AssertionError("lossless words changed")
                begin = time.perf_counter(); wire = packet.export()
                export_ms = (time.perf_counter() - begin) * 1000
                packet_sha = hashlib.sha256(wire).hexdigest()
                compatible = packet_sha == expected.get("packet_sha256")
                if q.get("require_original_packet") and not compatible:
                    raise AssertionError("B64 control differs from original packet bytes")
                begin = time.perf_counter(); imported = Packet.restore(wire, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                upload_ms = (time.perf_counter() - begin) * 1000
                if word_sha(codec.decode(imported)) != decoded_sha:
                    raise AssertionError("serialized packet differs")
                packet_path = out / (ident + ".kvc"); packet_path.write_bytes(wire)
                row = dict(id=ident, method=q["method"], document=e["document"], kappa=kappa, mode=mode,
                    shape=list(source.shape), values=source.numel(), original_word_sha256=e["original_word_sha256"],
                    decoded_word_sha256=decoded_sha, original_exact=exact, packet_bytes=len(wire),
                    complete_bpv=8 * len(wire) / source.numel(), saved_percent=100 * (1 - len(wire) / source.numel() / 2),
                    packet=str(packet_path), packet_sha256=packet_sha, packet_matches_original=compatible,
                    state_sha256=packet.header["state_sha256"],
                    encode=enc, decode=dec, serialized_export_wall_ms=export_ms, upload_wall_ms=upload_ms,
                    shared_predictor_gpu_bytes=getattr(codec, "shared_bytes", None),
                    quality=expected.get("quality"), quality_scope="reused short continuation evidence; identical decoded word hash required")
                if q["codec"]["kind"] == "lite":
                    groups = codec.layers[0]["groups"]
                    row["coder_geometry"] = [dict(columns=g["columns"], values_per_stream=codec.V,
                        lanes_per_program=codec.B, programs=codec.lossless.layout(source.shape[1] * g["columns"])[1]) for g in groups]
                elif q["codec"]["kind"] == "exans":
                    row["accounting"] = codec.accounting(packet)
                rows.append(row)
                entries.append(dict(id=ident, packet=str(packet_path), packet_sha256=packet_sha,
                                    expected_word_sha256=decoded_sha))
                log(f"{ident}: {row['complete_bpv']:.5f} bpv; encode {enc['wall_median_ms']:.3f} ms; decode {dec['wall_median_ms']:.3f} ms")
                del packet, restored, imported, wire; gc.collect()
            del source; gc.collect()
    finally:
        if hasattr(codec, "close"):
            codec.close()
    del codec; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    request = dict(codec=q["codec"], entries=entries, output=str(out / "receiver.json"))
    write(out / "receiver_request.json", request)
    argv = [sys.executable, Path(__file__).resolve(), "--device", str(device), "--receive", out / "receiver_request.json"]
    if device.type != "cuda":
        argv.append("--software-check")
    launch(argv, out / "receiver.log")
    proof = json.loads((out / "receiver.json").read_text())
    checks = {r["id"]: r for r in proof["checks"]}
    if set(checks) != {r["id"] for r in rows} or proof["gpu_measured"] != (device.type == "cuda"):
        raise ValueError("receiver coverage/device differs")
    for row in rows:
        if not checks[row["id"]]["verified"] or checks[row["id"]]["word_sha256"] != row["decoded_word_sha256"]:
            raise ValueError("independent replay failed")
        row["status"] = "verified"
    write(out / "worker_results.json", dict(status="completed" if device.type == "cuda" else "software_checked",
        gpu_measured=device.type == "cuda", rows=rows, cold_initialization_wall_ms=cold_ms,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU interpreter",
        software={n: importlib.metadata.version(n) for n in ("torch", "triton", "numpy")},
        codec_provenance=q["codec"],
        receiver_sha256=digest(out / "receiver.json")))


def report(r, out):
    lines = ["# Complete Lite packets versus retimed baselines", "",
        f"Status: **{r['status']}**. GPU measured: **{r.get('gpu_measured', False)}**.", "",
        "All methods use identical original BF16 words, all 28 layers. Complete packet bytes include framing and directories.",
        "Each method runs in a fresh process, with independent packet-only replay. Baseline implementations and fitted state are unchanged.",
        "Means below average per-document synchronized wall medians. Shared fixed state is additional.", "",
        "| Method | Kappa | Saved | Encode ms | Decode ms | Exact original |", "|---|---:|---:|---:|---:|---|"]
    for row in r.get("aggregates", []):
        lines.append(f"| {row['method']} | {row['kappa']:g} | {row['saved_percent']:.2f}% | {row['encode_ms']:.3f} | {row['decode_ms']:.3f} | {row['original_exact']} |")
    lines += ["", "Previously measured short continuation quality, reused only after identical reconstructed words.", "",
        "| Lossy method | Kappa | Document | PPL change |", "|---|---:|---|---:|"]
    for row in r.get("aggregates", []):
        if row["mode"] != "lossy":
            continue
        for q in row["quality"]:
            value = q.get("ppl_change_percent")
            lines.append(f"| {row['method']} | {row['kappa']:g} | {q['document']} | "
                         + ("unmeasured" if value is None else f"{value:+.3f}%") + " |")
    lines += ["", "Time ratios >1 mean slower than the baseline; packet ratios <1 mean fewer bytes.", "",
        "| Lossless method | Versus | Packet ratio | Encode time ratio | Decode time ratio |",
        "|---|---|---:|---:|---:|"]
    for row in r.get("aggregates", []):
        if row["method"].startswith("lite") and row["original_exact"]:
            for name, v in row["vs_baseline"].items():
                lines.append(f"| {row['method']} | {name} | {v['packet_bytes_ratio']:.3f} | {v['encode_time_ratio']:.2f} | {v['decode_time_ratio']:.2f} |")
    if r.get("network_model"):
        lines += ["", "Calculated serial network model — not measured transfer or serving latency.", "",
            "| Effective Gb/s | Best fresh encode + transfer + decode | Best precompressed transfer + decode |",
            "|---:|---|---|"]
        for e in r["network_model"]["links"]:
            lines.append(f"| {e['effective_gbps']:g} | {e['best_fresh']} | {e['best_precompressed']} |")
    if r.get("error"):
        lines += ["", "Failure: " + r["error"]]
    lines += ["", "B64 controls must match original packets byte for byte. B8/B16/B32 change stream grouping and charge their actual bytes; all decoded words must match the original reconstruction.",
        "B64 is measured before and after the sweep to expose drift. Timing workers never run the profiler.",
        "Lossy quality is reused only after matching the prior reconstructed-word hash; it is a short continuation smoke check, not full task validation.",
        "Official AQUA remains blocked; no AQUA-style/page adapter substitutes for it. ExANS native allocation peaks are unmeasured.",
        "No SSD, actual network, overlap, pool attention or serving result is measured by this benchmark."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-run", type=Path, default=ROOT / "runs/llama_causal_a4500_20261027_184615_1307629")
    p.add_argument("--baseline-run", type=Path, default=ROOT / "runs/llama_baselines_a4500_20261027_191350_1897426")
    p.add_argument("--profile-run", type=Path, default=ROOT / "runs/llama_profile_official_a4500_20261027_194103_2478724")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--receive", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--software-check", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker or args.receive:
        q = json.loads((args.worker or args.receive).read_text())
        spec = q["codec"]
        if spec["kind"] == "lite":
            torch, device = device_check(args.device, args.software_check)
        else:
            # Preserve the baseline harness' original runtime configuration;
            # Lite's deterministic matmul policy belongs to Lite workers.
            from scripts.bench_llama_baselines import device_check as baseline_device_check
            torch, device = baseline_device_check(args.device, args.software_check)
        (worker if args.worker else receive)(q, torch, device)
        return 0
    if args.output is None:
        p.error("--output required")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    result = dict(format="lite_baselines_v1", status="running", gpu_measured=False, jobs={}, rows=[])
    writable = not (out / "results.json").exists()
    def save():
        result["rows"] = [row for job in result["jobs"].values() if job.get("status") == "completed" for row in job["rows"]]
        result["aggregates"] = aggregate(result["rows"])
        result["network_model"] = network_model(result["aggregates"])
        write(out / "results.json", result); report(result, out)
    try:
        torch, device = device_check(args.device)
        if args.warmup < 1 or args.repeats < 2:
            raise ValueError("warmup >=1 and repeats >=2 required")
        base, inputs = verified_inputs(args.base_run.resolve())
        baseline = json.loads((args.baseline_run / "results.json").read_text())
        if baseline["status"] != "completed" or baseline["base"]["results_sha256"] != digest(args.base_run / "results.json"):
            raise ValueError("original baseline evidence belongs to another/incomplete run")
        versions = {n: importlib.metadata.version(n) for n in ("torch", "triton", "numpy", "transformers")}
        if (versions != base["software"] or torch.version.cuda != base["cuda_runtime"] or
                torch.cuda.get_device_name(device) != base["device"]):
            raise ValueError("root software/GPU differs from the verified Lite run")
        for name, wanted in base["protocol"]["code"].items():
            if digest(ROOT / name) != wanted:
                raise ValueError("original measured core changed: " + name)
        build = json.loads((args.profile_run / "exans_build.json").read_text())
        if not build.get("gpu_author_smoke_passed") or digest(build["library"]) != build["library_sha256"]:
            raise ValueError("checked native ExANS library missing/changed")
        if importlib.metadata.version("nvidia-libnvcomp-cu12") != build["nvcomp"]:
            raise ValueError("use the original pinned nvCOMP runtime")
        split_lut = (args.baseline_run / "splitzip_official_framed/splitzip_luts.npz").resolve()
        split_rows = [r for r in baseline["rows"] if r["method"] == "splitzip_official_framed"]
        if not split_rows or any(digest(split_lut) != r["state_sha256"] for r in split_rows):
            raise ValueError("original SplitZip TRAIN LUT identity differs")
        expected_base = {f"{r['document']}:{r['kappa']:g}": r for r in base["rows"] if r["policy"] == "lite"}
        if set(expected_base) != {f"{e['document']}:{k:g}" for e in inputs for k in (0., .5, 1.)}:
            raise ValueError("requires the six previously verified Lite cases")
        code = {f: digest(ROOT / f) for f in FILES}
        recipe = dict(base_run=str(args.base_run.resolve()), base_results_sha256=digest(args.base_run / "results.json"),
            baseline_results_sha256=digest(args.baseline_run / "results.json"), exans_build_sha256=digest(args.profile_run / "exans_build.json"),
            exans_library_sha256=build["library_sha256"], nvcomp=build["nvcomp"],
            toolchain_sha256=digest(args.profile_run / "toolchain.json"),
            code=code, original_code=base["protocol"]["code"], splitzip_state_sha256=digest(split_lut),
            warmup=args.warmup, repeats=args.repeats, software=versions, device=base["device"], cuda_runtime=torch.version.cuda)
        identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        if not writable:
            old = json.loads((out / "results.json").read_text())
            if old.get("identity") != identity:
                raise ValueError("resume identity differs; preserve this run and choose a fresh output")
            result = old; writable = True
        result.update(identity=identity, recipe=recipe, status="running", gpu_measured=True,
            model=base["protocol"]["source"],
            hardware=dict(name=base["device"], multiprocessors=torch.cuda.get_device_properties(device).multi_processor_count))
        result.pop("error", None); result.pop("traceback", None)
        write(out / "source_manifest.json", recipe)
        save()
        cache = ROOT / "runs/_llama_official"
        exans_source = cache / "sources/exans"
        split_source = cache / "sources/splitzip"
        jobs = []
        for method in ("raw_bf16", "splitzip_official_framed", "fp8_e4m3fn_saturating"):
            rows = [r for r in baseline["rows"] if r["method"] == method]
            spec = dict(kind=method)
            python = sys.executable
            if method == "splitzip_official_framed":
                spec.update(source=str(split_source), state=str(split_lut), state_sha256=digest(split_lut))
                python = str(cache / "envs/splitzip/bin/python")
            jobs.append((method, spec, [0.], {f"{r['document']}:0": r for r in rows}, python, None))
        exans = json.loads((args.profile_run / "exans.json").read_text())
        native_spec = dict(kind="exans", source=str(exans_source), library=build["library"],
            library_sha256=build["library_sha256"], blocks=max(math.ceil(e["values"] * 2 / 999424) for e in inputs))
        from kv_compare.profile_recovery import runtime_environment
        toolchain = json.loads((args.profile_run / "toolchain.json").read_text())
        nvcomp = importlib.util.find_spec("nvidia.libnvcomp")
        if nvcomp is None:
            raise ValueError("existing pinned nvCOMP runtime missing")
        locations = list(nvcomp.submodule_search_locations or [])
        native_env = runtime_environment(os.environ, toolchain["cuda_home"], [Path(s) / "lib" for s in locations])
        jobs.append(("exans_official_fixed_framed", native_spec, [0.],
            {f"{r['document']}:0": r for r in exans["rows"]}, sys.executable, native_env))
        options = {k: base["protocol"][k] for k in ("predictor_format", "values_per_stream", "block")}
        for tag, block in (("b64_before", 64), ("b8", 8), ("b16", 16), ("b32", 32), ("b64_after", 64)):
            jobs.append(("lite_" + tag, dict(kind="lite", state=base["state_directories"]["lite"],
                engine_options=dict(options, block=block)), [0., .5, 1.], expected_base, sys.executable, None))
        result["selected_jobs"] = [j[0] for j in jobs]; save()
        for method, spec, kappas, expected, python, env in jobs:
            directory = out / method; directory.mkdir(exist_ok=True)
            old = result["jobs"].get(method, {})
            if old.get("status") == "completed":
                for name, wanted in old["artifacts"].items():
                    if digest(confined(out, name)) != wanted:
                        raise ValueError("completed worker evidence changed")
                for r in old["rows"]:
                    if digest(r["packet"]) != r["packet_sha256"]:
                        raise ValueError("completed packet changed")
                log("reuse completed method: " + method); continue
            q = dict(method=method, codec=spec, inputs=inputs, kappas=kappas, expected=expected,
                warmup=args.warmup, repeats=args.repeats, output=str(directory),
                require_original_packet=method.startswith("lite_b64"))
            write(directory / "request.json", q)
            launch([python, Path(__file__).resolve(), "--device", str(device), "--worker", directory / "request.json"],
                   directory / "worker.log", env=env)
            data = json.loads((directory / "worker_results.json").read_text())
            if data["status"] != "completed" or not data["gpu_measured"]:
                raise ValueError("worker did not complete on CUDA")
            result["jobs"][method] = dict(status="completed", rows=data["rows"],
                artifacts={str((directory / f).relative_to(out)): digest(directory / f)
                           for f in ("worker_results.json", "receiver.json")})
            save()
        validate_comparison(result["rows"])
        result["status"] = "completed"; save()
        log("completed: " + str(out / "summary.md")); return 0
    except Exception as error:
        if writable:
            result.update(status="failed", error=str(error), traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__ == "__main__":
    raise SystemExit(main())

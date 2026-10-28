#!/usr/bin/env python3
"""Compare unchanged Lite with host-keyed RoPE using existing state and packets."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_llama_causal import device_check, log, timed
from scripts.bench_llama_baselines import call, confined, verified_inputs

ARMS = ("reference", "host_rope")
FILES = ("kv_compare/lite_fast.py", "scripts/bench_lite_fast.py",
         "scripts/package_lite_fast.py", "scripts/run_lite_fast_a4500.sh",
         "kv_compare/causal_profile.py", "scripts/bench_llama_baselines.py")


def paired_timed(functions, torch, device, warmup, repeats):
    """Alternate AB/BA at each round; both arms have the same resident setup."""
    values = {a: None for a in ARMS}
    data = {a: dict(samples=[], warmup_samples=[]) for a in ARMS}
    order_log = []
    for index in range(warmup + repeats):
        order = ARMS if index % 2 == 0 else ARMS[::-1]
        order_log.append(list(order))
        for arm in order:
            values[arm] = None
            value, measurement = timed(functions[arm], torch, device, 0, 1)
            values[arm] = value
            data[arm]["warmup_samples" if index < warmup else "samples"].append(measurement["samples"][0])
    for arm in ARMS:
        data[arm]["wall_median_ms"] = statistics.median(x["wall_ms"] for x in data[arm]["samples"])
    return values, data, order_log


def compare_case(engines, words, wire, expected, options, torch, device):
    from kv_compare.causal_packet import Packet, word_sha
    h = Packet.restore(wire, device).header
    packets, encode, enc_order = paired_timed({a: lambda a=a: engines[a].encode(
        words, start=h["start"], kappa=h["kappa"]) for a in ARMS},
        torch, device, options.warmup, options.repeats)
    for arm in ARMS:
        if packets[arm].export() != wire:
            raise AssertionError(arm + " changed original packet bytes")
        if word_sha(engines[arm].last_reconstruction) != expected["decoded_word_sha256"]:
            raise AssertionError(arm + " changed encoder reconstruction")
        engines[arm].last_reconstruction = None
    # Use the same already-uploaded original packet for both receivers. Full
    # packet-directory validation stays inside every timed decode invocation.
    packet = Packet.restore(wire, device)
    del packets
    recovered, decode, dec_order = paired_timed({a: lambda a=a: engines[a].decode(packet)
        for a in ARMS}, torch, device, options.warmup, options.repeats)
    for arm in ARMS:
        if word_sha(recovered[arm]) != expected["decoded_word_sha256"]:
            raise AssertionError(arm + " changed decoded BF16 words")
        if not h["kappa"] and not torch.equal(recovered[arm], words):
            raise AssertionError(arm + " lossless reconstruction differs from original")
    original_exact = bool(torch.equal(recovered["host_rope"], words))
    del recovered
    return dict(id=expected["id"], document=expected["document"], kappa=h["kappa"],
        shape=h["shape"], start=h["start"], values=words.numel(), packet_bytes=len(wire),
        complete_bpv=8 * len(wire) / words.numel(), saved_percent=100 * (1 - len(wire) / words.numel() / 2),
        packet_sha256=hashlib.sha256(wire).hexdigest(), state_sha256=h["state_sha256"],
        original_word_sha256=expected["original_word_sha256"],
        decoded_word_sha256=expected["decoded_word_sha256"], packet_matches_original=True,
        original_exact=original_exact, encode=encode, decode=decode,
        encode_speedup=encode["reference"]["wall_median_ms"] / encode["host_rope"]["wall_median_ms"],
        decode_speedup=decode["reference"]["wall_median_ms"] / decode["host_rope"]["wall_median_ms"],
        sample_order=dict(encode=enc_order, decode=dec_order), status="matched")


def receive(path, torch, device):
    """Receiver never loads source KV, source captures or prediction arrays."""
    from kv_compare.causal_packet import Packet, sha, word_sha, write_json
    from kv_compare.lite_fast import FastEngine
    q = json.loads(path.read_text())
    engine = FastEngine(q["state_dir"], device, **q["engine_options"])
    checks = []
    for e in q["entries"]:
        if sha(e["packet"]) != e["packet_sha256"]:
            raise ValueError("receiver packet changed")
        words = engine.decode(Packet.restore(Path(e["packet"]).read_bytes(), device))
        digest = word_sha(words)
        if digest != e["expected_word_sha256"]:
            raise AssertionError("fresh host-RoPE receiver changed decoded words")
        checks.append(dict(id=e["id"], verified=True, packet_sha256=e["packet_sha256"],
                           word_sha256=digest, shape=list(words.shape)))
        log("fresh host-RoPE receiver passed: " + e["id"])
    write_json(q["output"], dict(gpu_measured=device.type == "cuda", arm="host_rope",
        no_source_kv_or_prediction_arrays=True,
        inputs="fixed fitted state + original packet bytes; expected digests are assertions", checks=checks))


def diagnostic_case(engine, words, wire, expected, device, out, arm):
    from kv_compare.causal_packet import Packet, sha, word_sha
    from kv_compare.causal_profile import profile_call
    packet = Packet.restore(wire, device)
    profiles = {}
    for direction, fn in (("encode", lambda: engine.encode(words, start=packet.header["start"],
                                                           kappa=packet.header["kappa"])),
                          ("decode", lambda: engine.decode(packet))):
        trace = out / "traces" / f"{expected['id']}_{arm}_{direction}.json.gz"
        value, data = profile_call(fn, engine, packet.header["kappa"], device, trace)
        if direction == "encode":
            if value.export() != wire:
                raise AssertionError("profiling changed packet bytes")
            engine.last_reconstruction = None
        elif word_sha(value) != expected["decoded_word_sha256"]:
            raise AssertionError("profiling changed decoded words")
        data["trace"] = str(trace.relative_to(out)); data["trace_sha256"] = sha(trace)
        data["top_self_device"] = sorted((r for r in data["top_self_device"]
            if not r["name"].startswith("kv/")), key=lambda r: r["self_device_ms"], reverse=True)
        data["cuda_stream_synchronize_calls"] = sum(r["calls"] for r in data["operators"]
            if r["name"] == "cudaStreamSynchronize")
        profiles[direction] = data
        del value
    return profiles


def report(result, out):
    lines = ["# Lite host-RoPE optimization", "",
        f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured', False)}**.", "",
        "Existing Lite fitted state and original packets; both arms must reproduce exactly the same bytes.",
        "AB/BA alternating ordinary synchronized wall samples. Speedup is reference / host-RoPE; below 1 is slower.", "",
        "| Document | Kappa | Complete bpv | Reference encode ms | Host-RoPE encode ms | Encode speedup | Reference decode ms | Host-RoPE decode ms | Decode speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result.get("rows", []):
        enc, dec = r["encode"], r["decode"]
        lines.append(f"| {r['document']} | {r['kappa']:g} | {r['complete_bpv']:.5f} | "
            f"{enc['reference']['wall_median_ms']:.3f} | {enc['host_rope']['wall_median_ms']:.3f} | "
            f"{r['encode_speedup']:.3f}x | {dec['reference']['wall_median_ms']:.3f} | "
            f"{dec['host_rope']['wall_median_ms']:.3f} | {r['decode_speedup']:.3f}x |")
    if result.get("profiles"):
        lines += ["", "Profiler diagnostics below are separate instrumented runs, not ordinary latency.", "",
            "| Case / arm | Encode stream synchronizations | Decode stream synchronizations |",
            "|---|---:|---:|"]
        for ident, arms in result["profiles"].items():
            for arm, p in arms.items():
                lines.append(f"| {ident} / {arm} | {p['encode']['cuda_stream_synchronize_calls']} | "
                    f"{p['decode']['cuda_stream_synchronize_calls']} |")
    if result.get("error"):
        lines += ["", "Failure: " + result["error"]]
    lines += ["", "Kappa 0 is lossless; nonzero kappas reproduce the existing lossy reconstruction exactly.",
        "Packet export/upload, offline fitting and model quality evaluation are outside codec timings.",
        "Both engines are resident during paired timing; Torch baselines/peaks include this paired setup.",
        "Cold state initialization is separate. Shared fitted state, packet bytes and sample distributions are in results.json.",
        "Full traces remain under runs/. No pool conversion, attention, network, SSD or serving result is implied."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-run", type=Path, default=ROOT / "runs/llama_causal_a4500_20261027_184615_1307629")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kappas", default="0,0.5,1")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--profile", action="store_true", help="profile lossless test_text after ordinary measurements")
    p.add_argument("--verify-request", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--software-check", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.verify_request:
        torch, device = device_check(args.device, args.software_check)
        receive(args.verify_request, torch, device)
        return 0
    if args.output is None:
        p.error("--output is required")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    result = dict(format="lite_fast_v1", status="running", gpu_measured=False, rows=[], profiles={})
    writable = not (out / "results.json").exists()
    def save():
        write_json(out / "results.json", result); report(result, out)
    try:
        import numpy as np
        from kv_compare.causal_packet import Engine, sha, word_sha, write_json
        from kv_compare.lite_fast import FastEngine
        torch, device = device_check(args.device)
        kappas = [float(x) for x in args.kappas.split(",")]
        if (len(kappas) != len(set(kappas)) or not kappas or
                any(not 0 <= x <= 16 for x in kappas) or args.warmup < 1 or args.repeats < 2):
            raise ValueError("choose unique measured kappas, warmup >=1 and repeats >=2")
        base_dir = args.base_run.resolve()
        base, inputs = verified_inputs(base_dir)
        versions = {n: importlib.metadata.version(n) for n in ("torch", "triton", "numpy", "transformers")}
        if (versions != base["software"] or torch.version.cuda != base["cuda_runtime"] or
                torch.cuda.get_device_name(device) != base["device"]):
            raise ValueError("use the same GPU model and root software as the original Lite run")
        for path, digest in base["protocol"]["code"].items():
            if sha(ROOT / path) != digest:
                raise ValueError("original measured core changed: " + path)
        selected = [r for r in base["rows"] if r["policy"] == "lite" and r["kappa"] in kappas]
        if {(r["document"], r["kappa"]) for r in selected} != {
                (e["document"], k) for e in inputs for k in kappas}:
            raise ValueError("every requested Lite mode needs original packets on every document")
        options = {k: base["protocol"][k] for k in ("predictor_format", "values_per_stream", "block")}
        state_dir = Path(base["state_directories"]["lite"]).resolve()
        recipe = dict(base_run=str(base_dir), base_results_sha256=sha(base_dir / "results.json"),
            code={f: sha(ROOT / f) for f in FILES}, original_code=base["protocol"]["code"],
            engine_options=options, kappas=kappas, warmup=args.warmup, repeats=args.repeats,
            profile=args.profile, software=versions, cuda_runtime=torch.version.cuda, device=base["device"])
        identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        if not writable:
            old = json.loads((out / "results.json").read_text())
            if old.get("identity") != identity:
                raise ValueError("resume identity differs; keep this run and use a fresh output")
            result = old; writable = True
        result.update(identity=identity, recipe=recipe, status="running", gpu_measured=True, inputs=inputs,
            state_dir=str(state_dir), selected_ids=[r["id"] for r in selected])
        result.pop("error", None); result.pop("traceback", None)
        shutil.copyfile(base_dir / "results.json", out / "base_results.json")
        write_json(out / "source_manifest.json", dict(files=recipe["code"], original_files=recipe["original_code"]))
        save()
        engines = {}; cold = {}
        for arm, cls in (("reference", Engine), ("host_rope", FastEngine)):
            log("load existing Lite state: " + arm)
            begin = time.perf_counter(); engines[arm] = cls(state_dir, device, **options)
            torch.cuda.synchronize(device)
            cold[arm] = dict(wall_ms=(time.perf_counter() - begin) * 1000,
                             shared_predictor_gpu_bytes=engines[arm].shared_bytes)
        result["cold_state_initialization"] = cold
        if any(r["state_sha256"] != engines["reference"].state_sha256 for r in selected):
            raise ValueError("saved fitted-state identity differs")
        for kappa in kappas:
            if kappa:
                for engine in engines.values():
                    engine.residual(kappa)
        saved = {r["id"]: r for r in result["rows"]}
        replay = dict(state_dir=str(state_dir), engine_options=options, entries=[],
                      output=str(out / "independent_replay.json"))
        for e in inputs:
            words = torch.from_numpy(np.load(e["path"], allow_pickle=False).view(np.int16).copy()).to(device)
            if word_sha(words) != e["original_word_sha256"]:
                raise ValueError("source words changed after validation")
            for original in (r for r in selected if r["document"] == e["document"]):
                ident = original["id"]
                path = confined(base_dir, original["packet"])
                wire = path.read_bytes()
                if hashlib.sha256(wire).hexdigest() != original["packet_sha256"]:
                    raise ValueError("original measured packet changed")
                if ident not in saved:
                    log("paired Lite encode/decode: " + ident)
                    row = compare_case(engines, words, wire, original, args, torch, device)
                    result["rows"].append(row); saved[ident] = row; save()
                elif any(saved[ident][key] != original[key] for key in (
                        "packet_sha256", "decoded_word_sha256", "state_sha256", "original_word_sha256")):
                    raise ValueError("resumed measured evidence differs from original")
                else:
                    log("reuse completed paired measurement: " + ident)
                replay["entries"].append(dict(id=ident, packet=str(path),
                    packet_sha256=original["packet_sha256"], expected_word_sha256=original["decoded_word_sha256"]))
                if args.profile and original["document"] == "test_text" and original["kappa"] == 0:
                    profiles = result["profiles"].setdefault(ident, {})
                    for arm in ARMS:
                        if arm not in profiles:
                            log("separate diagnostic profile: " + ident + " / " + arm)
                            profiles[arm] = diagnostic_case(engines[arm], words, wire, original, device, out, arm)
                            save()
                        else:
                            for data in profiles[arm].values():
                                if sha(confined(out, data["trace"])) != data["trace_sha256"]:
                                    raise ValueError("completed trace changed")
            del words; gc.collect()
        # The paired engines and all original words leave scope before the
        # independent receiver process is launched.
        engines.clear(); engine = None; gc.collect(); torch.cuda.empty_cache()
        write_json(out / "receiver_request.json", replay)
        call([sys.executable, Path(__file__).resolve(), "--device", str(device),
              "--verify-request", out / "receiver_request.json"], out / "receiver.log")
        proof = json.loads((out / "independent_replay.json").read_text())
        if (not proof.get("gpu_measured") or not proof.get("no_source_kv_or_prediction_arrays") or
                {c["id"] for c in proof["checks"]} != set(result["selected_ids"])):
            raise ValueError("fresh GPU receiver coverage incomplete")
        result["receiver_sha256"] = sha(out / "independent_replay.json")
        result["status"] = "completed"; save()
        log("completed: " + str(out / "summary.md"))
        return 0
    except Exception as error:
        if writable:
            result.update(status="failed", error=str(error), traceback=traceback.format_exc())
            # Early device failures must still produce an honest report even
            # when CUDA/import checks stopped before causal_packet was loaded.
            tmp = out / "results.json.part"
            tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            tmp.replace(out / "results.json"); report(result, out)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

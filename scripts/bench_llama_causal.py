#!/usr/bin/env python3
"""Matched complete causal GPU packets: Block32 policy vs the team's lite policy,
both losslessly and with quantized residuals. Reuses frozen captures and cached
fits. This is a predictor-policy comparison, not the historical INT64 bitstream
or an official AQUA/ExANS/SplitZip reproduction.
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
import json
from pathlib import Path
import statistics
import subprocess
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(message):
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


def device_check(device, software_check=False):
    import torch
    d = torch.device(device)
    if software_check:
        if d.type != "cpu" or os.environ.get("TRITON_INTERPRET") != "1":
            raise RuntimeError("software checks require CPU and TRITON_INTERPRET=1")
    elif d.type != "cuda" or os.environ.get("TRITON_INTERPRET") or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required; no CPU/interpreter benchmark substitution")
    if d.type == "cuda":
        torch.cuda.set_device(d)
        torch.ones(1, device=d).add_(1); torch.cuda.synchronize(d)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch, d


def cached_model_snapshot():
    """Fail before fitting if the original model's local weight shards are missing."""
    from huggingface_hub import snapshot_download
    from kv_compare.causal_study import MODEL, REVISION
    root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    cfg = json.loads((root / "config.json").read_text())
    if (cfg.get("model_type") != "llama" or cfg.get("num_hidden_layers") != 28 or
            cfg.get("quantization_config")):
        raise ValueError("cached model is not the pinned unquantized Llama")
    index = root / "model.safetensors.index.json"
    files = (set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else
             {"model.safetensors"})
    if not files or any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in files):
        raise ValueError("original model weight shards are missing from the local cache")
    return root


def timed(fn, torch, device, warmup, repeats):
    warm, samples = [], []
    value = None
    for index in range(warmup + repeats):
        value = None; gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        begin = time.perf_counter()
        value = fn()
        if device.type == "cuda":
            b.record(); torch.cuda.synchronize(device)
        row = dict(wall_ms=(time.perf_counter() - begin) * 1000)
        if device.type == "cuda":
            row.update(cuda_interval_ms=a.elapsed_time(b), baseline_torch_bytes=baseline,
                peak_torch_bytes=torch.cuda.max_memory_allocated(device),
                incremental_peak_torch_bytes=torch.cuda.max_memory_allocated(device) - baseline)
        (warm if index < warmup else samples).append(row)
    return value, dict(wall_median_ms=statistics.median(r["wall_ms"] for r in samples),
        samples=samples, warmup_samples=warm)


def verify_request(path, torch, device, options):
    """Fresh process: reads fitted state, packets and expected digests only."""
    import numpy as np
    from kv_compare.causal_packet import Engine, Packet, sha, word_sha, write_json
    request = json.loads(path.read_text())
    checks = []
    engine = None; current = None
    for e in request["entries"]:
        if current != e["state_dir"]:
            engine = None; gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            engine = Engine(e["state_dir"], device, **request["engine_options"])
            current = e["state_dir"]
        wire = Path(e["packet"]).read_bytes()
        if hashlib.sha256(wire).hexdigest() != e["packet_sha256"]:
            raise ValueError("saved packet changed before independent replay")
        recovered = engine.decode(Packet.restore(wire, device))
        digest = word_sha(recovered)
        if digest != e["expected_word_sha256"]:
            raise AssertionError("independent receiver recovered different words: " + e["id"])
        target = Path(e["decoded"]); target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, recovered.cpu().numpy().astype("<u2"), allow_pickle=False)
        checks.append(dict(id=e["id"], word_sha256=digest, decoded_file_sha256=sha(target), verified=True))
        log("independent packet replay passed: " + e["id"])
    write_json(Path(request["output"]), dict(format="causal_independent_replay_v1",
        gpu_measured=device.type == "cuda", received="fixed fitted state + wire packets + expected digests",
        no_source_kv_or_prediction_arrays=True, checks=checks))


def run_codec(args, result, study, tests, states, out, torch, device):
    import numpy as np
    from kv_compare.causal_packet import Engine, Packet, sha, word_sha, write_json
    from kv_compare.core import words_to_float
    engine_options = dict(predictor_format=args.predictor_format,
                          values_per_stream=args.values_per_stream, block=args.block)
    result.setdefault("rows", [])
    saved = {r["id"]: r for r in result["rows"]}
    request = dict(engine_options=engine_options, entries=[], output=str(out / "independent_replay.json"))
    def replay_entry(row, state_dir):
        return dict(id=row["id"], state_dir=str(state_dir),
            packet=str(out / row["packet"]), packet_sha256=row["packet_sha256"],
            expected_word_sha256=row["decoded_word_sha256"], decoded=str(out / row["decoded"]))
    modes = [0.] + args.kappas
    for name in args.policies:
        log("load fixed receiver state: " + name)
        begin = time.perf_counter()
        engine = Engine(states[name], device, **engine_options)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result.setdefault("cold_state_initialization_wall_ms", {})[name] = (time.perf_counter() - begin) * 1000
        state_files = ["state.json"] + [e["path"] for e in engine.meta["files"]]
        if engine.meta.get("runtime_spread"):
            state_files.append(engine.meta["runtime_spread"]["path"])
        state_bytes = sum((states[name] / p).stat().st_size for p in state_files)
        for kappa in modes:
            if kappa:
                engine.residual(kappa)
            for doc in tests:
                ident = f"{name}_k{kappa:g}_{doc.name}"
                if ident in saved:
                    row = saved[ident]
                    if sha(out / row["packet"]) != row["packet_sha256"] or row["state_sha256"] != engine.state_sha256:
                        raise ValueError("saved packet or fitted state changed")
                    original = torch.stack([doc.words(l, "cpu", args.tokens) for l in range(study.L)]).to(torch.int16)
                    if word_sha(original) != row["original_word_sha256"]:
                        raise ValueError("saved measurement uses different original words")
                    if not kappa and row["decoded_word_sha256"] != row["original_word_sha256"]:
                        raise ValueError("saved lossless reconstruction differs")
                    request["entries"].append(replay_entry(row, states[name]))
                    log("reuse measured packet: " + ident)
                    continue
                log("causal encode/decode: " + ident)
                source = torch.stack([doc.words(l, device, args.tokens) for l in range(study.L)]).to(torch.int16)
                original_sha = word_sha(source)
                packet, enc = timed(lambda: engine.encode(source, kappa=kappa),
                    torch, device, args.warmup, args.repeats)
                target_sha = word_sha(engine.last_reconstruction)
                engine.last_reconstruction = None
                restored, dec = timed(lambda: engine.decode(packet),
                    torch, device, args.warmup, args.repeats)
                if word_sha(restored) != target_sha:
                    raise AssertionError("receiver differs from encoder reconstruction")
                original_exact = bool(torch.equal(source, restored))
                if not kappa and not original_exact:
                    raise AssertionError("lossless round trip changed original BF16 words")
                x, y = words_to_float(source), words_to_float(restored)
                rel = float(torch.linalg.vector_norm(y - x) / torch.linalg.vector_norm(x).clamp_min(1e-30))
                word_match = float((source == restored).float().mean())
                begin = time.perf_counter(); wire = packet.export()
                export_ms = (time.perf_counter() - begin) * 1000
                begin = time.perf_counter(); reloaded = Packet.restore(wire, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                import_ms = (time.perf_counter() - begin) * 1000
                if not torch.equal(engine.decode(reloaded), restored):
                    raise AssertionError("serialized packet changed reconstruction")
                path = out / "packets" / (ident + ".kvc"); path.parent.mkdir(exist_ok=True)
                path.write_bytes(wire)
                target = out / "reconstructed" / (ident + ".npy")
                digest = hashlib.sha256(wire).hexdigest()
                values = source.numel()
                row = dict(id=ident, document=doc.name, policy=name, kappa=kappa,
                    mode="lossless" if not kappa else "lossy", tokens=args.tokens, layers=study.L,
                    values=values, packet_bytes=len(wire), payload_bytes=packet.payload.numel() * 2,
                    directory_bytes=packet.starts.numel() * 8, complete_bpv=8 * len(wire) / values,
                    saved_percent=100 * (1 - len(wire) / (values * 2)), original_exact=original_exact,
                    original_word_sha256=original_sha, decoded_word_sha256=target_sha,
                    word_match_fraction=word_match, relative_l2_error=rel,
                    state_sha256=engine.state_sha256, shared_predictor_gpu_bytes=engine.shared_bytes,
                    fixed_decoder_state_file_bytes=state_bytes,
                    first_packet_plus_fixed_state_bpv=8 * (len(wire) + state_bytes) / values,
                    raw_fallback=packet.header["raw"], encode=enc, decode=dec,
                    serialized_export_wall_ms=export_ms, import_and_upload_wall_ms=import_ms,
                    packet=str(path.relative_to(out)), packet_sha256=digest,
                    decoded=str(target.relative_to(out)), status="pending_independent_verification")
                result["rows"].append(row)
                request["entries"].append(replay_entry(row, states[name]))
                write_json(out / "results.json", result)
                del source, packet, restored, reloaded, x, y, wire
        del engine
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(out / "verification_request.json", request)
    argv = [sys.executable, str(Path(__file__).resolve()), "--device", str(device),
            "--verify-request", str(out / "verification_request.json")]
    if args.software_check:
        argv.append("--software-check")
    with (out / "independent_replay.log").open("wb") as handle:
        proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    code = proc.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    log("independent receiver is running; " + str(out / "independent_replay.log"))
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
        if code:
            raise RuntimeError("independent receiver failed; see independent_replay.log")
    proof = json.loads((out / "independent_replay.json").read_text())
    checks = {r["id"]: r for r in proof["checks"]}
    if set(checks) != {r["id"] for r in result["rows"]}:
        raise ValueError("independent replay coverage differs")
    for row in result["rows"]:
        check = checks[row["id"]]
        if not check["verified"] or check["word_sha256"] != row["decoded_word_sha256"]:
            raise AssertionError("independent replay failed")
        row.update(status="verified", decoded_file_sha256=check["decoded_file_sha256"])
    result["codec_completed"] = True
    write_json(out / "results.json", result)


def run_quality(args, result, tests, out, torch, device):
    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from kv_compare.causal_packet import sha
    from kv_compare.causal_study import MODEL, REVISION
    from kv_compare.quality import continuation_nll, matrices_from_doc
    snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    log("load the cached original BF16 model for continuation quality")
    model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to(device).eval()
    if getattr(model.config, "quantization_config", None):
        raise ValueError("quantized weights are not allowed")
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("original BF16 parameters required")
    references = {}
    for doc in tests:
        reference = matrices_from_doc(doc, args.tokens)
        nll = continuation_nll(model, doc.ids, args.tokens, args.continuation, device, reference)
        again = continuation_nll(model, doc.ids, args.tokens, args.continuation, device, reference)
        if nll != again:
            raise AssertionError("restoring the same exact KV twice changed continuation NLL")
        own = continuation_nll(model, doc.ids, args.tokens, args.continuation, device)
        if abs(own - nll) > .01 * abs(own) + .002:
            raise AssertionError("captured exact KV is far from the model's own prefill; check input/model/cache identity")
        references[doc.name] = dict(nll=nll, ppl=float(np.exp(nll)),
            own_prefill_nll=own, scored_tokens=args.continuation - 1)
        for row in [r for r in result["rows"] if r["document"] == doc.name]:
            p = out / row["decoded"]
            if sha(p) != row["decoded_file_sha256"]:
                raise ValueError("verified receiver output changed")
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            C = a.shape[-1] // 2
            mats = [(np.array(a[l, :, :C]), np.array(a[l, :, C:])) for l in range(a.shape[0])]
            value = continuation_nll(model, doc.ids, args.tokens, args.continuation, device, mats)
            if row["original_exact"] and abs(value - nll) > 1e-6:
                raise AssertionError("exact transferred KV changed continuation quality")
            row["quality"] = dict(nll=value, ppl=float(np.exp(value)),
                ppl_change_percent=float(100 * np.expm1(value - nll)), scored_tokens=args.continuation - 1)
            log(f"quality {row['id']}: {row['quality']['ppl_change_percent']:+.3f}%")
    result["quality_reference"] = references
    result["quality_completed"] = True
    del model


def report(result, out):
    lines = ["# Matched causal GPU KV packet comparison", "",
        f"Status: **{result.get('status', 'unknown')}**. GPU measured: **{result.get('gpu_measured', False)}**.", "",
        "All rates below are serialized complete packets; shared fitted state is additional.",
        "Block32 is a matched predictor-policy control in the new shared coder, not the historical INT64 bitstream.",
        "Lossless means every original BF16 word matches. Lossy packets must exactly reproduce the encoder's intended reconstruction.",
        "Receiver replay runs in a separate process with fixed state and packet bytes only.", "",
        "| Document | Policy | Mode / step | Complete bpv | Saved | Encode wall ms | Decode wall ms | Original exact | PPL change | Verified |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---|"]
    source = result.get("protocol", {}).get("source", {})
    if source:
        lines.insert(4, f"Model: {source['model']} at {source['revision']}; all 28 original BF16 layers.")
    for r in result.get("rows", []):
        q = r.get("quality", {}).get("ppl_change_percent")
        lines.append(f"| {r['document']} | {r['policy']} | {r['mode']} / {r['kappa']:g} | {r['complete_bpv']:.5f} | "
            f"{r['saved_percent']:.2f}% | {r['encode']['wall_median_ms']:.3f} | {r['decode']['wall_median_ms']:.3f} | "
            f"{r['original_exact']} | {'unmeasured' if q is None else f'{q:+.3f}%'} | {r['status']} |")
    if result.get("error"):
        lines += ["", "Failure: " + result["error"]]
    lines += ["", "Timing includes causal prediction, RoPE, adaptive widths, GPU entropy coding and compaction.",
        "One final size synchronization and packet-directory validation use CPU control. Serialization/export, upload, fitting and model quality evaluation are separate.",
        "Samples and Torch peaks are in results.json. Shared-state bytes exclude dynamic scratch; no resident-pool or serving claim.",
        "BF16 products are validated on the same GPU/software. Cross-GPU reproducibility is untested.",
        "Continuation quality is a short matched smoke evaluation, not full WikiText2 PPL or official AQUA/ExANS/SplitZip reproduction."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--policies", default="block32,lite")
    p.add_argument("--kappas", default="0.5,1")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--continuation", type=int, default=128)
    p.add_argument("--train-docs", type=int, default=4)
    p.add_argument("--fit-tokens", type=int, default=1024)
    p.add_argument("--scale-tokens", type=int, default=1024)
    p.add_argument("--predictor-format", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--values-per-stream", type=int, default=128)
    p.add_argument("--block", type=int, choices=[32, 64, 128], default=64)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--stages", default="fit,codec,quality,report")
    p.add_argument("--fit-cache", type=Path, default=ROOT / "runs/_causal_fit_cache")
    p.add_argument("--verify-request", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--software-check", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.verify_request:
        torch, device = device_check(args.device, args.software_check)
        verify_request(args.verify_request, torch, device, args)
        return 0
    if args.output is None or args.source_run is None:
        raise ValueError("--source-run and --output are required")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    if not stages or set(stages) - {"fit", "codec", "quality", "report"}:
        raise ValueError("invalid stages")
    path = out / "results.json"
    if stages == ["report"]:
        result = json.loads(path.read_text()); report(result, out); return 0
    from kv_compare.causal_packet import json_bytes, sha, write_json
    result = dict(format="causal_compare_v1", status="running", gpu_measured=False, rows=[])
    identity_ready = False
    try:
        torch, device = device_check(args.device, args.software_check)
        import numpy as np
        from kv_compare.causal_study import CODE, ROOT, source_inputs, bounded, fit_policy, calibrate_runtime
        from kv_compare.predict import Study
        args.policies = args.policies.split(",")
        args.kappas = [float(x) for x in args.kappas.split(",")]
        if (not args.policies or len(set(args.policies)) != len(args.policies) or
                set(args.policies) - {"block32", "lite", "strong_noatt"} or
                any(not 0 < k <= 16 for k in args.kappas) or len(set(args.kappas)) != len(args.kappas) or
                not 0 < args.tokens <= 8192 or args.continuation < 2 or args.repeats < 1 or args.warmup < 1 or
                min(args.fit_tokens, args.scale_tokens, args.train_docs, args.values_per_stream) < 1):
            raise ValueError("invalid experiment options")
        root, train, tables, tests_names, source = source_inputs(args.source_run, out, args.train_docs, log)
        code = {name: sha(ROOT / name) for name in CODE}
        software = {p: importlib.metadata.version(p) for p in ("torch","triton","numpy","transformers")}
        arithmetic = dict(software=software, cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu interpreter",
            capability=list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        protocol = dict(source=source, code=code, policies=args.policies, kappas=args.kappas,
            tokens=args.tokens, continuation=args.continuation, train=train,
            fit_tokens=args.fit_tokens, scale_tokens=args.scale_tokens,
            predictor_format=args.predictor_format, values_per_stream=args.values_per_stream,
            block=args.block, warmup=args.warmup, repeats=args.repeats, arithmetic=arithmetic)
        identity = hashlib.sha256(json_bytes(protocol)).hexdigest()
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("identity") != identity:
                raise ValueError("run identity changed; choose a fresh output directory")
            result = old
        identity_ready = True
        result.update(identity=identity, protocol=protocol, status="running", gpu_measured=device.type == "cuda",
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu interpreter")
        result["software"] = software
        result["cuda_runtime"] = torch.version.cuda
        write_json(out / "input_manifest.json", source)
        write_json(out / "source_manifest.json", dict(files=code))
        study = Study(root, device, train, tables, tests_names, log=log)
        study.train = [bounded(d, args.fit_tokens) for d in study.train]
        study.tables = bounded(study.tables, args.scale_tokens)
        if any(d.tokens < args.tokens + args.continuation for d in study.tests):
            raise ValueError("TEST captures are too short for prompt plus continuation")
        if study.L != 28:
            raise ValueError("production run requires all 28 layers")
        if "quality" in stages and not result.get("quality_completed"):
            cached_model_snapshot()
        recipe = dict(source=source, code=code, train=train,
            train_lengths=[d.tokens for d in study.train], scale_tokens=study.tables.tokens,
            predictor_format=args.predictor_format, arithmetic=arithmetic)
        fit_id = hashlib.sha256(json_bytes(recipe)).hexdigest()
        states = {name: args.fit_cache.resolve() / fit_id / name for name in args.policies}
        result["state_directories"] = {k: str(v) for k, v in states.items()}
        if "fit" in stages:
            for name in args.policies:
                log("fit/reuse fixed state: " + name)
                fit_policy(study, name, states[name], recipe, log)
                study._f.clear()
                calibrate_runtime(study, states[name], args.predictor_format, log)
            result["fit_completed"] = True
            write_json(path, result)
        if "codec" in stages:
            if result.get("codec_completed"):
                for row in result["rows"]:
                    if sha(out / row["packet"]) != row["packet_sha256"] or sha(out / row["decoded"]) != row["decoded_file_sha256"]:
                        raise ValueError("completed packet evidence changed")
                log("reuse verified codec results")
            else:
                run_codec(args, result, study, study.tests, states, out, torch, device)
        if "quality" in stages and not result.get("quality_completed"):
            if not result.get("codec_completed"):
                raise ValueError("quality requires verified complete packets")
            run_quality(args, result, study.tests, out, torch, device)
        result["status"] = "completed" if result.get("codec_completed") and result.get("quality_completed") else "partial"
        result.pop("error", None); result.pop("traceback", None)
        write_json(path, result); report(result, out)
        return 0 if result["status"] == "completed" else 1
    except BaseException as error:
        if not identity_ready and path.exists():
            log("Existing run preserved; startup validation failed: " + str(error))
            return 1
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error), traceback=traceback.format_exc())
        write_json(path, result); report(result, out)
        log(result["traceback"])
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())

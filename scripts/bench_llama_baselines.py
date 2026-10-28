#!/usr/bin/env python3
"""Add controls to an already verified causal run without refitting or retiming it."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(s):
    print(time.strftime("[%H:%M:%S] ") + s, flush=True)


def device_check(name, software=False):
    import torch
    device = torch.device(name)
    if software:
        if device.type != "cpu" or os.environ.get("TRITON_INTERPRET") != "1":
            raise RuntimeError("CPU software checks require TRITON_INTERPRET=1")
    elif device.type != "cuda" or os.environ.get("TRITON_INTERPRET") or not torch.cuda.is_available():
        raise RuntimeError("actual CUDA required; no CPU benchmark substitution")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
        torch.ones(1, device=device).add_(1); torch.cuda.synchronize(device)
    return torch, device


def call(argv, log_path):
    """Keep child logs and terminate the whole child group on interruption."""
    log("running: " + " ".join(map(str, argv)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as f:
        proc = subprocess.Popen(list(map(str, argv)), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    code = proc.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    log("still running; log: " + str(log_path))
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    if code:
        raise RuntimeError(f"exit {code}; see {log_path}")


def confined(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing/outside-run evidence: " + str(name))
    return path


def verified_inputs(base_dir):
    import numpy as np
    from kv_compare.causal_packet import sha
    from kv_compare.causal_study import MODEL, REVISION
    base = json.loads((base_dir / "results.json").read_text())
    if not (base.get("status") == "completed" and base.get("gpu_measured") and
            base.get("codec_completed") and base.get("quality_completed")):
        raise ValueError("a completed actual-GPU causal/quality run is required")
    source = base["protocol"]["source"]
    if source["model"] != MODEL or source["revision"] != REVISION:
        raise ValueError("wrong model/revision")
    proof = json.loads((base_dir / "independent_replay.json").read_text())
    checks = {c["id"]: c for c in proof["checks"]}
    if not proof.get("gpu_measured") or not proof.get("no_source_kv_or_prediction_arrays"):
        raise ValueError("independent GPU receiver proof required")
    inputs = []
    for name in sorted({r["document"] for r in base["rows"]}):
        rows = [r for r in base["rows"] if r["document"] == name]
        for r in rows:
            c = checks[r["id"]]
            if (r["status"] != "verified" or not c["verified"] or
                    r["decoded_word_sha256"] != c["word_sha256"] or
                    r["decoded_file_sha256"] != c["decoded_file_sha256"]):
                raise ValueError("base row is not independently verified")
        r = next(r for r in rows if r["mode"] == "lossless" and r["original_exact"])
        path = confined(base_dir, r["decoded"])
        if sha(path) != r["decoded_file_sha256"]:
            raise ValueError("base receiver array changed")
        words = np.load(path, mmap_mode="r", allow_pickle=False)
        if (words.dtype != np.uint16 or words.shape != (28, r["tokens"], 2048) or
                hashlib.sha256(words.astype("<u2").tobytes()).hexdigest() != r["original_word_sha256"] or
                r["original_word_sha256"] != r["decoded_word_sha256"] or
                any(x["original_word_sha256"] != r["original_word_sha256"] for x in rows)):
            raise ValueError("baseline input is not the same original BF16 cache")
        inputs.append(dict(document=name, path=str(path), file_sha256=sha(path),
            original_word_sha256=r["original_word_sha256"], shape=list(words.shape), values=words.size,
            tokens=r["tokens"], layers=28))
    return base, inputs


def capture_root(base_dir, base, explicit=None):
    from kv_compare.causal_packet import sha
    roots = [explicit] if explicit else [base_dir / "input_view", ROOT / "runs/llama32_3b_compare/captures"]
    for root in roots:
        if root and (root / "meta.json").is_file() and sha(root / "meta.json") == base["protocol"]["source"]["meta_sha256"]:
            return root.resolve()
    raise ValueError("matching capture metadata missing; pass --captures with the original captures directory")


def prepare_train(root, base, path, torch):
    """Exactly the author's default 4,194,304-value prefix of selected TRAIN words."""
    import numpy as np
    from kv_compare.causal_packet import word_sha
    from kv_compare.causal_study import array_word_digest
    need = 4_194_304
    pieces, selection = [], []
    for name in base["protocol"]["train"]:
        K = np.load(root / (name + "_K_bits.npy"), mmap_mode="r", allow_pickle=False)
        V = np.load(root / (name + "_V_bits.npy"), mmap_mode="r", allow_pickle=False)
        if array_word_digest(K, V) != base["protocol"]["source"]["documents"][name]["words_sha256"]:
            raise ValueError("TRAIN capture checksum changed")
        for layer in range(K.shape[0]):
            n = min(base["protocol"]["fit_tokens"], K.shape[1])
            words = np.concatenate([K[layer, :n], V[layer, :n]], axis=1).astype("<u2").reshape(-1)
            taken = min(need, words.size)
            pieces.append(words[:taken]); need -= taken
            selection.append(dict(document=name, layer=layer, tokens=n, prefix_values=taken))
            if not need:
                break
        if not need:
            break
    if need:
        raise ValueError("insufficient TRAIN values; author benchmark repetition is forbidden here")
    words = torch.from_numpy(np.concatenate(pieces).view(np.int16).copy())
    torch.save(words.view(torch.bfloat16), path)
    return dict(values=words.numel(), words_sha256=word_sha(words), selection=selection,
        rule="author default prefix cap, not a balanced sample; no TEST calibration")


def receiver(request, torch, device):
    import numpy as np
    from kv_compare.causal_packet import Packet, sha, word_sha, write_json
    from kv_compare.framed_baselines import Baseline, checked_splitzip
    q = json.loads(request.read_text())
    if q["method"] == "splitzip_official_framed":
        checked_splitzip(q["source"], software_check=q.get("software_check", False))
    codec = Baseline(q["method"], device, source=q.get("source"), state=q.get("state"))
    checks = []
    for e in q["entries"]:
        path = Path(e["packet"])
        if sha(path) != e["packet_sha256"]:
            raise ValueError("receiver wire checksum changed")
        restored = codec.decode(Packet.restore(path.read_bytes(), device))
        digest = word_sha(restored)
        if digest != e["expected_word_sha256"]:
            raise ValueError("receiver output differs from intended reconstruction")
        target = Path(e["decoded"])
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, restored.cpu().numpy().astype("<u2"), allow_pickle=False)
        checks.append(dict(id=e["id"], verified=True, word_sha256=digest, decoded_file_sha256=sha(target)))
    write_json(q["output"], dict(gpu_measured=device.type == "cuda", checks=checks,
        received="wire + fixed LUT state + expected digests", no_source_kv_or_predictions=True))


def worker(request, torch, device):
    import numpy as np
    from kv_compare.causal_packet import sha, word_sha, write_json
    from kv_compare.framed_baselines import Baseline, checked_splitzip
    from scripts.bench_llama_causal import timed
    q = json.loads(request.read_text()); out = Path(q["output"]); out.mkdir(parents=True, exist_ok=True)
    provenance = dict(python=sys.executable, packages={p: importlib.metadata.version(p) for p in ("torch","triton","numpy")},
        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU software check",
        gpu_measured=device.type == "cuda")
    if q["method"] == "splitzip_official_framed":
        provenance["official_source"] = checked_splitzip(q["source"], software_check=q.get("software_check", False))
    begin = time.perf_counter()
    codec = Baseline(q["method"], device, source=q.get("source"))
    provenance["codec_initialization_wall_ms"] = (time.perf_counter()-begin)*1000
    state = None
    if q["method"] == "splitzip_official_framed":
        train = torch.load(q["calibration"], map_location=device, weights_only=True).view(torch.int16)
        begin = time.perf_counter()
        state = out / "splitzip_luts.npz"; codec.calibrate(train, state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        provenance["calibration_and_state_export_wall_ms"] = (time.perf_counter()-begin)*1000
        del train
    rows, entries = [], []
    for e in q["inputs"]:
        if sha(e["path"]) != e["file_sha256"]:
            raise ValueError("worker input changed")
        a = np.load(e["path"], allow_pickle=False)
        source = torch.from_numpy(a.view(np.int16).copy()).to(device)
        if word_sha(source) != e["original_word_sha256"]:
            raise ValueError("worker original word identity differs")
        if q["method"].startswith("fp8") and not bool(torch.isfinite(source.view(torch.bfloat16).float()).all()):
            raise ValueError("the FP8 reference requires finite input")
        packet, enc = timed(lambda: codec.encode(source), torch, device, q["warmup"], q["repeats"])
        recovered, dec = timed(lambda: codec.decode(packet), torch, device, q["warmup"], q["repeats"])
        exact = bool(torch.equal(source, recovered))
        if q["method"] != "fp8_e4m3fn_saturating" and not exact:
            raise AssertionError("lossless baseline changed original BF16 words")
        ident = q["method"] + "_" + e["document"]
        begin = time.perf_counter(); wire = packet.export(); export_ms = (time.perf_counter()-begin)*1000
        from kv_compare.causal_packet import Packet
        begin = time.perf_counter(); imported = Packet.restore(wire, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        upload_ms = (time.perf_counter()-begin)*1000
        if not torch.equal(codec.decode(imported), recovered):
            raise AssertionError("serialized baseline packet changed reconstruction")
        packet_path = out / (ident + ".kvc"); packet_path.write_bytes(wire)
        decoded = out / (ident + ".npy")
        native = packet.header["extra"].get("native_payload_bytes")
        row = dict(e, id=ident, method=q["method"], policy=q["method"], kappa=0,
            mode="lossy" if q["method"].startswith("fp8") else "lossless",
            status="pending", original_exact=exact, decoded_word_sha256=word_sha(recovered),
            word_match_fraction=float((source==recovered).float().mean()),
            packet=str(packet_path), packet_sha256=sha(packet_path), decoded=str(decoded),
            packet_bytes=len(wire), complete_bpv=8*len(wire)/source.numel(),
            payload_bytes=sum(f["bytes"] for f in packet.header["fields"]),
            native_payload_bytes=native, native_payload_bpv=None if native is None else 8*native/source.numel(),
            saved_percent=100*(1-len(wire)/(source.numel()*2)), encode=enc, decode=dec,
            shared_predictor_gpu_bytes=codec.shared_bytes, fixed_decoder_state_file_bytes=0 if state is None else state.stat().st_size,
            state_sha256=codec.state_sha256, serialized_export_wall_ms=export_ms, import_and_upload_wall_ms=upload_ms,
            fp8_saturated_values=int((source.view(torch.bfloat16).float().abs()>448).sum()) if q["method"].startswith("fp8") else 0)
        rows.append(row)
        entries.append(dict(id=ident, packet=str(packet_path), packet_sha256=sha(packet_path),
                            expected_word_sha256=row["decoded_word_sha256"], decoded=str(decoded)))
        if q["method"] == "splitzip_official_framed" and not q.get("software_check"):
            tensor_path = out / (e["document"] + ".pt")
            torch.save(source.cpu().view(torch.bfloat16).flatten(), tensor_path)
            native_path = out / (e["document"] + "_native.json")
            call([sys.executable, Path(q["source"]) / "bench_codec_throughput.py", "--device", str(device),
                "--input", tensor_path, "--numel", str(source.numel()), "--calibration-input", q["calibration"],
                "--chunk-size", "1024", "--output", native_path], out / (e["document"] + "_native.log"))
            native_result = json.loads(native_path.read_text())
            if native_result["numel"] != source.numel() or native_result["compressed_bytes"] != native:
                raise ValueError("official native output has different input length or byte count")
            row["native_author_benchmark"] = native_result
            tensor_path.unlink()
        del source, packet, recovered, imported, wire
        gc.collect()
    replay = dict(method=q["method"], source=q.get("source"), state=None if state is None else str(state),
        software_check=q.get("software_check", False), entries=entries, output=str(out / "receiver.json"))
    write_json(out / "receiver_request.json", replay)
    argv = [sys.executable, Path(__file__).resolve(), "--device", str(device), "--receive", out / "receiver_request.json"]
    if q.get("software_check"):
        argv.append("--software-check")
    call(argv, out / "receiver.log")
    proof = json.loads((out / "receiver.json").read_text())
    checks = {e["id"]: e for e in proof["checks"]}
    for r in rows:
        r.update(status="verified", decoded_file_sha256=checks[r["id"]]["decoded_file_sha256"])
    write_json(out / "worker_results.json", dict(rows=rows, provenance=provenance, proof=proof))


def quality(rows, inputs, base, captures, torch, device):
    import numpy as np
    from transformers import AutoModelForCausalLM
    from kv_compare.quality import continuation_nll
    from kv_compare.causal_packet import sha
    from scripts.bench_llama_causal import cached_model_snapshot
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(cached_model_snapshot(), torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to(device).eval()
    if any(p.dtype != torch.bfloat16 for p in model.parameters()) or getattr(model.config, "quantization_config", None):
        raise ValueError("original BF16 model weights required")
    def matrices(path):
        a = np.load(path, mmap_mode="r", allow_pickle=False); C = a.shape[-1]//2
        return [(np.array(a[l,:,:C]), np.array(a[l,:,C:])) for l in range(a.shape[0])]
    for e in inputs:
        ids = np.load(captures / (e["document"] + "_ids.npy"), allow_pickle=False)
        expected = base["protocol"]["source"]["documents"][e["document"]]["token_ids_sha256"]
        if hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() != expected:
            raise ValueError("continuation token identity changed")
        args = (model, ids, e["tokens"], base["protocol"]["continuation"], device)
        reference = continuation_nll(*args, matrices(e["path"]))
        if abs(reference-base["quality_reference"][e["document"]]["nll"]) > 1e-6:
            raise ValueError("exact reference quality differs from base run; do not combine these settings")
        for r in [r for r in rows if r["document"] == e["document"]]:
            if sha(r["decoded"]) != r["decoded_file_sha256"]:
                raise ValueError("receiver output changed before quality evaluation")
            nll = continuation_nll(*args, matrices(r["decoded"]))
            if r["original_exact"] and abs(nll-reference)>1e-6:
                raise AssertionError("exact baseline changed quality")
            r["quality"] = dict(nll=nll, ppl=float(np.exp(nll)),
                ppl_change_percent=float(100*np.expm1(nll-reference)), scored_tokens=base["protocol"]["continuation"]-1)
            log(f"quality {r['id']}: {r['quality']['ppl_change_percent']:+.3f}%")
    del model


def report(result, out):
    rows = result.get("rows", []) + result.get("reused_rows", [])
    lines = ["# Matched causal run with baseline controls", "",
        f"Status: **{result['status']}**. New GPU measured: **{result.get('gpu_measured', False)}**.", "",
        "Completion covers the raw/FP8/SplitZip controls; official AQUA-KV and ExANS evaluations remain incomplete.",
        "Pinned Llama-3.2-3B, all 28 BF16 layers, identical frozen TEST words. Block32/Lite measurements are reused unchanged.",
        "Complete framed bytes; warm synchronized per-request wall medians. Shared state is additional.", "",
        "| Document | Method / step | Mode | Complete bpv | Saved | Encode ms | Decode ms | Exact | PPL change | Shared GPU MiB |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in sorted(rows, key=lambda r:(r["document"],r.get("method",r["policy"]),r.get("kappa",0))):
        q = r.get("quality",{}).get("ppl_change_percent")
        label = r.get("method",r["policy"]) + (f" / {r['kappa']:g}" if r.get("kappa") else "")
        lines.append(f"| {r['document']} | {label} | {r['mode']} | {r['complete_bpv']:.5f} | {r['saved_percent']:.2f}% | "
            f"{r['encode']['wall_median_ms']:.3f} | {r['decode']['wall_median_ms']:.3f} | {r['original_exact']} | "
            f"{'unmeasured' if q is None else f'{q:+.3f}%'} | {r['shared_predictor_gpu_bytes']/2**20:.3f} |")
    native = [r for r in result.get("rows",[]) if "native_author_benchmark" in r]
    if native:
        lines += ["", "Official unchanged SplitZip author benchmark, saved-tensor mode (separate native timing/byte scope):", "",
            "| Document | Native bpv | Encode GB/s | Decode GB/s |", "|---|---:|---:|---:|"]
        for r in native:
            n = r["native_author_benchmark"]
            lines.append(f"| {r['document']} | {r['native_payload_bpv']:.5f} | {n['encode_gbs']['mean']:.3f} | {n['decode_gbs']['mean']:.3f} |")
    lines += ["", "| Baseline | Status | Evidence / next requirement |", "|---|---|---|"]
    for method, entry in result.get("baseline_status",{}).items():
        lines.append(f"| {method} | {entry['status']} | {entry['detail'].replace('|','/').replace(chr(10),' ')} |")
    lines += ["", "SplitZip's algorithm and official throughput script are unmodified. The framed row adds local transport glue and validation; it is not the paper's full evaluation.",
        "SplitZip runs in the author's pinned environment; package versions and hashes are recorded. Its calibration uses the author's default 4,194,304-value TRAIN prefix cap, not TEST.",
        "FP8 is a native unscaled E4M3FN cast with saturation at ±448, not a named paper baseline. Model weights stay BF16.",
        "New packets are replayed by a separate receiver with wire and fixed LUT state only. Export/upload, codec setup and calibration/state export are recorded separately from warm codec timings.",
        "Quality uses the same short continuation and exact reference as the base run, not full WikiText PPL or LongBench.",
        "Historical AQUA adaptations and ExANS results on other inputs are not substituted. No compressed attention, T-to-P, SSD/network or serving throughput is measured."]
    if result.get("error"):
        lines += ["", "Failure: " + result["error"]]
    lines += ["", "Historical official evaluation: https://github.com/Kibnelson/kv_reconstruction/blob/cd5df510d42739d11470d50d99f6c1ccf4caec14/benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517/summary.md"]
    (out/"summary.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-run", type=Path); p.add_argument("--output", type=Path)
    p.add_argument("--captures", type=Path); p.add_argument("--device", default="cuda:0")
    p.add_argument("--splitzip-python", type=Path, default=ROOT/"runs/_llama_official/envs/splitzip/bin/python")
    p.add_argument("--splitzip-source", type=Path, default=ROOT/"runs/_llama_official/sources/splitzip")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--receive", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--software-check", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker or args.receive:
        torch, device = device_check(args.device, args.software_check)
        (worker if args.worker else receiver)(args.worker or args.receive, torch, device)
        return 0
    if not args.base_run or not args.output:
        p.error("--base-run and --output are required")
    from kv_compare.causal_packet import sha, write_json
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/"results.json").exists():
        raise ValueError("fresh output required; existing report preserved")
    result = dict(format="causal_baselines_v1", status="running", rows=[], reused_rows=[], gpu_measured=False,
        baseline_status={
            "AQUA-KV official": dict(status="pending official evaluation", commit="e744183f2f3a4a90381f20a6975456871fd1ae29", detail="Prior official training stopped before a successful HIGGS/PPL result; adapted Quanto rows are excluded."),
            "ExANS official": dict(status="prior build blocked", commit="de6c38bcc5438475bd1abf1e9b3d154cef414ef6", detail="Pinned official C++ build previously failed in expans_codec.cu; no matched measurement. See the earlier official report."),
        })
    try:
        torch, device = device_check(args.device)
        result.update(gpu_measured=True, device=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)))
        base_dir = args.base_run.resolve()
        base, inputs = verified_inputs(base_dir)
        if result["device"] != base["device"] or result["capability"] != base["protocol"]["arithmetic"]["capability"]:
            raise ValueError("use the same GPU model as the reused measurement")
        versions = {p: importlib.metadata.version(p) for p in ("torch","triton","numpy","transformers")}
        if versions != base["software"] or torch.version.cuda != base["cuda_runtime"]:
            raise ValueError("root environment changed since the base run; preserve its packages")
        root = capture_root(base_dir, base, args.captures)
        result.update(reused_rows=base["rows"], base=dict(path=str(base_dir), identity=base["identity"],
            results_sha256=sha(base_dir/"results.json"), replay_sha256=sha(base_dir/"independent_replay.json")),
            protocol=base["protocol"], inputs=inputs, quality_reference=base["quality_reference"])
        for name, digest in base["protocol"]["code"].items():
            if sha(ROOT/name) != digest:
                raise ValueError("measured causal/quality source changed: "+name)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != base["protocol"]["arithmetic"]["cublas_workspace"]:
            raise ValueError("CUBLAS workspace setting changed since base run")
        shutil.copyfile(base_dir/"results.json", out/"base_results.json")
        shutil.copyfile(base_dir/"independent_replay.json", out/"base_independent_replay.json")
        code_files = ["kv_compare/framed_baselines.py", "scripts/bench_llama_baselines.py", "scripts/package_llama_baselines.py",
                      "scripts/run_llama_baselines_a4500.sh"]
        write_json(out/"source_manifest.json", dict(files={name:sha(ROOT/name) for name in code_files}, reused_code=base["protocol"]["code"]))
        result["workers"] = {}
        for method in ("raw_bf16", "fp8_e4m3fn_saturating", "splitzip_official_framed"):
            try:
                python = sys.executable
                request = dict(method=method, inputs=inputs, output=str(out/method),
                    warmup=base["protocol"]["warmup"], repeats=base["protocol"]["repeats"])
                if method.startswith("splitzip"):
                    if not args.splitzip_python.is_file() or not args.splitzip_source.is_dir():
                        raise RuntimeError("pinned official SplitZip source/environment missing; pass --splitzip-python / --splitzip-source")
                    calibration = out/"splitzip_train.pt"
                    result["splitzip_train"] = prepare_train(root, base, calibration, torch)
                    request.update(source=str(args.splitzip_source.resolve()), calibration=str(calibration))
                    # Preserve the venv executable path; resolving its symlink loses the venv.
                    python = str(args.splitzip_python.absolute())
                request_path = out/(method+"_request.json"); write_json(request_path, request)
                call([python, Path(__file__).resolve(), "--device", str(device), "--worker", request_path], out/(method+".log"))
                data = json.loads((out/method/"worker_results.json").read_text())
                result["rows"].extend(data["rows"]); result["workers"][method] = data
                result["baseline_status"][method] = dict(status="measured", detail="identical TEST words, fresh-process packet receiver verified")
            except Exception as error:
                result["baseline_status"][method] = dict(status="blocked", detail=str(error))
                log(method+": "+str(error))
            write_json(out/"results.json", result); report(result,out)
        if not result["rows"]:
            raise RuntimeError("no new baseline completed")
        quality(result["rows"], inputs, base, root, torch, device)
        result["quality_completed"] = True
        result["status"] = "completed" if len(result["workers"])==3 else "partial"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        log(result["traceback"])
    write_json(out/"results.json", result); report(result,out)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

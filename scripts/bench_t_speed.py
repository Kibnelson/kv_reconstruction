#!/usr/bin/env python3
"""Frozen-byte GPU speed comparison. No fitting, policy selection or merging."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_head_spread import (check_base, file_sha, flattened_tensors,
    load_words, measure, raw_digest, select_device, Window, write_json)

SPECS = {
    "dense_original": ("dense", "legacy_global", "original"),
    "dense_fast": ("dense", "legacy_global", "fast"),
    "dense_learned": ("dense", "learned_selected", "head"),
    "hbh_original": ("hbh", "legacy_global", "original"),
    "hbh_learned": ("hbh", "learned_selected", "head"),
}


def rotate(values, offset):
    offset %= len(values)
    return values[offset:] + values[:offset]


def modeled_link_ms(pair_ms, packet_bytes, gbps):
    if not math.isfinite(gbps) or gbps <= 0 or packet_bytes < 0 or pair_ms < 0:
        raise ValueError("positive finite bandwidth and nonnegative costs required")
    return pair_ms + packet_bytes * 8 / (gbps * 1e6)


def aggregate(rows, cases, rounds, gpu):
    """Each case has equal weight; take the median round time per case first."""
    result = {}
    timing = "cuda_median_ms" if gpu else "wall_median_ms"
    for name, execution in sorted({(r["variant"], r["execution"]) for r in rows}):
        group = [r for r in rows if (r["variant"], r["execution"]) == (name, execution)]
        by_case = {c: [r for r in group if r["case"] == c] for c in cases}
        complete = all(len(g) == rounds and {r["round"] for r in g} == set(range(rounds)) for g in by_case.values())
        first = []
        for case, g in by_case.items():
            if not g:
                continue
            if len({(r["packet_sha256"], r["values"], r["packet_bytes"]) for r in g}) != 1:
                raise AssertionError("rounds changed frozen packet bytes: " + name + ":" + case)
            first.append(g[0])
        if not first:
            continue
        total_values, total_bytes = sum(r["values"] for r in first), sum(r["packet_bytes"] for r in first)
        row = {"complete": complete, "cases": len(first), "rounds": rounds,
            "values": total_values, "packet_bytes": total_bytes,
            "bpv": 8 * total_bytes / total_values, "saving_from_bf16": 1 - total_bytes / (2 * total_values),
            "all_exact": all(r["bit_exact"] and r["matches_reference_packet"] for r in group),
            "graph_input_change_test_passed": execution != "graph" or all(r["graph_liveness_passed"] for r in group),
            "extra_policy_tensor_bytes": first[0]["extra_policy_tensor_bytes"],
            "serialized_state_bytes": first[0]["serialized_state_bytes"]}
        for operation in ("encode", "decode", "pair"):
            case_times = [statistics.median(r[operation][timing] for r in g) for g in by_case.values() if g]
            row[operation + "_ms"] = statistics.mean(case_times)
            round_times = [statistics.mean(r[operation][timing] for r in group if r["round"] == n)
                           for n in range(rounds) if any(r["round"] == n for r in group)]
            row[operation + "_round_mean_range_ms"] = [min(round_times), max(round_times)]
        mean_raw = 2 * total_values / len(first)
        row["encode_raw_GBps"] = mean_raw / (row["encode_ms"] * 1e6)
        row["decode_raw_GBps"] = mean_raw / (row["decode_ms"] * 1e6)
        row["modeled_serial_link_ms"] = {str(b): modeled_link_ms(row["pair_ms"], total_bytes / len(first), b)
                                       for b in (3, 10, 25, 50, 100, 200, 400)} if gpu else None
        result[name + "/" + execution] = row
    return result


def decision(result):
    table = result["aggregate"]
    eligible = result["gpu_executed"] and result["full_reference_coverage"] and result["arguments"]["rounds"] >= 3
    original, fast = table.get("dense_original/eager"), table.get("dense_fast/eager")
    hbh_eager, hbh_graph = table.get("hbh_learned/eager"), table.get("hbh_learned/graph")
    dense_candidate = bool(eligible and original and fast and original["complete"] and fast["complete"]
        and fast["all_exact"] and fast["packet_bytes"] == original["packet_bytes"]
        and fast["encode_ms"] <= 0.95 * original["encode_ms"] and fast["pair_ms"] < original["pair_ms"])
    graph_candidate = bool(eligible and hbh_graph and hbh_eager and hbh_graph["complete"] and hbh_eager["complete"]
        and hbh_graph["all_exact"] and hbh_graph["graph_input_change_test_passed"]
        and hbh_graph["packet_bytes"] == hbh_eager["packet_bytes"] and hbh_graph["pair_ms"] <= 0.95 * hbh_eager["pair_ms"])
    return {"fast_dense_candidate_for_review": dense_candidate,
            "hbh_graph_candidate_for_review": graph_candidate, "automatic_merge": False,
            "criterion": "actual GPU, all reference windows, >=3 rounds, identical packets, >=5% relevant speed gain"}


def summary(result):
    lines = ["# T speed comparison with frozen compressed bytes", "",
        f"Status: **{result['status']}**. GPU executed: **{result.get('gpu_executed', False)}**. "
        f"Full reference coverage: **{result.get('full_reference_coverage', False)}**.", "",
        "| Implementation / execution | BF16 saved | bpv | Encode ms | Decode ms | Pair ms | Complete |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for key, r in result.get("aggregate", {}).items():
        lines.append(f"| {key} | {100*r['saving_from_bf16']:.4f}% | {r['bpv']:.6f} | "
            f"{r['encode_ms']:.3f} | {r['decode_ms']:.3f} | {r['pair_ms']:.3f} | {r['complete']} |")
    lines += ["", "Times are CUDA event medians within a round, then median across rounds per case, then "
        "mean across cases. Pair is measured sequential encode + decode on one GPU, with resident input and "
        "preallocated stream arena/output. Graph pair uses two replay launches. Setup, graph memory, input "
        "staging, packet packing/D2H, packet expansion/H2D and cold worker time are separate fields.", "",
        "Every packet must match the earlier run byte for byte. No predictor, spread, alpha or tile is refitted. "
        "The hbh_learned full run therefore preserves the earlier 48.568% saving. A smaller case subset can "
        "have a different aggregate rate. No accuracy or symbol bits are discarded.", ""]
    if result.get("gpu_executed") and result.get("aggregate"):
        lines += ["## Ideal serial link model", "", "| Implementation / execution | 3 Gbps | 25 Gbps | 100 Gbps | 400 Gbps |",
                  "|---|---:|---:|---:|---:|"]
        for key, r in result["aggregate"].items():
            if not r["complete"]:
                continue
            m = r["modeled_serial_link_ms"]
            lines.append(f"| {key} | {m['3']:.3f} | {m['25']:.3f} | {m['100']:.3f} | {m['400']:.3f} |")
        lines += ["", "Milliseconds = measured codec pair + complete packet bytes / offered link bandwidth. "
            "This is a model: network, SSD, queueing, contention, state loading and the separately recorded host "
            "packet path are not measured by this table. It assumes resident shared state and no overlap.", ""]
    if result.get("decisions"):
        lines += [json.dumps(result["decisions"], sort_keys=True), ""]
    if result.get("error"):
        lines += ["Failure: " + result["error"], ""]
    if result.get("graph_errors"):
        lines += [f"Graph capture/replay errors: {len(result['graph_errors'])}. They are not relabeled as eager results.", ""]
    return "\n".join(lines) + "\n"


def make_codec(spec, state, tokens):
    from kv_compare.fast_dense_encode import FastDenseEncode
    from kv_compare.head_spread import HeadSpreadCodec
    from kv_compare.layerwise import LayerwiseOneStep
    kind = SPECS[spec][2]
    constants = dict(state["consts"])
    constants["cos"], constants["sin"] = constants["cos"][:tokens], constants["sin"][:tokens]
    cls = {"original": LayerwiseOneStep, "fast": FastDenseEncode, "head": HeadSpreadCodec}[kind]
    kwargs = dict(tokens_per_program=state["TB"], mode=state["mode"])
    if kind == "head":
        kwargs["policy"] = state["loaded_policy"]
    elif state["loaded_policy"].kind != 0:
        raise ValueError("original/fast dense requires the frozen global policy")
    codec = cls(constants, state["C"], state["hd"], **kwargs)
    codec.policy = state["loaded_policy"]  # wire identity; original kernels never read this attribute
    return codec


def prove_packet(args, output, state_path, packet, key, expected, device):
    packet_path, report_path = output / "packets" / (key + ".bin"), output / "packets" / (key + ".json")
    packet_path.write_bytes(packet)
    command = [sys.executable, str(ROOT / "scripts/bench_head_spread.py"), "--decode-worker",
        "--state", str(state_path), "--packet", str(packet_path), "--worker-report", str(report_path),
        "--device", str(device), "--threads", str(args.threads)]
    if args.software_check:
        command += ["--software-check"]
    env = dict(os.environ)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in env["PYTHONPATH"].split(os.pathsep) if p)
    child = subprocess.run(command, cwd=output / "packets", env=env, capture_output=True, text=True)
    if child.returncode:
        raise RuntimeError("packet-only decoder failed: " + child.stderr[-5000:])
    proof = json.loads(report_path.read_text())
    if not (proof["bit_exact"] and not proof["known_source_words"]
            and proof["words_sha256"] == expected["source_words_sha256"]
            and proof["values"] == expected["values"]
            and proof["gpu_executed"] == (device.type == "cuda")):
        raise AssertionError("independent state-plus-packet proof differs from frozen reference")
    packet_path.unlink()  # transient file created above only; the JSON proof remains
    return proof


def benchmark(args, output, report, log):
    import torch
    import triton
    from kv_compare.fast_dense_encode import CudaReplay
    from kv_compare.head_spread_wire import load_state, pack, unpack, tensors_bytes
    from kv_compare.predict import Doc

    device = select_device(args.device, args.software_check)
    if device.type != "cuda" and "graph" in args.executions:
        raise ValueError("CPU software validation cannot execute CUDA graphs; request --executions eager")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference_dir = Path(args.reference_run).resolve()
    reference_path = reference_dir / "results.json"
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "completed" or (device.type == "cuda" and not reference.get("gpu_executed")):
        raise ValueError("reference must be a completed GPU run (CPU fixtures require --software-check)")
    if reference.get("test_used_for_selection") or reference.get("automatic_merge"):
        raise ValueError("reference is not the frozen, unmerged experiment")
    pins = check_base()
    if device.type == "cuda":
        from kv_compare.fast_dense_checks import check
        log("checking new dense forward pass against original T over all BF16 patterns")
        report["device_checks"] = check(device, all_patterns=True)
        write_json(output / "device_checks.json", report["device_checks"])
        torch.set_num_threads(args.threads)
    for record in reference["experiment_sources"]:
        if file_sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("reference experiment source changed: " + record["path"])
    frozen_path = reference_dir / "selection_frozen.json"
    if json.loads(frozen_path.read_text()) != reference["selected_on_dev"]:
        raise ValueError("reference DEV selection changed")
    captures = Path(args.captures or reference["arguments"]["captures"]).resolve()
    cases = args.cases.split(",") if args.cases else reference["split"]["test_docs"]
    if len(set(cases)) != len(cases) or not set(cases) <= set(reference["split"]["test_docs"]):
        raise ValueError("cases must be distinct windows from the frozen reference")
    report.update(gpu_executed=device.type == "cuda", device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU interpreter",
        torch=torch.__version__, triton=triton.__version__, python=platform.python_version(),
        reference_results_sha256=file_sha(reference_path), reference_selection_sha256=file_sha(frozen_path),
        reference_model=reference["model_identity"], baseline_sources=pins,
        full_reference_coverage=set(cases) == set(reference["split"]["test_docs"]) and reference["full_model"],
        cases=cases, reference_rate=reference["aggregate"], test_used_for_policy_selection=False,
        capture_or_model_refit=False, automatic_merge=False, state_inputs=[], input_files=[])
    reference_rows = {(x["mode"], x["variant"], x["case"]): x for x in reference["test_rows"]}
    expected_inputs = {Path(x["path"]).name: x["sha256"] for x in reference["inputs"]}
    input_names = ["meta.json"] + [c + "_" + suffix for c in cases for suffix in ("K_bits.npy", "V_bits.npy", "ids.npy")]
    for name in input_names:
        path = captures / name
        digest = file_sha(path)
        if digest != expected_inputs[name]:
            raise ValueError("capture changed since reference run: " + name)
        report["input_files"].append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
    states, state_paths = {}, {}
    for spec in args.variants:
        mode, source, _ = SPECS[spec]
        path = reference_dir / "states" / f"{mode}_{source}.pt"
        state_paths[spec] = path
        if path not in states:
            log("loading frozen state: " + path.name)
            states[path] = load_state(path, device)
            report["state_inputs"].append(dict(path=str(path), sha256=file_sha(path), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns, predictor_sha256=states[path]["predictor_sha256"],
                policy_sha256=states[path]["policy_sha256"]))
        expected = reference_rows[(mode, source, cases[0])]
        if (states[path]["predictor_sha256"], states[path]["policy_sha256"]) != (expected["predictor_sha256"], expected["policy_sha256"]):
            raise ValueError("state differs from the selected reference policy")
    report["resident_cached_state_tensor_bytes"] = tensors_bytes(t for state in states.values() for t in state["consts"].values())
    tokens, layers = reference["split"]["test_tokens_each"], reference["layers"]
    for case_index, case in enumerate(cases):
        doc = Window(Doc(captures, case), 0, tokens)
        source_hash = raw_digest(doc, layers)
        words = load_words(doc, layers, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for round_index in range(args.rounds):
            order = rotate(args.variants, case_index + round_index)
            log(f"case {case}, round {round_index+1}/{args.rounds}, order {','.join(order)}")
            for method_index, spec in enumerate(order):
                state_path = state_paths[spec]
                state = states[state_path]
                mode, source, _ = SPECS[spec]
                expected = reference_rows[(mode, source, case)]
                if source_hash != expected["source_words_sha256"]:
                    raise ValueError("source words differ from the frozen reference")
                codec = make_codec(spec, state, tokens)
                segments, capacity = codec.layout(tokens)
                arena = torch.zeros(segments * capacity, dtype=torch.int16, device=device)
                decoded = torch.empty_like(words)
                bufs = codec.buffers(tokens)
                holder = {}
                def eager_encode():
                    holder["encoded"] = codec.encode(words, out=arena)
                    return holder["encoded"]
                def eager_decode():
                    return codec.decode(arena, holder["encoded"][1], tokens, out=decoded, bufs=bufs)
                def eager_pair():
                    eager_encode()
                    return eager_decode()
                eager_pair()
                if not torch.equal(decoded, words):
                    raise AssertionError("warmup decode changed BF16 words")
                replays, graph_setup, graph_error = [], None, None
                graph_live = False
                if "graph" in args.executions:
                    try:
                        eg = CudaReplay(lambda: codec.encode(words, out=arena), device)
                        replays.append(eg)
                        dg = CudaReplay(lambda: codec.decode(arena, eg.output[1], tokens, out=decoded, bufs=bufs), device)
                        replays.append(dg)
                        def graph_pair():
                            eg()
                            return dg()
                        # A changed input and deliberately dirty output must be reconstructed.
                        flat = words.view(-1)
                        flat[0].bitwise_xor_(1)
                        flat[-1].bitwise_xor_(1)
                        try:
                            decoded.fill_(1234)
                            graph_pair()
                            if not torch.equal(decoded, words):
                                raise AssertionError("graph replay used stale/cached input or stream state")
                        finally:
                            flat[0].bitwise_xor_(1)
                            flat[-1].bitwise_xor_(1)
                        arena.zero_()  # clear padding touched by the changed-input probe, outside timing
                        graph_pair()
                        if not torch.equal(decoded, words):
                            raise AssertionError("graph replay failed after restoring the original input")
                        graph_live = True
                        graph_setup = {"encode": eg.setup, "decode": dg.setup}
                    except Exception as error:
                        graph_error = type(error).__name__ + ": " + str(error)
                        report["graph_errors"].append(dict(variant=spec, case=case, round=round_index, error=graph_error))
                        log("graph unavailable for " + spec + ": " + graph_error)
                        for replay in reversed(replays):
                            replay.close()
                        replays.clear()
                        arena.zero_()
                        eager_pair()
                last_packet = packet = starts = None
                try:
                    executions = rotate(args.executions, round_index + case_index + method_index)
                    for execution in executions:
                        if execution == "graph" and not graph_live:
                            continue
                        log(f"  {spec} / {execution}: encode, decode, pair; frozen packet check")
                        operations = {"encode": eager_encode, "decode": eager_decode, "pair": eager_pair}
                        if execution == "graph":
                            operations = {"encode": eg, "decode": dg, "pair": graph_pair}
                        times = {}
                        for operation in rotate(["encode", "decode", "pair"], round_index + method_index):
                            value, times[operation] = measure(operations[operation], device, args.warmup, args.repeats)
                            del value
                        operations["pair"]()
                        if not torch.equal(decoded, words):
                            raise AssertionError("timed operation changed BF16 words")
                        starts = eg.output[1] if execution == "graph" else holder["encoded"][1]
                        packet, cost = pack(codec, arena, starts, tokens, state["predictor_sha256"], codec.policy, source_hash)
                        digest = hashlib.sha256(packet).hexdigest()
                        if digest != expected["packet_sha256"] or len(packet) != expected["packet_bytes"]:
                            raise AssertionError("speed change altered the frozen reference packet: " + spec)
                        row = dict(variant=spec, execution=execution, case=case, round=round_index,
                            values=words.numel(), raw_bytes=words.numel()*2, **cost,
                            packet_sha256=digest, source_words_sha256=source_hash,
                            predictor_sha256=state["predictor_sha256"], policy_sha256=state["policy_sha256"],
                            bit_exact=True, matches_reference_packet=True,
                            graph_liveness_passed=graph_live if execution == "graph" else None,
                            graph_setup=graph_setup if execution == "graph" else None,
                            extra_policy_tensor_bytes=codec.policy.array_bytes,
                            shared_predictor_tensor_bytes=tensors_bytes(codec.k.values()),
                            serialized_state_bytes=state_path.stat().st_size,
                            preallocated_decode_scratch_bytes=tensors_bytes(flattened_tensors(bufs)),
                            stream_arena_bytes=arena.numel()*arena.element_size(), decoded_output_bytes=decoded.numel()*2,
                            **times)
                        report["rows"].append(row)
                        last_packet = packet
                        write_json(output / "checkpoint.json", report)
                    if round_index == 0 and last_packet is not None:
                        key = spec + "_" + case
                        begin = time.perf_counter()
                        received, received_starts, _ = unpack(codec, last_packet, state["predictor_sha256"])
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        unpack_ms = 1000 * (time.perf_counter() - begin)
                        out = codec.decode(received, received_starts, tokens, out=decoded, bufs=bufs)
                        if not torch.equal(out, words):
                            raise AssertionError("expanded compact packet failed reconstruction")
                        del received, received_starts, out
                        report["packet_proofs"][key] = prove_packet(args, output, state_path, last_packet, key, expected, device)
                        report["packet_proofs"][key]["parent_packet_expand_H2D_ms"] = unpack_ms
                finally:
                    for replay in reversed(replays):
                        replay.close()
                del codec, arena, decoded, bufs, holder, last_packet, packet, starts
                gc.collect()
            report["aggregate"] = aggregate(report["rows"], cases, args.rounds, report["gpu_executed"])
            write_json(output / "checkpoint.json", report)
        del words
    for record in report["input_files"] + report["state_inputs"]:
        path = Path(record["path"])
        if (path.stat().st_size, path.stat().st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
            raise ValueError("frozen input changed during benchmark: " + str(path))
    if file_sha(reference_path) != report["reference_results_sha256"] or file_sha(frozen_path) != report["reference_selection_sha256"]:
        raise ValueError("reference metadata changed during benchmark")
    check_base()
    report["decisions"] = decision(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--captures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--software-check", action="store_true")
    parser.add_argument("--variants", default=",".join(SPECS))
    parser.add_argument("--executions", default="eager,graph")
    parser.add_argument("--cases")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    args.variants, args.executions = args.variants.split(","), args.executions.split(",")
    if (not set(args.variants) <= set(SPECS) or len(set(args.variants)) != len(args.variants)
            or not set(args.executions) <= {"eager", "graph"} or len(set(args.executions)) != len(args.executions)
            or min(args.rounds, args.repeats, args.threads) < 1 or args.warmup < 0):
        parser.error("invalid variants, executions, rounds or repetitions")
    output, reference = Path(args.output).resolve(), Path(args.reference_run).resolve()
    if output == reference or output.is_relative_to(reference) or reference.is_relative_to(output):
        parser.error("output must be separate from the frozen reference run")
    try:
        frozen = json.loads((reference / "results.json").read_text())
        captures = Path(args.captures or frozen["arguments"]["captures"]).resolve()
    except (OSError, ValueError, KeyError) as error:
        parser.error("cannot read the reference run: " + str(error))
    if output == captures or output.is_relative_to(captures) or captures.is_relative_to(output):
        parser.error("output must be separate from input captures")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "packets").mkdir(exist_ok=True)
    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (output / "run.log").open("a") as stream:
            stream.write(line + "\n")
    report = dict(format="t_frozen_speed_v1", status="running", arguments=vars(args),
        gpu_executed=False, rows=[], graph_errors=[], packet_proofs={}, aggregate={})
    try:
        benchmark(args, output, report, log)
        report["status"] = "completed_with_graph_errors" if report["graph_errors"] else "completed"
        code = 2 if report["graph_errors"] else 0
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
        log(report["error"])
        code = 2
    paths = [ROOT / "kv_compare/fast_dense_encode.py", ROOT / "kv_compare/fast_dense_checks.py", Path(__file__).resolve()]
    report["speed_sources"] = [dict(path=str(p.relative_to(ROOT)), sha256=file_sha(p)) for p in paths]
    write_json(output / "results.json", report)
    (output / "summary.md").write_text(summary(report))
    with zipfile.ZipFile(output / "share.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("results.json", "summary.md", "run.log", "device_checks.json"):
            if (output / name).is_file():
                z.write(output / name, name)
    log(f"{report['status']}: {output/'summary.md'}; share {output/'share.zip'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

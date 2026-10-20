#!/usr/bin/env python3
"""Controlled CPU transfer-codec comparison using actual model prefill hooks.

Model capture may run on CPU or CUDA. The supplied team's probability coder and
all codec measurements in this study run on CPU. This is an independent
validation unless --act-dir names the team's exact original captures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + canonical(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_act(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def combined_words(act):
    keys, values = np.asarray(act["K_bits"]), np.asarray(act["V_bits"])
    if keys.dtype != np.uint16 or values.dtype != np.uint16 or keys.shape != values.shape or keys.ndim != 3:
        raise ValueError("ACT requires paired uint16 [layers,tokens,channels] K_bits/V_bits")
    return np.concatenate([keys, values], axis=2)


def slice_act(act, start, stop):
    count = np.asarray(act["K_bits"]).shape[1]
    if not 0 <= start < stop <= count:
        raise ValueError("invalid ACT slice")
    out = dict(act)
    for key in ("K_bits", "V_bits", "H"):
        if key in out:
            out[key] = np.asarray(out[key])[:, start:stop].copy()
    if "ids" in out:
        out["ids"] = np.asarray(out["ids"])[start:stop].copy()
    out["positions"] = np.asarray(act.get("positions", np.arange(count)))[start:stop].astype(np.int64)
    return out


def gather_act(act, indices):
    indices = np.asarray(indices, dtype=np.int64)
    out = dict(act)
    tokens = np.asarray(act["K_bits"]).shape[1]
    for key in ("K_bits", "V_bits", "H"):
        if key in out:
            out[key] = np.asarray(out[key])[:, indices].copy()
    if "ids" in out:
        out["ids"] = np.asarray(out["ids"])[indices].copy()
    out["positions"] = np.asarray(act.get("positions", np.arange(tokens)))[indices].copy()
    return out


def adjacent_key_order(keys, head_dim):
    """Convert standard HuggingFace split-half RoPE coordinates to adjacent pairs."""
    keys = np.asarray(keys)
    if head_dim < 2 or head_dim % 2 or keys.shape[-1] % head_dim:
        raise ValueError("invalid key head geometry")
    permutation = np.column_stack([np.arange(head_dim // 2), np.arange(head_dim // 2, head_dim)]).ravel()
    shaped = keys.reshape(*keys.shape[:-1], keys.shape[-1] // head_dim, head_dim)
    return shaped[..., permutation].reshape(keys.shape).copy()


def source_features(act, source):
    if source == "hidden":
        hidden = np.asarray(act.get("H"))
        if hidden.ndim != 3 or hidden.shape[:2] != np.asarray(act["K_bits"]).shape[:2]:
            raise ValueError("all-layer actual hidden hooks are required for hidden metadata")
        return hidden.astype(np.float64)
    if source != "kv":
        raise ValueError("unknown metadata source")
    words = combined_words(act)
    values = (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    _, tokens, width = values.shape
    channels = width // 2
    hd, heads, base = int(act["hd"]), int(act["Hk"]), float(act["base"])
    positions = np.asarray(act.get("positions", np.arange(tokens)))
    angle = positions[:, None] * base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    cosine, sine = np.cos(angle)[None, :, None, :], np.sin(angle)[None, :, None, :]
    keys = values[..., :channels].reshape(values.shape[0], tokens, heads, hd)
    a, b = keys[..., 0::2].copy(), keys[..., 1::2].copy()
    keys[..., 0::2] = a * cosine + b * sine
    keys[..., 1::2] = -a * sine + b * cosine
    return values


def fit_projector(captures, source, rank):
    """TRAIN-only PCA and symmetric int8 quantization; no TEST statistics."""
    train = np.concatenate([source_features(act, source) for act in captures], axis=1)
    layers, count, width = train.shape
    if rank < 1 or rank > min(width, count):
        raise ValueError("metadata rank exceeds TRAIN geometry")
    means = train.mean(axis=1)
    basis = np.empty((layers, width, rank), dtype=np.float64)
    scales = np.empty((layers, rank), dtype=np.float64)
    for layer in range(layers):
        centered = train[layer] - means[layer]
        covariance = centered.T @ centered / count
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        vectors = eigenvectors[:, -rank:][:, ::-1].copy()
        # Resolve eigenvector sign ambiguity for repeatable encoder artifacts.
        pivots = np.argmax(np.abs(vectors), axis=0)
        vectors *= np.where(vectors[pivots, np.arange(rank)] < 0, -1.0, 1.0)
        basis[layer] = vectors
        projected = centered @ vectors
        scales[layer] = np.maximum(3.0 * projected.std(axis=0) / 127.0, 1e-8)
    return {"mean": means, "basis": basis, "scale": scales, "source": source, "rank": rank}


def project_codes(projector, capture, *, shuffled=False):
    values = source_features(capture, projector["source"])
    projected = np.einsum("lnd,ldr->lnr", values - projector["mean"][:, None], projector["basis"])
    rounded = np.rint(projected / projector["scale"][:, None])
    clipped = int(np.count_nonzero((rounded < -127) | (rounded > 127)))
    codes = np.clip(rounded, -127, 127).astype(np.int8)
    if shuffled:
        # Fixed negative control: each capture rolls by half its own token count.
        # These bytes are still transmitted and charged.
        codes = np.roll(codes, max(1, codes.shape[1] // 2), axis=1)
    return codes, {"clipped_codes": clipped, "codes": int(codes.size),
                   "clipping_fraction": clipped / max(1, codes.size), "shuffled": shuffled}


def save_projector(projector, path):
    np.savez(path, mean=projector["mean"], basis=projector["basis"], scale=projector["scale"],
             source=np.array(projector["source"]), rank=np.array(projector["rank"]))
    return {"serialized_bytes": Path(path).stat().st_size,
            "array_bytes": sum(projector[key].nbytes for key in ("mean", "basis", "scale")),
            "sha256": sha_file(path), "role": "encoder-only; decoder receives int8 codes"}


def fixture_specs(args, tokenizer):
    if args.ids_manifest_root:
        root = Path(args.ids_manifest_root)
        result = {}
        for split, limit, count in (("train", args.train_tokens, args.train_cases),
                                     ("dev", args.capture_tokens, 1),
                                     ("test", args.capture_tokens, args.test_cases)):
            manifest = json.loads((root / f"{split}_manifest.json").read_text())
            # Choose one longest capture per fixture, retaining the source order.
            selected = {}
            for case in manifest["cases"]:
                name = case.get("fixture", case["name"])
                if name not in selected or len(case["input_token_ids"]) > len(selected[name]["input_token_ids"]):
                    selected[name] = case
            result[split] = []
            for case in list(selected.values())[:count]:
                ids = list(case["input_token_ids"][:limit])
                if len(ids) != limit:
                    raise ValueError("source manifest has too few tokens")
                result[split].append({"name": case.get("fixture", case["name"]), "kind": case.get("kind"),
                                      "ids": ids, "ids_source": str(root / f"{split}_manifest.json")})
        return result
    from qwen_kv_bench.fixtures import SPLITS, build_input_ids
    result = {}
    for split, limit, count in (("train", args.train_tokens, args.train_cases),
                                 ("dev", args.capture_tokens, 1),
                                 ("test", args.capture_tokens, args.test_cases)):
        result[split] = [{"name": case["name"], "kind": case["kind"],
                          "ids": build_input_ids(tokenizer, case, limit), "ids_source": "qwen_kv_bench.fixtures"}
                         for case in SPLITS[split][:count]]
    return result


def capture_case(model, specification, device, path, max_layers=None):
    import torch
    from qwen_kv_bench.model_runtime import _pairs
    layers = list(model.model.layers)
    total_model_layers = len(layers)
    if max_layers is not None:
        if not 1 <= max_layers <= len(layers):
            raise ValueError("max_layers must select a nonempty contiguous layer prefix")
        layers = layers[:max_layers]
    hidden = [None] * len(layers)
    handles = []
    for layer, block in enumerate(layers):
        def hook(module, positional, keyword, index=layer):
            value = keyword.get("hidden_states", positional[0] if positional else None)
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
                raise ValueError("unsupported attention hidden-state hook geometry")
            hidden[index] = value.detach().to(device="cpu", dtype=torch.float32).numpy().copy()[0]
            return None
        handles.append(block.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    inputs = torch.tensor([specification["ids"]], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(inputs, use_cache=True, logits_to_keep=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - begin
        pairs = _pairs(output.past_key_values)[:len(layers)]
        keys, values = [], []
        for key, value in pairs:
            if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
                raise ValueError("study requires original BF16 cache tensors")
            def words(tensor):
                return tensor.detach().transpose(1, 2).contiguous().view(torch.uint16).cpu().numpy()[0].reshape(len(specification["ids"]), -1).copy()
            keys.append(words(key)); values.append(words(value))
        head_dim = int(pairs[0][0].shape[-1])
        heads = int(pairs[0][0].shape[1])
        if any(value is None for value in hidden):
            raise ValueError("not every attention hidden hook fired")
        act = {"K_bits": adjacent_key_order(np.stack(keys), head_dim), "V_bits": np.stack(values),
               "H": np.stack(hidden), "sel": np.arange(len(layers), dtype=np.int64),
               "ids": np.asarray(specification["ids"], dtype=np.int64), "Hk": np.array(heads),
               "hd": np.array(head_dim), "base": np.array(float(model.config.rope_theta)),
               "positions": np.arange(len(specification["ids"]), dtype=np.int64)}
        np.savez(path, **act)
        report = {"name": specification["name"], "kind": specification.get("kind"),
                  "tokens": len(specification["ids"]), "ids_sha256": sha_array(act["ids"]),
                  "kv_sha256": sha_array(combined_words(act)), "hidden_sha256": sha_array(act["H"]),
                  "capture_seconds_with_hooks": elapsed, "capture_file": str(path),
                  "model_layers": total_model_layers, "captured_layers": len(layers),
                  "layer_scope": "full model" if len(layers) == total_model_layers else "contiguous prefix; partial-layer study",
                  "hook": "self_attn forward_pre_hook hidden_states; observational, returns None",
                  "key_order": "HF split halves converted to adjacent pairs; values unchanged",
                  "shape": list(act["K_bits"].shape), "hidden_shape": list(act["H"].shape)}
        write_json(path.with_suffix(".json"), report)
        return act, report
    finally:
        for handle in handles:
            handle.remove()


def prepare_captures(args, output):
    captures_dir = output / "captures"
    captures_dir.mkdir(exist_ok=True)
    if args.act_dir:
        root = Path(args.act_dir)
        fit_paths = sorted(root.glob("train*.npz"))
        test_paths = [root / "test_text.npz", root / "test_code.npz"][:args.test_cases]
        train = [load_act(path) for path in fit_paths[:args.train_cases]]
        dev = load_act(root / "tables.npz")
        tests = [(path.stem, load_act(path)) for path in test_paths]
        if args.max_layers is not None:
            for act in train + [dev] + [value for _, value in tests]:
                for key in ("K_bits", "V_bits"):
                    act[key] = act[key][:args.max_layers].copy()
                if "H" in act and np.asarray(act["H"]).shape[0] >= args.max_layers:
                    act["H"] = act["H"][:args.max_layers].copy()
        used_paths = fit_paths[:args.train_cases] + [root / "tables.npz"] + test_paths
        return train, dev, tests, {"source": "supplied ACT files",
                                   "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in used_paths],
                                   "exact_original_dataset_identity": "not established by filenames alone"}
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("capture supports explicit cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA capture but CUDA is unavailable")
    local = Path(args.model).is_dir()
    local_weights = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha_file(path)}
                     for path in sorted(Path(args.model).glob("*.safetensors"))] if local else []
    capture_identity = {"model": args.model, "revision": args.revision,
                        "config_sha256": sha_file(Path(args.model) / "config.json") if local else None,
                        "local_weights": local_weights, "device": str(device),
                        "torch": torch.__version__, "transformers": transformers.__version__, "max_layers": args.max_layers}
    capture_fingerprint = hashlib.sha256(canonical(capture_identity).encode()).hexdigest()
    loading = {"local_files_only": local}
    if args.revision and not local:
        loading["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **loading)
    specifications = fixture_specs(args, tokenizer)
    model = None
    records = []
    all_captures = {}
    for split, specs in specifications.items():
        all_captures[split] = []
        for specification in specs:
            path = captures_dir / f"{split}_{specification['name']}_{len(specification['ids'])}.npz"
            sidecar = path.with_suffix(".json")
            if args.reuse_captures and path.is_file() and sidecar.is_file():
                act = load_act(path)
                report = json.loads(sidecar.read_text())
                if (not np.array_equal(act["ids"], specification["ids"]) or
                        report.get("model_fingerprint") != capture_fingerprint or
                        report["kv_sha256"] != sha_array(combined_words(act)) or
                        report["hidden_sha256"] != sha_array(act["H"])):
                    raise ValueError("reused capture does not match IDs/content hashes")
            else:
                if model is None:
                    print(f"Loading {args.model} for actual hidden/KV capture on {device}", flush=True)
                    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                        attn_implementation="sdpa", **loading).to(device).eval()
                print(f"Capture {split}: {specification['name']}, {len(specification['ids'])} tokens", flush=True)
                act, report = capture_case(model, specification, device, path, args.max_layers)
                report["model_fingerprint"] = capture_fingerprint
                write_json(sidecar, report)
            report["split"] = split
            records.append(report)
            all_captures[split].append((specification["name"], act))
    model_identity = {"model": args.model, "revision": args.revision,
                      "config_sha256": sha_file(Path(args.model) / "config.json") if local else None,
                      "local_weights": local_weights,
                      "torch": torch.__version__, "transformers": transformers.__version__,
                      "capture_device": str(device), "codec_device": "cpu", "captures": records}
    del model
    return [value for _, value in all_captures["train"]], all_captures["dev"][0][1], all_captures["test"], model_identity


def decode_child(args):
    from qwen_kv_bench.transfer_reference import load_model, decode
    begin = time.perf_counter()
    model = load_model(args.state)
    load_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    words, stats = decode(model, Path(args.decode_packet).read_bytes())
    wall = time.perf_counter() - begin
    np.save(args.decoded_output, words, allow_pickle=False)
    write_json(args.decode_report, {"load_seconds": load_seconds, "decode_wall_seconds": wall,
                                   "decoded_sha256": sha_array(words), "stats": stats,
                                   "inputs": [str(args.state), str(args.decode_packet)]})
    return 0


def metadata_for(specification, capture, start=0, count=None, partition=None):
    if specification["projector"] is None:
        return None, {}
    total = np.asarray(capture["K_bits"]).shape[1]
    count = total - start if count is None else count
    indices = np.arange(start, start + count)
    shuffled = specification.get("shuffled", False)
    if shuffled:
        first, stop = (0, total) if partition is None else partition
        if first < 0 or stop > total or not first <= start < start + count <= stop:
            raise ValueError("metadata shuffle partition does not contain requested page")
        indices = first + (indices - first - max(1, (stop - first) // 2)) % (stop - first)
    codes, diagnostics = project_codes(specification["projector"], gather_act(capture, indices))
    diagnostics["shuffled"] = shuffled
    diagnostics["source_token_indices"] = indices.tolist() if shuffled else None
    return codes, diagnostics


def measure(specification, capture, start, count, args, output, phase, name, *, subprocess_decode=False):
    from qwen_kv_bench.transfer_reference import encode, decode
    begin = time.perf_counter()
    partition = (np.asarray(capture["K_bits"]).shape[1] // 2, np.asarray(capture["K_bits"]).shape[1]) if phase == "DEV" else None
    codes, metadata_diagnostics = metadata_for(specification, capture, start, count, partition)
    metadata_prepare = time.perf_counter() - begin
    words = combined_words(capture)[:, start:start + count].copy()
    begin = time.perf_counter()
    packet, stats = encode(specification["model"], words, start=start, metadata=codes,
                           correction=specification.get("correction", "direct"),
                           residue_bits=specification.get("residue_bits", 0))
    encode_codec_wall = time.perf_counter() - begin
    encode_wall = metadata_prepare + encode_codec_wall
    if subprocess_decode:
        sandbox = output / "packet_checks"
        sandbox.mkdir(exist_ok=True)
        stem = f"{phase}_{name}_{specification['name']}_{start}"
        packet_path, decoded_path, report_path = sandbox / (stem + ".bin"), sandbox / (stem + ".npy"), sandbox / (stem + ".json")
        packet_path.write_bytes(packet)
        command = [sys.executable, str(Path(__file__).resolve()), "--decode-packet", str(packet_path.resolve()),
                   "--state", str(specification["state_path"].resolve()), "--decoded-output", str(decoded_path.resolve()),
                   "--decode-report", str(report_path.resolve())]
        begin = time.perf_counter()
        environment = dict(os.environ)
        if environment.get("PYTHONPATH"):
            environment["PYTHONPATH"] = os.pathsep.join(str(Path(item).resolve())
                for item in environment["PYTHONPATH"].split(os.pathsep) if item)
        try:
            subprocess.run(command, cwd=sandbox, env=environment, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("packet-only decoder failed: " + error.stderr) from error
        subprocess_wall = time.perf_counter() - begin
        decoded = np.load(decoded_path, allow_pickle=False)
        child = json.loads(report_path.read_text())
        decode_wall = child["decode_wall_seconds"]
        decoder_inputs = child["inputs"]
    else:
        begin = time.perf_counter()
        decoded, decoded_stats = decode(specification["model"], packet)
        decode_wall = time.perf_counter() - begin
        subprocess_wall, decoder_inputs = None, None
    exact = bool(np.array_equal(decoded, words))
    if not exact:
        raise AssertionError(f"bit-exact decode failed: {specification['name']} {phase} {name}")
    count_values, raw_bytes = int(words.size), int(words.nbytes)
    shared = specification["shared_state_bytes"]
    return {"phase": phase, "case": name, "variant": specification["name"], "start": start,
            "tokens": count, "values": count_values, "raw_bytes": raw_bytes,
            "packet_bytes": len(packet), "bits_per_value": len(packet) * 8 / count_values,
            "saving_fraction": 1 - len(packet) / raw_bytes, "bit_exact": exact,
            "encode_wall_seconds": encode_wall, "decode_wall_seconds": decode_wall,
            "metadata_prepare_seconds": metadata_prepare, "encode_codec_wall_seconds": encode_codec_wall,
            "subprocess_wall_seconds": subprocess_wall, "packet_only_subprocess": subprocess_decode,
            "decoder_inputs": decoder_inputs, "shared_state_bytes": shared,
            "shared_state_wire_bytes": specification["shared_state_wire_bytes"],
            "resident_decoder_state_bytes": specification["resident_decoder_state_bytes"],
            "resident_encoder_projector_bytes": specification["resident_encoder_projector_bytes"],
            "serialized_decoder_state_bytes": specification["serialized_decoder_state_bytes"],
            "serialized_encoder_projector_bytes": specification["serialized_encoder_projector_bytes"],
            "cold_bits_per_value": (len(packet) + specification["serialized_decoder_state_bytes"]) * 8 / count_values,
            "cold_both_endpoints_bits_per_value": (len(packet) + specification["shared_state_wire_bytes"]) * 8 / count_values,
            "metadata_diagnostics": metadata_diagnostics, "codec_stats": stats,
            "metadata_rank": specification.get("rank", 0), "residue_bits": specification.get("residue_bits", 0),
            "correction": specification.get("correction", "direct")}


def render_summary(result):
    lines = ["# Transfer and paid prefill-metadata comparison", "",
             "Independent controlled validation of the supplied transfer algorithm. "
             "Capture uses a real model; the codec is a NumPy/constriction CPU reference. "
             "These are not GPU codec timings and do not reproduce the team's checkpoint/documents.", "",
             f"Model: `{result['provenance']['model_identity'].get('model', 'supplied ACT')}`. "
             f"All {len(result['test_rows'])} TEST packet round trips passed bit-exact comparison in a separate decoder process.", "",
             "TRAIN alone fits predictors and metadata PCA/quantizers. First half of DEV fits scales; "
             "a disjoint DEV page chooses metadata rank and coset width. TEST reports the frozen choices.", "",
             "| Variant | Packet bpv | Reduction | Decoder state MiB | Encoder projector MiB | Encode s | Decode s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in result["aggregate"].items():
        lines.append(f"| {name} | {value['bits_per_value']:.4f} | {100 * value['saving_fraction']:.2f}% | "
                     f"{value['resident_decoder_state_bytes'] / 2**20:.2f} | {value['resident_encoder_projector_bytes'] / 2**20:.2f} | "
                     f"{value['encode_seconds']:.3f} | {value['decode_seconds']:.3f} |")
    lines += ["", "Packet rates include transmitted int8 metadata, framing and the entropy stream. "
              "Decoder resident state and encoder-only projector state are charged separately on their respective machines. "
              "Cold rates and amortized capacity figures are in results.json. Dense predictor arrays are physical bytes; "
              "a compact coefficient estimate is only a diagnostic.", "",
              "`coset_model` factorizes the same joint symbol distribution and is a control; "
              "`coset_uniform` sends residues under a uniform model before conditional quotient coding. "
              "Coset savings must be compared on total packet bytes rather than multiplied into another reduction.", "",
              "Actual hidden metadata is extracted from attention input hidden_states during prefill. "
              "The same-budget KV-derived control and fixed token-roll hidden control test whether hidden hooks add useful information.", "",
              "The entropy stream is serial in this CPU reference. Parallel-token prediction does not establish a parallel GPU entropy decoder. "
              "Cross-machine probability/coder reproducibility and GPU speed remain untested.", ""]
    return "\n".join(lines)


def run(args):
    from qwen_kv_bench.transfer_reference import fit, save_model, model_statistics
    import importlib.metadata
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states = output / "states"
    states.mkdir(exist_ok=True)
    train, dev, tests, identity = prepare_captures(args, output)
    if args.capture_only:
        write_json(output / "capture_manifest.json", identity)
        print(f"Capture manifest: {output / 'capture_manifest.json'}", flush=True)
        return 0
    if not train or not tests:
        raise ValueError("at least one TRAIN and TEST capture is required")
    metadata_skip_reason = None
    if args.no_metadata:
        metadata_skip_reason = "--no-metadata selected"
    elif any("H" not in act or np.asarray(act["H"]).shape[:2] != np.asarray(act["K_bits"]).shape[:2]
             for act in train + [dev] + [value for _, value in tests]):
        metadata_skip_reason = "ACT files do not contain actual attention-input H hooks for every layer; baseline/coset tests remain enabled"
        print("Metadata skipped: " + metadata_skip_reason, flush=True)
    dev_tokens = np.asarray(dev["K_bits"]).shape[1]
    scale_tokens = dev_tokens // 2
    if scale_tokens < 2 or dev_tokens - scale_tokens < args.page_tokens:
        raise ValueError("DEV must hold disjoint scale and selection subsets")
    scales = slice_act(dev, 0, scale_tokens)
    max_positions = max([np.asarray(act["K_bits"]).shape[1] for act in train + [dev] + [act for _, act in tests]])
    specifications = []
    development = []
    selected = {}

    def prepare(name, mode, projector=None, shuffled=False, rank=0):
        metadata_train = None
        metadata_scale = None
        projector_report = {"serialized_bytes": 0, "array_bytes": 0}
        if projector is not None:
            metadata_train = [project_codes(projector, act, shuffled=shuffled)[0] for act in train]
            # Shuffle only inside the scale subset, never using DEV-selection inputs.
            metadata_scale = project_codes(projector, scales, shuffled=shuffled)[0]
            projector_report = save_projector(projector, states / (name + "_encoder.npz"))
        print(f"TRAIN fit: {name}", flush=True)
        begin = time.perf_counter()
        model = fit(train, scales, mode=mode, block_size=args.block_size,
                    metadata_train=metadata_train, metadata_scale=metadata_scale, max_positions=max_positions)
        fit_seconds = time.perf_counter() - begin
        path = states / (name + ".npz")
        save_model(model, path)
        statistics = model_statistics(model)
        spec = {"name": name, "model": model, "projector": projector, "shuffled": shuffled,
                "rank": rank, "state_path": path,
                "shared_state_bytes": statistics["frozen_array_bytes"] + projector_report["array_bytes"],
                "shared_state_wire_bytes": path.stat().st_size + projector_report["serialized_bytes"],
                "resident_decoder_state_bytes": statistics["frozen_array_bytes"],
                "resident_encoder_projector_bytes": projector_report["array_bytes"],
                "serialized_decoder_state_bytes": path.stat().st_size,
                "serialized_encoder_projector_bytes": projector_report["serialized_bytes"],
                "model_statistics": statistics, "projector_statistics": projector_report,
                "fit_seconds": fit_seconds, "state_sha256": sha_file(path)}
        return spec

    for mode in ("mean", "previous", "aqua", "block"):
        spec = prepare(mode if mode != "block" else f"block{args.block_size}", mode)
        row = measure(spec, dev, scale_tokens, args.page_tokens, args, output, "DEV", "selection")
        development.append(row)
        specifications.append(spec)
    block = specifications[-1]
    metadata_winners = {}
    metadata_candidates = {}
    metadata_grid = () if metadata_skip_reason else (("kv", False), ("hidden", False), ("hidden", True))
    for source, shuffled in metadata_grid:
        label = source + ("_shuffled" if shuffled else "")
        candidates = []
        metadata_candidates[label] = {}
        for rank in args.metadata_ranks:
            projector = fit_projector(train, source, rank)
            spec = prepare(f"block_{label}_r{rank}", "block", projector, shuffled, rank)
            row = measure(spec, dev, scale_tokens, args.page_tokens, args, output, "DEV", "selection")
            development.append(row)
            candidates.append((row["packet_bytes"], rank, spec))
            metadata_candidates[label][rank] = spec
        if candidates:
            _, rank, winner = min(candidates, key=lambda item: (item[0], item[1]))
            metadata_winners[label] = winner
            specifications.append(winner)
            selected[label] = {"rank": rank, "variant": winner["name"], "rule": "minimum complete DEV packet bytes; smaller-rank tie break"}
    if "hidden" in metadata_winners:
        hidden_rank = metadata_winners["hidden"]["rank"]
        for label in ("kv", "hidden_shuffled"):
            matched = metadata_candidates[label][hidden_rank]
            if matched["name"] != metadata_winners[label]["name"]:
                specifications.append(dict(matched, name=matched["name"] + "_matched_hidden_rank"))
            selected[label + "_matched_hidden_rank"] = {"rank": hidden_rank,
                "variant": matched["name"] if matched["name"] == metadata_winners[label]["name"] else matched["name"] + "_matched_hidden_rank",
                "rule": "same transmitted-code budget as the DEV-selected hidden variant"}
    for base_label, original in (("plain", block), ("hidden", metadata_winners.get("hidden"))):
        if original is None or not args.coset_bits:
            continue
        candidates = []
        for bits in args.coset_bits:
            spec = dict(original, name=f"{original['name']}_coset_uniform_r{bits}", correction="coset_uniform", residue_bits=bits)
            row = measure(spec, dev, scale_tokens, args.page_tokens, args, output, "DEV", "selection")
            development.append(row)
            candidates.append((row["packet_bytes"], bits, spec))
        _, bits, winner = min(candidates, key=lambda item: (item[0], item[1]))
        specifications.append(winner)
        selected["coset_" + base_label] = {"residue_bits": bits, "variant": winner["name"],
                                            "rule": "minimum complete DEV packet bytes; smaller-width tie break"}
        if base_label == "plain":
            specifications.append(dict(original, name=f"{original['name']}_coset_model_r{bits}",
                                       correction="coset_model", residue_bits=bits))
    test_rows = []
    for name, capture in tests:
        tokens = np.asarray(capture["K_bits"]).shape[1]
        starts = sorted({value if value >= 0 else tokens - args.page_tokens for value in args.page_starts})
        for start in starts:
            if start < 0 or start + args.page_tokens > tokens:
                raise ValueError("TEST page outside capture")
            for spec in specifications:
                print(f"TEST packet: {name} start={start} {spec['name']}", flush=True)
                row = measure(spec, capture, start, args.page_tokens, args, output, "TEST", name, subprocess_decode=True)
                test_rows.append(row)
                write_json(output / "checkpoint.json", {"test_rows": test_rows, "selected": selected})
    aggregate = {}
    for spec in specifications:
        rows = [row for row in test_rows if row["variant"] == spec["name"]]
        values = sum(row["values"] for row in rows)
        packet_bytes = sum(row["packet_bytes"] for row in rows)
        per_token_values = combined_words(train[0]).shape[0] * combined_words(train[0]).shape[2]
        bpv = packet_bytes * 8 / values
        amortized = {}
        for tokens in (1024, 8192, 32768, 131072):
            effective = bpv + spec["resident_decoder_state_bytes"] * 8 / (tokens * per_token_values)
            combined_effective = bpv + spec["shared_state_bytes"] * 8 / (tokens * per_token_values)
            amortized[str(tokens)] = {"bits_per_value": effective, "saving_fraction": 1 - effective / 16,
                                     "decoder_state_only": True, "both_endpoints_bits_per_value": combined_effective}
        aggregate[spec["name"]] = {"values": values, "packet_bytes": packet_bytes,
            "bits_per_value": bpv, "saving_fraction": 1 - bpv / 16,
            "shared_state_bytes": spec["shared_state_bytes"], "shared_state_wire_bytes": spec["shared_state_wire_bytes"],
            "resident_decoder_state_bytes": spec["resident_decoder_state_bytes"],
            "resident_encoder_projector_bytes": spec["resident_encoder_projector_bytes"],
            "serialized_decoder_state_bytes": spec["serialized_decoder_state_bytes"],
            "serialized_encoder_projector_bytes": spec["serialized_encoder_projector_bytes"],
            "model_statistics": spec["model_statistics"],
            "projector_statistics": spec["projector_statistics"], "fit_seconds": spec["fit_seconds"],
            "encode_seconds": sum(row["encode_wall_seconds"] for row in rows),
            "metadata_prepare_seconds": sum(row["metadata_prepare_seconds"] for row in rows),
            "encode_codec_seconds": sum(row["encode_codec_wall_seconds"] for row in rows),
            "decode_seconds": sum(row["decode_wall_seconds"] for row in rows), "state_amortization_by_tokens": amortized}
    source_files = [Path(__file__), ROOT / "qwen_kv_bench/transfer_reference.py", ROOT / "qwen_kv_bench/transfer_coset.py"]
    result = {"format": "transfer_extension_study_v1", "status": "completed", "exact_original_reproduction": False,
        "reason": "Independent real-model captures; original GGUF/source documents/captures are not supplied with the team scripts.",
        "arguments": vars(args), "selected_on_dev": selected, "dev_rows": development,
        "development_split": {"scale_start": 0, "scale_stop": scale_tokens,
                              "selection_start": scale_tokens, "selection_tokens": args.page_tokens,
                              "dev_capture_tokens": dev_tokens},
        "test_rows": test_rows, "aggregate": aggregate,
        "metadata_skip_reason": metadata_skip_reason,
        "provenance": {"model_identity": identity, "python": platform.python_version(), "numpy": np.__version__,
                       "constriction": importlib.metadata.version("constriction"),
                       "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha_file(path)} for path in source_files]},
        "scope": {"capture_device": args.device, "codec_device": "cpu", "gpu_codec_measured": False,
                  "cross_machine_determinism_tested": False, "token_parallel_entropy_decode": False}}
    write_json(output / "results.json", result)
    write_json(output / "source_manifest.json", result["provenance"])
    (output / "summary.md").write_text(render_summary(result))
    fields = ["phase", "case", "variant", "start", "tokens", "values", "raw_bytes", "packet_bytes", "bits_per_value",
              "saving_fraction", "bit_exact", "encode_wall_seconds", "decode_wall_seconds", "shared_state_bytes", "cold_bits_per_value"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(development + test_rows)
    print(f"Report: {output / 'summary.md'}", flush=True)
    return 0


def integer_list(value):
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ids-manifest-root")
    p.add_argument("--act-dir")
    p.add_argument("--train-tokens", type=int, default=512)
    p.add_argument("--capture-tokens", type=int, default=512)
    p.add_argument("--page-tokens", type=int, default=32)
    p.add_argument("--page-starts", type=integer_list, default=[0])
    p.add_argument("--train-cases", type=int, default=3)
    p.add_argument("--test-cases", type=int, default=2)
    p.add_argument("--metadata-ranks", type=integer_list, default=[4, 8])
    p.add_argument("--no-metadata", action="store_true", help="Allow original ACT datasets with partial/no hidden hooks")
    p.add_argument("--coset-bits", type=integer_list, default=[2, 4])
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--output", default="runs/transfer_extensions")
    p.add_argument("--reuse-captures", action="store_true")
    p.add_argument("--capture-only", action="store_true")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Bound fitting/copies to a contiguous layer prefix; model forward remains complete")
    p.add_argument("--decode-packet")
    p.add_argument("--state")
    p.add_argument("--decoded-output")
    p.add_argument("--decode-report")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.decode_packet:
        if not args.state or not args.decoded_output or not args.decode_report:
            raise ValueError("packet-only decoder needs --state, --decoded-output, --decode-report")
        return decode_child(args)
    if min(args.train_tokens, args.capture_tokens, args.page_tokens, args.train_cases, args.test_cases, args.threads) < 1:
        raise ValueError("positive token/case/thread counts required")
    if any(rank < 1 for rank in args.metadata_ranks) or any(not 1 <= bits <= 7 for bits in args.coset_bits):
        raise ValueError("metadata ranks must be positive and coset widths within 1..7")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

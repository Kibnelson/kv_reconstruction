"""One separately frozen fresh smoke test for independent whole-record coding.

Run once from the project root.  This extension is separate from the preceding
main freeze.  It fixes all nine candidates, the original 2048-token fixture,
the wrapper level, source hashes and model hashes BEFORE prefill.  It never
reads earlier final data, performs no tuning, and reuses encoded inner records
for each wrapped candidate.  Raw KV is kept outside the deliverable package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import time
from datetime import datetime, timezone

import numpy as np

from kv_coset_fields import combined, controls
from kv_research import container, learned_codec
from kv_residual import learned, learned_e8
from kv_smoke.download_model import MODEL_ID, REVISION
from kv_smoke.prompts import build_input_ids
from kv_stacked import wrapper


FIXTURE = dict(
    name="wrapped_extension_urban_beekeeping_inspection",
    kind="text",
    prefix="Urban beekeeping inspection log: a rooftop apiary above the city library.\n",
    body=(
        "Inspection <cycle>: Mira placed the hive tool beside the notebook and "
        "watched the returning bees land on the narrow wooden entrance. The "
        "roof garden offered flowering thyme, while rainwater collected in a "
        "shallow tray filled with smooth stones. She lifted one frame slowly, "
        "checked the capped brood, and marked the position of an unfinished "
        "honey comb. Traffic sounded below the parapet, but the colony remained "
        "calm. Her partner recorded the weather, the weight of the hive, and "
        "whether the nearby flower beds needed watering. They closed the lid "
        "before discussing the observations with the building caretaker.\n"
    ),
    suffix="\nAt the next rooftop inspection, Mira will first check",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def equal(left, right):
    return len(left) == len(right) and all(np.array_equal(a, b)
        for lp, rp in zip(left, right) for a, b in zip(lp, rp))


def configuration():
    common = dict(step_log2=10, group_size=3072, correction="best", assignment="radius", force=True)
    return dict(experiment="separate_fresh_whole_record_extension", tokens=2048, tile_tokens=16,
        wrapper_level=6, model="kv_research/models/learned64.bin",
        candidates={
            "learned_e8_entropy": dict(kind="learned_e8", options=dict(common, family="e8")),
            "learned_e8_entropy_wrapped": dict(kind="wrapped", source="learned_e8_entropy"),
            "learned_scalar_entropy": dict(kind="learned_e8", options=dict(common, family="scalar")),
            "learned_scalar_entropy_wrapped": dict(kind="wrapped", source="learned_scalar_entropy"),
            "learned_radius_entropy": dict(kind="learned", options=dict(correction="best_entropy", assignment="radius", force=True)),
            "learned_radius_entropy_wrapped": dict(kind="wrapped", source="learned_radius_entropy"),
            "learned_l1_entropy": dict(kind="learned", options=dict(correction="best_entropy", assignment="l1", force=True)),
            "learned_l1_entropy_wrapped": dict(kind="wrapped", source="learned_l1_entropy"),
            "huffman_lower_deflate": dict(kind="huffman_lower_deflate"),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".smoke_model")
    parser.add_argument("--raw-dir", default="kv_residual_wrap_data")
    parser.add_argument("--output", default="kv_residual/results/wrapped_extension.json")
    args = parser.parse_args()
    manifest_path = Path("kv_residual/wrapped_extension_manifest.json")
    selection_path = Path("kv_residual/wrapped_extension_selection.json")
    config_path = Path("kv_residual/wrapped_extension_config.json")
    if manifest_path.exists() or selection_path.exists():
        raise RuntimeError("this one-case extension has already been frozen/captured; inspect its existing artifacts")
    config = configuration()
    model_path = Path(config["model"])
    if model_path.stat().st_size != 70095:
        raise AssertionError("unexpected frozen learned-model size")
    dictionary = learned_codec.load_model(model_path)
    pretrained_manifest = json.loads((Path(args.model_dir) / "download_manifest.json").read_text())
    if pretrained_manifest["model_id"] != MODEL_ID or pretrained_manifest["revision"] != REVISION:
        raise AssertionError("pretrained checkpoint is not the pinned model")
    pretrained_hashes = {}
    for row in pretrained_manifest["files"]:
        actual = sha256(Path(args.model_dir) / row["name"])
        if actual != row["sha256"]:
            raise AssertionError("pretrained file hash mismatch: " + row["name"])
        pretrained_hashes[row["name"]] = actual
    sources = [Path(__file__), Path("kv_residual/learned.py"), Path("kv_residual/learned_e8.py"),
        Path("kv_residual/entropy.py"), Path("kv_stacked/wrapper.py"), Path("kv_coset_fields/combined.py"),
        Path("kv_coset_fields/controls.py"), Path("kv_stacked/huffman_codec.py"),
        Path("kv_research/container.py"), Path("kv_research/e8_codec.py"),
        Path("kv_research/learned_codec.py"), Path("kv_smoke/prompts.py")]
    source_hashes = {str(path): sha256(path) for path in sources}
    write_json(config_path, config)
    selection = dict(experiment=config["experiment"], frozen_utc=now(),
        rationale="Prespecified confirmation of the prior development-only whole-record wrapping probe; no new grid or tuning.",
        fixture=FIXTURE, fixture_sha256=hashlib.sha256(json.dumps(FIXTURE, sort_keys=True).encode()).hexdigest(),
        config_sha256=sha256(config_path), source_sha256=source_hashes,
        learned_model=dict(path=str(model_path), sha256=sha256(model_path), serialized_bytes=70095,
            model_id=dictionary.model_id),
        pretrained_model=dict(model_id=MODEL_ID, revision=REVISION, file_sha256=pretrained_hashes),
        split="one entirely new synthetic 2048-token fixture, separate from main final evaluation")
    write_json(selection_path, selection)
    print("extension frozen before capture", flush=True)

    # No tokenization, model forward, or cache capture occurs before freeze.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_research.model_utils import cache_matrices, mapping
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(19)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    input_ids = build_input_ids(tokenizer, FIXTURE, 2048)
    capture_started = now()
    started = time.perf_counter()
    with torch.inference_mode():
        output = lm(torch.tensor([input_ids], dtype=torch.long), use_cache=True, logits_to_keep=1)
        raw = cache_matrices(output.past_key_values)
    prefill_seconds = time.perf_counter() - started
    raw_payload = b"".join(w.astype("<u2").tobytes() for pair in raw for w in pair)
    raw_size = len(raw_payload)
    raw_digest = hashlib.sha256(raw_payload).hexdigest()
    first_token = int(output.logits[0, -1].argmax())
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / (FIXTURE["name"] + "_2048.npz")
    np.savez(raw_path, **mapping(raw))
    manifest = dict(experiment=config["experiment"], selection_file=str(selection_path),
        selection_sha256=sha256(selection_path), frozen_utc=selection["frozen_utc"],
        capture_started_utc=capture_started, capture_completed_utc=now(),
        model_id=MODEL_ID, revision=REVISION, fixture=FIXTURE, tokens=2048,
        input_token_ids=input_ids, first_token=first_token, raw_payload_bytes=raw_size,
        raw_sha256=raw_digest, raw_cache_path=str(raw_path), raw_cache_sha256=sha256(raw_path),
        prefill_seconds=prefill_seconds,
        runtime=dict(python=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, numpy=np.__version__, device="cpu",
            dtype="bfloat16", attention_backend="sdpa", threads=4))
    if not selection["frozen_utc"] < capture_started:
        raise AssertionError("capture did not occur after extension freeze")
    write_json(manifest_path, manifest)
    del lm, output, raw_payload
    print("captured fresh urban beekeeping 2048-token KV", flush=True)

    caches, decoders, states = {}, {}, {}
    results = dict(experiment=config["experiment"], phase="separate fresh confirmation extension",
        configuration=config, selection_sha256=sha256(selection_path), manifest_sha256=sha256(manifest_path),
        capture_manifest=str(manifest_path), raw_payload_bytes=raw_size, raw_sha256=raw_digest,
        codecs={}, timing_scope="CPU Python diagnostic timings while other work may run; no GPU or serving claim.",
        limitation="One new synthetic prompt, not a representative corpus or high-concurrency/admission measurement.")
    for name, spec in config["candidates"].items():
        kind = spec["kind"]
        source = spec.get("source")
        wrap_verified = 0
        dependency_seconds = 0.0
        if kind == "wrapped":
            inner = caches[source]
            current = [0]
            def encoder(words, layer, kvkind):
                nonlocal wrap_verified
                index = current[0]
                current[0] += 1
                a, b = struct.unpack_from("<QQ", inner.data, container.HEADER.size + 8 * index)
                record = inner.data[a:b]
                encoded = wrapper.wrap(record, level=6)
                if wrapper.unwrap(encoded.data) != record:
                    raise AssertionError("whole-record wrapper did not restore inner bytes")
                wrap_verified += 1
                return encoded
            inner_decoder = decoders[source]
            decoder = lambda blob, l, k: inner_decoder(wrapper.unwrap(blob), l, k)
            state = states[source]
            dependency_seconds = results["codecs"][source]["encode_seconds"]
        elif kind in ("learned", "learned_e8"):
            module = learned if kind == "learned" else learned_e8
            options = spec["options"]
            encoder = lambda words, l, k, module=module, options=options: module.encode_matrix(
                words, dictionary, f"layer_{l:02d}_{k}", **options)
            decoder = lambda blob, l, k, module=module: module.decode_matrix(blob, dictionary)
            state = 70095
        else:
            encoder = lambda words, l, k: combined.encode_matrix(words,
                lambda lower, exponent: controls.encode_lower(lower, exponent, "deflate"), 2)
            decoder = lambda blob, l, k: combined.decode_matrix(blob,
                lambda data, exponent: controls.decode_lower(data, exponent, "deflate"), 2)
            state = 0
        started = time.perf_counter()
        encoded = container.encode_cache(raw, encoder, tile_tokens=16)
        encode_seconds = time.perf_counter() - started
        caches[name], decoders[name], states[name] = encoded, decoder, state
        started = time.perf_counter()
        recovered = container.decode_cache(encoded.data, decoder)
        decode_seconds = time.perf_counter() - started
        if not equal(raw, recovered):
            raise AssertionError("full KV recovery failed: " + name)
        del recovered
        damaged = bytearray(encoded.data)
        shape = container.shape(damaged)
        damaged[shape["payload_start"]] ^= 255
        final_tile = shape["ntiles"] - 1
        later = container.decode_tile(damaged, shape["layers"] - 1, final_tile, decoder)
        if not all(np.array_equal(words, original[final_tile * 16:]) for words, original in zip(later, raw[-1])):
            raise AssertionError("later tile depends on an earlier record")
        if kind == "wrapped" and wrap_verified != shape["nrecords"]:
            raise AssertionError("not every wrapped record was byte-verified")
        row = dict(encoded.stats, roundtrip_exact=True, independent_later_tile=True,
            raw_bf16_words_compared=raw_size // 2, encoded_sha256=hashlib.sha256(encoded.data).hexdigest(),
            shared_model_bytes=state, total_bytes_including_model_n1=len(encoded.data) + state,
            savings_with_model_n1_percent=100 * (1 - (len(encoded.data) + state) / raw_size),
            encode_seconds=encode_seconds, decode_seconds=decode_seconds,
            inner_dependency_encode_seconds=dependency_seconds,
            total_encode_seconds_with_dependency=encode_seconds + dependency_seconds,
            wrapped_inner_records_verified=wrap_verified, inner_source=source)
        results["codecs"][name] = row
        write_json(args.output, results)
        print(f"{name}: {len(encoded.data)}B + {state} modelB = {row['savings_with_model_n1_percent']:.4f}% saved, exact", flush=True)
    if source_hashes != {str(path): sha256(path) for path in sources}:
        raise AssertionError("source changed after extension freeze")
    if sha256(config_path) != selection["config_sha256"] or sha256(model_path) != selection["learned_model"]["sha256"]:
        raise AssertionError("configuration/model changed after extension freeze")
    results["completed_utc"] = now()
    results["all_nine_exact"] = len(results["codecs"]) == 9 and all(r["roundtrip_exact"] for r in results["codecs"].values())
    results["source_config_model_unchanged_after_freeze"] = True
    results["wrap_cost_reuses_encoded_inner_records"] = True
    write_json(args.output, results)
    print("wrapped extension complete " + args.output, flush=True)


if __name__ == "__main__":
    main()

"""Post-freeze development diagnostic for independent whole-record Deflate.

This probe neither changes frozen codecs/configuration nor reads evaluation
captures.  It asks whether the existing record wrapper helps the E8 coordinate
stream and matched controls.  Such gains belong to generic whole-record coding,
not specifically to E8.  Every wrapper includes its complete 13-byte header.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kv_research.learned_codec import load_model
from kv_residual import learned, learned_e8
from kv_stacked import wrapper


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", default=".kv_research_data/dev_sensor_pipeline_1024.npz")
    parser.add_argument("--model", default="kv_research/models/learned64.bin")
    parser.add_argument("--output", default="kv_residual/results/record_wrapping_dev.json")
    args = parser.parse_args()
    if Path(args.development).name != "dev_sensor_pipeline_1024.npz":
        raise ValueError("this post-freeze probe is restricted to the original development capture")
    model = load_model(args.model)
    model_size = Path(args.model).stat().st_size
    keys = [f"layer_{layer:02d}_{kind}" for layer in (0, 15, 29) for kind in ("K", "V")]
    source_paths = (Path("kv_residual/learned.py"), Path("kv_residual/learned_e8.py"),
                    Path("kv_residual/entropy.py"), Path("kv_stacked/wrapper.py"),
                    Path("kv_research/e8_codec.py"), Path("kv_research/learned_codec.py"))
    frozen_hashes = {str(path): sha256(path) for path in source_paths}
    capture = np.load(args.development)
    specifications = {
        "learned_e8_step10_entropy": (learned_e8, dict(step_log2=10, family="e8",
            correction="best", assignment="radius", force=True, group_size=3072)),
        "learned_scalar_step10_entropy": (learned_e8, dict(step_log2=10, family="scalar",
            correction="best", assignment="radius", force=True, group_size=3072)),
        "learned_radius_entropy": (learned, dict(correction="best_entropy", assignment="radius", force=True)),
        "learned_l1_entropy": (learned, dict(correction="best_entropy", assignment="l1", force=True)),
    }
    results = {}
    for name, (codec, options) in specifications.items():
        inner_total = outer_total = raw = records = 0
        inner_components, outer_components, modes, mechanisms = Counter(), Counter(), Counter(), Counter()
        elapsed = time.perf_counter()
        for key in keys:
            for offset in range(0, 256, 16):
                words = capture[key][offset:offset + 16]
                inner = codec.encode_matrix(words, model, key, **options)
                if not np.array_equal(codec.decode_matrix(inner.data, model), words):
                    raise AssertionError((name, key, offset, "inner"))
                outer = wrapper.wrap(inner, level=6)
                restored_inner = wrapper.unwrap(outer.data)
                if restored_inner != inner.data:
                    raise AssertionError((name, key, offset, "inner bytes"))
                if not np.array_equal(codec.decode_matrix(restored_inner, model), words):
                    raise AssertionError((name, key, offset, "wrapped"))
                raw += words.nbytes
                inner_total += len(inner.data)
                outer_total += len(outer.data)
                records += 1
                inner_components.update(inner.stats["component_bytes"])
                outer_components.update(outer.stats["component_bytes"])
                modes.update(outer.stats["wrapper_mode_counts"])
                mechanisms.update(inner.stats.get("mechanism_counts", inner.stats.get("mechanism_counts", {})))
        if raw != 589824 or records != 96:
            raise AssertionError("unexpected development subset")
        assert sum(inner_components.values()) == inner_total
        assert sum(outer_components.values()) == outer_total
        assert outer_components["wrapper_header"] == records * 13
        entry = dict(options=options, matrix_records=records, raw_payload_bytes=raw,
            inner_serialized_bytes=inner_total, wrapped_serialized_bytes=outer_total,
            inner_space_saving_percent=100 * (1 - inner_total / raw),
            wrapped_space_saving_percent=100 * (1 - outer_total / raw),
            wrapper_saved_bytes=inner_total - outer_total,
            shared_model_bytes=model_size,
            inner_plus_model_bytes=inner_total + model_size,
            wrapped_plus_model_bytes=outer_total + model_size,
            inner_plus_model_saving_percent=100 * (1 - (inner_total + model_size) / raw),
            wrapped_plus_model_saving_percent=100 * (1 - (outer_total + model_size) / raw),
            inner_component_bytes=dict(inner_components), wrapped_component_bytes=dict(outer_components),
            wrapper_mode_counts=dict(modes), mechanism_counts=dict(mechanisms),
            exact_inner_roundtrip=True, exact_wrapper_bytes=True, exact_wrapped_roundtrip=True,
            cpu_encode_wrap_and_double_decode_seconds=time.perf_counter() - elapsed)
        results[name] = entry
        print(json.dumps({"candidate": name, "inner_bytes": inner_total,
            "wrapped_bytes": outer_total, "wrapped_saving_percent": entry["wrapped_space_saving_percent"],
            "exact": True}), flush=True)
    current_hashes = {str(path): sha256(path) for path in source_paths}
    if current_hashes != frozen_hashes:
        raise AssertionError("a codec source changed during the diagnostic")
    best_learned = min((name for name in results if name in ("learned_radius_entropy", "learned_l1_entropy")),
                       key=lambda name: results[name]["wrapped_serialized_bytes"])
    e8 = results["learned_e8_step10_entropy"]["wrapped_serialized_bytes"]
    scalar = results["learned_scalar_step10_entropy"]["wrapped_serialized_bytes"]
    learned_size = results[best_learned]["wrapped_serialized_bytes"]
    report = dict(phase="post-freeze development-only diagnostic; not final evaluation or selection",
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=dict(development=args.development, keys=keys, first_tokens=256, tile_tokens=16,
            matrix_records=96, deflate_level=6, wrapper_header_bytes_per_record=13,
            parameter_search=False, frozen_codecs_unchanged=True),
        provenance=dict(development_sha256=sha256(args.development), model_path=args.model,
            model_sha256=sha256(args.model), model_id=model.model_id, shared_model_bytes=model_size,
            shared_dictionary_prototype_bytes=sum(book.nbytes for book in model.codebooks.values()),
            source_sha256=frozen_hashes, probe_sha256=sha256(__file__)),
        results=results,
        comparison=dict(best_wrapped_learned_control=best_learned,
            wrapped_e8_minus_wrapped_scalar_bytes=e8 - scalar,
            wrapped_e8_minus_best_wrapped_learned_bytes=e8 - learned_size),
        interpretation_limits=[
            "All bytes are sums of independent matrix records on the same small development subset; no outer cache offset index is included.",
            "Shared learned state is reported both separately and charged once per candidate on this subset; no generalization/admission claim is made.",
            "The wrapper codes every inner byte, including headers and corrections; any saving is not isolated to E8 coordinates.",
            "CPU timings are diagnostic Python timings while other work may run; they do not establish cheap GPU execution.",
            "A promising post-freeze diagnostic requires another fresh test split before any confirmation claim."])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run only the supplied team's pool functions on independently captured real KV.

This is a CPU byte/round-trip reference. It measures packed block payloads and
does not implement a GPU allocator, escape-rank index, or fused attention kernel.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = ("from_bits", "words", "zz", "unzz", "fit_pool", "pack_block", "unpack_block")
TRAIN_NAMES = ("train_train_railway_512", "train_train_graph_python_512", "train_train_catalog_512")
TEST_NAMES = ("test_test_antarctic_story_512", "test_test_rust_log_parser_512")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path, fit, channels, block_tokens=16):
    """Compile selected original definitions; never run its imports/main body."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    definitions = {name: [] for name in FUNCTIONS}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in definitions:
            definitions[node.name].append(node)
    if any(len(nodes) != 1 for nodes in definitions.values()):
        raise ValueError("expected exactly one definition of each original pool function")
    selected = [definitions[name][0] for name in FUNCTIONS]
    if any(node.decorator_list or node.args.defaults or any(node.args.kw_defaults) for node in selected):
        raise ValueError("original pool definitions must have no executable defaults/decorators")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np, "FIT": fit, "NV": channels, "PB": block_tokens}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_capture(path):
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key].copy() for key in ("K_bits", "V_bits", "Hk", "hd", "base", "ids")}
    keys, values = out["K_bits"], out["V_bits"]
    if keys.ndim != 3 or keys.shape != values.shape or keys.dtype != np.uint16 or values.dtype != np.uint16:
        raise ValueError("expected paired original BF16 uint16 [layer,token,channel] arrays")
    return out


def exponent_layout(words, reference, widths):
    exponent = (words.astype(np.int64) >> 7) & 255
    delta = exponent - reference[None, :]
    zigzag = np.where(delta >= 0, 2 * delta, -2 * delta - 1)
    escapes = zigzag > (1 << widths[None, :]) - 2
    dense_bits = int(len(words) * widths.sum())
    return dict(escape_count=int(escapes.sum()), sign_mantissa_bytes=int(words.size),
                dense_exponent_bits=dense_bits, dense_exponent_bytes=(dense_bits + 7) // 8,
                escape_header_bytes=2)


def group_min_formula_bits(words):
    """Team's 8-value exponent-group baseline formula, not encoded bytes."""
    if words.size % 8:
        raise ValueError("baseline needs groups of eight values")
    exponent = ((words.astype(np.int64) >> 7) & 255).reshape(-1, 8)
    span = np.ptp(exponent, axis=1)
    width = np.where(span > 0, np.floor(np.log2(np.maximum(span, 1))).astype(np.int64) + 1, 0)
    return int(words.size * 8 + (8 + 4 + 8 * width).sum())


def capture_identity(path):
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return dict(file=path.name, file_bytes=path.stat().st_size, file_sha256=file_sha256(path),
                metadata_file_sha256=file_sha256(metadata_path) if metadata_path.exists() else None,
                name=metadata.get("name"), kind=metadata.get("kind"),
                model_fingerprint=metadata.get("model_fingerprint"),
                kv_sha256=metadata.get("kv_sha256"), ids_sha256=metadata.get("ids_sha256"),
                key_order=metadata.get("key_order"))


def run_study(source, capture_dir, starts=(0, 480), page_tokens=32):
    if page_tokens <= 0 or page_tokens % 16:
        raise ValueError("page token count must be a positive multiple of 16")
    started = time.perf_counter()
    train = [load_capture(capture_dir / (name + ".npz")) for name in TRAIN_NAMES]
    layers, train_tokens, key_channels = train[0]["K_bits"].shape
    if any(item["K_bits"].shape != train[0]["K_bits"].shape for item in train):
        raise ValueError("training capture geometries differ")
    combined_channels = 2 * key_channels
    reference = load_reference(source, train, combined_channels)
    fitted = [reference["fit_pool"](layer) for layer in range(layers)]
    fitting_seconds = time.perf_counter() - started
    rows = []
    block_rows = []
    test_identities = []
    for name in TEST_NAMES:
        path = capture_dir / (name + ".npz")
        capture = load_capture(path)
        if capture["K_bits"].shape[::2] != (layers, key_channels):
            raise ValueError("test geometry differs from training")
        test_identities.append(capture_identity(path))
        for start in starts:
            stop = start + page_tokens
            if start < 0 or stop > capture["K_bits"].shape[1]:
                raise ValueError("test page lies outside captured tokens")
            packed_bytes = baseline_bits = escapes = headers = dense_bytes = sm_bytes = word_count = 0
            roundtrip = True
            for layer, (ref, wid) in enumerate(fitted):
                page = reference["words"](capture, layer)[start:stop]
                baseline_bits += group_min_formula_bits(page)
                word_count += page.size
                for local_start in range(0, page_tokens, 16):
                    block = page[local_start:local_start + 16]
                    record = reference["pack_block"](block, ref, wid)
                    decoded = reference["unpack_block"](record, ref, wid)
                    exact = bool(np.array_equal(decoded, block))
                    layout = exponent_layout(block, ref, wid)
                    expected_bytes = (layout["escape_header_bytes"] + layout["sign_mantissa_bytes"]
                                      + layout["dense_exponent_bytes"] + layout["escape_count"])
                    if len(record) != expected_bytes:
                        raise AssertionError("actual record bytes disagree with its exact size formula")
                    if not exact:
                        raise AssertionError("original pool failed exact BF16 round trip")
                    packed_bytes += len(record)
                    escapes += layout["escape_count"]
                    headers += layout["escape_header_bytes"]
                    dense_bytes += layout["dense_exponent_bytes"]
                    sm_bytes += layout["sign_mantissa_bytes"]
                    roundtrip &= exact
                    block_rows.append(dict(case=name, layer=layer, token_start=start + local_start,
                                           words=int(block.size), packed_bytes=int(len(record)),
                                           expected_bytes=int(expected_bytes), exact=exact, **layout))
            rows.append(dict(case=name, token_start=start, token_count=page_tokens, layer_count=layers,
                             words=int(word_count), raw_bf16_bytes=int(word_count * 2),
                             packed_bytes=int(packed_bytes), packed_bits_per_value=8 * packed_bytes / word_count,
                             reduction_fraction=1 - packed_bytes / (word_count * 2),
                             baseline_formula_bits=int(baseline_bits),
                             baseline_formula_bits_per_value=baseline_bits / word_count,
                             escape_count=int(escapes), escape_fraction=escapes / word_count,
                             escape_header_bytes=int(headers), dense_exponent_bytes=int(dense_bytes),
                             sign_mantissa_bytes=int(sm_bytes), exact=bool(roundtrip)))
    total_words = sum(row["words"] for row in rows)
    total_bytes = sum(row["packed_bytes"] for row in rows)
    total_baseline_bits = sum(row["baseline_formula_bits"] for row in rows)
    report = dict(format="team_original_pool_cpu_reference_v1", status="completed",
                  scope="independent real-model validation; not reproduction of original team dataset",
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, device="cpu"),
                  source=dict(file=source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
                              sha256=file_sha256(source), selected_original_functions=list(FUNCTIONS),
                              runner_sha256=file_sha256(__file__), original_source_unchanged=True),
                  geometry=dict(layers=layers, tokens_per_training_document=train_tokens,
                                key_channels=key_channels, combined_kv_channels=combined_channels,
                                block_tokens=16, values_per_block=16 * combined_channels,
                                kv_heads=int(train[0]["Hk"]), head_dimension=int(train[0]["hd"]),
                                rope_base=float(train[0]["base"])),
                  training=[capture_identity(capture_dir / (name + ".npz")) for name in TRAIN_NAMES],
                  tests=test_identities, fitting_seconds=fitting_seconds,
                  shared_calibration_numpy_bytes=sum(ref.nbytes + wid.nbytes for ref, wid in fitted),
                  accounting=dict(measured="actual concatenated variable-size original pack_block records",
                                  included=["raw sign/mantissa bytes", "packed exponent codes and padding",
                                            "one 2-byte escape count per 16-token block", "raw escape exponents"],
                                  excluded=["shared calibration references/widths", "page/block offset directory",
                                            "escape-rank lookup index", "GPU allocation and alignment",
                                            "fixed escape reservation and overflow allocation", "serving integration"],
                                  baseline="formula only: 8 sign/mantissa bits/value plus 8-bit min, 4-bit width and packed deltas per eight exponents"),
                  aggregate=dict(pages=len(rows), blocks=len(block_rows), words=int(total_words),
                                 raw_bf16_bytes=int(total_words * 2), packed_bytes=int(total_bytes),
                                 packed_bits_per_value=8 * total_bytes / total_words,
                                 reduction_fraction=1 - total_bytes / (total_words * 2),
                                 baseline_formula_bits_per_value=total_baseline_bits / total_words,
                                 exact=all(row["exact"] for row in rows),
                                 all_block_byte_counts_match_formula=all(row["packed_bytes"] == row["expected_bytes"] for row in block_rows)),
                  pages=rows, blocks=block_rows)
    report["total_seconds"] = time.perf_counter() - started
    return report


def markdown(report):
    total = report["aggregate"]
    lines = ["# Original team pool functions: independent CPU validation", "",
             "The original supplied packing/unpacking functions were executed on fresh real-model KV. "
             "This is not an exact reproduction of the team's Instruct Q4_1 checkpoint and documents.", "",
             "Model: HuggingFaceTB/SmolLM2-135M base, revision "
             "93efa2f097d58c2a74874c7e644dbc9b0cee75a2; native Hugging Face BF16 CPU capture. "
             "Keys were converted from split-half RoPE channel order to adjacent-pair order; values were unchanged.", "",
             f"All {total['words']:,} original BF16 words round-tripped exactly across "
             f"{total['pages']} pages and {total['blocks']} blocks, using all {report['geometry']['layers']} layers.", "",
             "| Test | Start | Original packed pool, bpv | Reduction vs BF16 | Group-min formula, bpv | Exact |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in report["pages"]:
        lines.append(f"| {row['case']} | {row['token_start']} | {row['packed_bits_per_value']:.4f} | "
                     f"{100 * row['reduction_fraction']:.2f}% | {row['baseline_formula_bits_per_value']:.4f} | {row['exact']} |")
    lines.extend([f"| **Weighted aggregate** | | **{total['packed_bits_per_value']:.4f}** | "
                  f"**{100 * total['reduction_fraction']:.2f}%** | **{total['baseline_formula_bits_per_value']:.4f}** | **{total['exact']}** |", "",
                  "Actual packed bytes include one 2-byte escape count per 16-token block, raw sign/mantissa, "
                  "packed exponent codes and raw escape exponents. Every actual block size matched its exact formula. "
                  "The comparison baseline is a size formula, not an independently executed encoder.", "",
                  "Shared calibration arrays, block/page indexes, escape-rank lookup indexes, GPU alignment/allocation, "
                  "fixed escape reservation and overflow allocation are excluded. No GPU kernel or attention throughput "
                  "was measured. The results therefore establish the original variable-size payload representation's "
                  "correctness and size on these captures, not its serving performance.", "",
                  "Three TRAIN documents fit references and widths; the two TEST documents are disjoint. "
                  "Only the first and final 32-token pages of each 512-token TEST capture were measured. "
                  "Results JSON retains source and capture SHA-256 identities and every block's byte accounting.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/team_transfer_original/combined.py")
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "runs/transfer_smoke_cpu/captures")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results/team_pool_reference_cpu_20261020")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be fresh or empty")
    result = run_study(args.source.resolve(), args.capture_dir.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "summary.md").write_text(markdown(result))
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()

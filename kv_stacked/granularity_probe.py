"""Development-only temporal granularity sweep with exact byte accounting.

This probe reads one existing development capture. Larger records change the
access/workspace contract; they are not comparisons at the fixed 16-token tile
contract. No final evaluation data, GPU kernel, or serving performance is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kv_research import container
from kv_stacked import field_codec, huffman_codec, wrapper


def load_cache(path):
    with np.load(path) as data:
        layers = sum(name.endswith("_K") for name in data.files)
        return [
            (data[f"layer_{i:02d}_K"].copy(), data[f"layer_{i:02d}_V"].copy())
            for i in range(layers)
        ]


def codecs():
    return {
        "field_canonical_huffman": (
            lambda words, layer, kind: huffman_codec.encode_matrix(words),
            lambda data, layer, kind: huffman_codec.decode_matrix(data),
        ),
        "raw_byteshuffle_deflate6": (
            lambda words, layer, kind: wrapper.wrap(
                wrapper.raw_matrix(words, shuffle=True), level=6
            ),
            lambda data, layer, kind: wrapper.decode_raw_matrix(wrapper.unwrap(data)),
        ),
        "field_best_deflate6_wrapper6": (
            lambda words, layer, kind: wrapper.wrap(
                field_codec.encode_matrix(words, strategy="best", level=6, group_size=64),
                level=6,
            ),
            lambda data, layer, kind: field_codec.decode_matrix(wrapper.unwrap(data)),
        ),
    }


def run(source: Path, output: Path, tile_sizes=(16, 64, 256)):
    matrices = load_cache(source)
    if not matrices:
        raise ValueError("empty development capture")
    tokens, channels = matrices[0][0].shape
    raw_bytes = sum(words.nbytes for pair in matrices for words in pair)
    result = {
        "scope": "development only; no final-dataset access",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape": {"layers": len(matrices), "tokens": tokens, "channels": channels},
        "raw_payload_bytes": raw_bytes,
        "selection_note": (
            "Exploratory granularity sweep before final configuration freeze. "
            "All coding choices and record metadata are charged to serialized bytes."
        ),
        "workspace_note": (
            "Decoded_KV_word_tile_bytes counts only returned uint16 K+V words for "
            "one layer. It excludes encoded resident state, entropy/Deflate tables, "
            "decompression temporaries, Python objects, attention, and model state. "
            "Larger tiles change random-access granularity and workspace, so their "
            "ratios are not apples-to-apples comparisons with the 16-token contract."
        ),
        "timing_note": (
            "Single CPU Python invocations, no warmup distribution. Full-cache "
            "decode is timed for correctness; these are not GPU or serving timings."
        ),
        "results": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for tile_tokens in tile_sizes:
        for name, (encoder, decoder) in codecs().items():
            start = time.perf_counter()
            encoded = container.encode_cache(matrices, encoder, tile_tokens=tile_tokens)
            encode_seconds = time.perf_counter() - start
            start = time.perf_counter()
            restored = container.decode_cache(encoded.data, decoder)
            decode_seconds = time.perf_counter() - start
            if len(restored) != len(matrices) or not all(
                np.array_equal(original, recovered)
                for pair, decoded_pair in zip(matrices, restored)
                for original, recovered in zip(pair, decoded_pair)
            ):
                raise AssertionError(f"non-exact cache: {name} tile={tile_tokens}")
            shape = container.shape(encoded.data)
            checked_tiles = sorted({0, shape["ntiles"] // 2, shape["ntiles"] - 1})
            for layer in {0, len(matrices) // 2, len(matrices) - 1}:
                for tile in checked_tiles:
                    restored_pair = container.decode_tile(encoded.data, layer, tile, decoder)
                    start_row = tile * tile_tokens
                    for original, recovered in zip(matrices[layer], restored_pair):
                        if not np.array_equal(
                            original[start_row:start_row + recovered.shape[0]], recovered
                        ):
                            raise AssertionError("independent random-access tile mismatch")
            stats = encoded.stats
            if sum(stats["component_bytes"].values()) != len(encoded.data):
                raise AssertionError("serialized component accounting mismatch")
            result["results"].append({
                "method": name,
                "tile_tokens": tile_tokens,
                "exact": True,
                "independent_tile_checks": len(checked_tiles) * len({0, len(matrices) // 2, len(matrices) - 1}),
                "decoded_KV_word_tile_bytes": min(tile_tokens, tokens) * channels * 2 * 2,
                "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "raw_payload_bytes": raw_bytes,
                "serialized_bytes": len(encoded.data),
                "ratio_vs_raw_payload": raw_bytes / len(encoded.data),
                "space_saving_percent": 100 * (1 - len(encoded.data) / raw_bytes),
                "component_bytes": stats["component_bytes"],
                "K_record_bytes": sum(row["serialized_bytes"] for row in stats["per_layer"] if row["kind"] == "K"),
                "V_record_bytes": sum(row["serialized_bytes"] for row in stats["per_layer"] if row["kind"] == "V"),
            })
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: result["results"][-1][key] for key in (
                "method", "tile_tokens", "serialized_bytes", "ratio_vs_raw_payload",
                "space_saving_percent", "decoded_KV_word_tile_bytes", "exact"
            )}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(".kv_research_data/dev_sensor_pipeline_1024.npz"))
    parser.add_argument("--output", type=Path, default=Path("kv_stacked/results/granularity_dev.json"))
    args = parser.parse_args()
    run(args.source, args.output)

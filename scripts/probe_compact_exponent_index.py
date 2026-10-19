#!/usr/bin/env python3
"""Prove an index-only space reduction for existing exact exponent packets.

This transforms actual serialized offset bytes, restores the original uint32
arrays, reinjects them into the original packet, and verifies exact BF16 word
reconstruction.  It does NOT define a new production packet format, measure a
GPU kernel, or count descriptor/version changes needed by a future format.

Example:
    python scripts/probe_compact_exponent_index.py --output /tmp/index_probe.json
Optional existing-capture examples:
    python scripts/probe_compact_exponent_index.py --sample sample_raw.npz \
        --sample-records 4 --output /tmp/index_probe_with_sample.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_kv_bench import exponent_recipe as recipe
from qwen_kv_bench import gpu_ops as g


_OFFSET_FIELDS = (("secondary_offsets", "secondary_count"),
                  ("rare_offsets", "rare_count"))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def compact_start_bytes(packet):
    """Serialize only starts, choosing width from geometry, never observed data.

    Before segment j, at most j*segment_size symbols can have escaped.  Thus
    every stored start fits uint16 if the last segment starts at most 65535.
    The terminal value is already secondary_count/rare_count in the descriptor.
    Omitting it matters: a 65536-symbol all-escape packet has terminal 65536,
    although its last segment START is only 65280 at segment size 256.
    """
    meta = packet.metadata
    if meta["method"] != "prefill_exponent":
        raise ValueError("this probe transforms prefill exponent packets only")
    count = int(np.prod(meta["shape"], dtype=np.int64))
    size = meta["segment_size"]
    last_start = ((count - 1) // size) * size if count else 0
    width = 16 if last_start <= 65535 else 32
    dtype = np.dtype("<u2" if width == 16 else "<u4")
    blobs = {}
    for name, terminal_name in _OFFSET_FIELDS:
        original = packet.fields[name].view(torch.uint32).detach().cpu().numpy()
        _check(len(original) == meta["segment_count"] + 1, "original index length mismatch")
        _check(int(original[-1]) == meta[terminal_name], "terminal differs from descriptor")
        starts = original[:-1]
        _check(not len(starts) or int(starts.max()) <= last_start, "geometry bound violated")
        blobs[name] = starts.astype(dtype).tobytes()
    return blobs, width, last_start


def restore_uint32_arrays(blobs, width, metadata, device):
    """Recover canonical original arrays from serialized starts plus metadata."""
    dtype = np.dtype("<u2" if width == 16 else "<u4")
    restored = {}
    for name, terminal_name in _OFFSET_FIELDS:
        starts = np.frombuffer(blobs[name], dtype=dtype)
        _check(len(starts) == metadata["segment_count"], "compact index length mismatch")
        array = np.concatenate((starts.astype("<u4"),
                                np.array([metadata[terminal_name]], dtype="<u4")))
        restored[name] = g._wire(torch.from_numpy(array.copy()).to(device))
    return restored


def prove_case(name, words, *, predictor="median", layout="time", segment_size=256):
    packet = recipe.encode(words, predictor=predictor, layout=layout,
                           segment_size=segment_size)
    original_fields = {key: value.clone() for key, value in packet.fields.items()}
    blobs, width, last_start = compact_start_bytes(packet)
    restored = restore_uint32_arrays(blobs, width, packet.metadata, words.device)
    for key in restored:
        _check(torch.equal(restored[key], original_fields[key]),
               f"{name}: offset bytes changed after restore: {key}")
    fields = dict(packet.fields)
    fields.update(restored)
    reinjected = g.Packet(fields, packet.header.clone(), copy.deepcopy(packet.metadata),
                          packet._descriptor_bytes, packet._header_verified)
    g.validate_packet(reinjected, check_header=True)
    decoded = g.decode(reinjected)
    _check(torch.equal(decoded.to(torch.int32), words.to(torch.int32)),
           name + ": BF16 reconstruction mismatch")
    _check(all(torch.equal(packet.fields[key], value) for key, value in original_fields.items()),
           name + ": original packet mutated")
    baseline = sum(packet.fields[key].numel() for key, _ in _OFFSET_FIELDS)
    compact = sum(len(blob) for blob in blobs.values())
    starts_dtype = np.dtype("<u2" if width == 16 else "<u4")
    maximum_start = max((int(np.frombuffer(blob, dtype=starts_dtype).max())
                         for blob in blobs.values() if blob), default=0)
    raw = words.numel() * 2
    return dict(name=name, shape=list(words.shape), predictor=predictor, layout=layout,
                device=str(words.device), values=words.numel(), raw_KV_bytes=raw,
                segment_size=segment_size, segments=packet.metadata["segment_count"],
                last_segment_start=last_start, geometry_selected_offset_bits=width,
                maximum_actual_serialized_start=maximum_start,
                terminal_secondary_count=packet.metadata["secondary_count"],
                terminal_rare_count=packet.metadata["rare_count"],
                baseline_index_bytes=baseline, stored_start_index_bytes=compact,
                index_byte_gain=baseline - compact,
                index_only_gain_percentage_points_of_raw=(100 * (baseline - compact) / raw if raw else None),
                original_packet_bytes=packet.physical_nbytes,
                checks=dict(original_offset_bytes_restored_exactly=True,
                            original_wire_descriptor_validated=True,
                            original_packet_unchanged=True, BF16_words_exact=True),
                descriptor_and_version_growth_accounted=False,
                GPU_timing_measured=False)


def built_in_cases(device):
    generator = torch.Generator(device="cpu").manual_seed(91973)
    def random(shape):
        return torch.randint(0, 65536, shape, generator=generator, dtype=torch.int64).to(torch.uint16).to(device)
    yield "all_65536_BF16_patterns", torch.arange(65536, dtype=torch.int64).to(torch.uint16).reshape(256, 256).to(device), {}
    # A median of an all-identical channel is exact and would have no escapes;
    # this deliberately uses the existing fixed127 CONTROL to exercise a
    # terminal 65536 in both streams. It is not an adaptation-quality result.
    yield "all_rare_65536_terminal_needs_17_bits", torch.zeros((64, 1024), dtype=torch.uint16, device=device), {"predictor": "fixed127"}
    yield "random_tail", random((17, 79)), {}
    yield "empty_tokens", torch.empty((0, 7), dtype=torch.uint16, device=device), {}
    yield "empty_channels", torch.empty((7, 0), dtype=torch.uint16, device=device), {}
    yield "empty_both", torch.empty((0, 0), dtype=torch.uint16, device=device), {}
    yield "66560_values_geometry_requires_uint32", random((65, 1024)), {}
    yield "random_65536_values", random((64, 1024)), {}
    yield "singleton", torch.full((1, 1), 65535, dtype=torch.uint16, device=device), {}
    yield "65536_single_symbol_segments_uint16_max_start", torch.zeros((64, 1024), dtype=torch.uint16, device=device), {"predictor": "fixed127", "segment_size": 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    parser.add_argument("--sample", type=Path, help="Optional NPZ containing uint16 [tokens,channels] KV arrays")
    parser.add_argument("--sample-records", type=int, default=4)
    parser.add_argument("--sample-tile-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if args.sample_records < 1 or args.sample_tile_tokens < 1:
        parser.error("sample record count and tile length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but no CUDA device is available")
    torch.set_num_threads(2)
    rows = [prove_case(name, words, **options) for name, words, options in built_in_cases(args.device)]
    by_name = {row["name"]: row for row in rows}
    all_rare = by_name["all_rare_65536_terminal_needs_17_bits"]
    _check(all_rare["terminal_secondary_count"] == all_rare["terminal_rare_count"] == 65536,
           "all-rare test did not exercise terminal 65536")
    _check(all_rare["geometry_selected_offset_bits"] == 16, "all-rare starts should fit uint16")
    _check(by_name["66560_values_geometry_requires_uint32"]["geometry_selected_offset_bits"] == 32,
           "larger geometry must use uint32 regardless of observed offsets")
    _check(by_name["65536_single_symbol_segments_uint16_max_start"]["last_segment_start"] == 65535,
           "uint16 boundary case was not reached")
    _check(by_name["65536_single_symbol_segments_uint16_max_start"]["maximum_actual_serialized_start"] == 65535,
           "largest uint16 start value was not serialized")
    if args.sample:
        count = 0
        with np.load(args.sample, allow_pickle=False) as sample:
            for key in sample.files:
                array = sample[key]
                if array.dtype != np.uint16 or array.ndim != 2 or not array.size:
                    continue
                for start in range(0, len(array), args.sample_tile_tokens):
                    words = torch.from_numpy(array[start:start + args.sample_tile_tokens].copy()).to(args.device)
                    rows.append(prove_case(f"sample:{key}:{start}", words))
                    count += 1
                    if count == args.sample_records:
                        break
                if count == args.sample_records:
                    break
        _check(count > 0, "sample had no nonempty uint16 [tokens,channels] arrays")
    report = dict(format="compact_exponent_index_transformation_proof_v1", status="passed",
                  device=args.device, cases=rows,
                  scope=("Actual start-array bytes are compacted, expanded to original uint32 arrays, "
                         "and reinjected into the existing packet before exact BF16 decoding. "
                         "Index-only byte gains exclude any future descriptor/version growth. "
                         "No new production packet format or GPU speed claim."),
                  optional_sample=str(args.sample) if args.sample else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], device=args.device, cases=len(rows),
                          output=str(args.output) if args.output else None,
                          index_gain_for_65536_values=all_rare["index_byte_gain"],
                          index_only_gain_percentage_points=all_rare["index_only_gain_percentage_points_of_raw"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

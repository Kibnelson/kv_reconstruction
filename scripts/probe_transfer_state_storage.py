#!/usr/bin/env python3
"""Measure exact INT32 storage of frozen transfer predictor coefficients.

This probe does not change the running codec or claim new KV-stream savings.
It compares predictor outputs after storing A as INT32, while features and
accumulation remain INT64. Inputs and state are explicit and never refitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qwen_kv_bench import transfer_reference as reference


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def compact_coefficients(model):
    """Copy frozen A arrays to INT32 only when every value is exactly retained."""
    result = copy.deepcopy(model)
    for blocks in result.predictors:
        for block in blocks:
            array = block["A"]
            if array.dtype.kind != "i" or np.any(array < -(1 << 31)) or np.any(array >= 1 << 31):
                raise ValueError("coefficient array cannot be stored exactly as INT32")
            block["A"] = array.astype(np.int32)
    result.state_sha256 = reference._state_digest(result)
    return result


def storage(model):
    arrays = reference._arrays(model)
    matrices = [block["A"] for blocks in model.predictors for block in blocks]
    minimum_counts = {}
    mixed_candidate_bytes = 0
    mixed_candidate_supported = True
    for matrix in matrices:
        if not matrix.size:
            selected = "empty"
        else:
            low, high = int(matrix.min()), int(matrix.max())
            selected = next(name for bits, name in ((8, "int8"), (16, "int16"), (32, "int32"), (64, "int64"))
                            if -(1 << (bits - 1)) <= low and high < 1 << (bits - 1))
        minimum_counts[selected] = minimum_counts.get(selected, 0) + 1
        if selected in ("empty", "int8", "int16"):
            mixed_candidate_bytes += matrix.size * 2
        elif selected == "int32":
            mixed_candidate_bytes += matrix.size * 4
        else:
            mixed_candidate_supported = False
    frozen_bytes = int(sum(x.nbytes for x in arrays.values()))
    coefficient_bytes = int(sum(x.nbytes for x in matrices))
    return dict(coefficient_elements=int(sum(x.size for x in matrices)),
                coefficient_bytes=coefficient_bytes,
                all_frozen_array_bytes=frozen_bytes,
                coefficient_dtypes=sorted({str(x.dtype) for x in matrices}),
                minimum_exact_signed_dtype_block_counts=minimum_counts,
                INT16_or_INT32_candidate_coefficient_bytes=int(mixed_candidate_bytes) if mixed_candidate_supported else None,
                INT16_or_INT32_candidate_all_frozen_array_bytes=frozen_bytes - coefficient_bytes + int(mixed_candidate_bytes) if mixed_candidate_supported else None,
                state_sha256=model.state_sha256)


def compare_predictions(original, compact, capture, start=0, tokens=16):
    """Replay exact prior/earlier-channel state in the encoder's causal order."""
    if original.metadata_rank or compact.metadata_rank:
        raise ValueError("this concrete probe requires a metadata-free frozen state")
    if original.geometry != compact.geometry:
        raise ValueError("state geometry differs")
    count = np.asarray(capture["K_bits"]).shape[1]
    if not 0 <= start < start + tokens <= count:
        raise ValueError("invalid capture token slice")
    capture_positions = reference._positions(capture)
    positions = capture_positions[start:start + tokens]
    source = reference.words(capture)[:, start:start + tokens]
    layers, _, width = source.shape
    previous = np.empty((tokens, 0), dtype=np.int64)
    digest = hashlib.sha256()
    checked_values = 0; checked_blocks = 0; products = set()
    all_equal = True
    for layer in range(layers):
        features = np.concatenate((previous, np.zeros((tokens, width), dtype=np.int64)), axis=1)
        offset = previous.shape[1]
        for index, (left, right) in enumerate(zip(original.predictors[layer], compact.predictors[layer])):
            channel = index * original.block_size
            expected = reference._predict(features, left, channel, positions, original)
            actual = reference._predict(features, right, channel, positions, compact)
            product = features[:, right["columns"]] @ right["A"].T
            products.add(str(product.dtype))
            if product.dtype != np.int64:
                raise AssertionError("INT32 coefficient storage did not retain INT64 accumulation")
            matches = np.array_equal(expected, actual)
            all_equal = all_equal and matches
            if not matches:
                raise AssertionError(f"predictor differs at layer {layer}, channel {channel}")
            digest.update(np.ascontiguousarray(actual, dtype="<f8").tobytes())
            checked_values += actual.size; checked_blocks += 1
            selection = slice(channel, channel + original.block_size)
            recovered = reference._fixed(reference.from_bits(source[layer, :, selection]))
            if channel < original.geometry["channels"]:
                recovered = reference._rope_int(recovered, channel, positions, original, True)
            features[:, offset + channel:offset + channel + original.block_size] = recovered
        previous = features[:, offset:].copy()
    return dict(predictions_bit_identical=bool(all_equal), checked_values=int(checked_values),
                checked_blocks=checked_blocks, accumulation_dtypes=sorted(products),
                prediction_sha256=digest.hexdigest(), layers=layers, tokens=tokens,
                first_absolute_position=int(positions[0]), last_absolute_position=int(positions[-1]))


def verify_packet_compaction(original, compact, packet, expected=None):
    """Change only the state's identity header, then decode the original payload.

    The old packet intentionally requires its original state hash.  The probe
    explicitly reframes that identity; it does not pretend old packets are
    accepted unchanged. Entropy bytes and their checksum remain untouched.
    """
    if len(packet) < reference.PREFIX.size:
        raise ValueError("truncated source packet")
    magic, length = reference.PREFIX.unpack_from(packet)
    if magic != reference.MAGIC or len(packet) < reference.PREFIX.size + length:
        raise ValueError("invalid source packet")
    header = json.loads(packet[reference.PREFIX.size:reference.PREFIX.size + length])
    if header.get("state_sha256") != original.state_sha256 or header.get("kind") != "range":
        raise ValueError("source packet must be a range stream for the original state")
    payload = packet[reference.PREFIX.size + length:]
    replacement = dict(header, state_sha256=compact.state_sha256)
    encoded_header = reference._canonical(replacement)
    reframed = reference.PREFIX.pack(magic, len(encoded_header)) + encoded_header + payload
    actual, stats = reference.decode(compact, reframed)
    if expected is not None and not np.array_equal(actual, expected):
        raise AssertionError("compact-state packet decoding differs from the original decoded array")
    changed_fields = sorted(key for key in header if header[key] != replacement[key])
    if changed_fields != ["state_sha256"]:
        raise AssertionError("probe changed more than frozen-state identity")
    return reframed, dict(scope="Original entropy payload decoded with INT32 coefficients after explicit state-identity header replacement",
                          changed_header_fields=changed_fields, original_packet_sha256=hashlib.sha256(packet).hexdigest(),
                          reframed_packet_sha256=hashlib.sha256(reframed).hexdigest(),
                          original_state_sha256=original.state_sha256, compact_state_sha256=compact.state_sha256,
                          payload_bytes_unchanged=True, payload_sha256=hashlib.sha256(payload).hexdigest(),
                          decoded_sha256=header["decoded_sha256"], entropy_stream_bytes=header["stream_bytes"],
                          metadata_bytes=header["metadata_bytes"], values=int(actual.size), shape=list(actual.shape),
                          exact_decoder_checksum_verified=bool(stats["exact_checksum_verified"]),
                          equals_original_decoded_array=expected is not None,
                          compact_decode_seconds=stats["decode_seconds"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-state", type=Path)
    parser.add_argument("--packet", type=Path, help="Optional original real range-coded packet")
    parser.add_argument("--expected-decoded", type=Path, help="Optional original decoder .npy output")
    parser.add_argument("--reframed-packet", type=Path, help="Optional output; stores explicit state-identity reframing")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be fresh or empty")
    model = reference.load_model(args.state)
    original_sha = reference._state_digest(model)
    candidate = compact_coefficients(model)
    with np.load(args.capture, allow_pickle=False) as archive:
        capture = {key: archive[key].copy() for key in archive.files}
    checked = compare_predictions(model, candidate, capture, args.start, args.tokens)
    before, after = storage(model), storage(candidate)
    if reference._state_digest(model) != original_sha:
        raise AssertionError("source state was modified")
    candidate_file = None
    if args.candidate_state:
        saved = reference.save_model(candidate, args.candidate_state)
        candidate_file = dict(path=str(args.candidate_state), bytes=saved["serialized_bytes"], sha256=sha_file(args.candidate_state))
    packet_check = None
    if args.packet:
        expected = np.load(args.expected_decoded, allow_pickle=False) if args.expected_decoded else None
        reframed, packet_check = verify_packet_compaction(model, candidate, args.packet.read_bytes(), expected)
        packet_check["source_packet_path"] = str(args.packet)
        if args.expected_decoded:
            packet_check["expected_decoded_path"] = str(args.expected_decoded)
            packet_check["expected_decoded_file_sha256"] = sha_file(args.expected_decoded)
        if args.reframed_packet:
            args.reframed_packet.parent.mkdir(parents=True, exist_ok=True)
            args.reframed_packet.write_bytes(reframed)
            packet_check["reframed_packet_path"] = str(args.reframed_packet)
    report = dict(scope="CPU frozen-state storage probe; no new KV entropy stream or GPU performance result",
                  source_state=dict(path=str(args.state), bytes=args.state.stat().st_size, sha256=sha_file(args.state)),
                  capture=dict(path=str(args.capture), sha256=sha_file(args.capture), role="TRAIN; frozen predictors were not refitted"),
                  before=before, after=after, checks=checked,
                  coefficient_bytes_saved=before["coefficient_bytes"] - after["coefficient_bytes"],
                  frozen_array_bytes_saved=before["all_frozen_array_bytes"] - after["all_frozen_array_bytes"],
                  frozen_array_reduction_fraction=1 - after["all_frozen_array_bytes"] / before["all_frozen_array_bytes"],
                  candidate_state=candidate_file, source_state_unchanged=True, packet_check=packet_check,
                  arithmetic_INT16_INT32_candidate=dict(
                      scope="Storage arithmetic only: each frozen block uses INT16 when its entire coefficient range fits, otherwise INT32; no mixed-dtype packet validation",
                      coefficient_bytes=before["INT16_or_INT32_candidate_coefficient_bytes"],
                      all_frozen_array_bytes=before["INT16_or_INT32_candidate_all_frozen_array_bytes"],
                      block_range_counts=before["minimum_exact_signed_dtype_block_counts"],
                      packet_validated=False),
                  limitations=["Tested 16 token positions from one real TRAIN capture by default, across every layer/channel block.",
                               "No coefficient quantization, truncation, retraining, or change to CDF tables.",
                               "INT32 storage is exact only after checking all frozen coefficients fit the signed range.",
                               "Numpy INT64 features times INT32 coefficients produced INT64 accumulation in this CPU runtime.",
                               "Packet checks, when present, explicitly replace only state identity and preserve the original entropy payload; separate GPU validation remains necessary."])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    text = ("# Exact transfer predictor state storage probe\n\n"
            "Frozen INT64 predictor coefficients were copied to INT32 after checking their entire range. "
            "The original state was retained unchanged. This changes shared-state storage, not KV stream bitrate.\n\n"
            "| Quantity | Original | INT32 coefficients |\n| --- | ---: | ---: |\n"
            f"| Coefficient bytes | {before['coefficient_bytes']:,} | {after['coefficient_bytes']:,} |\n"
            f"| All frozen array bytes | {before['all_frozen_array_bytes']:,} | {after['all_frozen_array_bytes']:,} |\n\n"
            f"All **{checked['checked_values']:,} predictions** in **{checked['checked_blocks']} blocks** "
            f"across **{checked['layers']} layers** matched bit for bit on {checked['tokens']} real TRAIN tokens. "
            "INT64 feature inputs and INT32 coefficient storage produced INT64 matrix products.\n\n"
            f"Frozen arrays shrink by **{100 * report['frozen_array_reduction_fraction']:.2f}%**, "
            f"saving **{report['frozen_array_bytes_saved']:,} bytes**.\n\n"
            "No GPU performance, cross-platform reproducibility, or additional KV-stream savings were measured.\n")
    mixed_bytes = before["INT16_or_INT32_candidate_all_frozen_array_bytes"]
    if mixed_bytes is not None:
        text += ("\nA further **storage arithmetic candidate** selects INT16 per block when every coefficient fits, "
                 f"otherwise INT32: **{before['INT16_or_INT32_candidate_coefficient_bytes']:,} coefficient bytes** "
                 f"and **{mixed_bytes:,} total frozen-array bytes**. "
                 "This mixed-dtype candidate has not been validated through packet decoding.\n")
    if packet_check:
        text += ("\nAn original held-out TEST range-coded packet was also decoded using the compact state. "
                 "Only its `state_sha256` header field was replaced; the entropy payload and decoded-word checksum were preserved. "
                 f"All **{packet_check['values']:,} decoded BF16 words** matched the original decoder output. "
                 f"The **{packet_check['entropy_stream_bytes']:,}-byte entropy stream** was unchanged.\n")
    (args.output / "summary.md").write_text(text)
    print(json.dumps({"report": str(args.output), **checked, "frozen_array_bytes_saved": report["frozen_array_bytes_saved"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

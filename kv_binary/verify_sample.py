"""NumPy-only verification of the actual delivered binary reconstruction paths.

Checks encoded/raw/model/state file hashes, every stored binary branch, actual
float32/BF16 amplitudes and identities, then every recovered BF16 word. This does not encode,
train, select candidates, load a language model, or benchmark serving.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from kv_binary import binary_codec
from kv_binary.run_eval import factories
from kv_research import container, learned_codec
from kv_stacked import wrapper


@contextmanager
def project_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_record(record, name, config):
    """Unwrap exactly the packaging stages named by the frozen config."""
    seen = set()
    while config["candidates"][name]["kind"] == "record_deflate":
        if name in seen:
            raise ValueError("cyclic sample wrapper dependency")
        seen.add(name)
        record = wrapper.unwrap(record)
        name = config["candidates"][name]["base"]
    return record, name


def inspect_binary_records(blob, name, config, model, state):
    """Verify actual serialized branches and read stored amplitude bits."""
    shape = container.shape(blob)
    if shape["tile_tokens"] != 16:
        raise AssertionError("sample changed the independent16-token access unit")
    counts = Counter()
    for index in range(shape["nrecords"]):
        start, end = struct.unpack_from("<QQ", blob, container.HEADER.size + 8 * index)
        if not shape["payload_start"] <= start < end <= len(blob):
            raise AssertionError("sample has invalid record offsets")
        record, base = inner_record(blob[start:end], name, config)
        if config["candidates"][base]["kind"] != "binary":
            raise AssertionError("selected sample does not resolve to a binary candidate")
        info = binary_codec.record_info(record)  # validates full inner CRC
        if not info["is_binary"] or info["mode"] != "binary":
            raise AssertionError("sample skipped the actual binary predictor")
        if info["model_id"] != model.model_id or info["state_id"] != state.state_id:
            raise AssertionError("sample binary model/transform identity mismatch")
        layer = index // (2 * shape["ntiles"])
        kind = "V" if (index // shape["ntiles"]) % 2 else "K"
        tile = index % shape["ntiles"]
        rows = min(16, shape["tokens"] - tile * 16)
        if info["dictionary_key"] != f"layer_{layer:02d}_{kind}" or info["shape"] != [rows, shape["channels"]]:
            raise AssertionError("sample record does not match its indexed layer/tile")
        if info["dimension"] != state.dimension:
            raise AssertionError("sample binary dimension differs from transform state")
        precision = info.get("amplitude_precision", "f32")
        expected_precision = config["candidates"][base].get("options", {}).get("amplitude_precision", "f32")
        if precision not in ("f32", "bf16") or precision != expected_precision:
            raise AssertionError("sample stored amplitude precision differs from configured arm")
        itemsize = 2 if precision == "bf16" else 4
        if info.get("amplitude_itemsize", itemsize) != itemsize:
            raise AssertionError("sample amplitude byte width is inconsistent")
        values = rows * shape["channels"]
        padded = ((values + state.dimension - 1) // state.dimension) * state.dimension
        learned_vectors, binary_vectors = padded // 8, padded // state.dimension
        qbits = (model.n_centroids - 1).bit_length()
        indices_bytes = (learned_vectors * qbits + 7) // 8
        original_sign_bytes = (padded + 7) // 8
        binary_sign_bytes = (padded + 7) // 8
        amplitude_offset = (binary_codec._RECORD.size + len(info["dictionary_key"].encode("utf8"))
                            + indices_bytes + original_sign_bytes + binary_sign_bytes)
        if amplitude_offset + binary_vectors * itemsize > len(record) - 4:
            raise AssertionError("sample truncated its stored amplitudes")
        if precision == "bf16":
            words = np.frombuffer(record, dtype="<u2", count=binary_vectors, offset=amplitude_offset)
            amplitudes = (words.astype(np.uint32) << 16).view(np.float32)
        else:
            amplitudes = np.frombuffer(record, dtype="<f4", count=binary_vectors, offset=amplitude_offset)
        if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
            raise AssertionError("sample has invalid stored amplitudes")
        counts.update(binary_records=1, learned_vectors=learned_vectors, binary_vectors=binary_vectors,
                      binary_sign_bits=padded, original_sign_bits=padded, amplitudes=binary_vectors,
                      nonzero_amplitudes=int(np.count_nonzero(amplitudes)))
    if counts["binary_records"] != shape["nrecords"] or counts["nonzero_amplitudes"] <= 0:
        raise AssertionError("sample did not exercise nonzero binary predictions")
    return shape, dict(counts)


def verify(metadata_path=None):
    binary_root = Path(__file__).resolve().parent
    project = binary_root.parent
    metadata_path = Path(metadata_path).resolve() if metadata_path else binary_root / "results/sample_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    samples, config = metadata["samples"], metadata["config"]
    if not isinstance(samples, dict) or not samples:
        raise AssertionError("no actual samples listed")
    results = []
    with project_directory(project):
        _, decoders, state_charges, details = factories(config)
        for name, sample in samples.items():
            if name not in config["candidates"]:
                raise AssertionError("sample candidate absent from frozen configuration")
            blob_path = metadata_path.parent / sample["encoded_file"]
            raw_path = metadata_path.parent / sample["raw_file"]
            if digest(blob_path) != sample["encoded_sha256"] or digest(raw_path) != sample["raw_file_sha256"]:
                raise AssertionError("sample encoded/raw file hash mismatch")
            model_path = Path(sample.get("model_path") or details[name]["model_path"])
            state_path = Path(sample.get("state_path") or details[name]["state_path"])
            expected_model = sample.get("model_sha256") or details[name]["model_sha256"]
            expected_state = sample.get("state_sha256") or details[name]["transform_sha256"]
            if digest(model_path) != expected_model or digest(state_path) != expected_state:
                raise AssertionError("sample frozen dictionary/transform file hash mismatch")
            if model_path.stat().st_size + state_path.stat().st_size != sample["decoder_state_bytes"]:
                raise AssertionError("sample external decoder state byte charge mismatch")
            if state_charges[name] != sample["decoder_state_bytes"]:
                raise AssertionError("sample state charge differs from configured decoder")
            model, state = learned_codec.load_model(model_path), binary_codec.load_state(state_path)
            if state.model_id != model.model_id:
                raise AssertionError("transform belongs to another dictionary")
            blob = blob_path.read_bytes()
            shape, counts = inspect_binary_records(blob, name, config, model, state)
            # Wrapped candidates expose the base mechanism separately; compare
            # actual stored counts with whichever diagnostic map was recorded.
            expected_counts = (sample.get("inner_diagnostics") or sample.get("diagnostics") or {}).get("mechanism_counts", {})
            for key, value in counts.items():
                if key in expected_counts and expected_counts[key] != value:
                    raise AssertionError("sample mechanism count differs from serialized record")
            for key in ("raw_nonfinite_records", "raw_fallback_records"):
                if expected_counts.get(key, 0):
                    raise AssertionError("sample diagnostics contain a skipped binary branch")
            recovered = container.decode_cache(blob, decoders[name])
            recovered_sha, total_bytes = hashlib.sha256(), 0
            with np.load(raw_path) as raw:
                expected_keys = {f"layer_{layer:02d}_{kind}" for layer in range(shape["layers"]) for kind in ("K", "V")}
                if set(raw.files) != expected_keys:
                    raise AssertionError("sample raw cache has unexpected layer/KV keys")
                for layer, pair in enumerate(recovered):
                    for kind, words in zip(("K", "V"), pair):
                        expected = raw[f"layer_{layer:02d}_{kind}"]
                        if expected.dtype != np.uint16:
                            raise AssertionError("sample raw reference lost original BF16 word dtype")
                        np.testing.assert_array_equal(words, expected)
                        encoded_words = words.astype("<u2").tobytes()
                        recovered_sha.update(encoded_words); total_bytes += len(encoded_words)
            if recovered_sha.hexdigest() != sample["raw_sha256"]:
                raise AssertionError("reconstructed canonical KV hash mismatch")
            if "raw_payload_bytes" in sample and total_bytes != sample["raw_payload_bytes"]:
                raise AssertionError("sample raw payload byte count mismatch")
            results.append({"candidate": name, "records": shape["nrecords"], "bf16_words": total_bytes // 2,
                            "decoder_state_bytes": sample["decoder_state_bytes"], "mechanism_counts": counts})
            print(f"PASS {name}: {shape['nrecords']:,} actual binary records; {total_bytes // 2:,} exact BF16 words; frozen state {sample['decoder_state_bytes']:,}B")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", help="Defaults to kv_binary/results/sample_metadata.json")
    args = parser.parse_args()
    verify(args.metadata)


if __name__ == "__main__":
    main()

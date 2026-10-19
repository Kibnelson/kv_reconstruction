"""Gate the completed frozen binary experiment, report measured data, package.

This reporting helper is deliberately excluded from the experiment's source
freeze. It never encodes, trains, selects settings, or runs fresh comparisons.
Run only after kv_binary/results/final.json has all six cases:
    python -m kv_binary.finalize
"""
from __future__ import annotations

from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np

from kv_binary import binary_codec
from kv_research import container, learned_codec


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_files(project, selection):
    for name, expected in {**selection["source_and_model_sha256"],
                           **selection["development_evidence_sha256"]}.items():
        check(sha(project / name) == expected, "frozen source/model/evidence changed: " + name)


def base_name(name, config):
    seen = set()
    while config["candidates"][name]["kind"] == "record_deflate":
        check(name not in seen, "cyclic wrapper dependency")
        seen.add(name); name = config["candidates"][name]["base"]
    return name


def validate(project, root, out, selection, result, config, capture):
    """Check complete protocol/accounting evidence; return reporting totals."""
    frozen_files(project, selection)
    selection_sha = sha(root / "selection.json")
    check(not selection["fresh_final_manifest_existed"], "final capture preceded the freeze")
    check(capture["selection_sha256"] == selection_sha, "capture used another source/settings freeze")
    check(result["test_job_execution"]["freeze_sha256"] == selection_sha, "final jobs used another freeze")
    check(result["config"] == config and result["capture_manifest"] == capture, "final config/capture mismatch")
    check(result["config_sha256"] == sha(root / "final_config.json"), "final configuration file changed")
    check(config["tile_tokens"] == 16, "changed independent access granularity")
    names = list(config["candidates"])
    check(len(names) == 12 and len(result["cases"]) == len(capture["cases"]) == 6, "incomplete12-by6 comparison")
    check(len({case["name"] for case in result["cases"]}) == 6, "duplicate final cases")
    check(sorted(case["tokens"] for case in result["cases"]) == [512] * 3 + [2048] * 3, "unexpected final lengths")
    unit = (out / "unit_tests.txt").read_text()
    matches = re.findall(r"Ran\s+(\d+)\s+tests?", unit)
    check(bool(matches) and max(map(int, matches)) >= 11 and "\nOK" in unit and "FAILED" not in unit,
          "required unit-test evidence missing")
    model_path = project / config.get("model", "kv_research/models/learned64.bin")
    model = learned_codec.load_model(model_path)
    check(model_path.stat().st_size == 70095 and model.model_id == sha(model_path), "frozen learned dictionary identity/bytes changed")
    binary = [name for name in names if config["candidates"][name]["kind"] == "binary"]
    wrapped = [name for name in names if config["candidates"][name]["kind"] == "record_deflate"]
    check(len(binary) == 3 and len(wrapped) == 5, "expected three binary arms and five matched wrappers")
    check({config["candidates"][name]["transform"] for name in binary} == {"identity", "dense", "hadamard"},
          "missing transform ablation")
    expected_states, loaded_states = {}, {}
    def state_charge(name):
        if name in expected_states:
            return expected_states[name]
        spec = config["candidates"][name]
        if spec["kind"] == "record_deflate":
            value = state_charge(spec["base"])
        elif spec["kind"] == "binary":
            mp = project / spec.get("model", config.get("model", str(model_path)))
            sp = project / spec["state"]
            state = binary_codec.load_state(sp); loaded_states[name] = state
            check(state.model_id == model.model_id and state.state_id == sha(sp), "transform state identity mismatch")
            check(state.dimension == spec["dimension"] and state.transform == spec["transform"], "transform differs from selected arm")
            check(sp.stat().st_size == 52 + (8 * state.dimension ** 2 if state.transform == "dense" else 0),
                  "transform state cost omitted matrix/header bytes")
            detail = result["state_details"][name]
            check(detail["dictionary_bytes"] == mp.stat().st_size and detail["transform_bytes"] == sp.stat().st_size,
                  "binary external state byte count mismatch")
            check(detail["model_sha256"] == sha(mp) and detail["transform_sha256"] == sha(sp), "binary state hash metadata mismatch")
            check(detail["runtime_rotation_array_bytes"] == state.runtime_rotation_array_bytes == state.dimension ** 2 * 8,
                  "runtime rotation array cost mismatch")
            value = mp.stat().st_size + sp.stat().st_size
        elif spec["kind"] in ("learned", "learned_e8"):
            value = model_path.stat().st_size
        else:
            value = 0
        expected_states[name] = value
        return value
    for name in names:
        check(state_charge(name) == result["state_bytes"][name], "decoder state omitted or double-counted: " + name)
    parts = {name: Counter() for name in names}
    mechanisms = {name: Counter() for name in names}
    residuals = {name: Counter() for name in names}
    modes = {name: Counter() for name in names}
    total_records, total_raw = 0, 0
    for case, source in zip(result["cases"], capture["cases"]):
        check(all(case[key] == source[key] for key in ("name", "tokens", "raw_sha256", "raw_payload_bytes")), "case/capture identity mismatch")
        check(set(case["codecs"]) == set(names), "missing final candidate")
        records = 60 * ((case["tokens"] + 15) // 16)
        words = case["raw_payload_bytes"] // 2
        total_records += records; total_raw += case["raw_payload_bytes"]
        for name in names:
            stats, spec = case["codecs"][name], config["candidates"][name]
            check(stats["roundtrip_exact"] and stats["independent_later_tile"], "failed exactness/access check: " + name)
            check(stats["raw_payload_bytes"] == case["raw_payload_bytes"], "wrong raw denominator")
            check(sum(stats["component_bytes"].values()) == stats["serialized_bytes"], "unaccounted record/index bytes")
            check(stats["decoder_state_bytes"] == expected_states[name], "per-case state charge mismatch")
            calculated = 100 * (1 - (stats["serialized_bytes"] + expected_states[name]) / stats["raw_payload_bytes"])
            check(abs(calculated - stats["saving_with_state_percent"]) < 1e-9, "incorrect state-inclusive saving")
            diagnostic = stats.get("diagnostics", {})
            mechanism = diagnostic.get("mechanism_counts", {})
            parts[name].update(stats["component_bytes"]); mechanisms[name].update(mechanism)
            modes[name].update(diagnostic.get("record_modes", {})); residuals[name].update(diagnostic.get("residual_summary", {}))
            if spec["kind"] == "learned":
                check(mechanism.get("stored_learned_vectors") == words // 8 and mechanism.get("stored_entropy_records") == records,
                      "learned arm skipped its predictor/correction path")
            if spec["kind"] == "learned_e8":
                check(mechanism.get("learned_vectors") == words // 8 and mechanism.get("e8_vectors") == words // 8
                      and mechanism.get("e8_groups") == records, "E8 control skipped its real description")
                check(stats["component_bytes"].get("matrix_raw_words", 0) == 0, "E8 control silently used raw words")
            if spec["kind"] == "binary":
                check(spec["options"].get("force", True), "main binary arm permits hidden fallback")
                dimension = loaded_states[name].dimension
                check(mechanism.get("binary_records") == records and diagnostic.get("record_modes", {}).get("binary") == records,
                      "binary arm skipped actual records")
                check(not mechanism.get("raw_nonfinite_records", 0) and not mechanism.get("raw_fallback_records", 0), "binary arm stored raw fallback")
                check(mechanism.get("learned_vectors") == words // 8 and mechanism.get("binary_vectors") == words // dimension,
                      "binary predictor vector counts wrong")
                check(mechanism.get("binary_sign_bits") == mechanism.get("original_sign_bits") == words,
                      "binary/original signs were not all stored")
                check(mechanism.get("amplitudes") == words // dimension and 0 < mechanism.get("nonzero_amplitudes", 0) <= words // dimension,
                      "binary arm did not store nonzero amplitudes")
                check(diagnostic.get("residual_summary", {}).get("values") == words, "missing measured residual diagnostics")
            if spec["kind"] == "record_deflate":
                base = case["codecs"][spec["base"]]
                check(stats.get("wrapper_inner_byte_identity") and stats.get("wrapped_record_count") == records,
                      "wrapper did not audit every inner record")
                check(stats.get("inner_container_sha256") == base["encoded_sha256"], "wrapper changed/re-encoded its base")
                check(stats.get("inner_diagnostics") == base.get("diagnostics", {}), "wrapper's measured inner path differs from base")
    check(total_records == 28800, "incorrect total access records")
    for name in binary:
        check(mechanisms[name]["binary_records"] == 28800, "incomplete actual binary path")
    generation = selection["generation_candidates"]
    check(len(generation) == 2 and len(set(generation)) == 2, "expected two development-selected model checks")
    for name in generation:
        check(config["candidates"][base_name(name, config)]["kind"] == "binary", "model check did not use a binary candidate")
        gen = result["cases"][0]["codecs"][name]["generation"]
        check(gen["steps"] == 8 and gen["all_step_logits_bitwise_equal"] and gen["max_abs_logit_difference"] == 0
              and len(gen["output_token_ids"]) == 8, "failed8-step exact model continuation")
    # The frozen generation harness asserts equality of every newly appended
    # KV tensor at each step in addition to exact logits; tokens use argmax.
    harness = (project / "kv_research/run_eval.py").read_text()
    check("equal_matrices(refkv,testkv)" in harness, "generation harness no longer checks appended KV")
    for name in names:
        aggregate = result["aggregate"][name]
        raw = sum(case["codecs"][name]["raw_payload_bytes"] for case in result["cases"])
        encoded = sum(case["codecs"][name]["serialized_bytes"] for case in result["cases"])
        check(aggregate["all_exact"] and aggregate["raw_payload_bytes"] == raw == total_raw
              and aggregate["serialized_bytes"] == encoded, "aggregate byte mismatch")
        for n in (1, 4, 16):
            expected = 100 * (1 - (encoded + 6 * expected_states[name] / n) / raw)
            check(abs(expected - aggregate["savings_including_state_percent"][str(n)]) < 1e-9, "sharing scenario state amortized across unrelated cases")
    return dict(total_records=total_records, total_raw_bytes=total_raw, total_bf16_words=total_raw // 2,
                unit_tests=max(map(int, matches)), parts=parts, mechanisms=mechanisms, residuals=residuals,
                states=expected_states, loaded_states=loaded_states, binary=binary, wrapped=wrapped,
                names=names, generation=generation, selection_sha256=selection_sha)


def augment_samples(project, root, out, selection, result, config, audit):
    metadata = read(out / "sample_metadata.json")
    check(metadata["config"] == config and set(metadata["samples"]) == set(audit["generation"]), "missing/mismatched selected real samples")
    first = result["cases"][0]
    raw_path = out / "sample_raw.npz"
    with np.load(raw_path) as raw:
        words = b"".join(raw[f"layer_{layer:02d}_{kind}"].astype("<u2").tobytes()
                         for layer in range(30) for kind in ("K", "V"))
    check(len(words) == first["raw_payload_bytes"] and hashlib.sha256(words).hexdigest() == first["raw_sha256"], "sample raw cache differs from final capture")
    for name, sample in metadata["samples"].items():
        stats = first["codecs"][name]
        check(sha(out / sample["encoded_file"]) == sample["encoded_sha256"] == stats["encoded_sha256"], "sample encoded bytes changed")
        check(container.shape((out / sample["encoded_file"]).read_bytes())["nrecords"] == 1920, "unexpected sample access-record count")
        base = base_name(name, config); detail = result["state_details"][name]
        sample.update(raw_payload_bytes=len(words), raw_sha256=first["raw_sha256"], raw_file_sha256=sha(raw_path),
                      model_path=detail["model_path"], state_path=detail["state_path"],
                      model_sha256=sha(project / detail["model_path"]), state_sha256=sha(project / detail["state_path"]),
                      decoder_state_bytes=audit["states"][name], diagnostics=stats.get("diagnostics", {}),
                      inner_diagnostics=stats.get("inner_diagnostics", {}), resolved_binary_candidate=base,
                      resolved_binary_options=config["candidates"][base].get("options", {}))
    metadata["selection_sha256"] = audit["selection_sha256"]
    metadata["note"] = "Actual selected final binary paths. File/state identities and every BF16 word are checked by python -m kv_binary.verify_sample."
    (out / "sample_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for source, target in (
        (project / ".kv_binary_data/final_manifest.json", out / "fresh_capture_manifest.json"),
        (project / ".smoke_model/download_manifest.json", out / "checkpoint_manifest.json"),
        (project / ".kv_research_data/train_manifest.json", out / "training_capture_manifest.json"),
        (project / ".kv_research_data/dev_manifest.json", out / "development_capture_manifest.json")):
        shutil.copyfile(source, target)


def correction_bytes(parts):
    return sum(size for key, size in parts.items() if key.startswith("matrix_correction_"))


def write_review(project, root, out, selection, result, config, audit):
    names, aggregate = audit["names"], result["aggregate"]
    saving = lambda name: aggregate[name]["savings_including_state_percent"]["1"]
    best = max(names, key=saving)
    binary = audit["binary"]
    by_transform = {config["candidates"][name]["transform"]: name for name in binary}
    radius = next(name for name in names if config["candidates"][name]["kind"] == "learned"
                  and config["candidates"][name].get("options", {}).get("assignment", "radius") == "radius")
    l1 = next(name for name in names if config["candidates"][name]["kind"] == "learned"
              and config["candidates"][name].get("options", {}).get("assignment") == "l1")
    strong = next(name for name in names if config["candidates"][name]["kind"] == "huffman_lower_deflate")
    values = audit["total_bf16_words"]
    bits = lambda count: count * 8 / values
    lines = ["# Binary residual reconstruction: measured exact KV records", "",
        f"All72 final cache comparisons recovered every original BF16 word. The best complete result was **{best}**, saving **{saving(best):.3f}%** including its full decoder state. The development-selected identity and rotated paths saved **{saving(selection['best_identity']):.3f}%** and **{saving(selection['best_rotated']):.3f}%**. These are measured CPU storage results; exactness does not establish a cheap GPU reconstruction path.", "",
        "## Representation and borrowed idea", "",
        "Each input is an original BF16 word: retain its sign and treat exponent plus fraction as a reversible15-bit magnitude integer m. The frozen learned64 dictionary supplies an8D prediction c. For groups of D residual coordinates, form r=m−c, binary signs b=sign(Rᵀr), direction u=Rb/√D and a stored projection amplitude a. Reproduce p=clip(roundEven(c+float64(a)u),0,32767), then encode e=m−p exactly. The amplitude is float32 or BF16 as identified by the record. Its actual stored round trip defines p before corrections are computed. Dense transforms are explicit serialized float64 matrices; arithmetic has a canonical multiply/add order rather than BLAS-dependent reductions.", "",
        "The borrowed idea is a mathematical rule describing an implicit rotated-hypercube codebook. This implementation is inspired by RaBitQ, not a reproduction of its ANN distance estimator. Projection amplitude differs from its inverse-alignment estimator. ANN error theorems do not imply exact KV, unchanged attention, or benefits here; exact signed corrections establish bit recovery. Hadamard, identity and dense rotation are distinct empirical controls.", "",
        "## Frozen fresh comparison", "",
        "Source, frozen dictionary, transform states, development evidence, and candidate settings were frozen before six new captures: clock-repair prose, Kotlin bracket code, and tidal-station records at512 and2,048 tokens. These repeated synthetic length variants are correlated mechanism coverage on native-BF16 SmolLM2-135M, not representative workloads or larger-model generalization. Every arm retains independently indexed16-token layer/K-or-V access records.", "",
        "| Candidate | Complete saving at N=1 | External decoder state |", "|---|---:|---:|"]
    for name in names:
        lines.append(f"| {name} | {saving(name):.3f}% | {audit['states'][name]:,}B |")
    lines += ["", "Percentages are weighted by original KV bytes. N=1 charges the entire required model and transform once per cache. N=4/16 JSON figures share one decoder state arithmetically across that many equal-shaped caches; they are not concurrent-serving measurements. Wrapper variants compress byte-identical complete inner records and count every wrapper header. Their gain combines packaging redundancy and does not isolate the binary predictor.", "",
        f"The strong exponent-Huffman/lower-byte-Deflate baseline saved **{saving(strong):.3f}%** without learned state. Matched learned-radius correction coding saved **{saving(radius):.3f}%**, and the separate learned-L1 control saved **{saving(l1):.3f}%**. Binary arms retain radius assignment so their secondary predictor is compared against the same prototype selection. A strong L1 control must also be considered when judging the complete design.", "",
        "## Where the bytes went", "",
        "The following table uses unwrapped arms and actual serialized bytes per original BF16 value. Correction figures include the full local entropy stream, its parameters and checksums. Other bytes include record/cache framing and temporal offsets. External state is charged separately in the complete table above.", "",
        "| Arm | Prototype IDs | Original signs | Binary signs | Amplitudes | Exact correction stream | Other/index |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in [radius] + binary:
        p = audit["parts"][name]
        ids = p.get("matrix_prototype_indices", 0) + p.get("matrix_learned_indices", 0)
        original = p.get("matrix_exact_signs", 0) + p.get("matrix_exact_original_signs", 0)
        secondary = p.get("matrix_binary_direction_signs", 0)
        amplitude = p.get("matrix_float32_amplitudes", 0) + p.get("matrix_bf16_amplitudes", 0)
        correction = correction_bytes(p)
        other = sum(p.values()) - ids - original - secondary - amplitude - correction
        lines.append(f"| {name} | {bits(ids):.4f} | {bits(original):.4f} | {bits(secondary):.4f} | {bits(amplitude):.4f} | {bits(correction):.4f} | {bits(other):.4f} |")
    lines += ["", "The unit in every numeric column above is bits per original value. A smaller remaining error must save enough correction bytes to pay for binary signs, amplitude and any extra identities/framing.", ""]
    baseline_correction = correction_bytes(audit["parts"][radius])
    for name in binary:
        change = baseline_correction - correction_bytes(audit["parts"][name])
        lines.append(f"- **{name}:** exact correction streams were {'smaller' if change >= 0 else 'larger'} by **{abs(change):,}B** than matched learned-radius; the complete state-inclusive saving changed by **{saving(name)-saving(radius):+.3f} percentage points**.")
    lines += ["", "## Prediction geometry diagnostics", "",
        "These diagnostics operate on BF16 integer symbols, not float KV error, perplexity, or attention quality. They are aggregated from the actual stored-amplitude predictions. Improved geometric error can coexist with a worse total record.", "",
        "| Binary arm | D / amplitude | Mean absolute error, before → after | Mean squared error, before → after | MAE reduction | MSE reduction |", "|---|---|---:|---:|---:|---:|"]
    for name in binary:
        r = audit["residuals"][name]; count = r["values"]
        mae0, mae1 = r["before_abs_sum"] / count, r["after_abs_sum"] / count
        mse0, mse1 = r["before_squared_sum"] / count, r["after_squared_sum"] / count
        spec = config["candidates"][name]
        precision = spec.get("options", {}).get("amplitude_precision", "f32")
        lines.append(f"| {name} | {spec['dimension']} / {precision} | {mae0:.3f} → {mae1:.3f} | {mse0:.3f} → {mse1:.3f} | {100*(1-mae1/mae0):.3f}% | {100*(1-mse1/mse0):.3f}% |")
    identity, dense = by_transform["identity"], by_transform["dense"]
    lines += ["", f"Dense versus identity changes complete N=1 saving by **{saving(dense)-saving(identity):+.3f} percentage points**. These are the development-selected configurations: compare their D and amplitude precision above before attributing a difference exclusively to rotation. Dense matrix storage is included in this difference.", "",
        "## Decoder state and actual mechanisms", "",
        "| Transform arm | Serialized dictionary | Serialized transform | Runtime rotation array |", "|---|---:|---:|---:|"]
    for name in binary:
        detail = result["state_details"][name]
        lines.append(f"| {name} | {detail['dictionary_bytes']:,}B | {detail['transform_bytes']:,}B | {detail['runtime_rotation_array_bytes']:,}B |")
    lines += ["", "Identity/Hadamard use procedural52-byte transform storage but this reference decoder still materializes a float64 D×D rotation array. Dense files contain that entire explicit matrix plus header/checksum. Runtime arrays, Python objects, model activations and reconstruction scratch are not a peak-memory measurement.", "",
        "Each of the three unwrapped binary arms stored its actual binary description in **28,800 records**. All original signs, binary signs, prototype IDs and amplitudes were present; no final binary record used nonfinite/raw-size fallback. Nonzero amplitudes were observed. All five wrappers audited every byte-identical inner record and its corresponding base-container hash. Two development-selected binary paths completed eight consecutive model steps each with exact logits and derived tokens; the frozen harness also asserted every newly appended KV tensor. The two included real samples independently exercise those actual branches.", "",
        "## Development precision refinement", ""]
    precision_path = root / "development/precision/grid_results.json"
    if precision_path.exists():
        grid = read(precision_path)
        lines += ["The float32-versus-BF16 refinement used only the sensor development prompt at D=64. The table below contains **subset-based full-cache estimates**, not fresh final results. Each estimate extrapolates18 actual records from layers0/15/29 and temporal positions0/32/63, adds the complete index and external state, and includes stored precision metadata. The frozen full-development shortlist is separate from these estimates.", "",
                  "| Transform | f32 estimate | BF16 estimate | f32 + record Deflate | BF16 + record Deflate |", "|---|---:|---:|---:|---:|"]
        for transform in ("identity", "dense", "hadamard"):
            keys = [f"binary_{transform}_d64_{p}{suffix}" for suffix in ("", "_deflate") for p in ("f32", "bf16")]
            values_grid = [grid["results"][key]["estimated_full_saving_with_state_percent"] for key in keys]
            lines.append(f"| {transform} | " + " | ".join(f"{v:.3f}%" for v in values_grid) + " |")
    lines += ["", "The initial dimensions8/32/64 grid, the separate D=64 precision grid, recorded original f32 source snapshot, full-development evidence and frozen choices are included. No dictionary was retrained. The BF16 refinement rounds the amplitude, reconstructs from that stored value, and recalculates exact corrections; reducing metadata alone is not assumed to improve total bytes.", "",
        "## Validation and limits", "",
        f"- All72 final cache roundtrips and independent later-tile checks passed; {audit['unit_tests']} unit tests passed. Every component sum, full dictionary/transform charge, source hash, evidence hash, capture/config identity and selected model continuation was gated before packaging.",
        "- The transform has fixed-order float64 multiply/add semantics and stored amplitude bits. Floating-point rotation followed by its inverse is never assumed to recover BF16 words by itself; signed integer corrections close the exact gap.",
        "- This is not a RaBitQ reproduction, a published exact KV codec, or evidence that ANN guarantees transfer to attention. It is one measured implicit-codebook adaptation.",
        "- CPU model checks restore the full prefix. No cheap GPU kernel, bounded GPU reconstruction buffer, SSD/PCIe improvement, TTFT/TPOT reduction, admission benefit or high-concurrency serving benefit was established.",
        "- Independent CPU jobs overlap and share resources. Encoding includes reference audits for some controls, and canonical dense transforms are unoptimized. Timing fields are diagnostics, not a clean serial production benchmark.",
        "- Future improvement requires a better complete-byte tradeoff or a cheaper execution path, selected on development data and checked on another fresh split. Final data must not be reused to tune these settings.", "",
        "Run `python -m kv_binary.verify_sample` from the extracted project root for NumPy-only verification of the two real selected paths. See `../README.md`, `selection.json`, `final_config.json` and `results/final.json` for reproduction and full evidence.", ""]
    report = "\n".join(lines)
    for old, new in {"All72": "All 72", "reversible15-bit": "reversible 15-bit", "an8D": "an 8D",
                     "at512": "at 512", "and2,048": "and 2,048", "indexed16-token": "indexed 16-token",
                     "procedural52-byte": "procedural 52-byte", "dimensions8/32/64": "dimensions 8/32/64",
                     "extrapolates18": "extrapolates 18", "layers0/15/29": "layers 0/15/29", "positions0/32/63": "positions 0/32/63"}.items():
        report = report.replace(old, new)
    (out / "review.md").write_text(report)
    return dict(best_overall=best, best_overall_saving_percent=saving(best),
                best_identity=selection["best_identity"], identity_saving_percent=saving(selection["best_identity"]),
                best_rotated=selection["best_rotated"], rotated_saving_percent=saving(selection["best_rotated"]),
                strong_baseline_saving_percent=saving(strong), learned_radius_saving_percent=saving(radius),
                learned_l1_saving_percent=saving(l1))


def package(project, root, selection):
    """Bundle the new namespace and frozen required historical dependencies."""
    files = {path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts
             and path.suffix != ".pyc" and path.name != "package_manifest.json"}
    for name in selection["source_and_model_sha256"]:
        path = project / name
        if path.suffix in (".py", ".cpp", ".bin") or name in ("kv_smoke/requirements.txt", ".smoke_model/download_manifest.json"):
            files.add(path)
    # Imports resolve through namespace packages too; retain explicit package
    # markers where present, even when an old source-freeze omitted them.
    for namespace in ("kv_smoke", "kv_research", "kv_stacked", "kv_coset_fields", "kv_residual", "kv_binary"):
        marker = project / namespace / "__init__.py"
        if marker.exists():
            files.add(marker)
    manifest = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="New binary namespace plus frozen source/model dependencies; includes two selected encoded samples and one original raw sample. No installed dependencies, language-model weights, or six full raw final captures.",
        files=[dict(path=str(path.relative_to(project)), bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(files)])
    mpath = root / "package_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n"); files.add(mpath)
    archive = project / "kv_binary_residual_tests.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(files):
            zipped.write(path, str(path.relative_to(project)))
    with zipfile.ZipFile(archive) as zipped:
        check(zipped.testzip() is None, "archive CRC/integrity failure")
        check(len(zipped.namelist()) == len(files), "duplicate/missing archive paths")
    check(archive.stat().st_size < 50 * 1024 ** 2, "archive exceeds50MiB")
    frozen_files(project, selection)
    return archive, manifest


def main():
    root = Path(__file__).resolve().parent; project = root.parent; out = root / "results"
    selection = read(root / "selection.json")
    config = read(root / "final_config.json")
    result = read(out / "final.json")
    capture = read(project / ".kv_binary_data/final_manifest.json")
    audit = validate(project, root, out, selection, result, config, capture)
    augment_samples(project, root, out, selection, result, config, audit)
    summary = write_review(project, root, out, selection, result, config, audit)
    validation = dict(validated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        all72_cache_roundtrips_exact=True, all_later_tile_checks_passed=True,
        frozen_source_model_evidence_unchanged=True, complete_state_and_component_accounting=True,
        access_records_per_binary_arm=audit["total_records"], actual_binary_arms=len(audit["binary"]),
        wrapped_candidates_byte_identity_checked=len(audit["wrapped"]), unit_tests=audit["unit_tests"],
        model_check_candidates=audit["generation"], model_steps_per_candidate=8,
        all_selected_step_logits_tokens_and_appended_kv_exact=True,
        selection_sha256=audit["selection_sha256"], final_config_sha256=sha(root / "final_config.json"),
        source_scope="Gates inspect completed recorded tests and frozen harness; no training, selection or new experiments.", **summary)
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    archive, manifest = package(project, root, selection)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, files=len(manifest["files"]) + 1, **summary), indent=2))


if __name__ == "__main__":
    main()

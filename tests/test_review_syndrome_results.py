"""Stdlib evidence-audit checks: state, escapes, selection and gzip integrity."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coset_review", ROOT / "scripts" / "review_syndrome_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def accounting(raw, packet, neural=0, anchor=0):
    return dict(sample=dict(raw_bytes=raw, packet_bytes=packet,
        payload_reduction_pct=100 * (1 - packet / raw)),
        state=dict(neural=dict(serialized_bytes=neural, resident_bytes=0),
            anchor=dict(serialized_bytes=anchor, resident_bytes=0), total_charged_bytes=neural + anchor),
        full_cache_estimate=dict(raw_bytes=raw, estimated_packet_bytes=packet,
            charged_state_bytes=neural + anchor, net_reduction_pct=100 * (1 - (packet + neural + anchor) / raw)))


def fixture(root):
    """A small internally consistent recorded report, not simulated KV evidence."""
    calibration = root / "calibration"; calibration.mkdir()
    source = dict(files={}); source["sources_sha256"] = audit.sha(audit.canonical(source["files"]))
    target = dict(model="synthetic-audit-fixture")
    captures = [dict(split="train", producer_records_sha256=str(i) * 64,
                     producer_inputs_sha256=chr(97 + i) * 64) for i in range(3)]
    models = {}
    for mode in ("none", "hidden"):
        metadata = dict(training_only=True, model_identity=target, training=dict(total_optimizer_updates=1))
        payload = b"fitted codec parameters"
        shard = dict(file=mode + ".bin", bytes=len(payload), sha256=audit.sha(payload))
        descriptor = dict(format="syndrome_model_artifact_v1", metadata=metadata, shard=shard,
            manifest_bytes=0, total_serialized_bytes=0,
            metadata_and_shard_crc32=zlib.crc32(audit.canonical(dict(metadata=metadata, shard=shard))))
        for _ in range(8):
            raw = audit.canonical(descriptor)
            if descriptor["manifest_bytes"] == len(raw) and descriptor["total_serialized_bytes"] == len(raw) + len(payload):
                break
            descriptor.update(manifest_bytes=len(raw), total_serialized_bytes=len(raw) + len(payload))
        files = []
        for suffix, data in (("json", raw), ("bin", payload)):
            path = calibration / (mode + "." + suffix); path.write_bytes(data)
            files.append(dict(file="calibration/" + path.name, bytes=len(data), sha256=audit.sha(data)))
        models[mode] = dict(model_id=files[0]["sha256"], serialized_state_bytes=sum(row["bytes"] for row in files),
            metadata=metadata, files=files)
    field = dict(file="calibration/centers.bin", serialized_bytes=8, resident_bytes=4, sha256=audit.sha(b"12345678"))
    (root / field["file"]).write_bytes(b"12345678")
    anchor = dict(file="calibration/anchor.bin", serialized_bytes=8, resident_bytes=4, sha256=audit.sha(b"87654321"))
    (root / anchor["file"]).write_bytes(b"87654321")
    selected = dict(none=6, hidden=6)
    frozen = dict(model_identity=target, source_sha256=source["sources_sha256"], models=models,
        field_state=field, anchor_state=anchor, training_hashes=[row["producer_inputs_sha256"] for row in captures],
        training_producer_input_hashes=[row["producer_inputs_sha256"] for row in captures],
        training_capture_hashes=[row["producer_records_sha256"] for row in captures], residue_selection=selected,
        selection_provenance=dict(split="dev", producer_records_sha256="d" * 64, tokens=31))
    report = dict(format=audit.FORMAT, status="completed", config=dict(modes=["none", "hidden"],
        dev_tokens=31, generation_steps=1, steps=1, repeats=1, rank=2, residue_bits=[6]),
        source_manifest=source, freeze=frozen, model_identity=target, models=models,
        training_captures=captures, field_state=field, anchor_state=anchor,
        selection=selected, development=dict(producer_records_sha256="d" * 64),
        development_grid={}, generation_checks=[], cases=[])

    def measure(mode, field_control=False):
        features = 0 if mode == "none" else 124
        parts = dict(feature_bytes=features, exponent_bytes=20, sign_bytes=16,
            residue_bytes=109 if field_control else 93, plane_bytes=0 if field_control else 16, framing_bytes=40)
        packet = sum(parts.values())
        hist = [2, 1, 1] + [0] * 29
        plane = dict(bit=6, values=124, bit_errors=3, groups=4, errors_per_group=hist,
            zero_error_groups=2, one_error_groups=1, multi_error_groups=1, recoverable_groups=3,
            tail_bits=0, mode_bytes=0, short_bytes=0, raw_group_bytes=16, tail_bytes=0,
            retained_payload_bytes=16, candidate_payload_saving_bytes=9, raw_plane_bytes=16,
            candidate_payload_bytes=7, retained_short_groups=0, retained_raw_groups=4, retained_mode="raw")
        trace = dict(layer=0, start=0, tokens=31, raw_bytes=248, packet_bytes=packet,
            component_bytes=parts.copy(), bitplanes=[] if field_control else [plane],
            feature_words_supplied=not field_control, plane_bytes=parts["plane_bytes"],
            retained_short_groups=0, candidate_groups=0 if field_control else 4,
            planes_used_syndrome=0, planes_used_positions=0)
        decoded = dict(maximum_dependency_depth=0, raw_prerequisite_pages=0, source_arrays_used=False,
            records=[dict(layer=0, start=0)])
        parts["container_framing_bytes"] = 8
        packet += 8
        return dict(raw_sample_bytes=248, packet_bytes=packet, component_bytes=parts, records=[trace],
            records_measured=1, independent_packet_decode=decoded, roundtrip_exact=True,
            container_roundtrip_verified=True, all_repeats_exact=True, maximum_dependency_depth=0,
            raw_prerequisite_pages=0, repeats=1, wall_seconds_samples=[.1], median_wall_seconds=.1,
            inline_codes_used=not field_control, residue_bits=7 if field_control else 6,
            packet_artifact=dict(file="packets/omitted.bin", bytes=packet, sha256="e" * 64))

    inline = dict(observed_inside_attention_forward=True, methods={
        mode: dict(calls=1, feature_bytes=0 if mode == "none" else 124) for mode in models})
    case = dict(name="synthetic", tokens=31, selected_starts=[0], matched_record_addresses=[[0, 0]],
        capture=dict(geometry=dict(layers=1), raw_payload_bytes=248, inline_side_codes=inline,
            producer_audit=dict(all_stage_targets_exact=True, hooks_removed=True)),
        results=[], syndrome_positions={})
    for name, state in (("frozen_exponent_control", anchor), ("field_baseline", field)):
        measured = dict(raw_sample_bytes=248, packet_bytes=180) if name.startswith("frozen") else measure("none", True)
        charge = state["serialized_bytes"]
        case["results"].append(dict(name=name, status="ok", measurement=measured,
            accounting=accounting(248, measured["packet_bytes"], anchor=charge) if name.startswith("frozen")
                       else accounting(248, measured["packet_bytes"], neural=charge)))
    for mode in models:
        measured = measure(mode)
        report["development_grid"][mode] = [dict(residue_bits=6, packet_bytes=measured["packet_bytes"],
            roundtrip_exact=True, measurement=copy.deepcopy(measured))]
        report["generation_checks"].append(dict(name="coset_" + mode + "_syndrome", status="ok",
            complete_prefix_tokens=31, result=dict(all_tokens_equal=True, all_logits_bitwise_equal=True,
                all_appended_KV_bitwise_equal=True, per_step=[dict(logits_finite=True)]),
            packet_artifact=copy.deepcopy(measured["packet_artifact"])))
        for correction in ("syndrome", "positions"):
            case["results"].append(dict(name="coset_" + mode + "_" + correction, status="ok",
                measurement=copy.deepcopy(measured), accounting=accounting(248, measured["packet_bytes"],
                    neural=models[mode]["serialized_state_bytes"])))
        case["syndrome_positions"][mode] = dict(syndrome=measured["packet_bytes"], positions=measured["packet_bytes"], packet_byte_difference=0)
    report["cases"] = [case]
    for name, value in (("results.json", report), ("source_manifest.json", source), ("frozen_manifest.json", frozen)):
        (root / name).write_bytes(audit.canonical(value))
    return report


class ReviewCosetResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = fixture(self.root)

    def save(self):
        (self.root / "results.json").write_bytes(audit.canonical(self.report))

    def test_valid_complete_report_and_actual_fitted_state_hashes(self):
        analysis, validation = audit.review(self.root)
        self.assertTrue(validation["all_passed"], [x for x in validation["checks"] if not x["passed"]])
        self.assertIn("calibration/hidden.bin", analysis["verified_artifact_files"])
        self.assertEqual(analysis["omitted_packet_files"], ["packets/omitted.bin"])
        (self.root / "calibration" / "hidden.bin").write_bytes(b"different")
        _, validation = audit.review(self.root)
        self.assertFalse(validation["all_passed"])
        self.assertTrue(any(not x["passed"] and "artifact bytes/hash" in x["check"] for x in validation["checks"]))

    def test_corrupt_ledger_histogram_and_hidden_dependency_are_detected(self):
        row = self.report["cases"][0]["results"][2]
        row["measurement"]["component_bytes"]["feature_bytes"] += 1
        row["measurement"]["records"][0]["bitplanes"][0]["errors_per_group"][0] += 1
        row["measurement"]["independent_packet_decode"]["source_arrays_used"] = True
        self.save()
        _, validation = audit.review(self.root)
        failed = [x["check"] for x in validation["checks"] if not x["passed"]]
        self.assertTrue(any("component ledger" in x for x in failed))
        self.assertTrue(any("histogram" in x for x in failed))
        self.assertTrue(any("independent restore" in x for x in failed))

    def test_frozen_selection_and_mismatched_positions_are_detected(self):
        self.report["selection"]["hidden"] = 4
        self.report["cases"][0]["syndrome_positions"]["hidden"]["positions"] += 1
        self.save()
        _, validation = audit.review(self.root)
        failed = [x["check"] for x in validation["checks"] if not x["passed"]]
        self.assertTrue(any("DEV minimum" in x for x in failed))
        self.assertTrue(any("syndrome/position" in x for x in failed))

    def test_delivered_manifest_metadata_is_crosschecked_against_report(self):
        self.report["models"]["hidden"]["metadata"]["training"]["total_optimizer_updates"] = 9
        self.report["config"]["steps"] = 9
        self.report["models"]["none"]["metadata"]["training"]["total_optimizer_updates"] = 9
        # Even a report and frozen manifest changed together cannot rewrite
        # what was actually stored inside the delivered fitted model file.
        self.save()
        (self.root / "frozen_manifest.json").write_bytes(audit.canonical(self.report["freeze"]))
        _, validation = audit.review(self.root)
        self.assertTrue(any(not row["passed"] and "actual model metadata" in row["check"] for row in validation["checks"]))

    def test_common_residue_rows_do_not_change_frozen_selection(self):
        case = self.report["cases"][0]
        row = copy.deepcopy(next(row for row in case["results"] if row["name"] == "coset_hidden_syndrome"))
        row["name"] = "coset_hidden_common_syndrome"
        case["results"].append(row)
        case["common_residue_bits"] = 6
        case["common_residue_methods"] = dict(none="coset_none_syndrome", hidden=row["name"])
        self.save()
        _, validation = audit.review(self.root)
        self.assertTrue(validation["all_passed"], [x for x in validation["checks"] if not x["passed"]])

    def test_gzip_package_identity_and_complete_stream_integrity(self):
        raw = (self.root / "results.json").read_bytes()
        (self.root / "results.json").unlink()
        compressed = gzip.compress(raw, mtime=0); (self.root / "results.json.gz").write_bytes(compressed)
        manifest = dict(files=[dict(file="results.json.gz", bytes=len(compressed), sha256=audit.sha(compressed))],
            uncompressed_results=dict(bytes=len(raw), sha256=audit.sha(raw)))
        (self.root / "package_manifest.json").write_bytes(audit.canonical(manifest))
        _, validation = audit.review(self.root)
        self.assertTrue(validation["all_passed"])
        (self.root / "results.json.gz").write_bytes(compressed[:-1])
        with self.assertRaises((EOFError, OSError)):
            audit.review(self.root)

    def test_unsafe_artifact_path_rejected(self):
        self.report["models"]["hidden"]["files"][0]["file"] = "../outside.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unsafe artifact"):
            audit.review(self.root)


if __name__ == "__main__":
    unittest.main()

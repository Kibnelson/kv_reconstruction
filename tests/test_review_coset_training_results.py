"""Cross-arm evidence checks, independently of large packet-audit fixtures."""
import copy
import gzip
import importlib.util
import struct
import zlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("training_audit", ROOT / "scripts/review_coset_training_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def metric(probability=True):
    value = dict(values=62, bit_errors=1, original_ones=1, predicted_ones=0,
                 groups=2, tail_bits=0, errors_per_group=[1, 1] + [0]*30, recoverable_groups=2)
    if probability:
        value.update(BCE_sum_bits=31., BCE_bits=.5, Brier_sum=6.2, Brier_mean=.1,
            calibration=dict(counts=[0, 62] + [0]*8, probability_sums=[0., 6.2] + [0.]*8,
                             truth_sums=[0., 1.] + [0.]*8))
    return value


def shard(value):
    payload = struct.pack("<f", float(value))
    metadata = audit.canonical(dict(format="neural_parameter_shard_v1", group="syndrome_parameters",
        array_manifest=[dict(name="prior", shape=[1], dtype="<f4", offset=0, nbytes=4)]))
    body = metadata + payload
    return struct.pack("<4sBIII", b"NPKV", 1, len(metadata), len(payload), zlib.crc32(body)) + body


def fixture(directory):
    config = dict(short_steps=2, long_steps=4)
    source = dict(files={})
    source["sources_sha256"] = audit.sha(audit.canonical(source["files"]))
    geometry = dict(layers=1, channels=1)
    probe = dict(split="dev", sha256="probe", records=[dict(layer=0, start=0, tokens=31, channels=2)])
    captures = [dict(producer_inputs_sha256="train")]
    arms = []
    for sampling in ("random", "balanced"):
        for budget, steps in (("short", 2), ("long", 4)):
            name = sampling + "_" + budget
            root = directory / name
            root.mkdir()
            short_wire = shard(1 if sampling == "random" else 2)
            wire = short_wire if budget == "short" else shard(3 if sampling == "random" else 4)
            common = audit._parameter_hash(short_wire)
            final = audit._parameter_hash(wire)
            calibration = root / "calibration"; calibration.mkdir()
            (calibration / "none.bin").write_bytes(wire)
            updates = [0, 2] if budget == "short" else [0, 2, 4]
            points = []
            for update in updates:
                learned = dict(omitted=metric(), by_bitplane={"6":metric()})
                metrics = dict(format="prefill_coset_fixed_dev_diagnostics_v1", probe=probe,
                    prior_logits_sha256="frozen-prior", by_residue={"6":dict(known_fields_sha256="known-fields",
                        learned=learned, controls=dict(constant_zero=dict(omitted=metric(False)),
                            train_prior=dict(omitted=metric())), ablations={})})
                points.append(dict(update=update, parameter_sha256="initial" if update == 0 else common if update == 2 else final,
                    sampling_prefix_sha256="empty" if update == 0 else common + "-schedule" if update == 2 else final + "-schedule", metrics=metrics))
            coverage = dict(optimizer_updates_by_layer_channel=[[steps, steps]], word_examples_by_layer_channel=[[steps*31, steps*31]])
            diag = dict(format="qwen_prefill_coset_training_diagnostics_v1", probe_manifest=probe,
                capture=dict(producer_inputs_sha256="dev",input_ids_sha256="dev-input",kv_words_sha256="dev-kv"), evaluation=dict(updates=updates),
                modes=dict(none=dict(checkpoints=points, coverage=coverage)))
            raw = audit.canonical(diag)
            (root / "training_diagnostics.json").write_bytes(raw)
            training = dict(channel_sampling=sampling, learning_rate_horizon=4,
                coverage_sha256=audit.sha(audit.canonical(coverage)), total_word_examples=steps*62,
                channels_per_step=2, initial_parameter_sha256="initial", final_parameter_sha256=final,
                sampling_sha256=final + "-schedule", checkpoints=[{k:point[k] for k in ("update", "parameter_sha256", "sampling_prefix_sha256")} for point in points if point["update"]])
            child_cfg = dict(rank=8, width=32, batch_tokens=31, channels_per_step=2, seed=71,
                train_tokens=64, dev_tokens=64, lengths=[2048], cases=1, tile_tokens=64,
                chunk_tokens=512, max_pairs=72, modes=["none"], residue_bits=[6], steps=steps)
            child = dict(config=child_cfg, source_manifest=source, model_identity="qwen-fixture",
                models=dict(none=dict(metadata=dict(geometry=geometry, training=training),
                    files=[dict(file="calibration/none.bin",bytes=len(wire),sha256=audit.sha(wire))])),
                training_captures=captures, development=dict(producer_inputs_sha256="dev"),
                cases=[dict(name="fresh-test", tokens=2048, selected_starts=[0], matched_record_addresses=[[0,0]],
                    capture=dict(input_ids_sha256="test-input", kv_words_sha256="test-kv", producer_inputs_sha256="test-producer"))],
                training_diagnostics=dict(file="training_diagnostics.json", bytes=len(raw), sha256=audit.sha(raw)))
            (root / "results.json").write_bytes(audit.canonical(child))
            arms.append(dict(name=name, sampling=sampling, steps=steps, directory=name, status="completed",
                report_sha256=audit.sha(audit.canonical(child)), diagnostics_sha256=audit.sha(raw)))
    report = dict(format=audit.FORMAT, status="completed", config=config, arms=arms)
    plan = dict(format="prefill_coset_training_plan_v1", config=config, lr_horizon=4,
                arms=[{k:a[k] for k in ("name", "sampling", "steps")} for a in arms])
    for name, value in (("results.json",report), ("source_manifest.json",source), ("frozen_manifest.json",plan)):
        (directory/name).write_bytes(audit.canonical(value))
    return report


def package_fixture(directory, report):
    children = []
    for arm in report["arms"]:
        root = directory / arm["directory"]
        raw = (root / "results.json").read_bytes()
        (root / "results.json.gz").write_bytes(gzip.compress(raw, mtime=0))
        (root / "results.json").unlink()
        uncompressed = dict(name="results.json", bytes=len(raw), sha256=audit.sha(raw), compressed_as="results.json.gz", gzip_roundtrip_verified=True)
        child_manifest = dict(format="qwen_results_package_v1", uncompressed_results=uncompressed, files=[])
        data = audit.canonical(child_manifest)
        (root / "package_manifest.json").write_bytes(data)
        children.append(dict(directory=arm["directory"], package_manifest=dict(bytes=len(data),sha256=audit.sha(data)), uncompressed_results=uncompressed))
    for name in ("results.csv", "summary.md"):
        (directory / name).write_text("report\n")
    raw = (directory / "results.json").read_bytes()
    (directory / "results.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    (directory / "results.json").unlink()
    files = [dict(file=name,bytes=len(data),sha256=audit.sha(data)) for name in (
        "results.json.gz", "results.csv", "summary.md", "source_manifest.json", "frozen_manifest.json") for data in [(directory/name).read_bytes()]]
    root_manifest = dict(format="qwen_results_package_v1", files=files,
        uncompressed_results=dict(name="results.json",bytes=len(raw),sha256=audit.sha(raw),compressed_as="results.json.gz",gzip_roundtrip_verified=True))
    data = audit.canonical(root_manifest)
    (directory / "package_manifest.json").write_bytes(data)
    chain = dict(format="qwen_coset_training_package_v1", arms=children,
        root_package_manifest=dict(bytes=len(data),sha256=audit.sha(data)))
    (directory / "training_package_manifest.json").write_bytes(audit.canonical(chain))


class CosetTrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = fixture(self.root)
        # The existing packet auditor has its own corruption/integrity suite.
        # Here it is independently mocked so these fixtures target cross-arm
        # provenance and diagnostic arithmetic rather than fake KV packets.
        def child_review(root, source_root=None):
            _, raw = audit._report(root)
            return dict(results_json_sha256=audit.sha(raw)), dict(all_passed=True, passed=1, total=1)
        self.patch = mock.patch.object(audit.packet_audit, "review", side_effect=child_review)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def mutate(self, arm, callback, *, diagnostic=False):
        row = next(x for x in self.report["arms"] if x["name"] == arm)
        path = self.root / arm / ("training_diagnostics.json" if diagnostic else "results.json")
        obj, _ = audit.read_json(path)
        callback(obj)
        raw = audit.canonical(obj); path.write_bytes(raw)
        if diagnostic:
            row["diagnostics_sha256"] = audit.sha(raw)
            cp = self.root / arm / "results.json"
            child, _ = audit.read_json(cp)
            child["training_diagnostics"].update(bytes=len(raw),sha256=audit.sha(raw))
            cp.write_bytes(audit.canonical(child))
        row["report_sha256"] = audit.sha((self.root / arm / "results.json").read_bytes())
        (self.root / "results.json").write_bytes(audit.canonical(self.report))

    def failures(self):
        _, validation = audit.review(self.root)
        return [x["check"] for x in validation["checks"] if not x["passed"]]

    def test_valid_factorial_does_not_require_savings(self):
        self.assertEqual(self.failures(), [])

    def test_changed_training_prefix_is_rejected(self):
        self.mutate("random_long", lambda d:d["modes"]["none"]["checkpoints"][1].update(parameter_sha256="wrong"), diagnostic=True)
        self.assertTrue(any("identical short numerical prefix" in x for x in self.failures()))

    def test_changed_fixed_prior_is_rejected(self):
        self.mutate("balanced_long", lambda d:d["modes"]["none"]["checkpoints"][-1]["metrics"].update(prior_logits_sha256="changing-prior"), diagnostic=True)
        self.assertTrue(any("fixed probe/truth/prior" in x for x in self.failures()))

    def test_calibration_and_group_accounting_corruption_is_rejected(self):
        def corrupt(d):
            row=d["modes"]["none"]["checkpoints"][-1]["metrics"]["by_residue"]["6"]["learned"]["omitted"]
            row["calibration"]["counts"][1] += 1
            row["errors_per_group"][0] += 1
        self.mutate("random_short", corrupt, diagnostic=True)
        failed=self.failures()
        self.assertTrue(any("calibration counts" in x for x in failed))
        self.assertTrue(any("group histogram" in x for x in failed))

    def test_actual_parameter_shard_matches_numerical_hash(self):
        (self.root/"random_short"/"calibration"/"none.bin").write_bytes(shard(99))
        self.assertTrue(any("final numerical state matches delivered shard" in x for x in self.failures()))

    def test_test_capture_changes_are_rejected(self):
        self.mutate("balanced_long",lambda d:d["cases"][0]["capture"].update(kv_words_sha256="different"))
        self.assertTrue(any("matched sources, probes, TEST addresses" in x for x in self.failures()))

    def test_raw_child_identity_corruption_is_rejected(self):
        path=self.root/"random_short"/"results.json"
        path.write_bytes(path.read_bytes()+b" ")
        self.assertTrue(any("child raw report identity" in x for x in self.failures()))

    def test_unsafe_arm_paths_are_rejected(self):
        self.report["arms"][0]["directory"]="../outside"
        (self.root/"results.json").write_bytes(audit.canonical(self.report))
        with self.assertRaisesRegex(ValueError,"Unsafe artifact path"):
            audit.review(self.root)

    def test_gzip_packages_and_all_manifest_links_are_verified(self):
        package_fixture(self.root,self.report)
        self.assertEqual(self.failures(),[])
        (self.root/"summary.md").write_text("changed delivery")
        self.assertTrue(any("root delivered file identity: summary.md" in x for x in self.failures()))

    def test_changed_child_manifest_chain_is_rejected(self):
        package_fixture(self.root,self.report)
        path=self.root/"random_short"/"package_manifest.json"
        obj,_=audit.read_json(path);obj["scope"]="changed"
        path.write_bytes(audit.canonical(obj))
        self.assertTrue(any("training package child manifest identity" in x for x in self.failures()))

    def test_missing_training_package_manifest_is_rejected(self):
        package_fixture(self.root,self.report)
        (self.root/"training_package_manifest.json").unlink()
        self.assertTrue(any("complete training package manifest present" in x for x in self.failures()))

    def test_cli_refuses_existing_output(self):
        output=self.root/"review"; output.mkdir(); (output/"keep").write_text("preserve")
        with self.assertRaises(SystemExit) as error:
            audit.main([str(self.root),"--output",str(output)])
        self.assertEqual(error.exception.code,2)
        self.assertEqual((output/"keep").read_text(),"preserve")


if __name__=="__main__":
    unittest.main()

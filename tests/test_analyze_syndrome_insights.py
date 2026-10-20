"""Independent NumPy-only schedule reconstruction and coverage integrity."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coset_insights", ROOT / "scripts" / "analyze_syndrome_insights.py")
insights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(insights)


def fixture(strategy="random", modern=False):
    """Fixed small reference schedule, including channel-epoch wraps.

    These channel tables were recorded from the producer schedule helper;
    reconstructing them here never imports Torch or calls that helper.
    """
    channels = {
        "random": [[1,2,3,4], [0,3,4,5], [1,2,3,4], [1,2,3,4],
                   [1,2,3,4], [0,2,3,5], [2,3,4,5], [0,1,2,4]],
        "balanced": [[0,1,3,4], [0,2,3,4], [1,2,4,5], [1,2,4,5],
                     [0,2,3,5], [0,1,3,5], [0,2,3,5], [2,3,4,5]],
    }[strategy]
    offsets, budgets = [6,2,0,3,4,4,5,1], [0,6,6,6,6,2,2,0]
    counts = np.zeros((2,6), np.int64)
    words = np.zeros_like(counts)
    omitted = np.zeros((2,6,7), np.int64)
    digest = hashlib.sha256(); checkpoints = []
    prefixes = {0: digest.hexdigest()}
    for i, indices in enumerate(channels):
        layer = i % 2
        row = dict(update=i+1, layer=layer, capture=0, record=0, offset=offsets[i],
            start=offsets[i], tokens=4, residue_bits=budgets[i], channel_indices=indices)
        digest.update(insights.canonical(row))
        prefixes[i+1] = digest.hexdigest()
        checkpoints.append(dict(row, sampling_prefix_sha256=digest.hexdigest()))
        counts[layer, indices] += 1; words[layer, indices] += 4; omitted[layer, indices, budgets[i]:] += 4
    summary = insights._coverage_summary(counts, words, omitted)
    coverage = dict(optimizer_updates_by_layer_channel=counts.tolist(), word_examples_by_layer_channel=words.tolist(), summary=summary)
    train = dict(seed=7, total_optimizer_updates=8, batch_tokens=4, channels_per_step=4,
        total_word_examples=128, total_token_examples=32, checkpoints=checkpoints, sampling_sha256=digest.hexdigest())
    if modern:
        train.update(channel_sampling=strategy, coverage_sha256=hashlib.sha256(insights.canonical(coverage)).hexdigest(),
            coverage_summary={k:v for k,v in summary.items() if k != "by_layer"})
    else:
        for row in checkpoints: row.pop("sampling_prefix_sha256")
    metadata = dict(geometry=dict(layers=2, channels=3), training=train, config=dict(training_residue_bits=[0,2,4,6]))
    report = dict(models={mode:dict(metadata=copy.deepcopy(metadata)) for mode in ("none", "hidden")},
        training_captures=[dict(tokens=10, producer_audit=dict(coverage="full", tile_tokens=10))],
        config=dict(tile_tokens=10))
    external = dict(format="qwen_prefill_coset_training_diagnostics_v1", modes={mode:dict(coverage=copy.deepcopy(coverage),
        checkpoints=[dict(update=u, sampling_prefix_sha256=prefixes[u]) for u in (0,4,8)]) for mode in report["models"]})
    return report, external


class SyndromeInsightTests(unittest.TestCase):
    def test_old_random_without_new_metadata_remains_supported(self):
        report, _ = fixture()
        result = insights.sampling_coverage(report)
        self.assertEqual(result["channel_sampling"], "random")
        self.assertEqual(result["never_selected_by_optimizer"], 1)
        self.assertEqual(result["actual_word_examples"], 128)
        self.assertEqual(result["compact_coverage_verified_models"], [])
        report["models"]["hidden"]["metadata"]["training"]["sampling_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "full sampling digest"):
            insights.sampling_coverage(report)

    def test_balanced_compact_and_full_coverage_are_independently_verified(self):
        report, external = fixture("balanced", modern=True)
        result = insights.sampling_coverage(report, external)
        self.assertEqual(result["never_selected_by_optimizer"], 0)
        self.assertEqual(result["max_minibatches_per_slot"], 3)
        self.assertEqual(set(result["compact_coverage_verified_models"]), {"none", "hidden"})
        self.assertEqual(set(result["external_coverage_verified_models"]), {"none", "hidden"})
        changed = copy.deepcopy(external)
        changed["modes"]["hidden"]["coverage"]["optimizer_updates_by_layer_channel"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "External full optimizer coverage"):
            insights.sampling_coverage(report, changed)
        changed = copy.deepcopy(report)
        changed["models"]["hidden"]["metadata"]["training"]["coverage_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "compact coverage/hash"):
            insights.sampling_coverage(changed, external)
        changed = copy.deepcopy(report)
        changed["models"]["hidden"]["metadata"]["training"]["checkpoints"][3]["sampling_prefix_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "checkpoint prefix"):
            insights.sampling_coverage(changed, external)

    def test_companion_identity_missing_and_corrupt_are_distinguished(self):
        report, external = fixture("balanced", modern=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = json.dumps(external).encode()
            report["training_diagnostics"] = dict(file="training_diagnostics.json", bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())
            self.assertIsNone(insights.load_training_diagnostics(directory, report))
            result = insights.sampling_coverage(report)
            self.assertTrue(result["external_coverage_companion_recorded"])
            self.assertFalse(result["external_coverage_companion_loaded"])
            path = directory / "training_diagnostics.json"
            path.write_bytes(data)
            self.assertEqual(insights.load_training_diagnostics(directory / "results.json.gz", report), external)
            path.write_bytes(data+b" ")
            with self.assertRaisesRegex(ValueError, "byte/hash mismatch"):
                insights.load_training_diagnostics(directory, report)


if __name__ == "__main__":
    unittest.main()

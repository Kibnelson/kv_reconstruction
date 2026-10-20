"""Fixed probes, decoder-known input boundaries and diagnostic accounting."""
import copy
import json
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import syndrome_diagnostics as diagnostics, syndrome_models as models


def _words(value):
    x = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((x+np.uint32(0x7fff)+((x>>16)&1))>>16).astype(np.uint16)


def _fixture(tokens=64):
    g = dict(layers=1, channels=2, hidden_size=2, head_dim=2, decoder_rank=1,rank=1,width=2,source_width=2)
    p = {name:torch.zeros(shape) for name,shape in models._network_shapes(g,"hidden").items()}
    p.update(source_mu=torch.zeros((1,2)),source_sd=torch.ones((1,2)),
             rope_inv_freq=torch.ones(1),exponent_centers=torch.full((1,4),127.))
    p["encoder_W"][0,0] = 1.
    p["W0"][0,0] = 1.
    p["W1"][0,0] = 1.
    p["W2"][0,6] = 12.
    p["b2"][6] = -1.
    h = np.zeros((tokens,2),np.float32)
    h[:,0] = np.where(np.arange(tokens)%2,2.,-2.)
    bit = np.where(h[:,0]>0,64,0).astype(np.uint16)
    target = np.broadcast_to(np.uint16(127<<7)|bit[:,None],(tokens,2)).copy()
    r = dict(split="dev",layer=0,start=0,tokens=tokens,head_dim=2,
             hidden=_words(h),k=target.copy(),v=target.copy(),k_proj=target.copy(),k_norm_eps=1e-6)
    return p,g,[r]


class FixedSyndromeDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_fixed_head_tail_probe_all_channels_and_source_identity(self):
        p,g,rows = _fixture(128)
        probe,manifest = diagnostics.make_probe(rows,tokens_per_layer=64)
        self.assertEqual([(r["start"],r["tokens"]) for r in probe],[(0,32),(96,32)])
        self.assertEqual(manifest["word_values"],64*4)
        self.assertEqual(manifest["selected_layers"],[0])
        repeat,again = diagnostics.make_probe(copy.deepcopy(rows),tokens_per_layer=64)
        self.assertEqual(manifest,again)
        rows[0]["hidden"][0,0] ^= np.uint16(1)
        _,changed = diagnostics.make_probe(rows,tokens_per_layer=64)
        self.assertNotEqual(manifest["sha256"],changed["sha256"])
        self.assertNotEqual(manifest["source_records_sha256"],changed["source_records_sha256"])
        self.assertEqual(probe[0]["hidden"][0,0],repeat[0]["hidden"][0,0])

    def test_planted_signal_controls_ablations_and_calibration_counts(self):
        params,g,records = _fixture()
        result = diagnostics.evaluate(params,g,"hidden",records,residue_bits=(6,),prior_logits=np.zeros((1,4,7),np.float32))
        row = result["by_residue"]["6"]
        learned = row["learned"]["omitted"]
        self.assertEqual(learned["values"],256)
        self.assertEqual(learned["bit_errors"],0)
        self.assertEqual(learned["original_ones"],128)
        self.assertEqual(learned["predicted_ones"],128)
        self.assertEqual(learned["groups"],8)
        self.assertEqual(learned["recoverable_groups"],8)
        self.assertEqual(learned["errors_per_group"][0],8)
        self.assertEqual(learned["tail_bits"],8)
        self.assertEqual(learned["values"],31*learned["groups"]+learned["tail_bits"])
        self.assertEqual(sum(learned["calibration"]["counts"]),256)
        self.assertEqual(sum(learned["calibration"]["truth_sums"]),128.)
        self.assertAlmostEqual(learned["BCE_sum_bits"]/256,learned["BCE_bits"])
        self.assertEqual(row["controls"]["constant_zero"]["omitted"]["bit_errors"],128)
        self.assertNotIn("BCE_bits",row["controls"]["constant_zero"]["omitted"])
        self.assertEqual(row["controls"]["train_prior"]["omitted"]["bit_errors"],128)
        self.assertAlmostEqual(row["controls"]["train_prior"]["omitted"]["BCE_bits"],1.)
        self.assertGreater(row["ablations"]["zero"]["bit_errors_delta"],0)
        self.assertGreater(row["ablations"]["shuffled"]["prediction_changed_bits"],0)
        self.assertGreater(row["ablations"]["zero"]["BCE_bits_delta"],0)
        self.assertEqual(sum(x["values"] for x in row["learned"]["by_kind"].values()),256)
        json.dumps(result,allow_nan=False)

    def test_decoder_known_words_masked_before_model_and_source_hash_responds(self):
        params,g,records = _fixture()
        original_forward = models._forward
        seen = []
        def checked(*args,**kwargs):
            known,b = args[5:7]
            self.assertTrue(bool(torch.all((known.to(torch.int64)&(127^((1<<b)-1)))==0)))
            seen.append(True)
            return original_forward(*args,**kwargs)
        with mock.patch.object(models,"_forward",checked):
            a = diagnostics.evaluate(params,g,"hidden",records,residue_bits=(6,),ablations=False)
        changed = copy.deepcopy(records)
        changed[0]["k"] ^= np.uint16(64); changed[0]["v"] ^= np.uint16(64)
        b = diagnostics.evaluate(params,g,"hidden",changed,residue_bits=(6,),ablations=False)
        self.assertTrue(seen)
        self.assertEqual(a["by_residue"]["6"]["known_fields_sha256"],b["by_residue"]["6"]["known_fields_sha256"])
        self.assertNotEqual(a["probe"]["sha256"],b["probe"]["sha256"])
        self.assertEqual(b["by_residue"]["6"]["learned"]["omitted"]["bit_errors"],256)

    def test_full_diagnostic_does_not_mutate_inputs_or_rng_and_prior_is_explicit(self):
        params,g,records = _fixture(32)
        original_params = {name:value.clone() for name,value in params.items()}
        original_hash = diagnostics.producer._source_hash(records)
        state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        first = diagnostics.evaluate(params,g,"hidden",records,residue_bits=(0,2,4,6))
        second = diagnostics.evaluate(params,g,"hidden",records,residue_bits=(0,2,4,6))
        self.assertEqual(first,second)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        after = np.random.get_state()
        self.assertEqual(numpy_state[0],after[0]);np.testing.assert_array_equal(numpy_state[1],after[1]);self.assertEqual(numpy_state[2:],after[2:])
        self.assertEqual(original_hash,diagnostics.producer._source_hash(records))
        for name,value in params.items(): self.assertTrue(torch.equal(value,original_params[name]))
        for row in first["by_residue"].values(): self.assertNotIn("train_prior",row["controls"])
        self.assertEqual(first["by_residue"]["0"]["learned"]["omitted"]["values"],32*4*7)

    def test_frozen_train_prior_remains_fixed_when_learned_prior_changes(self):
        params,g,records = _fixture(32)
        prior = np.zeros((1,4,7),np.float32)
        a = diagnostics.evaluate(params,g,"hidden",records,residue_bits=(6,),ablations=False,prior_logits=prior)
        params["prior"] += 5.
        b = diagnostics.evaluate(params,g,"hidden",records,residue_bits=(6,),ablations=False,prior_logits=prior)
        self.assertEqual(a["prior_logits_sha256"],b["prior_logits_sha256"])
        self.assertEqual(a["by_residue"]["6"]["controls"],b["by_residue"]["6"]["controls"])
        self.assertNotEqual(a["by_residue"]["6"]["learned"],b["by_residue"]["6"]["learned"])

    def test_none_mode_has_no_features_or_ablations_and_zero_groups_have_tails(self):
        params,g,records = _fixture(8)
        g = dict(g,rank=0,source_width=0)
        result = diagnostics.evaluate(params,g,"none",records,residue_bits=(6,),prior_logits=np.zeros((1,4,7),np.float32))
        self.assertEqual(result["code_summary"]["rank"],0)
        self.assertEqual(result["code_summary"]["retained_BF16_bytes"],0)
        self.assertFalse(result["ablations_evaluated"])
        row = result["by_residue"]["6"]["learned"]["omitted"]
        self.assertEqual(row["groups"],0);self.assertEqual(row["tail_bits"],32)
        self.assertIsNone(row["recoverable_fraction"])

    def test_dev_only_nonoverlap_and_finite_prior_rejected(self):
        params,g,records = _fixture()
        bad = copy.deepcopy(records);bad[0]["split"]="test"
        with self.assertRaisesRegex(ValueError,"DEV"): diagnostics.evaluate(params,g,"hidden",bad)
        with self.assertRaisesRegex(ValueError,"overlap"): diagnostics.make_probe(records+records)
        with self.assertRaisesRegex(ValueError,"prior"): diagnostics.evaluate(params,g,"hidden",records,prior_logits=np.zeros((1,4,6)))
        with self.assertRaisesRegex(ValueError,"residue"): diagnostics.evaluate(params,g,"hidden",records,residue_bits=(6,6))
        with self.assertRaisesRegex(ValueError,"RoPE"): diagnostics.evaluate(params,g,"hidden",records,rope_scaling=0.)


if __name__ == "__main__":
    unittest.main()

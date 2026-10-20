"""TRAIN isolation, exact retained codes and decoder-known-input boundaries."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from qwen_kv_bench import syndrome_models as models, producer_models as producer


def words(value):
    bits=np.ascontiguousarray(value,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)


def captures(seed=71,tokens=32):
    rng=np.random.default_rng(seed);result=[]
    for _ in range(3):
        records=[]
        for layer in range(2):
            h=rng.standard_normal((tokens,12)).astype(np.float32)
            # Known fields contain no high-mantissa answer: exponent constant,
            # low mantissa zero. Hidden first coordinates determine bit6.
            bits=((h[:,0]>0).astype(np.uint16)*64)[:,None]
            k=np.broadcast_to(np.uint16(127<<7)|bits,(tokens,8)).copy()
            v=np.broadcast_to(np.uint16(127<<7)|(((h[:,1]>0).astype(np.uint16)*64)[:,None]),(tokens,8)).copy()
            projection=words(rng.standard_normal((tokens,8)).astype(np.float32))
            records.append(dict(layer=layer,start=0,tokens=tokens,head_dim=4,hidden=words(h),
                k=k,v=v,k_proj=projection,k_norm_eps=1e-6))
        result.append(records)
    return result


def fit(directory,data=None,**kwargs):
    return models.fit(captures() if data is None else data,directory,
        training_metadata=[dict(split="train",fixture=i) for i in range(3)],
        model_identity=dict(model="syndrome-synthetic",config=dict(num_hidden_layers=2,
            num_key_value_heads=2,head_dim=4,hidden_size=12)),
        inverse=np.array([1.,.1],np.float32),rope_scaling=1.,head_dim=4,rank=2,width=16,
        batch_tokens=16,channels_per_step=8,device="cpu",seed=71,**kwargs)


class SyndromeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        cls.directory=tempfile.TemporaryDirectory()
        cls.data=captures()
        cls.models=fit(Path(cls.directory.name)/"models",cls.data,steps=8)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup();torch.set_num_threads(cls.previous_threads)

    def test_complete_artifacts_reload_prediction_and_accounting(self):
        for name,model in self.models.items():
            self.assertEqual(model.state_bytes,sum(Path(p).stat().st_size for p in model.artifact_paths))
            self.assertEqual(model.model_id,hashlib.sha256(Path(model.path).read_bytes()).hexdigest())
            self.assertTrue(all(Path(p).suffix in (".json",".bin") for p in model.artifact_paths))
            self.assertEqual(model.metadata["training_hashes"],[producer._source_hash(r) for r in self.data])
            reloaded=models.load(model.path,expected_model_identity=model.metadata["model_identity"],
                expected_training_hashes=model.metadata["training_hashes"],expected_model_id=model.model_id,expected_mode=name)
            p,q=model.prepare("cpu"),reloaded.prepare("cpu")
            record=copy.deepcopy(self.data[0][0])
            features=p.encode_features(record)
            expected_rank=0 if name=="none" else 2
            self.assertEqual(features.dtype,np.uint16);self.assertEqual(features.shape,(32,expected_rank))
            self.assertEqual(p.last_feature_diagnostics["feature_bytes"],features.nbytes)
            known=np.concatenate((record["k"],record["v"]),axis=1)
            logits=p.predict_logits(0,features,13,known,2)
            record.clear()
            for key in ("encoder_W","encoder_b","source_mu","source_sd"):
                q.params.pop(key,None)
            np.testing.assert_array_equal(logits,q.predict_logits(0,features.copy(),13,known.copy(),2))
            bits=q.predict_bits(0,features,13,known,2)
            self.assertEqual(bits.dtype,np.uint8);self.assertEqual(bits.shape,(32,16,7))
            self.assertGreaterEqual(q.state_bytes,q.physical_nbytes)

    def test_unknown_target_bits_are_masked_even_for_unmasked_caller(self):
        target=np.concatenate((self.data[0][0]["k"],self.data[0][0]["v"]),axis=1)
        p=self.models["hidden"].prepare("cpu");features=p.encode_features(self.data[0][0])
        rng=np.random.default_rng(9)
        for b in (0,2,4,6,7):
            unknown=(np.uint16(127)^np.uint16((1<<b)-1))
            changed=(target&~unknown)|(rng.integers(0,128,target.shape,dtype=np.uint16)&unknown)
            np.testing.assert_array_equal(p.predict_logits(0,features,0,target,b),
                p.predict_logits(0,features,0,changed,b))
        with self.assertRaises(ValueError):p.predict_bits(0,features,0,target,8)
        bad=features.copy();bad[0,0]=0x7fc1
        with self.assertRaisesRegex(ValueError,"nonfinite"):p.predict_bits(0,bad,0,target,2)

    def test_modes_share_sampling_and_side_budget_and_balance_layers(self):
        logs=[m.metadata["training"] for m in self.models.values()]
        self.assertEqual(len({r["sampling_sha256"] for r in logs}),1)
        for row in logs:
            self.assertEqual(row["total_optimizer_updates"],8)
            self.assertEqual(row["updates_by_layer"],{"0":4,"1":4})
            self.assertEqual(row["total_word_examples"],8*16*8)
            self.assertEqual(row["word_examples_by_layer"],{"0":512,"1":512})
        for name,model in self.models.items():
            g=model.metadata["geometry"]
            self.assertEqual(g["decoder_rank"],2);self.assertEqual(g["rank"],0 if name=="none" else 2)
            self.assertIn("surrogate",model.metadata["config"]["objective"])
        self.assertEqual(self.models["hidden_norm"].metadata["geometry"]["source_width"],14)

    def test_actual_upstream_source_and_norm_ablation_are_distinct_from_final_kv(self):
        record=copy.deepcopy(self.data[0][0])
        p=self.models["hidden_norm"].prepare("cpu");expected=p.encode_features(record)
        record["k"]^=np.uint16(0x8001);record["v"]^=np.uint16(0x8002)
        np.testing.assert_array_equal(expected,p.encode_features(record))
        original=models.normalization_scales(record,4)
        record["k_proj"]=words(np.full((32,8),2.,np.float32))
        modified=models.normalization_scales(record,4)
        self.assertTrue(np.any(original!=modified))
        changed=p.encode_features(record)
        self.assertEqual(changed.shape,expected.shape)
        self.assertTrue(np.any(changed!=expected))
        self.assertTrue(p.last_feature_diagnostics["normalization_scale_recomputation"])
        base=producer._source_hash(self.data[0])
        changed=copy.deepcopy(self.data[0]);changed[0]["k_norm_eps"]*=2
        self.assertNotEqual(base,producer._source_hash(changed))
        with self.assertRaises(ValueError):
            record["k_norm_eps"]=0;models.normalization_scales(record,4)

    def test_bf16_rounding_has_straight_through_encoder_gradient(self):
        params=dict(source_mu=torch.zeros((1,2)),source_sd=torch.ones((1,2)),
            encoder_W=torch.eye(2,requires_grad=True),encoder_b=torch.zeros(2,requires_grad=True))
        x=torch.tensor([[.123456,-1.234567]])
        z=models._encode(params,x,0,straight_through=True)
        np.testing.assert_array_equal(z.detach().numpy(),x.to(torch.bfloat16).float().numpy())
        z.sum().backward()
        self.assertGreater(float(params["encoder_W"].grad.abs().sum()),0)

    def test_train_isolation_and_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad=copy.deepcopy(self.data);bad[0][0]["split"]="test"
            with self.assertRaisesRegex(ValueError,"TRAIN"):
                fit(Path(temporary)/"test",bad,steps=1,modes=("hidden",))
            bad=copy.deepcopy(self.data);bad[0].append(copy.deepcopy(bad[0][0]))
            with self.assertRaisesRegex(ValueError,"overlap"):
                fit(Path(temporary)/"overlap",bad,steps=1,modes=("hidden",))
            model=self.models["hidden"]
            manifest=Path(temporary)/Path(model.path).name
            manifest.write_bytes(Path(model.path).read_bytes())
            shard=Path(temporary)/Path(model.artifact_paths[1]).name
            data=Path(model.artifact_paths[1]).read_bytes()
            shard.write_bytes(data[:-1]+bytes([data[-1]^1]))
            with self.assertRaisesRegex(ValueError,"SHA"):models.load(manifest)
            with self.assertRaisesRegex(ValueError,"TRAIN"):models.load(model.path,expected_training_hashes=["wrong"])

    def test_hidden_training_can_reduce_held_fixed_training_bit_loss(self):
        # Plant two recoverable signs in an exactly two-dimensional source,
        # matching the retained rank. The known fields reveal neither target
        # bit, so improvement requires the encoder-to-decoder gradient path.
        learnable=copy.deepcopy(self.data)
        for records in learnable:
            for record in records:
                record["hidden"][:,2:]=0
        with tempfile.TemporaryDirectory() as temporary:
            a=fit(Path(temporary)/"initial",learnable,steps=1,modes=("hidden",))["hidden"].prepare("cpu")
            b=fit(Path(temporary)/"trained",learnable,steps=800,modes=("hidden",))["hidden"].prepare("cpu")
            def loss(p):
                total=[]
                for records in learnable:
                    for r in records:
                        target=np.concatenate((r["k"],r["v"]),axis=1)
                        logits=p.predict_logits(r["layer"],p.encode_features(r),0,target,6)[...,6]
                        truth=((target>>6)&1).astype(np.float32)
                        total.append(np.mean(np.maximum(logits,0)-logits*truth+np.log1p(np.exp(-np.abs(logits)))))
                return float(np.mean(total))
            self.assertLess(loss(b),loss(a)-.05)

    def test_replay_contract_rejects_changed_cpu_thread_setting(self):
        p=self.models["none"].prepare("cpu")
        torch.set_num_threads(1)
        try:
            with self.assertRaisesRegex(ValueError,"contract"):p.encode_features(self.data[0][0])
        finally:torch.set_num_threads(2)


if __name__=="__main__":
    unittest.main()


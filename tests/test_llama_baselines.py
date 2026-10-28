"""Baseline transport/protocol checks; CPU software results are never GPU measurements."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
    raise unittest.SkipTest("PyTorch/Triton required for baseline checks")
import torch
from kv_compare.causal_packet import Packet, sha, word_sha, write_json
from kv_compare.framed_baselines import Baseline, checked_splitzip
from scripts import bench_llama_baselines as bench
from scripts import package_llama_baselines as package


class BaselineTests(unittest.TestCase):
    def test_raw_all_bf16_patterns_and_checked_framing(self):
        x = torch.arange(65536,dtype=torch.int32).to(torch.int16).reshape(2,128,256)
        codec = Baseline("raw_bf16","cpu")
        wire = codec.encode(x).export()
        y = codec.decode(Packet.restore(wire,"cpu"))
        self.assertTrue(torch.equal(x,y))
        self.assertGreater(len(wire),x.numel()*2)
        damaged = bytearray(wire); damaged[-33] ^= 1
        with self.assertRaises(ValueError):
            Packet.restore(damaged,"cpu")
        p = Packet.restore(wire,"cpu"); p.starts[-1] -= 4
        with self.assertRaisesRegex(ValueError,"directory"):
            codec.decode(p)
        p = Packet.restore(wire,"cpu"); p.header["state_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError,"identity"):
            codec.decode(p)

    def test_fp8_saturation_rounding_and_complete_accounting(self):
        x = torch.tensor([-600.,-448.,-1.1,-0.,0.,.1,1.1,448.,600.],dtype=torch.bfloat16)
        codec = Baseline("fp8_e4m3fn_saturating","cpu")
        wire = codec.encode(x.view(torch.int16)).export()
        y = codec.decode(Packet.restore(wire,"cpu")).view(torch.bfloat16)
        expected = x.float().clamp(-448,448).to(torch.float8_e4m3fn).to(torch.bfloat16)
        self.assertTrue(torch.equal(y.view(torch.int16),expected.view(torch.int16)))
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertGreater(len(wire),x.numel())
        self.assertFalse(torch.equal(y,x))

    def test_actual_cuda_required_for_measurement(self):
        with self.assertRaisesRegex(RuntimeError,"actual CUDA required"):
            bench.device_check("cpu")

    def test_verified_input_identity_and_reject_changed_receiver(self):
        from kv_compare.causal_study import MODEL, REVISION
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.arange(28*2*2048,dtype=np.uint16).reshape(28,2,2048)
            path = root/"decoded.npy"; np.save(path,a)
            digest = hashlib.sha256(a.astype("<u2").tobytes()).hexdigest()
            r = dict(id="lite_k0_test",document="test",mode="lossless",status="verified",original_exact=True,
                decoded="decoded.npy",decoded_file_sha256=sha(path),decoded_word_sha256=digest,original_word_sha256=digest,tokens=2)
            base = dict(status="completed",gpu_measured=True,codec_completed=True,quality_completed=True,
                rows=[r],protocol=dict(source=dict(model=MODEL,revision=REVISION)))
            write_json(root/"results.json",base)
            write_json(root/"independent_replay.json",dict(gpu_measured=True,no_source_kv_or_prediction_arrays=True,
                checks=[dict(id=r["id"],verified=True,word_sha256=digest,decoded_file_sha256=sha(path))]))
            _,inputs = bench.verified_inputs(root)
            self.assertEqual(inputs[0]["original_word_sha256"],digest)
            with path.open("ab") as f: f.write(b"changed")
            with self.assertRaisesRegex(ValueError,"array changed"):
                bench.verified_inputs(root)

    def test_package_allowlist_and_evidence_tamper_rejection(self):
        # Synthetic packaging fixture, not a GPU execution result.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/"run"; method="raw_bf16"; worker=run/method
            worker.mkdir(parents=True)
            packet=worker/"wire.kvc"; packet.write_bytes(b"test packet")
            decoded=worker/"received.npy"; np.save(decoded,np.zeros(4,dtype=np.uint16))
            (worker/"must_not_publish_tensor.pt").write_bytes(b"private tensor")
            row=dict(id="raw_test",method=method,policy=method,document="test",status="verified",mode="lossless",
                original_exact=True,decoded_word_sha256="fixture",decoded_file_sha256=sha(decoded),
                packet=str(packet),packet_sha256=sha(packet),decoded=str(decoded),values=4,
                packet_bytes=packet.stat().st_size,complete_bpv=8*packet.stat().st_size/4)
            write_json(worker/"receiver.json",dict(gpu_measured=True,no_source_kv_or_predictions=True,checks=[dict(
                id=row["id"],verified=True,word_sha256="fixture",decoded_file_sha256=sha(decoded))]))
            write_json(run/"results.json",dict(status="partial",gpu_measured=True,rows=[row],workers={method:{}},baseline_status={}))
            (run/"summary.md").write_text("synthetic package unit fixture\n")
            dest=root/"report"
            self.assertEqual(package.main([str(run),"--destination",str(dest)]),0)
            self.assertFalse(any(p.suffix in (".pt",".npy",".kvc") for p in dest.rglob("*")))
            packet.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"evidence changed"):
                package.main([str(run),"--destination",str(root/"bad_report")])

    @unittest.skipUnless(os.environ.get("TRITON_INTERPRET")=="1","explicit CPU software check")
    def test_worker_separate_receiver_quality_and_small_package(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from kv_compare.capture import capture_ids
        from kv_compare.quality import continuation_nll
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); captures = root/"captures"; out = root/"out"; out.mkdir()
            torch.manual_seed(1)
            cfg = LlamaConfig(vocab_size=37,hidden_size=8,intermediate_size=16,num_hidden_layers=2,
                num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=64)
            model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
            ids = (np.arange(10)*3%37).tolist()
            capture_ids(model,ids,captures,"test",torch.device("cpu"))
            K=np.load(captures/"test_K_bits.npy"); V=np.load(captures/"test_V_bits.npy")
            words=np.concatenate([K[:,:4],V[:,:4]],axis=2)
            source=root/"input.npy"; np.save(source,words)
            e=dict(document="test",path=str(source),file_sha256=sha(source),
                original_word_sha256=hashlib.sha256(words.astype("<u2").tobytes()).hexdigest(),
                values=words.size,tokens=4,layers=2,shape=list(words.shape))
            rows=[]; workers={}
            for method in ("raw_bf16","fp8_e4m3fn_saturating"):
                req=dict(method=method,inputs=[e],output=str(out/method),warmup=1,repeats=2,software_check=True)
                path=root/(method+".json"); write_json(path,req)
                bench.worker(path,torch,torch.device("cpu"))
                w=json.loads((out/method/"worker_results.json").read_text())
                self.assertFalse(w["provenance"]["gpu_measured"])
                self.assertTrue(w["proof"]["no_source_kv_or_predictions"])
                self.assertEqual(w["rows"][0]["status"],"verified")
                rows.extend(w["rows"]); workers[method]=w
            ref=continuation_nll(model,ids,4,4,torch.device("cpu"),[(words[l,:,:4],words[l,:,4:]) for l in range(2)])
            base=dict(quality_reference=dict(test=dict(nll=ref)),protocol=dict(continuation=4,source=dict(documents=dict(test=dict(
                token_ids_sha256=hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest())))))
            with mock.patch("scripts.bench_llama_causal.cached_model_snapshot",return_value="tiny"), mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",return_value=model):
                bench.quality(rows,[e],base,captures,torch,torch.device("cpu"))
            self.assertEqual(rows[0]["quality"]["ppl_change_percent"],0)
            result=dict(status="partial",gpu_measured=False,rows=rows,workers=workers,reused_rows=[],baseline_status={})
            write_json(out/"results.json",result); bench.report(result,out)
            # Packaging must refuse CPU evidence as an actual GPU report.
            with self.assertRaisesRegex(ValueError,"GPU receiver"):
                package.main([str(out),"--destination",str(root/"report")])

    @unittest.skipUnless(os.environ.get("SPLITZIP_TEST_SOURCE") and os.environ.get("TRITON_INTERPRET")=="1",
                         "pinned author checkout and CPU interpreter required")
    def test_unchanged_splitzip_odd_escape_alignment_and_fresh_state(self):
        source=Path(os.environ["SPLITZIP_TEST_SOURCE"])
        checked_splitzip(source,software_check=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"lut.npz"
            codec=Baseline("splitzip_official_framed","cpu",source=source)
            train=torch.full((16,),127<<7,dtype=torch.int16)
            codec.calibrate(train,path)
            # Three escapes produce odd byte counts in the uint16 / uint8 fields.
            x=torch.tensor([127<<7,126<<7,125<<7,124<<7],dtype=torch.int16)
            packet=codec.encode(x)
            self.assertEqual(packet.header["extra"]["n_esc"],3)
            self.assertEqual(packet.header["extra"]["native_payload_bytes"],27)
            other=Baseline("splitzip_official_framed","cpu",source=source,state=path)
            self.assertTrue(torch.equal(other.decode(Packet.restore(packet.export(),"cpu")),x))
            self.assertEqual(other.shared_bytes,784)
            for n in (3,6,8,1032):
                x=torch.arange(n,dtype=torch.int32).mul(7919).to(torch.int16)
                self.assertTrue(torch.equal(other.decode(Packet.restore(codec.encode(x).export(),"cpu")),x))
            # Exercise the native worker and its separate receiver, not just an object reload.
            root=Path(tmp); sample=root/"sample.npy"; np.save(sample,x.numpy().astype("<u2"))
            calibration=root/"train.pt"; torch.save(train.view(torch.bfloat16),calibration)
            req=dict(method="splitzip_official_framed",source=str(source),calibration=str(calibration),
                output=str(root/"worker"),warmup=1,repeats=1,software_check=True,
                inputs=[dict(document="test",path=str(sample),file_sha256=sha(sample),
                    original_word_sha256=word_sha(x),values=x.numel(),tokens=1,layers=1,shape=list(x.shape))])
            request=root/"request.json"; write_json(request,req)
            bench.worker(request,torch,torch.device("cpu"))
            w=json.loads((root/"worker/worker_results.json").read_text())
            self.assertTrue(w["rows"][0]["original_exact"])
            self.assertEqual(w["rows"][0]["status"],"verified")
            self.assertFalse(w["proof"]["gpu_measured"])


if __name__ == "__main__":
    unittest.main()

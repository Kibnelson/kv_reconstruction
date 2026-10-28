"""Software tests of profiling, official-protocol preservation and native glue."""
import ctypes
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import numpy as np
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
    raise unittest.SkipTest("PyTorch/Triton required")
import torch
from kv_compare.causal_packet import Engine,Packet,sha
from kv_compare.causal_profile import profile_call,scopes
from kv_compare import causal_packet
from kv_compare.framed_baselines import frame
from kv_compare.official_exans import Exans,RAW,RECORD,METHOD,STATE
from scripts import run_llama_profile_baselines as runner
from scripts import package_llama_profile as package
from tests.test_llama_causal import tiny_state


class ProtocolTests(unittest.TestCase):
    def test_child_environment_device_selection_and_failure_restore_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=runner.Suite.__new__(runner.Suite); suite.out=Path(tmp); suite.env=dict(os.environ)
            suite.result={}; suite.save=lambda:None; suite.aqua_visible_device="GPU-test-device"
            before=dict(os.environ); cwd=Path.cwd()
            def launch(argv,path):
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"],"GPU-test-device")
                self.assertEqual(Path.cwd(),suite.out)
                path.parent.mkdir(parents=True,exist_ok=True); path.write_text("fixture")
            with mock.patch.object(runner,"call",side_effect=launch):
                suite.command("aqua_test",["fixture"],suite.out)
            self.assertEqual(dict(os.environ),before); self.assertEqual(Path.cwd(),cwd)
            with mock.patch.object(runner,"call",side_effect=RuntimeError("fixture error")):
                with self.assertRaises(RuntimeError): suite.command("aqua_failed",["fixture"],suite.out)
            self.assertEqual(dict(os.environ),before); self.assertEqual(Path.cwd(),cwd)
            self.assertEqual(suite.result["commands"][-1]["status"],"failed")

    def test_exans_official_geometry_charged_raw_tail_and_directory_validation(self):
        self.assertEqual(RAW,999424); self.assertEqual(RECORD,770048)
        # Exercise a charged raw-overflow packet on CPU without calling a native kernel.
        obj=Exans.__new__(Exans); obj.device=torch.device("cpu"); obj.blocks=1
        source=torch.arange(6,dtype=torch.int16).reshape(1,2,3)
        raw=torch.zeros(RAW,dtype=torch.uint8); raw[:source.numel()*2]=source.view(torch.uint8).flatten()
        p=frame(METHOD,source.shape,STATE,[("fit_indices",torch.empty(0,dtype=torch.int32)),
            ("raw_indices",torch.tensor([0],dtype=torch.int32)),("records",torch.empty(0,dtype=torch.uint8)),("raw",raw)],
            dict(blocks=1,raw_block_bytes=RAW,record_bytes=RECORD,input_padding_bytes=RAW-source.numel()*2))
        wire=p.export(); restored=Packet.restore(wire,"cpu")
        self.assertTrue(torch.equal(obj.decode(restored),source))
        self.assertGreater(len(wire),RAW)
        self.assertEqual(obj.accounting(p)["charged_raw_fallback_bytes"],RAW)
        p.header["extra"]["input_padding_bytes"]-=1
        with self.assertRaisesRegex(ValueError,"geometry"):
            obj.decode(p)
        with self.assertRaisesRegex(RuntimeError,"actual CUDA"):
            Exans("missing","missing","cpu",1)

    def test_official_aqua_commands_preserve_readme_recipe_and_ppl_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); suite=runner.Suite.__new__(runner.Suite)
            suite.out=root; suite.cache=root/"cache"; suite.result={"commands":[]}
            suite.env={}
            suite.args=mock.Mock(aqua_python=Path("/aqua/bin/python"))
            suite.aqua_verify=lambda:None
            def dataset(training=False):
                if training: (root/"aqua_train_data.json").write_text("{}")
                return {}
            suite.aqua_data=dataset
            suite.aqua_common=lambda:["--model_name","/pinned-llama","--model_seqlen","8192","--edenn_d","2","--edenn_n","16","--hadamard_groupsize","1024"]
            commands=[]
            def command(name,argv,cwd=None,env=None):
                commands.append((name,list(map(str,argv))))
                target=root/(name+".log"); target.write_text("WikiText-2 perplexity: 7.25\n")
                if name=="aqua_train_official": (root/"aqua_predictors.pt").write_bytes(b"fixture")
                suite.result["commands"].append(dict(wall_seconds=1.))
                return target
            suite.command=command
            with mock.patch.object(runner,"check_source"):
                suite.aqua_train(); suite.aqua_eval(False); suite.aqua_eval(True)
            train=commands[0][1]
            for key,value in (("--dataset","pajama"),("--total_nsamples","256"),("--valid_nsamples","32"),("--model_seqlen","8192")):
                self.assertEqual(train[train.index(key)+1],value)
            self.assertIn("--offload_activations",train)
            for _,argv in commands[1:]:
                self.assertEqual(argv[argv.index("--prefix_size")+1],"4")
                self.assertEqual(argv[argv.index("--recent_buffer_size")+1],"128")
                self.assertNotIn("--not_quantize_first_layer",argv)
            self.assertIn("--no_quant",commands[1][1]); self.assertIn("--predictors_input_path",commands[2][1])
            data=json.loads((root/"aqua_ppl_higgs2_official.json").read_text())
            self.assertEqual(data["physical_cache_bytes"],"unmeasured")
            self.assertIn("not transfer codec",data["scope"])
            (root/"bad.log").write_text("WikiText-2 perplexity: nan\n")
            with self.assertRaises(ValueError): runner.ppl(root/"bad.log")

    def test_dataset_identity_script_uses_the_official_text_join(self):
        import types
        with tempfile.TemporaryDirectory() as tmp:
            suite=runner.Suite.__new__(runner.Suite); suite.out=Path(tmp)
            suite.args=mock.Mock(aqua_python=Path("/aqua/bin/python"))
            suite.aqua_common=lambda:["--model_name","/pinned-llama"]
            texts=[]
            class Dataset:
                _fingerprint="fixture"
                def __len__(self): return 2
                def __getitem__(self,key): return ["alpha","beta"]
            class Tokenizer:
                @staticmethod
                def from_pretrained(*a,**k):
                    def tokenize(text,**kw):
                        texts.append(text); return {"input_ids":torch.tensor([[1,2,3]])}
                    return tokenize
            def command(name,argv):
                code=argv[2]
                with mock.patch.dict(sys.modules,{"datasets":types.SimpleNamespace(load_dataset=lambda *a,**k:Dataset()),
                        "transformers":types.SimpleNamespace(AutoTokenizer=Tokenizer)}),mock.patch.object(sys,"argv",["-c",*map(str,argv[3:])]):
                    exec(compile(code,"<identity preflight>","exec"),{})
            import sys
            suite.command=command
            data=suite.aqua_data()
            self.assertEqual(texts,["alpha\n\nbeta"])
            self.assertEqual(data["tokens"],3)
            self.assertTrue(suite.aqua_data(training=True)["training"])

    def test_stage_reuse_retry_and_tamper_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=runner.Suite.__new__(runner.Suite); suite.out=Path(tmp)
            suite.result={"stages":{}}; suite.save=lambda:None
            path=suite.out/"proof.json"; path.write_text("{}")
            action=mock.Mock(return_value=dict(artifacts=suite.artifacts(path)))
            self.assertTrue(suite.step("profile",action)); self.assertTrue(suite.step("profile",action))
            self.assertEqual(action.call_count,1)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError,"artifact changed"): suite.step("profile",action)
            error=mock.Mock(side_effect=RuntimeError("build failed"))
            self.assertFalse(suite.step("exans",error))
            self.assertEqual(suite.result["stages"]["exans"]["status"],"blocked")
            self.assertTrue(suite.step("exans",lambda:{}))

    def test_package_rejects_cpu_success_and_keeps_tensors_and_traces_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/"run"; run.mkdir()
            data=dict(status="completed",gpu_measured=False,stages={},recipe={"stages":["profile"]})
            (run/"results.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"CPU evidence"): package.main([str(run),"--destination",str(root/"bad")])
            data["status"]="failed"; (run/"results.json").write_text(json.dumps(data))
            (run/"model.pt").write_bytes(b"not for publication")
            (run/"traces").mkdir()
            with gzip.open(run/"traces/tiny.json.gz","wb") as f: f.write(b"{}")
            self.assertEqual(package.main([str(run),"--destination",str(root/"report")]),0)
            names={str(p.relative_to(root/"report")) for p in (root/"report").rglob("*")}
            self.assertNotIn("model.pt",names); self.assertNotIn("traces/tiny.json.gz",names)

    @unittest.skipUnless(os.environ.get("CUDA_TEST_INCLUDE") and shutil.which("g++"),"CUDA headers and host compiler required")
    def test_compatibility_overload_forwards_error_to_original_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); header=Path(__file__).resolve().parents[1]/"native/exans_build_compat.h"
            program='#include "'+str(header)+'"\n'+r'''
struct openlake_expans_codec {int value=7;};
namespace {int fail_cuda(openlake_expans_codec* c,cudaError_t status,const char*) {return c->value+int(status);}}
int main() {auto c=std::make_unique<openlake_expans_codec>(); return fail_cuda(c,cudaErrorMemoryAllocation,"allocate")==9?0:1;}
'''
            (root/"test.cc").write_text(program)
            p=subprocess.run(["g++","-std=c++17","-I"+os.environ["CUDA_TEST_INCLUDE"],root/"test.cc","-o",root/"test"],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr[-3000:])
            self.assertEqual(subprocess.run([root/"test"]).returncode,0)

    @unittest.skipUnless(os.environ.get("EXANS_TEST_SOURCE"),"pinned official checkout required")
    def test_pinned_exans_author_source_is_unchanged(self):
        self.assertEqual(runner.check_source(os.environ["EXANS_TEST_SOURCE"],"exans"),runner.PINS["exans"][1])


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET")=="1","explicit CPU interpreter required")
class ProfileTests(unittest.TestCase):
    def test_profile_stage_reuses_fitted_state_and_matches_existing_control_packets(self):
        import hashlib
        from kv_compare.causal_packet import word_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state=root/"state"; state.mkdir(); tiny_state(state)
            words=(torch.arange(48).reshape(2,3,8).float()/53-.4).to(torch.bfloat16).view(torch.int16)
            path=root/"words.npy"; np.save(path,words.numpy().astype("<u2"))
            options=dict(predictor_format="bf16",values_per_stream=2,block=8)
            engine=Engine(state,"cpu",**options); rows=[]
            for kappa in (0.,1.):
                packet=engine.encode(words,kappa=kappa)
                rows.append(dict(policy="lite",document="test",kappa=kappa,
                    packet_sha256=hashlib.sha256(packet.export()).hexdigest(),decoded_word_sha256=word_sha(engine.decode(packet))))
            suite=runner.Suite.__new__(runner.Suite); suite.out=root/"out"; suite.out.mkdir()
            suite.args=mock.Mock(document="test",kappas=[0.,1.]); suite.device=torch.device("cpu"); suite.torch=torch
            suite.inputs=[dict(document="test",path=str(path),original_word_sha256=word_sha(words))]
            suite.base=dict(state_directories={"lite":str(state)},rows=rows,protocol=dict(options,warmup=1,repeats=1))
            fitted=sha(state/"state.json")
            result=suite.profile()
            data=json.loads((suite.out/"profile.json").read_text())
            self.assertFalse(data["gpu_measured"]); self.assertEqual(len(data["rows"]),2)
            self.assertEqual(sha(state/"state.json"),fitted)
            self.assertTrue(all((suite.out/p).is_file() for p in result["artifacts"]))

    def test_profile_records_scopes_preserves_bytes_and_restores_instrumentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root)
            engine=Engine(root,"cpu",predictor_format="bf16",values_per_stream=2,block=8)
            torch.manual_seed(3); words=(torch.randn(2,3,8)*.1).to(torch.bfloat16).view(torch.int16)
            original_compact=causal_packet.compact
            for kappa in (0.,1.):
                control=engine.encode(words,kappa=kappa); intended=engine.decode(control)
                p,enc=profile_call(lambda:engine.encode(words,kappa=kappa),engine,kappa,"cpu",root/f"k{kappa}_encode.json.gz")
                y,dec=profile_call(lambda:engine.decode(control),engine,kappa,"cpu",root/f"k{kappa}_decode.json.gz")
                self.assertEqual(control.export(),p.export()); self.assertTrue(torch.equal(y,intended))
                self.assertFalse(enc["gpu_measured"]); self.assertEqual(enc["cuda_device_event_count"],0)
                names={r["name"] for r in enc["semantic_scopes"]}
                self.assertIn("kv/entropy_encode",names); self.assertIn("kv/compaction",names)
                self.assertIn("kv/rope_forward",names)
                self.assertTrue(json.loads(gzip.decompress(Path(enc["trace"]).read_bytes()))["traceEvents"])
            self.assertIs(causal_packet.compact,original_compact)
            with self.assertRaisesRegex(RuntimeError,"fixture"):
                with scopes(engine,0): raise RuntimeError("fixture")
            self.assertIs(causal_packet.compact,original_compact)
            self.assertNotIn("rotate",engine.__dict__)


if __name__=="__main__": unittest.main()

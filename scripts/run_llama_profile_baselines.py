#!/usr/bin/env python3
"""Profile existing Lite packets, exercise official ExANS, and run official AQUA.

Author code stays unchanged; native build compatibility is disclosed. Existing
fits and completed baseline rows are read-only. Each stage saves its own proof.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_llama_baselines import call, device_check, verified_inputs, log
from scripts.package_llama_causal import sha
from kv_compare.profile_recovery import recover_partial,runtime_environment

PINS={"aqua":("goodevening13/aquakv","e744183f2f3a4a90381f20a6975456871fd1ae29"),
      "exans":("openlake-project/openlake","de6c38bcc5438475bd1abf1e9b3d154cef414ef6")}
FILES=["scripts/run_llama_profile_baselines.py","scripts/run_llama_profile_a4500.sh",
       "scripts/package_llama_profile.py","scripts/setup_qwen_cuda_toolchain.py",
       "kv_compare/causal_profile.py","kv_compare/official_exans.py","native/exans_build_compat.h",
       "scripts/bench_llama_baselines.py","kv_compare/framed_baselines.py","scripts/package_llama_causal.py",
       "kv_compare/profile_recovery.py","scripts/prepare_aqua_calibration.py",
       "benchmark_fixtures/aqua_redpajama_backup_manifest.json"]


def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".part")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n"); tmp.replace(path)


def check_source(path,name):
    commit=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(path),"status","--porcelain","--untracked-files=no"],text=True)
    if commit!=PINS[name][1] or dirty:
        raise ValueError("official source changed: "+name)
    return commit


def ppl(path):
    values=re.findall(r"WikiText-2 perplexity:\s*([0-9.eE+-]+)",Path(path).read_text())
    if len(values)!=1 or not math.isfinite(float(values[0])) or float(values[0])<=0:
        raise ValueError("official evaluation did not report one finite perplexity")
    return float(values[0])


def receive(request,torch,device):
    import numpy as np
    from kv_compare.causal_packet import Packet,word_sha
    from kv_compare.official_exans import Exans
    q=json.loads(request.read_text()); check_source(q["source"],"exans")
    if sha(q["library"])!=q["library_sha256"]:
        raise ValueError("native receiver library changed")
    codec=Exans(q["source"],q["library"],device,q["blocks"])
    try:
        checks=[]
        for e in q["entries"]:
            if sha(e["packet"])!=e["packet_sha256"]:
                raise ValueError("receiver packet changed")
            words=codec.decode(Packet.restore(Path(e["packet"]).read_bytes(),device))
            if word_sha(words)!=e["expected_word_sha256"]:
                raise AssertionError("official ExANS receiver differs from original BF16 words")
            np.save(e["decoded"],words.cpu().numpy().astype("<u2"),allow_pickle=False)
            checks.append(dict(id=e["id"],verified=True,word_sha256=word_sha(words),decoded_file_sha256=sha(e["decoded"])))
        write(q["output"],dict(gpu_measured=True,no_source_kv_or_predictions=True,checks=checks,
            inputs="wire and pinned native decoder library only; expected digest is an assertion"))
    finally:
        codec.close()


class Suite:
    def __init__(self,args):
        self.args=args; self.out=args.output.resolve(); self.out.mkdir(parents=True,exist_ok=True)
        self.cache=ROOT/"runs/_llama_official"
        self.env=dict(os.environ,OMP_NUM_THREADS="16",OPENBLAS_NUM_THREADS="16",MAX_JOBS="2",TORCH_CUDA_ARCH_LIST="8.6")
        self.env.pop("TRITON_INTERPRET",None)
        code={f:sha(ROOT/f) for f in FILES}
        recipe=dict(base_results_sha256=sha(args.base_run/"results.json"),
            baseline_results_sha256=sha(args.baseline_run/"results.json"),code=code,pins=PINS,
            stages=args.stages,document=args.document,kappas=args.kappas,policy="lite",
            aqua_calibration=getattr(args,"aqua_calibration","original"))
        identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        self.result=dict(format="llama_profile_official_v1",status="running",gpu_measured=False,
            identity=identity,recipe=recipe,stages={})
        if (self.out/"results.json").exists():
            old=json.loads((self.out/"results.json").read_text())
            if old["identity"]!=identity:
                old=recover_partial(old,recipe,self.out)
            self.result=old
        self.save()

    def save(self):
        write(self.out/"results.json",self.result)
        self.report()

    def command(self,name,argv,cwd=None,env=None):
        path=self.out/"logs"/(name+".log")
        attempt=1
        while path.exists():
            path=self.out/"logs"/(name+f".attempt{attempt}.log"); attempt+=1
        begin=time.perf_counter()
        entry=dict(name=name,argv=list(map(str,argv)),log=str(path.relative_to(self.out)),status="running")
        self.result.setdefault("commands",[]).append(entry); self.save()
        # call() owns process-group cleanup and prints a heartbeat every 30 s.
        old=os.environ.copy()
        try:
            child_env=dict(env or self.env)
            if name.startswith("exans") and child_env.get("CUDA_HOME"):
                child_env=runtime_environment(child_env,child_env["CUDA_HOME"],getattr(self,"nvcomp_runtime",()))
            if name.startswith("aqua") and getattr(self,"aqua_visible_device",None) is not None:
                child_env["CUDA_VISIBLE_DEVICES"]=self.aqua_visible_device
            os.environ.update(child_env)
            # The child may need the official checkout as its import directory.
            previous=Path.cwd()
            if cwd:
                os.chdir(cwd)
            try:
                call(argv,path)
            finally:
                os.chdir(previous)
            entry["status"]="completed"
        except BaseException as error:
            entry.update(status="failed",error=str(error)); raise
        finally:
            os.environ.clear(); os.environ.update(old)
            entry["wall_seconds"]=time.perf_counter()-begin
        self.save()
        return path

    def step(self,name,action):
        old=self.result["stages"].get(name,{})
        if old.get("status")=="completed":
            for path,digest in old.get("artifacts",{}).items():
                if sha(self.out/path)!=digest:
                    raise ValueError("completed artifact changed: "+path)
            log("reuse verified completed stage: "+name); return True
        self.result["stages"][name]=dict(status="running"); self.save()
        try:
            info=action() or {}
            self.result["stages"][name]=dict(status="completed",**info)
            log(name+": completed")
            return True
        except Exception as error:
            self.result["stages"][name]=dict(status="blocked",reason=str(error),traceback=traceback.format_exc())
            log(name+": "+str(error)); return False
        finally:
            gc.collect()
            if hasattr(self,"torch"):
                self.torch.cuda.empty_cache()
            self.save()

    def artifacts(self,*paths):
        return {str(Path(p).relative_to(self.out)):sha(p) for p in paths}

    def checkout(self,name):
        source=self.cache/"sources"/name
        if not source.exists():
            source.parent.mkdir(parents=True,exist_ok=True)
            self.command(name+"_clone",["git","clone","--no-checkout","https://github.com/"+PINS[name][0]+".git",source])
            self.command(name+"_checkout",["git","-C",source,"checkout","--detach",PINS[name][1]])
        check_source(source,name)
        return source

    def hardware(self):
        self.torch,self.device=device_check(self.args.device)
        visible=self.env.get("CUDA_VISIBLE_DEVICES")
        self.aqua_visible_device=visible.split(",")[self.device.index] if visible else str(self.device.index)
        base,inputs=verified_inputs(self.args.base_run.resolve())
        self.base,self.inputs=base,inputs
        self.baselines=json.loads((self.args.baseline_run/"results.json").read_text())
        if self.baselines["status"]!="completed" or self.baselines["base"]["results_sha256"]!=sha(self.args.base_run/"results.json"):
            raise ValueError("baseline controls belong to another/incomplete causal run")
        versions={p:importlib.metadata.version(p) for p in ("torch","triton","numpy","transformers")}
        if (versions!=base["software"] or self.torch.version.cuda!=base["cuda_runtime"] or
            self.torch.cuda.get_device_name(self.device)!=base["device"]):
            raise ValueError("profile on the same GPU model and root software as the completed causal run")
        for path,digest in base["protocol"]["code"].items():
            if sha(ROOT/path)!=digest:
                raise ValueError("measured core changed: "+path)
        self.torch.use_deterministic_algorithms(True)
        self.torch.backends.cuda.matmul.allow_tf32=False; self.torch.backends.cudnn.allow_tf32=False
        self.result.update(gpu_measured=True,device=base["device"],software=versions,inputs=inputs,
            base_run=str(self.args.base_run.resolve()),baseline_run=str(self.args.baseline_run.resolve()))
        for src,dest in ((self.args.base_run/"results.json","base_results.json"),
                         (self.args.baseline_run/"results.json","baseline_results.json")):
            shutil.copyfile(src,self.out/dest)
        self.save()

    def profile(self):
        import numpy as np
        from kv_compare.causal_packet import Engine,word_sha
        from kv_compare.causal_profile import profile_call
        from scripts.bench_llama_causal import timed
        e=next(e for e in self.inputs if e["document"]==self.args.document)
        words=self.torch.from_numpy(np.load(e["path"],allow_pickle=False).view(np.int16).copy()).to(self.device)
        engine=Engine(Path(self.base["state_directories"]["lite"]),self.device,
            **{k:self.base["protocol"][k] for k in ("predictor_format","values_per_stream","block")})
        rows=[]; traces=[]
        warmup,repeats=(self.base["protocol"][k] for k in ("warmup","repeats"))
        for kappa in self.args.kappas:
            original=next(r for r in self.base["rows"] if r["policy"]=="lite" and r["document"]==e["document"] and r["kappa"]==kappa)
            if kappa: engine.residual(kappa)
            packet,enc=timed(lambda:engine.encode(words,kappa=kappa),self.torch,self.device,warmup,repeats)
            if hashlib.sha256(packet.export()).hexdigest()!=original["packet_sha256"]:
                raise ValueError("paired profile control differs from the saved measured packet")
            decoded,dec=timed(lambda:engine.decode(packet),self.torch,self.device,warmup,repeats)
            if word_sha(decoded)!=original["decoded_word_sha256"]:
                raise AssertionError("paired profile receiver changed")
            row=dict(kappa=kappa,document=e["document"],encode=enc,decode=dec,
                packet_sha256=original["packet_sha256"],state_sha256=engine.state_sha256,
                original_word_sha256=e["original_word_sha256"],decoded_word_sha256=word_sha(decoded),profiles={})
            for direction,fn in (("encode",lambda:engine.encode(words,kappa=kappa)),("decode",lambda:engine.decode(packet))):
                trace=self.out/"traces"/f"lite_k{kappa:g}_{direction}.json.gz"
                value,diagnostic=profile_call(fn,engine,kappa,self.device,trace)
                if direction=="encode":
                    if hashlib.sha256(value.export()).hexdigest()!=original["packet_sha256"]:
                        raise AssertionError("instrumentation changed packet bytes")
                elif word_sha(value)!=original["decoded_word_sha256"]:
                    raise AssertionError("instrumentation changed reconstruction")
                diagnostic["trace"]=str(trace.relative_to(self.out))
                diagnostic["trace_sha256"]=sha(trace)
                diagnostic["profiled_over_unprofiled_wall"]=diagnostic["profiled_wall_ms"]/row[direction]["wall_median_ms"]
                row["profiles"][direction]=diagnostic; traces.append(trace)
            rows.append(row)
            log(f"Lite kappa={kappa:g}: paired and instrumented packet/reconstruction digests agree")
        write(self.out/"profile.json",dict(gpu_measured=self.device.type=="cuda",rows=rows,scope="diagnostic trace separate from unprofiled timing"))
        return dict(artifacts=self.artifacts(self.out/"profile.json",*traces))

    def toolchain(self):
        self.command("cuda_toolchain",[sys.executable,ROOT/"scripts/setup_qwen_cuda_toolchain.py"])
        data=json.loads((ROOT/"runs/_matched_baselines/cuda12_8/toolchain.json").read_text())
        home=Path(data["cuda_home"])
        self.env.update(CUDA_HOME=str(home),CUDA_PATH=str(home),PATH=str(home/"bin")+os.pathsep+self.env.get("PATH",""))
        write(self.out/"toolchain.json",data)
        return dict(artifacts=self.artifacts(self.out/"toolchain.json"))

    def exans_build(self):
        source=self.checkout("exans"); build=self.out/"exans_build"
        if importlib.metadata.version("nvidia-libnvcomp-cu12")!="5.3.0.16":
            raise ValueError("use the existing pinned nvidia-libnvcomp-cu12==5.3.0.16 SDK")
        spec=importlib.util.find_spec("nvidia.libnvcomp")
        sdk=Path(next(iter(spec.submodule_search_locations)))
        home=Path(self.env["CUDA_HOME"])
        self.nvcomp_runtime=(sdk/"lib64",sdk/"lib")
        header=ROOT/"native/exans_build_compat.h"
        self.command("exans_configure",["cmake","-S",source/"crates/openlake_kv_client/cuda","-B",build,
            "-DOPENLAKE_EXPANS_BUILD=ON","-DOPENLAKE_CUDA_CODEC_BUILD_TESTS=ON","-DCMAKE_CUDA_ARCHITECTURES=86",
            "-DNVCOMP_ROOT="+str(sdk),"-DCMAKE_CUDA_FLAGS=--pre-include="+str(header)])
        self.command("exans_build",["cmake","--build",build,"--target","openlake_expans_test","-j","2"])
        self.command("exans_runtime_links",["ldd",build/"openlake_expans_test"])
        self.command("exans_author_smoke",[build/"openlake_expans_test"])
        check_source(source,"exans")
        library=build/"libopenlake_expans.so"
        write(self.out/"exans_build.json",dict(commit=PINS["exans"][1],author_sources_unchanged=True,
            compatibility_header_sha256=sha(header),library=str(library),library_sha256=sha(library),
            nvcomp="5.3.0.16",gpu_author_smoke_passed=True,
            compatibility="compiler-injected overload forwards allocation-error unique_ptr to existing raw-pointer handler only"))
        return dict(artifacts=self.artifacts(self.out/"exans_build.json",library))

    def exans(self):
        import numpy as np
        from kv_compare.causal_packet import Packet,word_sha
        from kv_compare.official_exans import Exans,RAW
        from scripts.bench_llama_causal import timed
        source=self.cache/"sources/exans"; build=json.loads((self.out/"exans_build.json").read_text())
        blocks=max(math.ceil(e["values"]*2/RAW) for e in self.inputs)
        codec=Exans(source,build["library"],self.device,blocks)
        directory=self.out/"exans"; directory.mkdir(exist_ok=True)
        rows=[]; entries=[]; validations=[]
        warmup,repeats=(self.base["protocol"][k] for k in ("warmup","repeats"))
        try:
            for e in self.inputs:
                if sha(e["path"])!=e["file_sha256"]:
                    raise ValueError("matched frozen input changed before ExANS")
                words=self.torch.from_numpy(np.load(e["path"],allow_pickle=False).view(np.int16).copy()).to(self.device)
                if word_sha(words)!=e["original_word_sha256"]:
                    raise ValueError("ExANS input differs from the other matched methods")
                packet,enc=timed(lambda:codec.encode(words),self.torch,self.device,warmup,repeats)
                recovered,dec=timed(lambda:codec.decode(packet),self.torch,self.device,warmup,repeats)
                if not self.torch.equal(words,recovered):
                    raise AssertionError("official ExANS round trip differs from original words")
                wire=packet.export(); path=directory/(e["document"]+".kvc"); path.write_bytes(wire)
                restored=codec.decode(Packet.restore(wire,self.device))
                if not self.torch.equal(restored,words): raise AssertionError("serialized ExANS differs")
                target=directory/(e["document"]+".npy")
                row=dict(id="exans_"+e["document"],document=e["document"],values=e["values"],tokens=e["tokens"],
                    original_word_sha256=word_sha(words),decoded_word_sha256=word_sha(recovered),original_exact=True,
                    packet=str(path.relative_to(self.out)),packet_sha256=sha(path),packet_bytes=len(wire),
                    complete_bpv=8*len(wire)/words.numel(),saved_percent=100*(1-len(wire)/(words.numel()*2)),
                    encode=enc,decode=dec,accounting=codec.accounting(packet),decoded=str(target.relative_to(self.out)))
                rows.append(row)
                entries.append(dict(id=row["id"],packet=str(path),packet_sha256=sha(path),
                    expected_word_sha256=row["original_word_sha256"],decoded=str(target)))
                del words,packet,recovered,restored,wire
            # Correctness-only fixture: the author's uniform-exponent overflow pattern,
            # plus a partial final block. Never included in Llama performance rows.
            i=self.torch.arange(RAW//2+3,dtype=self.torch.int32,device=self.device)
            fixture=(((i%256)<<7)|(i&127)).to(self.torch.int16)
            packet=codec.encode(fixture); recovered=codec.decode(packet)
            if not self.torch.equal(recovered,fixture) or codec.accounting(packet)["raw_fallback_blocks"]<1:
                raise AssertionError("ExANS raw-overflow/tail transport validation failed")
            path=directory/"overflow_tail_validation.kvc"; path.write_bytes(packet.export())
            target=directory/"overflow_tail_validation.npy"
            validations.append(dict(id="overflow_tail_validation",correctness_only=True,
                original_word_sha256=word_sha(fixture),packet=str(path.relative_to(self.out)),packet_sha256=sha(path),
                decoded=str(target.relative_to(self.out)),accounting=codec.accounting(packet)))
            entries.append(dict(id="overflow_tail_validation",packet=str(path),packet_sha256=sha(path),
                expected_word_sha256=word_sha(fixture),decoded=str(target)))
            del i,fixture,packet,recovered
        finally:
            codec.close()
        del codec; gc.collect(); self.torch.cuda.empty_cache()
        q=dict(source=str(source),library=build["library"],library_sha256=build["library_sha256"],
            blocks=blocks,entries=entries,output=str(self.out/"exans_receiver.json"))
        write(self.out/"exans_receiver_request.json",q)
        self.command("exans_receiver",[sys.executable,Path(__file__).resolve(),"--device",str(self.device),
            "--receive",self.out/"exans_receiver_request.json"])
        proof=json.loads((self.out/"exans_receiver.json").read_text())
        checks={c["id"]:c for c in proof["checks"]}
        for row in rows:
            row.update(status="verified",decoded_file_sha256=checks[row["id"]]["decoded_file_sha256"])
        for v in validations:
            v.update(status="verified",decoded_file_sha256=checks[v["id"]]["decoded_file_sha256"])
        write(self.out/"exans_validation.json",validations)
        write(self.out/"exans.json",dict(rows=rows,gpu_measured=True,official_commit=PINS["exans"][1],
            scope="official ExANS native codec plus local complete fixed-record transport harness; not original vLLM evaluation"))
        return dict(artifacts=self.artifacts(self.out/"exans.json",self.out/"exans_receiver.json",self.out/"exans_validation.json",
            *[self.out/r[k] for r in rows+validations for k in ("packet","decoded")]))

    def aqua_setup(self):
        source=self.checkout("aqua"); py=self.args.aqua_python.absolute()
        if not py.is_file():
            raise ValueError("existing official AQUA environment missing: "+str(py))
        constraints=self.out/"aqua_constraints.txt"
        constraints.write_text("torch==2.5.1\ntransformers==4.46.3\ndatasets==2.20.0\nnumpy==1.26.4\npyarrow==20.0.0\n")
        self.command("aqua_accelerate",[py,"-m","pip","install","--no-input","-c",constraints,"accelerate==1.1.1"])
        self.command("aqua_pip_check",[py,"-m","pip","check"])
        script=("import importlib.metadata,json,torch,accelerate; from transformers.utils import is_accelerate_available; "
            "wanted={'torch':'2.5.1','transformers':'4.46.3','datasets':'2.20.0','numpy':'1.26.4','pyarrow':'20.0.0','accelerate':'1.1.1'}; "
            "got={p:importlib.metadata.version(p).split('+')[0] for p in wanted}; assert got==wanted,(got,wanted); "
            "assert torch.cuda.is_available() and is_accelerate_available(); "
            "from fast_hadamard_transform import hadamard_transform; "
            "x=hadamard_transform(torch.ones(2,1024,device='cuda')); torch.cuda.synchronize(); "
            "print(json.dumps({'packages':got,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))")
        path=self.command("aqua_preflight",[py,"-c",script],source)
        write(self.out/"aqua_environment.json",dict(commit=PINS["aqua"][1],python=str(py),author_sources_unchanged=True,
            missing_loader_dependency="accelerate==1.1.1",preflight=path.read_text()))
        return dict(artifacts=self.artifacts(self.out/"aqua_environment.json"))

    def aqua_common(self):
        from scripts.bench_llama_causal import cached_model_snapshot
        snapshot=cached_model_snapshot()
        return ["--model_name",snapshot,"--model_seqlen","8192","--edenn_d","2","--edenn_n","16","--hadamard_groupsize","1024"]

    def aqua_verify(self):
        source=self.cache/"sources/aqua"; check_source(source,"aqua")
        script=("import torch,importlib.metadata; from transformers.utils import is_accelerate_available; "
            "wanted={'torch':'2.5.1','transformers':'4.46.3','datasets':'2.20.0','numpy':'1.26.4','accelerate':'1.1.1'}; "
            "got={p:importlib.metadata.version(p).split('+')[0] for p in wanted}; assert got==wanted,(got,wanted); "
            "assert torch.cuda.is_available() and is_accelerate_available(); print(got)")
        self.command("aqua_verify_runtime",[self.args.aqua_python.absolute(),"-c",script],source)

    def aqua_data(self,training=False):
        # Cache/identify the same author's datasets before running unchanged entry points.
        snapshot=str(self.aqua_common()[1]); output=self.out/("aqua_train_data.json" if training else "aqua_test_data.json")
        script=("import json,sys,hashlib; from datasets import load_dataset; from transformers import AutoTokenizer; "
            "train=sys.argv[3]=='train'; ds=load_dataset('togethercomputer/RedPajama-Data-1T-Sample' if train else 'wikitext', "
            "**({'split':'train'} if train else {'name':'wikitext-2-raw-v1','split':'test'})); "
            "d={'dataset_fingerprint':ds._fingerprint,'rows':len(ds),'training':train}; "
            "text=None if train else '\\n\\n'.join(ds['text']); "
            "ids=None if train else AutoTokenizer.from_pretrained(sys.argv[1],padding_side='left')(text,return_tensors='pt')['input_ids']; "
            "d.update({} if train else {'text_sha256':hashlib.sha256(text.encode()).hexdigest(), "
            "'token_ids_sha256':hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest(),'tokens':ids.numel()}); "
            "open(sys.argv[2],'w').write(json.dumps(d,indent=2)+'\\n')")
        self.command("aqua_identify_train" if training else "aqua_identify_test",
            [self.args.aqua_python.absolute(),"-c",script,snapshot,output,"train" if training else "test"])
        return json.loads(output.read_text())

    def aqua_train(self):
        source=self.cache/"sources/aqua"; check_source(source,"aqua")
        self.aqua_verify()
        backup=getattr(self.args,"aqua_calibration","original")=="backup"
        data="pajama"; calibration=[]
        if backup:
            tokens=self.out/"aqua_calibration_tokens.pt"; receipt=self.out/"aqua_calibration_backup.json"
            self.command("aqua_prepare_backup",[self.args.aqua_python.absolute(),ROOT/"scripts/prepare_aqua_calibration.py",
                "--source",source,"--model-path",self.aqua_common()[1],"--output",tokens,"--receipt",receipt,
                "--cache",self.cache/"calibration_backup_cache"])
            data=str(tokens); calibration=[tokens,receipt]
        else:
            self.aqua_data(training=True)
            calibration=[self.out/"aqua_train_data.json"]
        target=self.out/"aqua_predictors.pt"
        self.command("aqua_train_official",[self.args.aqua_python.absolute(),source/"train_predictors.py",*self.aqua_common(),
            "--dataset",data,"--total_nsamples","256","--valid_nsamples","32","--offload_activations",
            "--predictors_output_path",target],source)
        check_source(source,"aqua")
        return dict(artifacts=self.artifacts(target,*calibration),calibration_source="pinned backup" if backup else "original",
            scope="unchanged author training; 256x8192, holdout32, HIGGS2/first-layer4; "+("explicit backup calibration, original equivalence unverified" if backup else "original RedPajama"))

    def aqua_eval(self,quantized):
        source=self.cache/"sources/aqua"; check_source(source,"aqua")
        self.aqua_verify(); data=self.aqua_data()
        reference=self.out/"aqua_ppl_raw_official.json"
        if quantized and reference.exists() and data!=json.loads(reference.read_text())["test_data"]:
            raise ValueError("official AQUA raw/quantized WikiText/token identities differ")
        name="aqua_ppl_higgs2_official" if quantized else "aqua_ppl_raw_official"
        command=[self.args.aqua_python.absolute(),source/"evaluate_perplexity.py",*self.aqua_common(),
            "--prefix_size","4","--recent_buffer_size","128"]
        command += ["--predictors_input_path",self.out/"aqua_predictors.pt"] if quantized else ["--no_quant"]
        # Both official PPL commands now consume the verified local dataset/model cache.
        env=dict(self.env,HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1")
        path=self.command(name,command,source,env=env)
        value=ppl(path); check_source(source,"aqua")
        write(self.out/(name+".json"),dict(perplexity=value,
            end_to_end_command_wall_seconds=self.result["commands"][-1]["wall_seconds"],
            scope="official full WikiText2 evaluation including model/cache work and startup, not transfer codec time",
            test_data=data,physical_cache_bytes="unmeasured",nominal_quantizer="HIGGS2 residual / HIGGS4 first layer" if quantized else "BF16"))
        return dict(artifacts=self.artifacts(path,self.out/(name+".json")),perplexity=value)

    def execute(self):
        self.hardware()
        if "profile" in self.args.stages: self.step("profile",self.profile)
        if set(self.args.stages)&{"exans","aqua"}:
            # Refresh env variables even when build tools are a completed checkpoint.
            tools=self.step("toolchain",self.toolchain)
            if tools:
                data=json.loads((self.out/"toolchain.json").read_text()); home=Path(data["cuda_home"])
                self.env.update(CUDA_HOME=str(home),CUDA_PATH=str(home),PATH=str(home/"bin")+os.pathsep+self.env.get("PATH",""))
        else: tools=False
        if "exans" in self.args.stages and tools and self.step("exans_build",self.exans_build):
            self.step("exans",self.exans)
        if "aqua" in self.args.stages and tools and self.step("aqua_setup",self.aqua_setup):
            raw=self.step("aqua_raw",lambda:self.aqua_eval(False))
            if self.step("aqua_train",self.aqua_train): self.step("aqua",lambda:self.aqua_eval(True))
        required=[s for s in self.args.stages]
        if "aqua" in required: required += ["aqua_raw"]
        self.result["status"]="completed" if all(self.result["stages"].get(s,{}).get("status")=="completed" for s in required) else "partial"
        self.save()

    def report(self):
        r=self.result
        lines=["# Lite profiling and official baseline follow-up","",f"Status: **{r['status']}**. Actual GPU checked: **{r['gpu_measured']}**.","",
            "Existing fits and measured results are preserved. Profile diagnostics are separate from unprofiled timing.","",
            "| Stage | Status | Detail |","|---|---|---|"]
        for name,e in r["stages"].items():
            lines.append(f"| {name} | {e['status']} | {e.get('reason',e.get('scope','')).replace('|','/')} |")
        if r.get("recipe",{}).get("aqua_calibration")=="backup":
            lines += ["","AQUA calibration explicitly uses a pinned third-party RedPajama backup. The author code and sampling settings are preserved; equivalence to the removed original dataset is unverified."]
        prior=self.out/"baseline_results.json"
        if prior.exists():
            saved=json.loads(prior.read_text()); groups={}
            for row in saved["rows"]+saved["reused_rows"]:
                label=row.get("method",row["policy"])+(f" / {row['kappa']:g}" if row.get("kappa") else "")
                groups.setdefault(label,[]).append(row)
            exans_path=self.out/"exans.json"
            if exans_path.exists(): groups["official ExANS, fixed framed"]=json.loads(exans_path.read_text())["rows"]
            lines += ["","Same frozen TEST inputs; existing measurements reused, ExANS added. Means of per-request warm wall medians; complete framed bytes.","",
                "| Method / step | Complete bpv | Saved | Encode ms | Decode ms | Exact |","|---|---:|---:|---:|---:|---|"]
            for label,group in groups.items():
                avg=lambda fn:sum(fn(e) for e in group)/len(group)
                lines.append(f"| {label} | {avg(lambda e:e['complete_bpv']):.5f} | {avg(lambda e:e['saved_percent']):.2f}% | "
                    f"{avg(lambda e:e['encode']['wall_median_ms']):.3f} | {avg(lambda e:e['decode']['wall_median_ms']):.3f} | {all(e['original_exact'] for e in group)} |")
        p=self.out/"profile.json"
        if p.exists():
            lines += ["","| Lite step | Direction | Ordinary ms | Profiled ms | CUDA events |","|---:|---|---:|---:|---:|"]
            data=json.loads(p.read_text())["rows"]
            for row in data:
                for direction,d in row["profiles"].items():
                    lines.append(f"| {row['kappa']:g} | {direction} | {row[direction]['wall_median_ms']:.3f} | {d['profiled_wall_ms']:.3f} | {d['cuda_device_event_count']} |")
            for row in data:
                for direction,d in row["profiles"].items():
                    lines += ["",f"Top self-device operators, Lite {row['kappa']:g} {direction}:",""]
                    # CUDA user-annotation spans include idle time and are not kernels.
                    top=[o for o in d["top_self_device"] if not o["name"].startswith("kv/")]
                    lines += [f"- {o['name']}: {o['self_device_ms']:.3f} ms across {o['calls']} calls" for o in top[:5]]
        if (self.out/"exans.json").exists():
            lines += ["","| ExANS document | Complete bpv | Saved | Encode ms | Decode ms | Raw blocks | Exact |","|---|---:|---:|---:|---:|---:|---|"]
            for e in json.loads((self.out/"exans.json").read_text())["rows"]:
                lines.append(f"| {e['document']} | {e['complete_bpv']:.5f} | {e['saved_percent']:.2f}% | {e['encode']['wall_median_ms']:.3f} | {e['decode']['wall_median_ms']:.3f} | {e['accounting']['raw_fallback_blocks']} | {e['original_exact']} |")
        if r["stages"].get("aqua",{}).get("status")=="completed" and r["stages"].get("aqua_raw",{}).get("status")=="completed":
            raw=r["stages"]["aqua_raw"]["perplexity"]; quant=r["stages"]["aqua"]["perplexity"]
            lines += ["",f"Official WikiText2 PPL: BF16 **{raw:.5f}**, AQUA HIGGS2 **{quant:.5f}** ({100*(quant/raw-1):+.3f}%)."]
        lines += ["","ExANS: unchanged pinned author kernels and native C ABI; documented compiler-injected allocation-error overload. Physical fixed slots, input padding and local raw overflow fallback are charged. Native allocator peaks are unmeasured.",
            "AQUA: unchanged author train/evaluation; calibration source is recorded explicitly and context/buffers are preserved. Its command wall time includes model inference and is not compared numerically with transfer codec latency. Physical AQUA wire bytes are unmeasured.",
            "Profiler scopes are nested/inclusive, not additive; GPU operator self-times and CUDA events are diagnostics. Ordinary wall samples are the performance control.",
            "No CPU-estimated GPU performance, pool attention, network/SSD, T-to-P or serving speedup is claimed."]
        if r.get("error"): lines += ["","Failure: "+r["error"]]
        (self.out/"summary.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-run",type=Path,default=ROOT/"runs/llama_causal_a4500_20261027_184615_1307629")
    p.add_argument("--baseline-run",type=Path,default=ROOT/"runs/llama_baselines_a4500_20261027_191350_1897426")
    p.add_argument("--output",type=Path); p.add_argument("--device",default="cuda:0")
    p.add_argument("--stages",default="profile,exans,aqua")
    p.add_argument("--document",default="test_text"); p.add_argument("--kappas",default="0,1")
    p.add_argument("--aqua-python",type=Path,default=ROOT/"runs/_llama_official/envs/aqua/bin/python")
    p.add_argument("--aqua-calibration",choices=["original","backup"],default="original",
        help="original keeps author Hub data; backup explicitly opts into pinned RedPajama backup calibration")
    p.add_argument("--receive",type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.receive:
        torch,device=device_check(args.device); receive(args.receive,torch,device); return 0
    if not args.output: p.error("--output is required")
    args.stages=args.stages.split(","); args.kappas=[float(k) for k in args.kappas.split(",")]
    if not args.stages or set(args.stages)-{"profile","exans","aqua"} or len(set(args.stages))!=len(args.stages):
        p.error("invalid stages")
    if not args.kappas or len(set(args.kappas))!=len(args.kappas) or any(k not in (0,.5,1) for k in args.kappas):
        p.error("kappas must select existing modes 0,0.5,1")
    suite=Suite(args)
    try:
        suite.execute()
    except Exception as error:
        suite.result.update(status="failed",error=str(error),traceback=traceback.format_exc()); log(suite.result["traceback"]); suite.save()
    return 0 if suite.result["status"]=="completed" else 1


if __name__=="__main__": sys.exit(main())

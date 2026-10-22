#!/usr/bin/env python3
"""Run unchanged official baselines and our codec on pinned BF16 Llama-3.2-3B.

Run with the existing project .venv. Baseline frameworks are installed into
separate environments under runs/. No baseline or dependency is monkey-patched.
Blocked methods produce logs and missing metrics, never replacement results.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import time
import traceback
import venv
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.llama_official_worker import MODEL, REVISION, write, sha

SOURCES={
    'aqua':('goodevening13/aquakv','e744183f2f3a4a90381f20a6975456871fd1ae29'),
    'splitzip':('Intelligent-Microsystems-Lab/SplitZip','42c743118a29db3270ae6a921a4a4eecb20cad04'),
    'exans':('openlake-project/openlake','de6c38bcc5438475bd1abf1e9b3d154cef414ef6'),
    'hadamard':('Dao-AILab/fast-hadamard-transform','e7706faf8d1c3b9f241e36860640ad1dac644ede')}
SOURCE_FILES=['scripts/run_llama_official_a4500.py','scripts/llama_official_worker.py',
    'qwen_kv_bench/transfer_reference.py','qwen_kv_bench/transfer_probability.py',
    'qwen_kv_bench/transfer_t1.py','qwen_kv_bench/transfer_t1_kernels.py',
    'qwen_kv_bench/transfer_gpu_predictor.py','scripts/setup_qwen_cuda_toolchain.py',
    'qwen_kv_bench/matched_transfer.py','qwen_kv_bench/model_runtime.py',
    'scripts/bench_qwen_matched_gpu.py']
AQUA_RUNTIME={'accelerate':'1.1.1'}


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def check_sources(directory, commit):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=directory,text=True)
    if head!=commit or dirty: raise RuntimeError('Author source commit changed or tracked files modified: '+str(directory))


def check_model_config(config):
    if (config.get('model_type')!='llama' or config.get('num_hidden_layers')!=28 or
            config.get('num_key_value_heads')!=8 or config.get('head_dim')!=128 or
            config.get('torch_dtype')!='bfloat16' or config.get('quantization_config')):
        raise ValueError('Expected the pinned unquantized BF16 Llama-3.2-3B configuration')


def same_tokens(a,b):
    keys=['text_sha256','token_ids_sha256','tokens']
    if any(a[k]!=b[k] for k in keys): raise ValueError('AQUA and codec evaluation text/token IDs differ')


class Run:
    def __init__(self, output, stages):
        self.out=output.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.cache=ROOT/'runs/_llama_official'; self.cache.mkdir(parents=True,exist_ok=True)
        self.env=dict(os.environ); self.env.pop('TRITON_INTERPRET',None)
        self.env.update(OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.6')
        self.logs=self.out/'logs';self.logs.mkdir(exist_ok=True)
        self.sources={name:self.cache/'sources'/name for name in SOURCES}
        self.python={name:self.cache/'envs'/name/'bin/python' for name in ['aqua','splitzip']}
        self.snapshot=None
        identity={p:sha(ROOT/p) for p in SOURCE_FILES}
        self.result=dict(format='llama32_official_gpu_v1',status='running',gpu_measured=False,
            created=now(),model=MODEL,revision=REVISION,layers=28,requested=stages,stages={},
            protocol=dict(aqua='official HIGGS-2, first layer HIGGS-4; 256x8192 pajama, holdout 32; WikiText2 PPL, prefix4/recent128',
                splitzip='official benchmark; chunk1024, warmup10, iters50, repeats5; saved original BF16 input; TRAIN calibration only',
                ours='Block32 A, fresh TRAIN fit; all 28 layers; first/last1024 of first8192 WikiText2 TEST tokens',
                exans='unchanged official CMake build/test only; no invented Qwen/Llama performance wrapper'),
            source_files=identity,scope='Official-code new-model evaluation; not reproduction of every paper setting or a serving benchmark')
        state=self.out/'results.json'
        if state.exists():
            old=json.loads(state.read_text())
            if any(old.get(k)!=self.result[k] for k in ['format','model','revision','requested','source_files','protocol']):
                raise ValueError('Resume identity differs; use a new output directory')
            self.result=old;self.result['status']='running'
        self.save()

    def command(self, name, argv, cwd=None):
        path=self.logs/(name+'.log'); attempt=1
        while path.exists(): path=self.logs/(name+f'.attempt{attempt}.log');attempt+=1
        argv=list(map(str,argv));print(f'[{now()}] {name}: log {path}',flush=True)
        self.result.setdefault('commands',[]).append(dict(name=name,argv=argv,cwd=str(cwd or ROOT),log=str(path.relative_to(self.out))))
        self.save()
        with path.open('wb') as log:
            proc=subprocess.Popen(argv,cwd=cwd or ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while True:
                    try: code=proc.wait(timeout=30);break
                    except subprocess.TimeoutExpired: print(f'[{now()}] {name}: running; {path}',flush=True)
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        if code: raise RuntimeError(f'{name} exited {code}; see {path.relative_to(self.out)}')
        print(f'[{now()}] {name}: passed',flush=True)
        return path

    def stage(self, name, action, dependencies=()):
        for dep in dependencies:
            if self.result['stages'].get(dep,{}).get('status')!='completed':
                self.result['stages'][name]=dict(status='blocked',reason='Dependency did not complete: '+dep)
                self.save();return
        old=self.result['stages'].get(name,{})
        if old.get('status') in {'completed','smoke_only'}:
            for rel,digest in old.get('artifacts',{}).items():
                if sha(self.out/rel)!=digest: raise ValueError('Completed artifact changed: '+rel)
            print(f'[{now()}] {name}: verified saved completion',flush=True);return
        self.result['stages'][name]=dict(status='running',started=now());self.save()
        try:
            info=action() or {}
            status=info.pop('status','completed')
            self.result['stages'][name].update(status=status,completed=now(),**info)
        except Exception as e:
            status='blocked' if name in {'hardware','model','toolchain','splitzip_env','aqua_env','exans'} else 'failed'
            self.result['stages'][name].update(status=status,completed=now(),reason=str(e),traceback=traceback.format_exc())
            print(f'[{now()}] {name}: {status.upper()}: {e}',flush=True)
        self.save()

    def save(self):
        write(self.out/'results.json',self.result)
        lines=['# Llama-3.2-3B official GPU evaluation','',f"Status: **{self.result['status']}**. GPU measured: **{self.result['gpu_measured']}**.",'',
            f'Model: `{MODEL}` at `{REVISION}`; original BF16, all 28 layers.','',
            '| Stage | Status | Detail |','|---|---|---|']
        for name,info in self.result['stages'].items():
            detail=info.get('reason',info.get('scope','')).replace('|','/').replace('\n',' ')
            lines.append(f"| {name} | {info['status']} | {detail} |")
        for name,info in self.result['stages'].items():
            if name=='aqua' and info.get('status')=='completed':
                lines+=['',f"Official WikiText2 PPL: BF16 **{info['raw_perplexity']:.6g}**; AQUA HIGGS-2 **{info['aqua_perplexity']:.6g}**.",
                    'HIGGS bit widths are configuration labels. Actual packed bytes, resident memory and transfer-codec throughput are unmeasured by this entry point.']
        native=[p for p in sorted(self.out.glob('splitzip_case_*.json')) if self.result['stages'].get('splitzip',{}).get('status')=='completed']
        if native:
            lines+=['','Official SplitZip native codec results:','','| Case | Native bpv | Saved | Encode GB/s | Decode GB/s |',
                '|---|---:|---:|---:|---:|']
            for path in native:
                d=json.loads(path.read_text())
                lines.append(f"| {path.stem} | {8*d['compressed_bytes']/d['numel']:.5f} | {100*(1-d['compressed_bytes']/d['raw_bytes']):.2f}% | {d['encode_gbs']['mean']:.3f} | {d['decode_gbs']['mean']:.3f} |")
        ours=self.out/'ours/results.json'
        if ours.is_file() and self.result['stages'].get('ours',{}).get('status')=='completed':
            lines+=['','Our transfer codec results (separate packet/timing scope):','',
                '| Method | Start | Payload bpv | Complete bpv | Encode wall ms | Decode wall ms | Exact |',
                '|---|---:|---:|---:|---:|---:|---|']
            for d in json.loads(ours.read_text())['rows']:
                lines.append(f"| {d['method']} | {d['start']} | {8*d['bytes']['payload_bytes']/d['values']:.5f} | {8*d['bytes']['complete_bytes']/d['values']:.5f} | {d['encode']['wall_median_ms']:.3f} | {d['decode']['wall_median_ms']:.3f} | {d['exact']} |")
        lines+=['','Official SplitZip results are native codec metrics. Our results include separate payload and framed bytes, synchronized wall samples and CUDA intervals.',
            'AQUA PPL, native codec throughput and our packet latency have different scopes; no cross-scope speed ranking is produced.',
            'Blocked stages have no substituted, synthetic, CPU-estimated or adapted performance results.',
            'This base-model test does not measure LongBench, compressed attention, T-to-P conversion, SSD/network or serving throughput.']
        (self.out/'summary.md').write_text('\n'.join(lines)+'\n')

    def artifacts(self,*paths): return {str(p.relative_to(self.out)):sha(p) for p in paths}

    def hardware(self):
        from scripts.llama_official_worker import cuda
        torch=cuda()
        data=dict(actual_cuda=True,name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),
            total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch=str(torch.__version__),cuda=torch.version.cuda)
        write(self.out/'hardware.json',data)
        return dict(artifacts=self.artifacts(self.out/'hardware.json'),scope='actual CUDA required before downloading model weights or installing baseline environments')

    def checkout(self,name):
        repo,commit=SOURCES[name]; path=self.sources[name]
        if not (path/'.git').exists():
            path.mkdir(parents=True,exist_ok=True)
            self.command(name+'_init',['git','init',path])
        remote=subprocess.run(['git','config','--get','remote.origin.url'],cwd=path,text=True,capture_output=True)
        if remote.returncode:
            self.command(name+'_remote',['git','remote','add','origin','https://github.com/'+repo+'.git'],path)
        elif remote.stdout.strip()!='https://github.com/'+repo+'.git':
            raise ValueError('Unexpected upstream remote: '+name)
        head=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=path,text=True,capture_output=True)
        if head.returncode:
            self.command(name+'_fetch',['git','fetch','--depth','1','origin',commit],path)
            self.command(name+'_checkout',['git','checkout','--detach','FETCH_HEAD'],path)
        check_sources(path,commit)
        return path

    def toolchain(self):
        self.command('cuda_tools',[sys.executable,ROOT/'scripts/setup_qwen_cuda_toolchain.py'])
        data=json.loads((ROOT/'runs/_matched_baselines/cuda12_8/toolchain.json').read_text())
        root=Path(data['cuda_home']);self.env['CUDA_HOME']=str(root);self.env['CUDA_PATH']=str(root)
        self.env['PATH']=str(root/'bin')+os.pathsep+self.env.get('PATH','')
        write(self.out/'toolchain.json',data)
        return dict(artifacts=self.artifacts(self.out/'toolchain.json'))

    def model(self):
        from huggingface_hub import snapshot_download
        snapshot=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=[
            '*.json','*.safetensors','*.model','*.txt','*.tiktoken','*.jinja']))
        config=json.loads((snapshot/'config.json').read_text());check_model_config(config)
        write(self.out/'model_manifest.json',dict(model=MODEL,revision=REVISION,snapshot=str(snapshot),config=config,
            config_sha256=sha(snapshot/'config.json'),checkpoint='unquantized checkpoint from official AQUA README'))
        self.snapshot=snapshot
        return dict(artifacts=self.artifacts(self.out/'model_manifest.json'))

    def setup_env(self,name):
        src=self.checkout(name);py=self.python[name];envroot=py.parent.parent
        stamp=envroot/'official_profile.json'
        original_profile=dict(source=SOURCES[name][1],requirements_sha256=sha(src/'requirements.txt'),
            extra_constraints={'pyarrow':'20.0.0','numpy':'1.26.4'},
            hadamard=SOURCES['hadamard'][1] if name=='aqua' else None)
        profile=dict(original_profile,runtime_requirements=AQUA_RUNTIME) if name=='aqua' else original_profile
        if stamp.exists():
            old=json.loads(stamp.read_text())
            # The only accepted migration adds the missing loader dependency.
            # Author sources, framework versions and all other constraints stay pinned.
            if old!=profile and not (name=='aqua' and old==original_profile):
                raise ValueError('Environment profile changed')
            if name=='aqua': self.ensure_aqua_runtime(py,envroot)
            self.command(name+'_pip_check',[py,'-m','pip','check'])
            self.verify_env(name)
            return self.record_environment(name,profile,stamp)
        if not py.exists(): venv.EnvBuilder(with_pip=True).create(envroot)
        constraints=envroot/'constraints.txt';constraints.write_text('numpy==1.26.4\npyarrow==20.0.0\n')
        self.command(name+'_build_prerequisites',[py,'-m','pip','install','packaging','wheel','ninja'])
        if name=='aqua':
            self.command(name+'_torch',[py,'-m','pip','install','torch==2.5.1','numpy==1.26.4'])
        self.command(name+'_requirements',[py,'-m','pip','install','--no-build-isolation','-c',constraints,'-r',src/'requirements.txt'])
        if name=='aqua':
            self.ensure_aqua_runtime(py,envroot)
            had=self.checkout('hadamard')
            self.command('hadamard_install',[py,'-m','pip','install','--no-build-isolation',had])
            check_sources(had,SOURCES['hadamard'][1])
            self.command('aqua_import',[py,'train_predictors.py','--help'],src)
        else:
            self.command('ours_dependencies',[py,'-m','pip','install','constriction==0.5.0'])
        self.command(name+'_pip_check',[py,'-m','pip','check'])
        self.verify_env(name)
        return self.record_environment(name,profile,stamp)

    def ensure_aqua_runtime(self,py,envroot):
        # Upstream calls from_pretrained(low_cpu_mem_usage=True) but omits
        # Accelerate from requirements.txt. Install it without relaxing author pins.
        self.verify_env('aqua',include_runtime=False)
        constraints=envroot/'loader_constraints.txt'
        constraints.write_text('torch==2.5.1\ntransformers==4.46.3\ndatasets==2.20.0\nnumpy==1.26.4\npyarrow==20.0.0\n')
        self.command('aqua_runtime_install',[py,'-m','pip','install','--no-input','-c',constraints,
            '--report',self.out/'aqua_runtime_install.json',
            *[f'{name}=={version}' for name,version in AQUA_RUNTIME.items()]])

    def record_environment(self,name,profile,stamp):
        script=("import importlib.metadata,json; "
            "print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},sort_keys=True))")
        log=self.command(name+'_environment',[self.python[name],'-c',script])
        path=self.out/(name+'_environment.json')
        write(path,dict(profile=profile,installed=json.loads(log.read_text())))
        write(stamp,profile)
        return dict(artifacts=self.artifacts(path))

    def preflight(self,name):
        self.verify_env(name)
        path=self.out/(name+'_preflight.json')
        self.command(name+'_gpu_preflight',[self.python[name],ROOT/'scripts/llama_official_worker.py',
            'preflight','--manifest',self.out/'model_manifest.json','--output',path])
        return dict(artifacts=self.artifacts(path),scope='actual CUDA kernel and full WikiText2 text/token identity; no codec performance claim')

    def verify_env(self,name,*,include_runtime=True):
        check_sources(self.sources[name],SOURCES[name][1])
        # Recheck pinned direct dependencies when resuming a partially run batch.
        script=("import importlib.metadata,json,sys; wanted=json.loads(sys.argv[1]); "
                "got={p:importlib.metadata.version(p).split('+')[0] for p in wanted}; "
                "assert got==wanted,(got,wanted)")
        wanted={'torch':'2.5.1','transformers':'4.46.3','datasets':'2.20.0'} if name=='aqua' else {
            'torch':'2.9.0','transformers':'4.57.1','triton':'3.5.0','datasets':'2.19.2'}
        if name=='aqua' and include_runtime:
            wanted.update(AQUA_RUNTIME)
            script+=("; import accelerate; from transformers.utils import is_accelerate_available; "
                "assert is_accelerate_available(), 'Transformers cannot use Accelerate'")
        self.command(name+'_verify_versions',[self.python[name],'-c',script,json.dumps(wanted)])

    def capture(self):
        self.verify_env('splitzip')
        path=self.out/'captures'
        self.command('capture_original_llama',[self.python['splitzip'],ROOT/'scripts/llama_official_worker.py',
            'capture','--manifest',self.out/'model_manifest.json','--output',path])
        data=json.loads((path/'manifest.json').read_text())
        same_tokens(data['datasets']['test'],json.loads((self.out/'splitzip_preflight.json').read_text())['test_data'])
        return dict(artifacts=self.artifacts(path/'manifest.json'),scope='all 28 original BF16 layers; TRAIN/validation/TEST separated')

    def splitzip(self):
        self.verify_env('splitzip')
        src=self.sources['splitzip'];check_sources(src,SOURCES['splitzip'][1])
        m=json.loads((self.out/'captures/manifest.json').read_text());directory=self.out/'captures';outputs=[]
        calibration=directory/m['calibration_input']['path']
        if sha(calibration)!=m['calibration_input']['file_sha256']: raise ValueError('TRAIN input changed')
        for i,case in enumerate(m['benchmark_inputs']):
            path=directory/case['path']
            if sha(path)!=case['file_sha256']: raise ValueError('TEST input changed')
            output=self.out/f'splitzip_case_{i}.json'
            self.command(f'splitzip_case_{i}',[self.python['splitzip'],src/'bench_codec_throughput.py',
                '--device','cuda:0','--input',path,'--calibration-input',calibration,'--numel',case['values'],
                '--chunk-size','1024','--warmup','10','--iters','50','--repeats','5','--output',output],src)
            data=json.loads(output.read_text())
            if data['numel']!=case['values'] or data['raw_bytes']!=case['values']*2: raise ValueError('Native benchmark truncated TEST')
            if data['calibration'].get('calibration_source')=='benchmark_input': raise ValueError('TEST calibration forbidden')
            outputs.append(output)
        check_sources(src,SOURCES['splitzip'][1]);self.result['gpu_measured']=True
        return dict(artifacts=self.artifacts(*outputs),scope='unchanged author benchmark; two identical-to-ours TEST word slices; native codec byte/timing scope')

    def ours(self):
        self.verify_env('splitzip')
        output=self.out/'ours'
        self.command('ours_full_model_gpu',[self.python['splitzip'],ROOT/'scripts/llama_official_worker.py',
            'ours','--manifest',self.out/'model_manifest.json','--captures',self.out/'captures','--output',output])
        self.result['gpu_measured']=True
        return dict(artifacts=self.artifacts(output/'results.json'),scope='fresh Block32 A fit and exact packet restore, all 28 layers; no AQUA adapter')

    def aqua(self):
        self.verify_env('aqua')
        src=self.sources['aqua'];check_sources(src,SOURCES['aqua'][1]);py=self.python['aqua']
        a=json.loads((self.out/'aqua_preflight.json').read_text())
        b=json.loads((self.out/'splitzip_preflight.json').read_text());same_tokens(a['test_data'],b['test_data'])
        common=['--model_name',self.snapshot,'--model_seqlen','8192','--edenn_d','2','--edenn_n','16','--hadamard_groupsize','1024']
        # Keep paper/README calibration intact. No smaller calibration fallback.
        predictors=self.out/'aqua_predictors.pt'
        self.command('aqua_train_official',[py,src/'train_predictors.py',*common,
            '--dataset','pajama','--total_nsamples','256','--valid_nsamples','32',
            '--offload_activations','--predictors_output_path',predictors],src)
        inference=['--prefix_size','4','--recent_buffer_size','128']
        offline={k:self.env.get(k) for k in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE']}
        self.env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        try:
            raw=self.command('aqua_ppl_raw_official',[py,src/'evaluate_perplexity.py',*common,*inference,'--no_quant'],src)
            quant=self.command('aqua_ppl_higgs2_official',[py,src/'evaluate_perplexity.py',*common,*inference,
                '--predictors_input_path',predictors],src)
        finally:
            for key,value in offline.items():
                if value is None:self.env.pop(key,None)
                else:self.env[key]=value
        def perplexity(path):
            values=re.findall(r'WikiText-2 perplexity:\s*([0-9.eE+-]+)',path.read_text())
            if len(values)!=1 or not 0<float(values[0])<float('inf'): raise ValueError('Missing finite official PPL output')
            return float(values[0])
        check_sources(src,SOURCES['aqua'][1]);self.result['gpu_measured']=True
        return dict(raw_perplexity=perplexity(raw),aqua_perplexity=perplexity(quant),
            artifacts=self.artifacts(raw,quant),scope='unchanged official HIGGS calibration and full WikiText2 PPL; nominal residual2/first-layer4; physical codec bytes unmeasured')

    def exans(self):
        src=self.checkout('exans');build=self.out/'exans_build'
        # Use the installed pinned nvCOMP SDK package without rewriting CMake.
        import importlib.util
        spec=importlib.util.find_spec('nvidia.libnvcomp')
        if spec is None: raise RuntimeError('Native nvCOMP SDK missing in the project .venv')
        sdk=Path(next(iter(spec.submodule_search_locations)))
        self.command('exans_configure_official',['cmake','-S',src/'crates/openlake_kv_client/cuda','-B',build,
            '-DOPENLAKE_EXPANS_BUILD=ON','-DOPENLAKE_CUDA_CODEC_BUILD_TESTS=ON',
            '-DCMAKE_CUDA_ARCHITECTURES=86','-DNVCOMP_ROOT='+str(sdk)])
        self.command('exans_build_official',['cmake','--build',build,'--target','openlake_expans_test','-j','2'])
        self.command('exans_smoke_official',[build/'openlake_expans_test'])
        check_sources(src,SOURCES['exans'][1])
        # The released test uses generated BF16 words. Do not call it Llama data.
        return dict(status='smoke_only',scope='official synthetic GPU correctness smoke only; no released Llama throughput result')

    def execute(self):
        self.stage('hardware',self.hardware)
        self.stage('model',self.model,['hardware'])
        if self.result['stages']['model']['status']!='completed': return
        self.snapshot=Path(json.loads((self.out/'model_manifest.json').read_text())['snapshot'])
        self.stage('toolchain',self.toolchain)
        if self.result['stages']['toolchain']['status']=='completed':
            data=json.loads((self.out/'toolchain.json').read_text());root=Path(data['cuda_home'])
            self.env['CUDA_HOME']=str(root);self.env['CUDA_PATH']=str(root)
            self.env['PATH']=str(root/'bin')+os.pathsep+self.env.get('PATH','')
        if set(self.result['requested'])&{'splitzip','ours','aqua'}:
            self.stage('splitzip_env',lambda:self.setup_env('splitzip'))
            self.stage('splitzip_preflight',lambda:self.preflight('splitzip'),['splitzip_env','model'])
        if set(self.result['requested'])&{'splitzip','ours'}:
            self.stage('capture',self.capture,['splitzip_preflight'])
        if 'splitzip' in self.result['requested']: self.stage('splitzip',self.splitzip,['capture'])
        if 'ours' in self.result['requested']: self.stage('ours',self.ours,['capture'])
        if 'exans' in self.result['requested']: self.stage('exans',self.exans,['toolchain'])
        if 'aqua' in self.result['requested']:
            self.stage('aqua_env',lambda:self.setup_env('aqua'),['toolchain'])
            self.stage('aqua_preflight',lambda:self.preflight('aqua'),['aqua_env','model'])
            self.stage('aqua',self.aqua,['aqua_preflight','splitzip_preflight'])


def package(output,destination):
    """Small explicit allowlist; never ship weights, tokens, KV or predictors."""
    if destination.exists() or destination.with_suffix('.zip').exists(): raise ValueError('Fresh report destination required')
    destination.mkdir(parents=True)
    import shutil
    files=['results.json','summary.md','source_manifest.json','hardware.json','toolchain.json','model_manifest.json',
        'splitzip_preflight.json','aqua_preflight.json','splitzip_case_0.json','splitzip_case_1.json','ours/results.json',
        'aqua_environment.json','splitzip_environment.json','aqua_runtime_install.json']
    for path in ['aqua_train_official.log','aqua_ppl_raw_official.log','aqua_ppl_higgs2_official.log',
            'aqua_runtime_install.log','aqua_pip_check.log','aqua_verify_versions.log','aqua_environment.log',
            'aqua_requirements.log','aqua_import.log','aqua_gpu_preflight.log','hadamard_install.log',
            'exans_build_official.log','exans_configure_official.log','exans_smoke_official.log']:
        files.extend(str(p.relative_to(output)) for p in (output/'logs').glob(path.replace('.log','*.log')))
    for relative in files:
        source=output/relative
        if not source.is_file(): continue
        if source.is_symlink() or not source.resolve().is_relative_to(output.resolve()):
            raise ValueError('Report symlink or escaping path forbidden')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    write(destination/'package_manifest.json',dict(files={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()},
        scope='explicit report allowlist; no model weights, tokens, raw/compressed KV or predictor tensors'))
    with zipfile.ZipFile(destination.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in destination.rglob('*'):
            if p.is_file(): archive.write(p,arcname=destination.name+'/'+str(p.relative_to(destination)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--stages',default='splitzip,ours,exans,aqua');p.add_argument('--publish-results',action='store_true')
    args=p.parse_args();stages=args.stages.split(',')
    if args.output and args.resume: p.error('Choose --output or --resume')
    if not stages or set(stages)-{'splitzip','ours','exans','aqua'} or len(set(stages))!=len(stages): p.error('Invalid stages')
    if args.publish_results:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        if branch!='experiment/llama32_official_gpu': p.error('Publishing requires experiment/llama32_official_gpu')
    output=args.resume or args.output or ROOT/'runs'/('llama32_official_a4500_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if args.output and args.output.exists(): p.error('Use --resume for an existing run directory')
    run=Run(output,stages)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(run.out/'source_manifest.json',dict(commit=head,files=run.result['source_files'],author_commits=SOURCES))
    interrupted=False
    try: run.execute()
    except KeyboardInterrupt: interrupted=True
    except Exception as error: run.result['fatal_error']=repr(error);run.result['fatal_traceback']=traceback.format_exc()
    completed=all(run.result['stages'].get(s,{}).get('status')=='completed' for s in stages)
    failed=bool(run.result.get('fatal_error')) or any(s['status']=='failed' for s in run.result['stages'].values())
    run.result['status']='interrupted' if interrupted else ('completed' if completed else ('failed' if failed else 'partial_or_blocked'))
    run.save();print((run.out/'summary.md').read_text())
    print('Resume: '+shlex.join([str(sys.executable),'scripts/run_llama_official_a4500.py','--resume',str(run.out),'--stages',args.stages]),flush=True)
    if args.publish_results and not interrupted:
        destination=ROOT/'benchmark_results'/(run.out.name+'_report_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S'))
        package(run.out,destination)
        subprocess.run([sys.executable,'scripts/build_standalone.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'scripts/build_standalone.py','--check'],cwd=ROOT,check=True)
        paths=['qwen_kv_gpu_bench.py',str(destination.relative_to(ROOT)),str(destination.with_suffix('.zip').relative_to(ROOT))]
        subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Add Llama official GPU report ('+run.result['status']+')','--',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','pull','--rebase','origin','experiment/llama32_official_gpu'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','HEAD:experiment/llama32_official_gpu'],cwd=ROOT,check=True)
        print('Report: '+str(destination/'summary.md'))
    return 130 if interrupted else (0 if completed else 1)


if __name__=='__main__': sys.exit(main())

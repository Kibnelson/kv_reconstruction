#!/usr/bin/env python3
"""Four matched exact Block32 arms with probability, hook and device diagnostics.

A: fixed width; B: causal-context width; C/D: B + equal-budget KV/actual-H hint.
Means remain frozen. CPU interval probabilities and range coding are explicit.
CUDA, when requested, computes integer means/width IDs and hint projections in
the *actual packet decode*. CPU causal context and transfers are included.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
from scripts import bench_transfer_extensions as common
from qwen_kv_bench import transfer_reference as ref
from qwen_kv_bench import transfer_probability as codec


def write(path,value):
    common.write_json(path,value)


def progress(message):
    print(time.strftime("[%H:%M:%S] ")+message,flush=True)


def make_backend(base,state,device):
    if device=="cpu":return codec.NumpyBackend(base,state)
    from qwen_kv_bench.transfer_probability_gpu import TorchBackend
    return TorchBackend(base,state,device)


def device_info(device):
    if device=="cpu":return dict(selected="cpu",gpu=False)
    import torch
    d=torch.device(device)
    if d.type!="cuda" or not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device unavailable; no CPU fallback")
    return dict(selected=str(d),gpu=True,name=torch.cuda.get_device_name(d),torch=torch.__version__,
                torch_cuda=torch.version.cuda,total_memory_bytes=torch.cuda.get_device_properties(d).total_memory)


def memory_begin(backend):
    if not backend.gpu:return None
    import torch
    backend.synchronize();before=torch.cuda.memory_allocated(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device);return before


def memory_end(backend,before):
    if not backend.gpu:return dict(measured=False)
    import torch
    backend.synchronize()
    return dict(measured=True,baseline_allocated_bytes=before,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device)-before,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(backend.device),
        scope="Torch allocation only; includes uploaded context/output parameters and retained hint tensors; CPU arrays/coder excluded")


def process_peak_bytes():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform=='darwin' else peak*1024)


def hints(base,encoder,capture,backend=None):
    if encoder is None:return None,dict(source_prepare_seconds=0.,projection_quantization_seconds=0.)
    begin=time.perf_counter();source=codec.hint_source(base,capture,encoder["kind"])
    prepare=time.perf_counter()-begin;begin=time.perf_counter()
    ids=codec.hint_ids(encoder,source,backend)
    return ids,dict(source_prepare_seconds=prepare,projection_quantization_seconds=time.perf_counter()-begin,
        projection_device="cpu" if backend is None else str(backend.device),
        scope="starts from retained CPU capture; GPU projection includes source uploads; not fused prefill-hook overhead")


def capture_inputs(args,output):
    if args.tiny:
        if args.act_dir or args.captures_from:raise ValueError('--tiny creates its own actual-hook captures')
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        if args.max_layers is not None and args.max_layers>2:raise ValueError('tiny model has two layers')
        config=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=2,head_dim=32,max_position_embeddings=256,
            bos_token_id=1,eos_token_id=2,pad_token_id=0)
        torch.manual_seed(args.seed)
        model=Qwen3ForCausalLM(config).to(device=args.device,dtype=torch.bfloat16).eval()
        weights=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            weights.update(name.encode());weights.update(tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())
        capture_dir=output/'captures';capture_dir.mkdir();entries=[];records=[];data={}
        for split,count in [('train',args.train_cases),('dev',1),('test',args.test_cases)]:
            data[split]=[]
            for i in range(count):
                # Functional random-model test, with real model hooks. No
                # pretrained-model or natural-workload compression inference.
                rng=np.random.default_rng(np.random.SeedSequence([args.seed,{'train':1,'dev':2,'test':3}[split],i]))
                n=max(64,2*args.page_tokens)
                if n>256:raise ValueError('tiny context must fit 256 positions')
                spec=dict(name=f'tiny_{split}_{i}',kind='random-token software check',ids=rng.integers(3,128,n).tolist())
                path=capture_dir/(spec['name']+'.npz')
                act,record=common.capture_case(model,spec,torch.device(args.device),path,args.max_layers)
                record['split']=split;records.append(record);data[split].append((spec['name'],act))
                entries.append(dict(split=split,name=spec['name'],file=str(path.relative_to(output)),sha256=common.sha_file(path)))
        identity=dict(model='random tiny Qwen3; software validation only',seed=args.seed,config=config.to_dict(),
            weights_sha256=weights.hexdigest(),capture_device=args.device,captures=records,
            torch=torch.__version__,transformers=importlib.metadata.version('transformers'))
        write(output/'capture_manifest.json',dict(format='block32_probability_captures_v1',entries=entries,
            model_identity=identity,hidden_provenance='actual observational self_attn pre-hooks on a random tiny Qwen3 model; not pretrained evidence'))
        del model
        return [a for _,a in data['train']],data['dev'][0][1],data['test'],identity
    if args.captures_from:
        root=Path(args.captures_from);manifest=json.loads((root/"capture_manifest.json").read_text())
        if manifest.get('format')!='block32_probability_captures_v1':
            raise ValueError('reuse the original self-contained capture directory, not a reference to it')
        entries=manifest["entries"];all_data={"train":[],"dev":[],"test":[]}
        for entry in entries:
            relative=Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:raise ValueError("invalid capture path")
            p=root/relative
            if common.sha_file(p)!=entry["sha256"]:raise ValueError("reused capture checksum mismatch")
            a=common.limit_act_layers(common.load_act(p),args.max_layers)
            common.source_features(a,"hidden")
            all_data[entry["split"]].append((entry["name"],a))
        if len(all_data["dev"])!=1:raise ValueError("one DEV capture required")
        identity=dict(manifest["model_identity"],reused_from=str(root))
        train=[a for _,a in all_data["train"]];dev=all_data["dev"][0][1];tests=all_data["test"]
        write(output/"capture_manifest.json",dict(format="block32_probability_capture_reference_v1",
             source_manifest=str(root/"capture_manifest.json"),source_sha256=common.sha_file(root/"capture_manifest.json"),
             entries=entries,model_identity=identity,scope="capture files remain under reused_from; not copied"))
        return train,dev,tests,identity
    train,dev,tests,identity=common.prepare_captures(args,output)
    entries=[]
    if args.act_dir:
        paths=sorted(Path(args.act_dir).glob("train*.npz"))[:args.train_cases]
        data=[("train",p.stem,a,p) for p,a in zip(paths,train)]
        data += [("dev","tables",dev,Path(args.act_dir)/"tables.npz")]
        data += [("test",n,a,Path(args.act_dir)/(n+".npz")) for n,a in tests]
        # Normalize imported arrays under this run, so reuse is self-contained.
        for split,name,a,origin in data:
            p=output/"captures"/(split+"_"+name+".npz");np.savez(p,**a)
            entries.append(dict(split=split,name=name,file=str(p.relative_to(output)),sha256=common.sha_file(p),
                                original_path=str(origin)))
    else:
        for report in identity["captures"]:
            p=Path(report["capture_file"])
            entries.append(dict(split=report["split"],name=report["name"],file=str(p.relative_to(output)),
                                sha256=common.sha_file(p)))
    write(output/"capture_manifest.json",dict(format="block32_probability_captures_v1",entries=entries,
        model_identity=identity,hidden_provenance="observational actual attention pre-hook for fresh model captures; imported ACT provenance belongs to its supplier"))
    return train,dev,tests,identity


def page(capture,start,tokens):
    if start<0:start=capture["K_bits"].shape[1]-tokens
    return common.slice_act(capture,start,start+tokens)


def calibrate(base,state,encoder,dev_pages):
    candidates=[]
    for strength in (0,2,4):
        trial=copy.deepcopy(state)
        if state.arm=="B":trial.context_strength=strength
        else:trial.hint_strength=strength
        trial.digest();total=0;streams=0;hint_bytes=0
        for capture in dev_pages:
            ids,_=hints(base,encoder,capture)
            packet,stats=codec.encode(base,trial,capture,ids)
            restored,_=codec.decode(base,trial,packet)
            if not np.array_equal(restored,ref.words(capture)):raise AssertionError("DEV round trip failed")
            total+=len(packet);streams+=stats["main_stream_bytes"];hint_bytes+=stats["hint_bytes"]
        candidates.append(dict(strength=strength,packet_bytes=total,main_stream_bytes=streams,hint_bytes=hint_bytes))
    winner=min(candidates,key=lambda r:(r["packet_bytes"],r["strength"]))
    if state.arm=="B":state.context_strength=winner["strength"]
    else:state.hint_strength=winner["strength"]
    state.digest()
    return dict(candidates=candidates,selected_strength=winner["strength"],
        rule="minimum complete CPU DEV selection-packet bytes; tie chooses less adaptation; hints remain charged even at strength zero")


def child(args):
    begin=time.perf_counter()
    base=ref.load_model(args.base_state);state=codec.load_state(args.adapter_state)
    load_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    backend=make_backend(base,state,args.parameter_device)
    setup_seconds=time.perf_counter()-begin
    output,report=codec.decode(base,state,Path(args.decode_packet).read_bytes(),backend)
    np.save(args.decoded_output,output)
    write(args.decode_report,dict(report,backend=backend.statistics(),decoder_inputs=[args.base_state,args.adapter_state,args.decode_packet],
        original_or_hidden_inputs=False,cold_state_load_seconds=load_seconds,
        cold_backend_prepare_seconds=setup_seconds,host_process_peak_rss_bytes=process_peak_bytes(),
        host_memory_scope='whole fresh decoder process including imports, cold state, packet and output; not isolated workspace'));return 0


def packet_process(base_path,state_path,packet_path,output,device):
    decoded=output.with_suffix(".decoded.npy");report=output.with_suffix(".decode.json")
    cmd=[sys.executable,str(Path(__file__).resolve()),"--base-state",str(base_path),"--adapter-state",str(state_path),
         "--decode-packet",str(packet_path),"--decoded-output",str(decoded),"--decode-report",str(report),
         "--parameter-device",device]
    begin=time.perf_counter()
    completed=subprocess.run(cmd,capture_output=True,text=True)
    if completed.returncode:
        raise RuntimeError('packet-only decoder failed: '+completed.stderr[-4000:])
    a=np.load(decoded,allow_pickle=False);r=json.loads(report.read_text());decoded.unlink()
    r['whole_process_wall_seconds']=time.perf_counter()-begin
    return a,r


def diagnostic_summary(rows,arm,layers,tokens,hint_bytes):
    result=[]
    for l in range(layers):
        for kind in ("K","V","KV"):
            selected=[r for r in rows if r["layer"]==l and (kind=="KV" or r["kind"]==kind)]
            n=sum(r["values"] for r in selected);gain=sum(r["ideal_bits_saved"] for r in selected)
            # Allocate only the layer's equal share of packed hint bits. This
            # screen excludes framing/finite-coder differences; packet totals win.
            paid=hint_bytes*8/layers*(1 if kind=="KV" else .5)
            result.append(dict(arm=arm,layer=l,kind=kind,values=n,ideal_bits_saved=gain,
                allocated_hint_bits=paid,net_ideal_bits_before_framing=gain-paid,
                net_ideal_bpv_before_framing=(gain-paid)/n,
                diagnostic_only=True))
    return result


def context_ablation(base,state,trace_rows,training_context_means):
    """Fixed-model ideal-rate attribution; not a separately tuned static codec."""
    total={'actual_context':0.,'TRAIN_mean_context':0.,'shuffled_context':0.}
    for r in trace_rows:
        l,g=r['layer'],r['group'];x=r['context']
        contexts={'actual_context':x,
            'TRAIN_mean_context':np.broadcast_to(training_context_means[l,g],x.shape),
            'shuffled_context':np.roll(x,max(1,len(x)//2),axis=0)}
        for name,features in contexts.items():
            off=codec.offsets(state,features,l,g)
            inv=r['inverse']*state.scale_table[off+codec.MAX_OFFSET,None]
            total[name]+=float(codec.symbol_bits(r['mu'],inv,r['target'],base.f_tab).sum())
    return dict(ideal_bits=total,mean_context_extra_bits=total['TRAIN_mean_context']-total['actual_context'],
        shuffled_context_extra_bits=total['shuffled_context']-total['actual_context'],
        scope='fixed selected B, TRAIN mean or within-page token roll; ideal costs only; not a best-refitted static-width baseline or encoded new arm')


def derive_insights(aggregate,rows,diagnostics,base):
    comparisons={}
    for arm in ('B','C','D'):
        control='A' if arm=='B' else 'B';x=aggregate[arm];y=aggregate[control]
        comparisons[arm+'_vs_'+control]=dict(
            packet_bytes_saved=y['packet_bytes']-x['packet_bytes'],
            main_stream_bits_saved=8*(y['main_stream_bytes']-x['main_stream_bytes']),
            added_hint_bits=8*(x['hint_bytes']-y['hint_bytes']),
            added_framing_bits=8*(x['framing_bytes']-y['framing_bytes']),
            added_decoder_numeric_bytes=x['decoder_numeric_bytes']-y['decoder_numeric_bytes'],
            added_encoder_numeric_bytes=x['encoder_numeric_bytes']-y['encoder_numeric_bytes'],
            packet_improves=x['packet_bytes']<y['packet_bytes'])
    patterns={}
    groups_per_kind=base.geometry['channels']//base.block_size
    for arm in codec.ARMS:
        selected=[r for r in diagnostics['groups'] if r['arm']==arm]
        by_layer={}
        for r in diagnostics['layer_rows']:
            if r['arm']==arm and r['kind']=='KV':
                item=by_layer.setdefault(r['layer'],dict(ideal_bits_saved=0.,allocated_hint_bits=0.))
                item['ideal_bits_saved']+=r['ideal_bits_saved'];item['allocated_hint_bits']+=r['allocated_hint_bits']
        ranked=sorted([dict(layer=l,**v,net_ideal_bits=v['ideal_bits_saved']-v['allocated_hint_bits']) for l,v in by_layer.items()],
                      key=lambda x:x['net_ideal_bits'],reverse=True)
        depth={}
        for kind in ('K','V'):
            for part in ('early','later'):
                chosen=[r for r in selected if r['kind']==kind and
                        ((r['group']%groups_per_kind < max(1,groups_per_kind//4))==(part=='early'))]
                n=sum(r['values'] for r in chosen)
                depth[kind+'_'+part]=dict(values=n,ideal_bpv_saved=(sum(r['ideal_bits_saved'] for r in chosen)/n if n else None))
        n=sum(r['values'] for r in selected);ng=sum(r['syndrome31_groups'] for r in selected)
        patterns[arm]=dict(ranked_layers=ranked,prefix_depth=depth,
            mantissa_bit6_error_fraction=sum(r['mantissa_bit6_errors'] for r in selected)/n,
            diagnostic_syndrome31_eligible_fraction=sum(r['syndrome31_eligible_groups'] for r in selected)/ng if ng else None,
            basic_single_error_group_break_even_fraction=1/26,
            scope='Descriptive TEST patterns only. Layer/group gates must be chosen on new TRAIN/DEV, and group-local syndrome screening is not an encoded coset result.')
    sensitivity={}
    for arm in ('C','D'):
        selected=[r for r in diagnostics['dev_sensitivity'] if r['arm']==arm]
        actual=sum(r['sizes']['actual']['main_stream_bytes'] for r in selected)
        sensitivity[arm]=dict(zero_main_bytes_minus_actual=sum(r['sizes']['zero']['main_stream_bytes'] for r in selected)-actual,
            shuffled_main_bytes_minus_actual=sum(r['sizes']['shuffled']['main_stream_bytes'] for r in selected)-actual)
    occupancy={}
    for arm in ('C','D'):
        hist=np.sum([r['hint_occupancy'] for r in rows if r['arm']==arm],axis=0)
        probabilities=hist/np.maximum(hist.sum(1,keepdims=True),1)
        entropy=-np.sum(np.where(probabilities>0,probabilities*np.log2(np.maximum(probabilities,1e-30)),0),axis=1)
        occupancy[arm]=dict(per_layer_histograms=hist.tolist(),per_layer_entropy_bits=entropy.tolist(),
            per_layer_active_ids=np.count_nonzero(hist,axis=1).tolist(),
            qualification='Occupancy/entropy diagnostic; hints use their full fixed packed bit budget, not this empirical entropy.')
    return dict(format='block32_probability_insights_v1',comparisons=comparisons,patterns=patterns,
        dev_fixed_model_hint_sensitivity=sensitivity,hint_occupancy=occupancy,
        context_attribution=diagnostics['context_ablation'],
        actual_hidden_beats_tested_KV_hint_on_packets=aggregate['D']['packet_bytes']<aggregate['C']['packet_bytes'],
        actual_hidden_beats_no_hint_B_on_packets=aggregate['D']['packet_bytes']<aggregate['B']['packet_bytes'],
        qualification='These comparisons concern the tested extractors and workload. They do not prove intrinsic information gain, novelty, GPU entropy throughput or serving capacity.')


def summarize(report):
    lines=["# Four-arm Block32 probability experiment","",f"Status: **{report['status']}**.",""]
    if report["status"]!="completed":
        lines += [str(report.get("error","unknown error"))];return "\n".join(lines)+"\n"
    lines += [f"Parameter device: `{report['device']['selected']}`. GPU parameters measured: **{report['coverage']['gpu_parameters']}**.",
        "CPU interval probabilities and range coding; CPU causal context with per-group device staging. Means are frozen.",
        "",f"DEV-selected storage winner: **{report['frozen_dev_winner']}**. TEST never changes selection.","",
        "| Arm | Packet bpv | Reduction | Hints (B) | Delta bytes vs A | CPU decoder state MiB | Encoder state KiB | Decode median s/page |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    a=report['aggregate']['A']['packet_bytes']
    for arm,v in report['aggregate'].items():
        lines.append(f"| {arm} | {v['bits_per_value']:.6f} | {v['reduction_percent']:.3f}% | {v['hint_bytes']} | {v['packet_bytes']-a:+d} | {v['decoder_numeric_bytes']/2**20:.3f} | {v['encoder_numeric_bytes']/1024:.3f} | {v['decode_median_seconds_per_page']:.6f} |")
    lines += ["","A uses the original Block32 entropy stream with the same fixed binary framing as B/C/D. Original legacy A packet sizes are reported separately. Shared state is outside warm packet rates.",
        "","A = fixed width; B = causal-context width; C = B + paid produced-KV hint; D = B + equally budgeted actual-prefill-H hint.",
        "","See `insights.json`, `layer_diagnostics.csv` and `diagnostics.json.gz` for paired byte gains versus hint cost, early/later channel groups, K/V, layer patterns, scale histograms, hint occupancy and DEV zero/shuffle sensitivity. Ideal group gains and syndrome screens are diagnostics, not measured group packets or new coset savings.",
        "","Every measured page is independently decoded in a fresh process from baseline state, adapter state and packet only. CPU and requested-device parameter digests, restored words and hint IDs are compared. Device timings include validation/staging; this is not a GPU entropy decoder, compressed attention or a serving speedup."]
    return "\n".join(lines)+"\n"


def run(args,output,report):
    report['device']=device_info(args.parameter_device)
    progress("preparing actual hidden/KV captures")
    train,dev,tests,identity=capture_inputs(args,output)
    report['model_identity']=identity
    write(output/'input_ids.json',dict(train=[np.asarray(a.get('ids',[])).tolist() for a in train],
        dev=np.asarray(dev.get('ids',[])).tolist(),test={name:np.asarray(a.get('ids',[])).tolist() for name,a in tests},
        arrays=[dict(split=split,index=i,kv_sha256=common.sha_array(ref.words(a)),hidden_sha256=common.sha_array(a['H']))
            for split,data in [('train',train),('dev',[dev]),('test',[a for _,a in tests])] for i,a in enumerate(data)]))
    if args.capture_only:
        report['status']='captures_only';return
    if len(train)<=args.base_fit_cases or args.base_fit_cases<1:
        raise ValueError("need disjoint baseline TRAIN and adapter TRAIN captures")
    for a in [*train,dev,*[a for _,a in tests]]:common.source_features(a,"hidden")
    identities=[common.sha_array(ref.words(a)) for a in [*train,dev,*[a for _,a in tests]]]
    if len(set(identities))!=len(identities):raise ValueError('identical KV captures cross fitting/evaluation split boundaries')
    base_train=train[:args.base_fit_cases];adapter_train=train[args.base_fit_cases:]
    half=dev['K_bits'].shape[1]//2
    if half<2 or dev['K_bits'].shape[1]-half<args.page_tokens:raise ValueError("DEV too short for disjoint scale/selection halves")
    scale=common.slice_act(dev,0,half)
    starts=sorted(set([half,dev['K_bits'].shape[1]-args.page_tokens]))
    dev_pages=[page(dev,s,args.page_tokens) for s in starts]
    # Geometry/positions from TEST only size the frozen RoPE table; no TEST
    # values enter fitting, mode selection or any emitter policy.
    maximum=max(int(ref._positions(a).max())+1 for a in [*train,dev,*[a for _,a in tests]])
    calibration=output/'calibration';calibration.mkdir();packets=output/'packets';packets.mkdir()
    progress("fitting frozen Block32 means on baseline TRAIN")
    begin=time.perf_counter()
    base=ref.fit(base_train,scale,mode='block',block_size=args.block_size,max_positions=maximum)
    report['calibration_seconds']={'base_mean_fit_and_scale_calibration':time.perf_counter()-begin}
    base_path=calibration/'base.npz';base_stats=ref.save_model(base,base_path)
    means_before=ref._state_digest(base)
    report['data_split']=dict(base_fit_captures=len(base_train),adapter_fit_captures=len(adapter_train),
        train_identity=[common.sha_array(ref.words(a)) for a in train],
        dev_scale_tokens=[0,half],dev_selection_tokens=[half,dev['K_bits'].shape[1]],dev_page_starts=starts,
        test_used_for_fitting=False,mean_weights_frozen=True,layer_scope=identity.get('captures',[{}])[0].get('layer_scope','imported capture geometry'))
    progress("fitting B on disjoint adapter TRAIN residuals")
    begin=time.perf_counter();b,traces=codec.fit_context(base,adapter_train)
    report['calibration_seconds']['B_context_fit_and_training_trace']=time.perf_counter()-begin
    selections={'B':calibrate(base,b,None,dev_pages)}
    states={'A':codec.empty_state(base),'B':b};encoders={'A':None,'B':None}
    for arm,kind in [('C','kv'),('D','hidden')]:
        progress(f"fitting {arm}: {args.hint_bits}-bit supervised {kind} scale-innovation hint")
        begin=time.perf_counter()
        states[arm],encoders[arm]=codec.fit_hint(base,b,adapter_train,traces,kind,args.hint_bits,args.sketch_width,seed=args.seed)
        report['calibration_seconds'][arm+'_hint_fit']=time.perf_counter()-begin
        selections[arm]=calibrate(base,states[arm],encoders[arm],dev_pages)
    report['selection']=selections;report['base_state']=base_stats
    context_means={}
    for i,first in enumerate(traces[0]):
        context_means[first['layer'],first['group']]=np.rint(np.concatenate([tr[i]['context'] for tr in traces]).mean(axis=0)).astype(np.int64)
    report['geometry']=dict(base.geometry,block_size=base.block_size)
    report['method']=dict(target='scale only; original Block32 mean parameters stay frozen',
        context_features=list(codec.FEATURE_NAMES),scale_offset_eighth_octave_range=[-16,16],
        hint_bits_per_token_layer=args.hint_bits,readout_sketch_width=args.sketch_width,
        hint_fit='ridge readout of fixed random sketch of signed/absolute source features, folded to one integer projection; supervised by adapter-TRAIN residual scale innovation',
        cpu_fit=True,decoder_context='previous-layer KV and error summaries from earlier recovered groups only',
        corpus_scope='existing synthetic fixture suite unless imported; exploratory comparisons, not broad-workload validation')
    write(calibration/'fit_diagnostics.json',dict(
        adapter_tokens_per_layer=sum(a['K_bits'].shape[1] for a in adapter_train),
        groups_per_layer=len(base.predictors[0]),
        all_layer_group_context_fits=True,
        context_coefficient_q12=b.coefficients.tolist(),
        response_rows=[dict(layer=r['layer'],group=r['group'],capture=i,
            samples=len(r['response']),response_mean=float(r['response'].mean()),
            response_std=float(r['response'].std()),response_clip_fraction=float(np.mean(np.abs(r['response'])>=32)),
            context_min=r['context'].min(axis=0).tolist(),context_max=r['context'].max(axis=0).tolist(),
            TRAIN_context_mean=context_means[r['layer'],r['group']].tolist())
            for i,tr in enumerate(traces) for r in tr],
        scope='adapter TRAIN only; no learned extractor convergence claim; every layer/group is fit in closed form'))
    state_files={};state_stats={};dev_sizes={};diagnostics=dict(dev_sensitivity=[],pages=[],layer_rows=[],context_ablation=[])
    for capture in dev_pages:
        diagnostics['context_ablation'].append(dict(split='DEV',start=int(ref._positions(capture)[0]),
            **context_ablation(base,b,codec.trace(base,capture,b),context_means)))
    for arm,state in states.items():
        state_files[arm]=calibration/(arm+'_decoder.npz');stats=codec.save_state(state,state_files[arm])
        encoder_stats=codec.save_encoder(encoders[arm],calibration/(arm+'_encoder.npz')) if encoders[arm] else dict(numeric_bytes=0,serialized_bytes=0)
        state_stats[arm]=dict(adapter=stats,encoder=encoder_stats)
        dev_total=0
        for capture in dev_pages:
            ids,_=hints(base,encoders[arm],capture);packet,_=codec.encode(base,state,capture,ids);dev_total+=len(packet)
        dev_sizes[arm]=dev_total
        if encoders[arm]:
            # Fixed-model sensitivity only, entirely within DEV selection.
            for capture in dev_pages:
                ids,_=hints(base,encoders[arm],capture)
                sizes={};forced_sizes={};forced=copy.deepcopy(state);forced.hint_strength=4;forced.digest()
                for name,z in [('actual',ids),('zero',np.zeros_like(ids)),('shuffled',np.roll(ids,max(1,ids.shape[1]//2),axis=1))]:
                    p,s=codec.encode(base,state,capture,z);sizes[name]=dict(packet_bytes=len(p),main_stream_bytes=s['main_stream_bytes'])
                    fp,fs=codec.encode(base,forced,capture,z);forced_sizes[name]=dict(packet_bytes=len(fp),main_stream_bytes=fs['main_stream_bytes'])
                diagnostics['dev_sensitivity'].append(dict(arm=arm,start=int(ref._positions(capture)[0]),sizes=sizes,
                    selected_hint_strength=state.hint_strength,forced_full_strength_sizes=forced_sizes,
                    scope="fixed selected model; code roll remains inside DEV-selection page; not used to choose TEST gates"))
    state_penalty={arm:base_stats['frozen_array_bytes']+state_stats[arm]['adapter']['numeric_bytes'] for arm in states}
    selection_values=sum(ref.words(p).size for p in dev_pages)
    # Compare warm packet rates and an explicitly prespecified state-amortized
    # storage scenario. Report both instead of letting hidden state be free.
    reference_values=args.reference_tokens*base.geometry['layers']*2*base.geometry['channels']
    selection_score={a:dev_sizes[a]/selection_values+state_penalty[a]/reference_values for a in states}
    report['frozen_dev_winner']=min(states,key=lambda a:(selection_score[a],a))
    report['dev_arm_comparison']=dict(packet_bytes=dev_sizes,bytes_per_value_plus_state_scenario=selection_score,
        reference_tokens=args.reference_tokens,scope="DEV only; decoder numeric state once per reference cache; encoder counted separately; not GPU capacity")
    report['state']=state_stats;report['test_rows']=[];all_group_rows=[]
    for name,capture in tests:
        chosen=sorted(set(s if s>=0 else capture['K_bits'].shape[1]-args.page_tokens for s in args.page_starts))
        for start in chosen:
            captured=page(capture,start,args.page_tokens);original=ref.words(captured)
            trace_rows=codec.trace(base,captured,b)
            diagnostics['context_ablation'].append(dict(split='TEST',case=name,start=start,
                **context_ablation(base,b,trace_rows,context_means)))
            legacy,legacy_stats=ref.encode(base,original,start=int(ref._positions(captured)[0]))
            _,legacy_header_len=ref.PREFIX.unpack_from(legacy)
            legacy_stream=legacy[ref.PREFIX.size+legacy_header_len:]
            for arm,state in states.items():
                progress(f"TEST {name} token {start}: arm {arm} CPU + {args.parameter_device} causal packet decode")
                begin=time.perf_counter();backend=make_backend(base,state,args.parameter_device)
                backend_prepare_seconds=time.perf_counter()-begin
                memory_before=memory_begin(backend)
                cpu_ids,hint_cpu=hints(base,encoders[arm],captured)
                ids,hint_device=hints(base,encoders[arm],captured,backend)
                if cpu_ids is not None and not np.array_equal(cpu_ids,ids):raise AssertionError("device hint IDs differ from CPU")
                cpu_packet,cpu_stats=codec.encode(base,state,captured,cpu_ids)
                packet,encoded=codec.encode(base,state,captured,ids,backend)
                if packet!=cpu_packet:raise AssertionError("requested-device encoded packet differs from CPU")
                if arm=='A' and not encoded['raw_fallback'] and codec.parse_packet(base,state,packet)[3]!=legacy_stream:
                    raise AssertionError("A entropy stream changed from original Block32")
                cpu_restored,cpu_decoded=codec.decode(base,state,packet)
                if not np.array_equal(cpu_restored,original):raise AssertionError("CPU packet round trip differs")
                for _ in range(args.warmup):codec.decode(base,state,packet,backend)
                samples=[]
                for _ in range(args.repeats):
                    restored,decoded=codec.decode(base,state,packet,backend)
                    if not np.array_equal(restored,original) or decoded['parameter_sha256']!=cpu_decoded['parameter_sha256']:
                        raise AssertionError("device decoded words or causal parameters differ")
                    samples.append(decoded)
                memory=memory_end(backend,memory_before)
                packet_path=packets/f'{name}_{start}_{arm}.bin';packet_path.write_bytes(packet)
                isolated,isolated_report=packet_process(base_path,state_files[arm],packet_path,packet_path,args.parameter_device)
                if not np.array_equal(isolated,original) or isolated_report['parameter_sha256']!=cpu_decoded['parameter_sha256']:
                    raise AssertionError("fresh process packet-only restoration differs")
                group_rows=codec.analyze_trace(base,state,trace_rows,cpu_ids)
                if encoders[arm]:
                    forced=copy.deepcopy(state);forced.hint_strength=4
                    diagnostics.setdefault('forced_full_hint_groups',[]).extend(
                        dict(r,arm=arm,case=name,start=start,diagnostic_only=True)
                        for r in codec.analyze_trace(base,forced,trace_rows,cpu_ids))
                layer_rows=diagnostic_summary(group_rows,arm,base.geometry['layers'],args.page_tokens,encoded['hint_bytes'])
                for r in layer_rows:r.update(case=name,start=start)
                all_group_rows += [dict(r,arm=arm,case=name,start=start) for r in group_rows]
                diagnostics['layer_rows']+=layer_rows
                occupancy=[np.bincount(z,minlength=1 << state.hint_bits).tolist() for z in ids] if ids is not None else None
                row=dict(case=name,start=start,arm=arm,values=original.size,raw_bytes=original.nbytes,
                    packet_sha256=hashlib.sha256(packet).hexdigest(),packet=str(packet_path.relative_to(output)),
                    encode=encoded,cpu_encode=cpu_stats,cpu_decode=cpu_decoded,decode_samples=samples,
                    decode_median_seconds=float(np.median([s['decode_seconds'] for s in samples])),
                    hints_cpu=hint_cpu,hints_device=hint_device,hint_occupancy=occupancy,
                    backend=backend.statistics(),torch_memory=memory,bit_exact=True,
                    cold_backend_prepare_seconds=backend_prepare_seconds,
                    host_process_peak_rss_bytes=process_peak_bytes(),
                    parameter_parity=True,hint_parity=True,packet_only_process=True,
                    packet_only_decoder_inputs=isolated_report['decoder_inputs'],
                    packet_only_process_report=isolated_report,
                    legacy_A_packet_bytes=len(legacy) if arm=='A' else None,
                    legacy_A_entropy_identical=(codec.parse_packet(base,state,packet)[3]==legacy_stream) if arm=='A' and not encoded['raw_fallback'] else None)
                report['test_rows'].append(row)
                if ids is not None:np.savez(output/f'hints_{name}_{start}_{arm}.npz',ids=ids)
                del backend
    if ref._state_digest(base)!=means_before:raise AssertionError("frozen Block32 state was modified")
    aggregate={}
    for arm in states:
        rows=[r for r in report['test_rows'] if r['arm']==arm];n=sum(r['values'] for r in rows)
        length=sum(r['encode']['packet_bytes'] for r in rows)
        aggregate[arm]=dict(values=n,packet_bytes=length,bits_per_value=length*8/n,reduction_percent=100*(1-length/(2*n)),
            hint_bytes=sum(r['encode']['hint_bytes'] for r in rows),framing_bytes=sum(r['encode']['framing_bytes'] for r in rows),
            main_stream_bytes=sum(r['encode']['main_stream_bytes'] for r in rows),
            decoder_numeric_bytes=state_penalty[arm],encoder_numeric_bytes=state_stats[arm]['encoder']['numeric_bytes'],
            decode_median_seconds_per_page=float(np.median([r['decode_median_seconds'] for r in rows])),
            all_exact=all(r['bit_exact'] and r['parameter_parity'] and r['packet_only_process'] for r in rows))
    diagnostics['groups']=all_group_rows
    diagnostics['scope']='Ideal coding costs, group-local hard-bit screens and descriptive TEST patterns; no TEST-selected policies or coset packets.'
    with gzip.GzipFile(filename=str(output/'diagnostics.json.gz'),mode='wb',mtime=0) as stream:stream.write(common.canonical(diagnostics).encode())
    with (output/'layer_diagnostics.csv').open('w',newline='') as stream:
        rows=diagnostics['layer_rows'];writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['aggregate']=aggregate;report['status']='completed';report['baseline_mean_state_unchanged']=True
    insights=derive_insights(aggregate,report['test_rows'],diagnostics,base)
    write(output/'insights.json',insights)
    report['coverage']=dict(actual_hook_capture=not bool(args.act_dir or args.captures_from),
        gpu_parameters=bool(report['device']['gpu'] and any(not r['encode']['raw_fallback'] for r in report['test_rows'])),
        raw_fallback_pages=sum(r['encode']['raw_fallback'] for r in report['test_rows']),
        cpu_entropy=True,actual_causal_packet_decode=True,
        gpu_entropy_decoder=False,compressed_attention=False,serving_speedup=False,
        all_layer_capture=(all(r.get('captured_layers')==r.get('model_layers') for r in identity['captures'])
                           if identity.get('captures') else None),
        scope='All imported capture identity belongs to its manifest. GPU timing includes serial CPU context and parameter staging.')
    report['host_process_peak_rss_bytes']=process_peak_bytes()
    report['state_and_packet_qualification']='Decoder state and encoder-only state are separate. No sum across experimental arms is a serving residency claim. Packets use common framing; legacy A framing is accounted separately.'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen3-4B-Instruct-2507')
    p.add_argument('--revision',default='cdbee75f17c01a7cc42f958dc650907174af0554')
    p.add_argument('--device',default='cuda:0',help='original model capture device')
    p.add_argument('--parameter-device',default='cuda:0',help='cpu or explicit CUDA; entropy remains CPU')
    p.add_argument('--act-dir');p.add_argument('--captures-from');p.add_argument('--ids-manifest-root')
    p.add_argument('--tiny',action='store_true',help='random two-layer Qwen3 actual-hook software check; bounded contexts, no model download')
    p.add_argument('--train-tokens',type=int,default=512);p.add_argument('--capture-tokens',type=int,default=2048)
    p.add_argument('--train-cases',type=int,default=3);p.add_argument('--test-cases',type=int,default=2)
    p.add_argument('--base-fit-cases',type=int,default=2)
    p.add_argument('--page-tokens',type=int,default=32);p.add_argument('--page-starts',type=common.integer_list,default=[0,-1])
    p.add_argument('--block-size',type=int,default=32);p.add_argument('--hint-bits',type=int,choices=(1,2,4),default=2)
    p.add_argument('--sketch-width',type=int,default=16);p.add_argument('--reference-tokens',type=int,default=8192)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=71)
    p.add_argument('--max-layers',type=int,default=None);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--capture-only',action='store_true')
    p.add_argument('--output',default='runs/block32_probability_arms')
    p.add_argument('--decode-packet');p.add_argument('--base-state');p.add_argument('--adapter-state')
    p.add_argument('--decoded-output');p.add_argument('--decode-report')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.decode_packet:return child(args)
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('use a fresh output directory; reuse captures with --captures-from')
    output.mkdir(parents=True,exist_ok=True)
    if args.repeats<1 or args.warmup<0 or args.page_tokens<1 or args.reference_tokens<1 or args.threads<1:
        raise ValueError('invalid repetition, page or thread count')
    if any(s < -1 for s in args.page_starts):raise ValueError('only -1 denotes the last page; other starts must be nonnegative')
    if args.act_dir and args.captures_from:raise ValueError('choose one capture source')
    args.reuse_captures=False
    report=dict(format=codec.FORMAT,status='running',arguments=vars(args),python=sys.version,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','constriction')},
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=common.sha_file(p)) for p in [Path(__file__),
            ROOT/'qwen_kv_bench/transfer_probability.py',ROOT/'qwen_kv_bench/transfer_probability_gpu.py',
            ROOT/'qwen_kv_bench/transfer_reference.py',ROOT/'qwen_kv_bench/transfer_coset.py',
            ROOT/'qwen_kv_bench/transfer_gpu_predictor.py',ROOT/'scripts/bench_transfer_extensions.py',
            ROOT/'qwen_kv_bench/model_runtime.py',ROOT/'qwen_kv_bench/fixtures.py']])
    try:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            if args.parameter_device!='cpu' or (not args.act_dir and not args.captures_from):raise
        run(args,output,report)
    except Exception as error:
        report['status']='failed';report['error']=dict(type=type(error).__name__,message=str(error))
        write(output/'results.json',report);(output/'summary.md').write_text(summarize(report));raise
    write(output/'results.json',report)
    if report['status']=='captures_only':
        (output/'summary.md').write_text('Captures prepared. No codec or GPU parameter result is claimed.\n')
    else:(output/'summary.md').write_text(summarize(report))
    progress(f"completed: {output/'summary.md'}");return 0


if __name__=='__main__':raise SystemExit(main())

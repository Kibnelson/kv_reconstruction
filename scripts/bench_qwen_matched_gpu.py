#!/usr/bin/env python3
"""Capture once, fit without TEST, then compare actual GPU codecs on Qwen KV.

The pilot is a partial-layer functional/compression comparison. It does not
reproduce any paper's full workload or implement the team's 51.4% method.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
import numpy as np
from qwen_kv_bench import transfer_reference as ref, transfer_probability as prob
from qwen_kv_bench import matched_transfer as matched

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
PREDICTORS = {
    'block32_A': ('block',32,'A'), 'block32_B': ('block',32,'B'),
    'previous32_B': ('previous',32,'B'), 'aqua_style32_B': ('aqua',32,'B'),
    'strong_layers32_B': ('strong_layers',32,'B'), 'strong_layers64_B': ('strong_layers',64,'B')}
DEFAULT_METHODS = 'raw_bf16,'+','.join(PREDICTORS)+',splitzip_native,nvcomp_bitcomp,nvcomp_cascaded'
PRESETS = {
    'pilot': dict(layers=4, train_cases=4, train_tokens=1024, scale_tokens=1024,
                  test_cases=2, test_tokens=2048, fit_rows=2048),
    'full': dict(layers=36, train_cases=128, train_tokens=1024, scale_tokens=8192,
                 test_cases=8, test_tokens=8192, fit_rows=8192)}


def log(message):print(time.strftime('[%H:%M:%S] ')+message, flush=True)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):h.update(block)
    return h.hexdigest()


def sha_words(words):return hashlib.sha256(np.ascontiguousarray(words).astype('<u2',copy=False).tobytes()).hexdigest()


def blocks(rows, tokenizer, count, tokens):
    """Deterministic stream blocks; discard a consumed row's unused suffix.

    A dataset row never supplies both fit and scale captures. WikiText rows
    are lines, not complete articles; TEST is its official held-out split.
    """
    for _ in range(count):
        ids = []; ranges = []
        while len(ids) < tokens:
            index, row = next(rows)
            text = row.get('text','')
            if not text.strip():continue
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:encoded.append(tokenizer.eos_token_id)
            take = min(len(encoded), tokens-len(ids))
            ranges.append(dict(row=index, token_start=0, token_end=take))
            ids.extend(encoded[:take])
        yield dict(ids=ids, ranges=ranges)


def make_token_plan(args, output, tokenizer):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    config = PRESETS[args.preset]
    protocol = dict(model=MODEL, model_revision=REVISION, tracks=args.tracks,
                    preset=args.preset, seed=args.seed, **config)
    path = output/'token_plan.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['protocol'] != protocol:raise ValueError('resume protocol differs from saved token plan')
        return result
    result = dict(protocol=protocol, datasets={}, entries=[])
    for track in args.tracks:
        repo = 'Salesforce/wikitext' if track=='wikitext2' else 'allenai/c4'
        revision = HfApi().dataset_info(repo).sha
        kwargs = dict(path=repo, name='wikitext-2-raw-v1' if track=='wikitext2' else 'en',
                      revision=revision, streaming=True)
        if track=='c4':
            kwargs['data_files'] = {'train':'en/c4-train.00000-of-01024.json.gz',
                                   'validation':'en/c4-validation.00000-of-00008.json.gz'}
        log(f'{track}: selecting disjoint fitting/calibration and held-out token blocks')
        train = enumerate(iter(load_dataset(**kwargs, split='train')))
        scale = enumerate(iter(load_dataset(**kwargs, split='validation'))) if track=='wikitext2' else train
        test_split = 'test' if track=='wikitext2' else 'validation'
        test = enumerate(iter(load_dataset(**kwargs, split=test_split)))
        result['datasets'][track] = dict(repository=repo, revision=revision,
            config=kwargs['name'], data_files=kwargs.get('data_files'), test_split=test_split,
            sampling='deterministic prefix stream blocks with EOS between rows; not the papers random sampling',
            scale_split='validation' if track=='wikitext2' else 'train rows following all fit rows')
        for split, iterator, count, n in [('train',train,config['train_cases'],config['train_tokens']),
                ('scale',scale,1,config['scale_tokens']),('test',test,config['test_cases'],config['test_tokens'])]:
            for i, block in enumerate(blocks(iterator,tokenizer,count,n)):
                result['entries'].append(dict(name=f'{track}_{split}_{i:03d}',track=track,split=split,**block))
    write(path,result)
    return result


def capture(args, output, torch, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen_kv_bench.model_runtime import _pairs
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    plan = make_token_plan(args, output, tokenizer)
    path = output/'capture_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else dict(
        format='qwen_matched_captures_v1', protocol=plan['protocol'], datasets=plan['datasets'],
        model_identity=None, entries=[])
    if manifest['protocol'] != plan['protocol']:raise ValueError('capture manifest protocol mismatch')
    completed = {e['name']:e for e in manifest['entries']}
    for e in completed.values():load_capture(output,e,verify=True)
    missing = [e for e in plan['entries'] if e['name'] not in completed]
    if not missing:return manifest
    log('loading pinned BF16 Qwen model; no weight quantization or automatic offload')
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', trust_remote_code=False).to(device).eval()
    config = model.config.to_dict()
    identity = dict(model=MODEL, revision=REVISION, config=config, dtype='bfloat16',
                    attention_backend='sdpa', tokenizer_revision=REVISION)
    if manifest['model_identity'] is not None and manifest['model_identity'] != identity:
        raise ValueError('model identity changed during resume')
    manifest['model_identity'] = identity
    count = PRESETS[args.preset]['layers']
    for e in missing:
        log(e['name']+': model prefill and original KV capture')
        ids = torch.tensor([e['ids']],dtype=torch.long,device=device)
        torch.cuda.synchronize(device); begin=time.perf_counter()
        with torch.inference_mode():result=model(ids,use_cache=True,logits_to_keep=1)
        torch.cuda.synchronize(device); elapsed=time.perf_counter()-begin
        pairs = _pairs(result.past_key_values)
        if len(pairs)<count:raise ValueError('fewer model layers than required')
        keys=[]; values=[]
        for k,v in pairs[:count]:
            if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16:raise ValueError('original BF16 cache required')
            for tensor,destination in [(k,keys),(v,values)]:
                destination.append(tensor.detach().transpose(1,2).contiguous().view(torch.uint16)
                    .cpu().numpy()[0].reshape(len(e['ids']),-1).copy())
        directory=output/'captures'/e['name'];directory.mkdir(parents=True,exist_ok=True)
        files={}; hashes={}
        for name,array in [('K_bits',np.stack(keys)),('V_bits',np.stack(values))]:
            p=directory/(name+'.npy');tmp=p.with_suffix('.part')
            with tmp.open('wb') as f:np.save(f,array,allow_pickle=False)
            tmp.replace(p);files[name]=str(p.relative_to(output));hashes[name]=sha_file(p)
        entry=dict(name=e['name'],track=e['track'],split=e['split'],files=files,hashes=hashes,
            shape=list(keys[0].shape), layers=count, model_layers=len(pairs),
            Hk=int(pairs[0][0].shape[1]),hd=int(pairs[0][0].shape[-1]),base=float(model.config.rope_theta),
            tokens=len(e['ids']),token_ids_sha256=hashlib.sha256(np.asarray(e['ids'],dtype='<i8').tobytes()).hexdigest(),
            ranges=e['ranges'],prefill_gpu_seconds=elapsed,
            key_order='original HF split halves; no hidden-state hooks retained')
        manifest['entries'].append(entry);write(path,manifest)
        del result,pairs,ids,keys,values;gc.collect()
    del model;gc.collect();torch.cuda.empty_cache()
    return manifest


def load_capture(output, entry, verify=False):
    result={}
    for name,relative in entry['files'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts:raise ValueError('invalid capture path')
        p=output/p
        if verify and sha_file(p)!=entry['hashes'][name]:raise ValueError('capture checksum mismatch: '+str(p))
        result[name]=np.load(p,mmap_mode='r',allow_pickle=False)
    result.update(Hk=np.array(entry['Hk']),hd=np.array(entry['hd']),base=np.array(entry['base']))
    return result


def select_rows(captures, limit, seed):
    lengths=[len(ref._positions(a)) for a in captures];total=sum(lengths)
    selected=np.sort(np.random.default_rng(seed).choice(total,min(limit,total),replace=False))
    result=[];offset=0
    for a,n in zip(captures,lengths):
        rows=selected[(selected>=offset)&(selected<offset+n)]-offset;offset+=n
        if not len(rows):continue
        result.append(dict(a,K_bits=a['K_bits'][:,rows],V_bits=a['V_bits'][:,rows],positions=rows))
    return result,dict(available_rows=total,selected_rows=len(selected),seed=seed,
                       selection_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest())


def fit(args, output, manifest):
    config=PRESETS[args.preset]; root=output/'calibration';root.mkdir(exist_ok=True)
    report={}
    for track in args.tracks:
        directory=root/track;directory.mkdir(exist_ok=True)
        entries=[e for e in manifest['entries'] if e['track']==track]
        originals=[load_capture(output,e) for e in entries if e['split']=='train']
        originals,selection=select_rows(originals,config['fit_rows'],args.seed)
        train=[matched.adjacent_capture(a) for a in originals]
        scale=matched.adjacent_capture(load_capture(output,next(e for e in entries if e['split']=='scale')))
        fit_identity=matched.digest(dict(entries=[e for e in entries if e['split']!='test'],selection=selection,
            model=manifest['model_identity'],predictors=PREDICTORS,
            fitting_source={p.name:sha_file(p) for p in [Path(__file__),Path(ref.__file__),Path(prob.__file__)]}))
        meta_path=directory/'fit_manifest.json'
        metadata=json.loads(meta_path.read_text()) if meta_path.exists() else dict(identity=fit_identity,selection=selection,arms={})
        if metadata['identity']!=fit_identity:raise ValueError('frozen fit identity differs on resume')
        write(meta_path,metadata)
        for name,(mode,group,arm) in PREDICTORS.items():
            if name not in args.methods:continue
            bp=directory/(mode+str(group)+'_base.npz');sp=directory/(name+'_state.npz')
            if name in metadata['arms']:
                if sha_file(bp)!=metadata['arms'][name]['base_file_sha256'] or sha_file(sp)!=metadata['arms'][name]['state_file_sha256']:
                    raise ValueError('frozen calibration file changed')
                continue
            begin=time.perf_counter();log(f'{track}/{name}: offline fit (TEST excluded)')
            base=ref.load_model(bp) if bp.exists() else ref.fit(train,scale,mode=mode,block_size=group,
                max_positions=max(config['train_tokens'],config['scale_tokens'],config['test_tokens']))
            if not bp.exists():ref.save_model(base,bp)
            if arm=='A':state=prob.empty_state(base,'A')
            else:
                # The same prespecified context rule and strength serve all B
                # arms. Do not tune candidate strengths on TEST.
                state,traces=prob.fit_context(base,train);del traces
            prob.save_state(state,sp)
            metadata['arms'][name]=dict(mode=mode,group=group,arm=arm,context_strength=state.context_strength,
                fit_seconds=time.perf_counter()-begin,base_file_sha256=sha_file(bp),state_file_sha256=sha_file(sp),
                base_serialized_bytes=bp.stat().st_size,state_serialized_bytes=sp.stat().st_size,
                base_statistics=ref.model_statistics(base),state_sha256=state.state_sha256)
            write(meta_path,metadata);del base,state;gc.collect()
        report[track]=metadata
        del train,scale,originals;gc.collect()
    return report


def adapter_for(name, track, output, device, train_entries=None, seed=71):
    if name=='raw_bf16':return matched.Raw(device)
    directory=output/'calibration'/track
    if name in PREDICTORS:
        mode,group,_=PREDICTORS[name]
        return matched.Predictive(name,ref.load_model(directory/(mode+str(group)+'_base.npz')),
                                  prob.load_state(directory/(name+'_state.npz')),device)
    if name.startswith('nvcomp_'):return matched.Nvcomp(name.split('_')[1].title(),device)
    if name=='splitzip_native':
        import torch
        sp=directory/'splitzip_state.npz';calibration=None
        if not sp.exists():
            if train_entries is None:raise ValueError('missing SplitZip TRAIN calibration')
            cfg=json.loads((output/'capture_manifest.json').read_text())['protocol']
            arrays,_=select_rows([load_capture(output,e) for e in train_entries],cfg['fit_rows'],seed)
            words=np.concatenate([ref.words(a).reshape(-1) for a in arrays])
            # Bounded deterministic sample of TRAIN words for the 16-entry
            # native codebook, never values from the held-out source.
            if len(words)>8_000_000:
                rows=np.random.default_rng(seed).choice(len(words),8_000_000,replace=False);words=words[rows]
            calibration=torch.from_numpy(words.copy()).to(device)
        return matched.SplitZip(device,ROOT/'runs/_matched_baselines/splitzip/codec_gpu.py',calibration,sp)
    raise ValueError('unknown method '+name)


def measure(fn, check, args, torch, device, raw_bytes):
    samples=[];warmup_samples=[];v=None
    for index in range(args.warmup+args.repeats):
        v=None;gc.collect();torch.cuda.synchronize(device)
        baseline=torch.cuda.memory_allocated(device);free_before=torch.cuda.mem_get_info(device)[0]
        torch.cuda.reset_peak_memory_stats(device)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter();start.record();v=fn();end.record();torch.cuda.synchronize(device)
        wall=(time.perf_counter()-begin)*1000
        sample=dict(wall_ms=wall,cuda_interval_ms=float(start.elapsed_time(end)),
            baseline_torch_bytes=baseline,peak_torch_bytes=torch.cuda.max_memory_allocated(device),
            incremental_torch_peak_bytes=torch.cuda.max_memory_allocated(device)-baseline,
            cuda_free_before_bytes=free_before,cuda_free_after_bytes=torch.cuda.mem_get_info(device)[0])
        check(v) # all equality/error checks are outside the measured interval
        if index>=args.warmup:samples.append(sample)
        else:warmup_samples.append(sample)
    walls=[s['wall_ms'] for s in samples];median=float(np.median(walls))
    return v,dict(samples=samples,warmup_samples=warmup_samples,wall_median_ms=median,wall_mean_ms=float(np.mean(walls)),
        wall_p95_ms=float(np.quantile(walls,.95)),cuda_interval_median_ms=float(np.median([s['cuda_interval_ms'] for s in samples])),
        original_GB_per_second=raw_bytes/1e6/median,
        incremental_torch_peak_bytes=max(s['incremental_torch_peak_bytes'] for s in samples),
        allocator_scope='Torch peak only; native nvCOMP allocation peaks excluded; free-memory snapshots are not peaks')


def diagnostic(adapter, original, start):
    if not isinstance(adapter,matched.Predictive):return []
    c=original.shape[-1]//2
    adjacent=matched.key_order(original,adapter.engine.base.geometry['head_dim'])
    capture=dict(K_bits=adjacent[...,:c],V_bits=adjacent[...,c:],positions=np.arange(start,start+original.shape[1]),
        Hk=adapter.engine.base.geometry['heads'],hd=adapter.engine.base.geometry['head_dim'],base=adapter.engine.base.geometry['base'])
    result={};state=adapter.engine.adapter
    for r in prob.trace(adapter.engine.base,capture,state):
        off=prob.offsets(state,r['context'],r['layer'],r['group']) if state.arm=='B' else np.zeros(len(r['target']),np.int64)
        inv=r['inverse']*state.scale_table[off+prob.MAX_OFFSET,None]
        b=prob.symbol_bits(r['mu'],inv,r['target'],adapter.engine.base.f_tab)
        kind='K' if r['group']*adapter.engine.base.block_size<c else 'V';key=(r['layer'],kind)
        row=result.setdefault(key,dict(layer=r['layer'],kind=kind,values=0,ideal_bits=0.,absolute_error_sum=0.,
                                      scope='CPU diagnostic replay; not measured GPU coding or packet size'))
        row['values']+=r['target'].size;row['ideal_bits']+=float(b.sum())
        row['absolute_error_sum']+=float(np.abs(ref.from_bits(r['target'])-r['mu']).sum())
    return list(result.values())


def requests(entries, sizes):
    result=[]
    for e in entries:
        for n in sizes:
            if n>e['tokens']:raise ValueError('request size exceeds captured context')
            for start in sorted({0,e['tokens']-n}):
                result.append((e,start,n))
    return result


def coverage(entries, sizes):
    words=0;contexts=[]
    for e in entries:
        intervals=sorted((start,start+n) for _,start,n in requests([e],sizes))
        merged=[]
        for a,b in intervals:
            if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
            else:merged.append([a,b])
        n=sum(b-a for a,b in merged);positions=e['layers']*n*2*e['shape'][1]
        contexts.append(dict(case=e['name'],merged_token_intervals=merged,word_positions=positions))
        words+=positions
    return dict(unique_test_word_positions=words,unique_test_source_bytes=2*words,contexts=contexts,
        scope='union of requested positions, counted once across sizes and methods; not every captured context word is timed')


def summarize(result):
    rows=result['rows']; groups={}
    for r in rows:
        key=(r['track'],r['method'],r['tokens']);g=groups.setdefault(key,[]);g.append(r)
    aggregates=[]
    for (track,method,tokens),rs in groups.items():
        values=sum(r['values'] for r in rs);wire=sum(r['bytes']['complete_bytes'] for r in rs)
        aggregates.append(dict(track=track,method=method,request_tokens=tokens,cases=len(rs),values=values,
            complete_bpv=wire*8/values,payload_bpv=sum(r['bytes']['payload_bytes'] for r in rs)*8/values,
            saving_percent=100*(1-wire/(2*values)),
            encode_wall_ms_mean_of_medians=float(np.mean([r['encode']['wall_median_ms'] for r in rs])),
            decode_wall_ms_mean_of_medians=float(np.mean([r['decode']['wall_median_ms'] for r in rs])),
            raw_chunks=sum(r['raw_chunks'] for r in rs),exact=all(r['exact'] for r in rs)))
    result['aggregates']=aggregates
    reference={(r['track'],r['request_tokens']):r for r in aggregates if r['method']=='block32_B'}
    comparisons=[]
    for r in aggregates:
        base=reference.get((r['track'],r['request_tokens']))
        if base is None or r is base or not r['exact']:continue
        comparisons.append(dict(track=r['track'],method=r['method'],request_tokens=r['request_tokens'],
            bpv_saved_vs_block32_B=base['complete_bpv']-r['complete_bpv'],
            saving_percentage_points_vs_block32_B=(base['complete_bpv']-r['complete_bpv'])*100/16,
            encode_speedup_vs_block32_B=base['encode_wall_ms_mean_of_medians']/r['encode_wall_ms_mean_of_medians'],
            decode_speedup_vs_block32_B=base['decode_wall_ms_mean_of_medians']/r['decode_wall_ms_mean_of_medians']))
    result['comparisons_vs_block32_B']=comparisons
    return aggregates


def save_report(output,result):
    aggregates=summarize(result);write(output/'results.json',result)
    lines=['# Matched Qwen GPU transfer comparison','',f"Status: **{result['status']}**. GPU measured: **{result['gpu_measured']}**.",'',
        'Original pinned Qwen BF16 cache; every method receives identical HF-layout words.',
        'Pilot uses four of 36 layers; full uses all layers. This is a new calibration/workload, so the earlier 44.67% is an anchor, not this table’s expected result.',
        '', '| Track | Method | Request tokens | Complete bpv | Saved | Encode ms | Decode ms | Exact |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in aggregates:
        lines.append(f"| {r['track']} | {r['method']} | {r['request_tokens']} | {r['complete_bpv']:.5f} | {r['saving_percent']:.2f}% | {r['encode_wall_ms_mean_of_medians']:.3f} | {r['decode_wall_ms_mean_of_medians']:.3f} | {r['exact']} |")
    stronger=[c for c in result['comparisons_vs_block32_B'] if c['method'].startswith('strong_layers')]
    if stronger:
        lines+=['','| Track | Added context | Tokens | bpv saved vs B | Encode speedup | Decode speedup |',
                '|---|---|---:|---:|---:|---:|']
        for c in stronger:
            lines.append(f"| {c['track']} | {c['method']} | {c['request_tokens']} | {c['bpv_saved_vs_block32_B']:+.5f} | {c['encode_speedup_vs_block32_B']:.3f}× | {c['decode_speedup_vs_block32_B']:.3f}× |")
        lines+=['','Positive bpv differences mean smaller complete packets. Speedup below 1 means slower. Inspect both tracks and shared state before promoting a candidate.']
    states=[(t,m,v) for t,methods in result['availability'].items() for m,v in methods.items() if v['status']=='available']
    if states:
        lines+=['','| Track | Method | Shared GPU state MiB |','|---|---|---:|']
        for t,m,v in states:
            n=v.get('persistent_device_bytes')
            size='unmeasured' if n is None else f'{n/2**20:.3f}'
            lines.append(f"| {t} | {m} | {size} |")
        lines+=['','nvCOMP native persistent state and allocation peaks are unmeasured.']
    lines+=['','Sizes include a common checked model/state/layout envelope and all required packet parts. Shared calibration is additional.',
        'Times are means of per-request synchronized wall medians; CUDA intervals, arithmetic means, samples and Torch peaks are in results.json.',
        'Input uploads, offline fitting and compressed-byte serialization are separate. Per-method file cycles include compressed export, fsync, read, upload and decode; filesystem cache remains warm.',
        'Native SplitZip charges its actual 7-byte escape entries. nvCOMP consumes the same exact words through an int16 view in lossless mode.',
        '', '**Scope:** strong_layers adds same-token layer 0 and l−2 to previous-layer plus earlier-group context. It does not implement the team’s attention predictor, pivot order, repeat-token context or value-slice coder; 51.4% is not an achieved GPU result.',
        'AQUA-style is a predictor ablation inside our exact coder, not the published lossy AQUA-KV system. FP8 speculative-model coding and ExANS remain pending adapters.',
        'This benchmark measures GPU transfer codecs. It does not measure P1 attention, T-to-P conversion, cold SSD, network or serving throughput.']
    unavailable=[(t,m,v) for t,methods in result['availability'].items() for m,v in methods.items() if v['status']!='available']
    if unavailable:
        lines+=['','| Track | Unavailable/failed method | Reason |','|---|---|---|']
        for t,m,v in unavailable:lines.append(f"| {t} | {m} | {v.get('reason','').replace('|','/').replace(chr(10),' ')[:300]} |")
    if result.get('error'):lines+=['','Error: '+result['error']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def bench(args,output,manifest,result,torch,device):
    for track in args.tracks:
        entries=[e for e in manifest['entries'] if e['track']==track]
        cases=requests([e for e in entries if e['split']=='test'],args.request_tokens)
        result['availability'].setdefault(track,{})
        for name in args.methods:
            adapter=None
            try:
                log(track+'/'+name+': initialize frozen decoder state')
                begin=time.perf_counter();adapter=adapter_for(name,track,output,device,
                    [e for e in entries if e['split']=='train'],args.seed)
                torch.cuda.synchronize(device)
                result['availability'][track][name]=dict(status='available',cold_initialization_seconds=time.perf_counter()-begin,
                                                         **adapter.statistics())
            except Exception as error:
                if name in PREDICTORS or name=='raw_bf16':raise
                result['availability'][track][name]=dict(status='unavailable',reason=repr(error))
                log(name+' unavailable: '+str(error));save_report(output,result);continue
            try:
                for i,(e,start,n) in enumerate(cases):
                    log(f'{track}/{name}: {e["name"]} tokens {start}:{start+n} ({i+1}/{len(cases)})')
                    a=load_capture(output,e);original=ref.words(a)[:,start:start+n].copy();raw_bytes=original.nbytes
                    torch.cuda.synchronize(device);begin=time.perf_counter()
                    source=torch.from_numpy(original).to(device);torch.cuda.synchronize(device)
                    upload_seconds=time.perf_counter()-begin
                    packet,enc=measure(lambda:adapter.encode(source,start),adapter.check_encode,args,torch,device,raw_bytes)
                    def check(v):
                        adapter.check_decode()
                        if not torch.equal(v,source):raise AssertionError('original BF16 words differ')
                    decoded,dec=measure(lambda:adapter.decode(packet),check,args,torch,device,raw_bytes)
                    del decoded
                    begin=time.perf_counter();wire,account=matched.export_record(adapter,packet,manifest['model_identity'])
                    export_seconds=time.perf_counter()-begin
                    # This reconstruction starts from only framed bytes plus
                    # frozen state, including pages at nonzero positions.
                    del packet;torch.cuda.synchronize(device);begin=time.perf_counter()
                    imported=matched.import_record(adapter,wire,manifest['model_identity']);restored=adapter.decode(imported)
                    torch.cuda.synchronize(device);import_decode_seconds=time.perf_counter()-begin;check(restored)
                    row=dict(track=track,method=name,case=e['name'],start=start,tokens=n,values=original.size,
                        original_sha256=sha_words(original),bytes=account,encode=enc,decode=dec,exact=True,
                        input_h2d_seconds=upload_seconds,compressed_export_seconds=export_seconds,
                        import_plus_decode_seconds=import_decode_seconds,packet_sha256=hashlib.sha256(wire).hexdigest(),
                        raw_chunks=imported.attributes.get('raw_chunks',0))
                    if i==0:
                        storage=output/'storage';storage.mkdir(exist_ok=True);p=storage/(track+'_'+name+'.t')
                        torch.cuda.synchronize(device);begin=time.perf_counter()
                        wire,_=matched.export_record(adapter,imported,manifest['model_identity'])
                        with p.open('wb') as f:f.write(wire);f.flush();os.fsync(f.fileno())
                        fd=os.open(storage,os.O_RDONLY)
                        try:os.fsync(fd)
                        finally:os.close(fd)
                        read=p.read_bytes();again=matched.import_record(adapter,read,manifest['model_identity'])
                        out=adapter.decode(again);torch.cuda.synchronize(device)
                        row['file_cycle_wall_ms']=(time.perf_counter()-begin)*1000;check(out)
                        row['diagnostic']=diagnostic(adapter,original,start)
                        del again,out
                    result['rows'].append(row);result['gpu_measured']=True;save_report(output,result)
                    del source,restored,imported,wire,original,a;gc.collect()
            except Exception as error:
                result['availability'][track][name]['status']='failed'
                result['availability'][track][name]['reason']=repr(error)
                result['failures'].append(dict(track=track,method=name,error=traceback.format_exc()))
                save_report(output,result)
            finally:
                del adapter;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preset',choices=PRESETS,default='pilot')
    p.add_argument('--tracks',default='wikitext2,c4');p.add_argument('--methods',default=DEFAULT_METHODS)
    p.add_argument('--request-tokens',default='32,128');p.add_argument('--warmup',type=int)
    p.add_argument('--repeats',type=int);p.add_argument('--seed',type=int,default=71)
    p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=['capture','fit','bench','all'],default='all')
    args=p.parse_args();args.tracks=args.tracks.split(',');args.methods=args.methods.split(',')
    if args.warmup is None:args.warmup=3 if args.preset=='full' else 1
    if args.repeats is None:args.repeats=10 if args.preset=='full' else 3
    args.request_tokens=list(map(int,args.request_tokens.split(',')))
    allowed=set(DEFAULT_METHODS.split(','))
    if (not args.tracks or len(set(args.tracks))!=len(args.tracks) or set(args.tracks)-{'wikitext2','c4'}
            or not args.methods or len(set(args.methods))!=len(args.methods) or set(args.methods)-allowed
            or args.warmup<0 or args.repeats<1 or not args.request_tokens or min(args.request_tokens)<1):
        p.error('invalid tracks, methods, sizes or timing count')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    result=dict(format='qwen_matched_gpu_report_v1',status='running',gpu_measured=False,
        protocol=vars(args)|{'output':str(output)},rows=[],availability={},failures=[],
        limits=['pilot/full presets are explicitly different from paper sampling',
                'strong_layers is a first controlled ablation, not full strong-and-fast',
                'ExANS and speculative FP8 model-copy adapters pending',
                'Torch peaks exclude native nvCOMP allocation peaks'])
    try:
        import torch
        device=torch.device(args.device)
        if os.environ.get('TRITON_INTERPRET')=='1' or device.type!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('actual CUDA required; CPU fallback/interpreter results are forbidden')
        torch.cuda.set_device(device)
        result['device']=dict(name=torch.cuda.get_device_name(device),capability=torch.cuda.get_device_capability(device),
            torch=torch.__version__,cuda=torch.version.cuda,triton=importlib.metadata.version('triton'),
            python=platform.python_version(),total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
        result['software']={}
        for package in ['transformers','datasets','numpy','huggingface-hub','nvidia-nvcomp-cu12']:
            try:result['software'][package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:result['software'][package]='not installed'
        try:
            snapshot=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.current.sm,memory.used,memory.total',
                '--format=csv'],capture_output=True,text=True,timeout=10)
            result['nvidia_smi_initial']=dict(returncode=snapshot.returncode,stdout=snapshot.stdout,stderr=snapshot.stderr,
                scope='all physical GPUs; Torch device selection follows CUDA_VISIBLE_DEVICES')
        except Exception as error:result['nvidia_smi_initial']=dict(unavailable=repr(error))
        if args.phase in ('capture','all'):manifest=capture(args,output,torch,device)
        else:manifest=json.loads((output/'capture_manifest.json').read_text())
        expected=dict(model=MODEL,model_revision=REVISION,tracks=args.tracks,preset=args.preset,seed=args.seed,**PRESETS[args.preset])
        if manifest['protocol']!=expected:raise ValueError('arguments differ from frozen capture protocol')
        for e in manifest['entries']:load_capture(output,e,verify=True)
        result['data']=manifest
        result['coverage']={t:coverage([e for e in manifest['entries'] if e['track']==t and e['split']=='test'],args.request_tokens) for t in args.tracks}
        if args.phase in ('fit','all'):result['calibration']=fit(args,output,manifest)
        if args.phase in ('bench','all'):
            if args.phase=='bench':
                result['calibration']={t:json.loads((output/'calibration'/t/'fit_manifest.json').read_text()) for t in args.tracks}
            # An explicit nondefault stream is accepted by nvCOMP and is also
            # the current Torch stream for all copies, kernels and events.
            bench_stream=torch.cuda.Stream(device=device)
            with torch.cuda.stream(bench_stream):
                bench(args,output,manifest,result,torch,device)
        result['status']='failed' if result['failures'] else ('completed' if args.phase in ('all','bench') else args.phase+'_completed')
        if result['status']=='completed' and any(v['status']=='unavailable' for ms in result['availability'].values() for v in ms.values()):
            result['status']='completed_with_unavailable_baselines'
    except Exception as error:
        result['status']='failed';result['error']=repr(error);result['traceback']=traceback.format_exc()
    finally:
        try:head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except Exception:head='unknown'
        write(output/'source_manifest.json',dict(commit=head,files={str(f.relative_to(ROOT)):sha_file(f) for f in [
            Path(__file__).resolve(),ROOT/'qwen_kv_bench/matched_transfer.py',ROOT/'qwen_kv_bench/transfer_t1.py',
            ROOT/'qwen_kv_bench/transfer_reference.py',ROOT/'qwen_kv_bench/transfer_probability.py',ROOT/'qwen_kv_bench/transfer_t1_kernels.py',
            ROOT/'qwen_kv_bench/transfer_gpu_predictor.py',ROOT/'qwen_kv_bench/model_runtime.py',
            ROOT/'scripts/run_qwen_matched_a4500.sh',ROOT/'scripts/setup_qwen_matched_baselines.py',
            ROOT/'scripts/package_qwen_matched_gpu.py',ROOT/'tests/test_qwen_matched_gpu.py']}))
        save_report(output,result)
    log('completed: '+str(output/'summary.md'))
    if result['status']=='failed':
        print(result.get('traceback') or json.dumps(result['failures'],indent=2),file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())

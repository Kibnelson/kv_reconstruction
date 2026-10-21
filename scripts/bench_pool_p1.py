#!/usr/bin/env python3
"""Task 1: bounded GPU P1 packing and head-tile reads, with strict coverage.

Use --cpu-oracle only for representation checks. It records no GPU timing.
T coding, T-to-P conversion, SSD I/O and attention are NOT implemented here.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from qwen_kv_bench import pool_p1 as p


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def log(s):print(time.strftime('[%H:%M:%S] ')+s,flush=True)


def safe_file(root,name):
    root=Path(root).resolve();f=(root/name).resolve()
    if not f.is_relative_to(root) or not f.is_file():raise ValueError('missing or unsafe input: '+str(name))
    return f


@dataclass
class Case:
    name:str
    words:np.ndarray
    layout:p.Layout
    state:p.Calibration
    provenance:dict
    software_case:bool=False


def fixture_cases(root,tile):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m.get('format')!='transfer_gpu_replay_fixture_v1':raise ValueError('unsupported frozen fixture format')
    required={m['pool_state_file'],*[entry['words_file'] for entry in m['pages']]}
    for name in sorted(required):
        item=m['files'][name]
        f=safe_file(root,name)
        if f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']:raise ValueError('fixture hash mismatch: '+name)
    with np.load(safe_file(root,m['pool_state_file']),allow_pickle=False) as z:state=p.calibration(z['refs'],z['widths'])
    g=m['state']['geometry'];out=[]
    for entry in m['pages']:
        f=safe_file(root,entry['words_file']);w=np.load(f,allow_pickle=False)
        layout=p.Layout(w.shape[0],w.shape[1],int(g['heads']),int(g['head_dim']),tile)
        if tuple(w.shape)!=layout.shape or w.dtype!=np.uint16:raise ValueError('wrong fixture word geometry')
        p.check_calibration(layout,state)
        out.append(Case(entry['name'],w,layout,state,dict(source='frozen SmolLM component fixture',
            fixture_manifest_sha256=sha(root/'manifest.json'),word_file_sha256=sha(f),
            original_words_sha256=hashlib.sha256(w.tobytes()).hexdigest(),token_start=entry['start'],
            model_identity=m['provenance']['model_identity'],key_order=entry['source_capture']['key_order'])))
    return out,dict(kind='frozen_fixture',path=str(root),manifest_sha256=sha(root/'manifest.json'),
                    required_files_verified=sorted(required),
                    scope='Same source words as historical P0 replay; verifies required words and pool calibration only, not unused T packets/predictor state; not fresh Qwen or serving execution')


def read_capture(path):
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in ['K_bits','V_bits','Hk','hd']}
    if a['K_bits'].ndim!=3 or a['K_bits'].shape!=a['V_bits'].shape or any(a[k].dtype!=np.uint16 for k in ['K_bits','V_bits']):
        raise ValueError('capture must contain original paired BF16 uint16 K/V')
    return a


def captured_cases(root,tile,page_tokens,starts):
    from scripts.bench_team_pool_reference import load_reference
    root=Path(root);manifest=json.loads((root/'capture_manifest.json').read_text())
    if manifest.get('format')!='block32_probability_captures_v1':
        raise ValueError('use the original self-contained probability run, not a capture-reference run')
    arrays={'train':[],'test':[]}
    for entry in manifest['entries']:
        f=safe_file(root,entry['file'])
        if sha(f)!=entry['sha256']:raise ValueError('capture hash mismatch: '+entry['file'])
        if entry['split'] in arrays:arrays[entry['split']].append((entry,read_capture(f)))
    if not arrays['train'] or not arrays['test']:raise ValueError('TRAIN and TEST captures required')
    train=[a for _,a in arrays['train']];layers,_,channels=train[0]['K_bits'].shape
    ref=load_reference(ROOT/'experiments/team_transfer_original/combined.py',train,2*channels)
    fitted=[ref['fit_pool'](l) for l in range(layers)]
    state=p.calibration(np.stack([x[0] for x in fitted]),np.stack([x[1] for x in fitted]))
    out=[]
    for entry,a in arrays['test']:
        heads=int(np.asarray(a['Hk']).reshape(-1)[0]);hd=int(np.asarray(a['hd']).reshape(-1)[0])
        if a['K_bits'].shape[::2]!=(layers,channels) or heads*hd!=channels:raise ValueError('capture geometries differ')
        tokens=a['K_bits'].shape[1]
        for start in sorted({s if s>=0 else tokens-page_tokens for s in starts}):
            if start<0 or start+page_tokens>tokens:raise ValueError('requested page lies outside capture')
            w=np.ascontiguousarray(np.concatenate([a['K_bits'][:,start:start+page_tokens],a['V_bits'][:,start:start+page_tokens]],axis=2))
            g=p.Layout(layers,page_tokens,heads,hd,tile)
            out.append(Case(entry['name']+'_'+str(start),w,g,state,dict(source='reused actual GPU probability-run capture',
                manifest_sha256=sha(root/'capture_manifest.json'),capture_file_sha256=entry['sha256'],
                original_words_sha256=hashlib.sha256(w.tobytes()).hexdigest(),token_start=start,
                model_identity=manifest['model_identity'],calibration_split='TRAIN captures only; original pool width objective',
                scope='same TEST pages as previous exploratory probability study; not a new holdout')))
    return out,dict(kind='verified_capture_reuse',path=str(root),manifest_sha256=sha(root/'capture_manifest.json'),
                    calibration='original per-channel exponent width objective on TRAIN; no P1 tuning on TEST')


def edge_cases():
    out=[];rng=np.random.default_rng(99)
    for name,tokens,wide in [('no_escapes_tail',19,False),('raw_escape_stress',32,True)]:
        g=p.Layout(1,tokens,2,16,4)
        w=(rng.integers(0,65536,g.shape,dtype=np.uint16) if wide else np.full(g.shape,127<<7,np.uint16))
        state=p.calibration(np.full((1,g.channels),127,np.uint8),np.full((1,g.channels),8 if wide else 1,np.uint8))
        out.append(Case(name,w,g,state,dict(scope='synthetic software case, not model rate'),True))
    g=p.Layout(1,512,8,16,4);w=np.empty(131072,np.uint16);w[::2]=127<<7;w[1::2]=np.arange(65536,dtype=np.uint16)
    s=p.calibration(np.full((1,g.channels),127,np.uint8),np.ones((1,g.channels),np.uint8))
    out.append(Case('all_bf16_patterns_packed',w.reshape(g.shape),g,s,dict(scope='all 65536 bit patterns in packed/escape sectors'),True))
    g=p.Layout(1,256,8,16,4);s=p.calibration(np.full((1,g.channels),127,np.uint8),np.full((1,g.channels),8,np.uint8))
    out.append(Case('all_bf16_patterns_raw',np.arange(65536,dtype=np.uint16).reshape(g.shape),g,s,dict(scope='all 65536 bit patterns in raw fallback sectors'),True))
    return out


def packet_statistics(desc,capacity,g,state_bytes):
    valid=desc[:,2]>=0;required=int(desc[:,1].sum());directory=desc.nbytes
    return dict(required_payload_bytes=required,arena_capacity_bytes=capacity,descriptor_bytes=directory,
        encode_scan_workspace_bytes=g.sectors*8,calibration_device_bytes=state_bytes,
        required_payload_bpv=required*8/(g.raw_bytes//2),
        allocated_packet_bytes=capacity+directory,allocated_packet_bpv=(capacity+directory)*8/(g.raw_bytes//2),
        allocated_packet_plus_calibration_bytes=capacity+directory+state_bytes,
        required_payload_plus_directory_bytes=required+directory,
        raw_original_bytes=g.raw_bytes,padded_raw_bytes=g.padded_raw_bytes,
        packed_sectors=int(np.count_nonzero(desc[:,2]==1)),raw_sectors=int(np.count_nonzero(desc[:,2]==0)),
        unpublished_sectors=int(np.count_nonzero(~valid)),stored_escape_bytes=int(desc[valid,3].sum()),
        fit=bool(np.all(valid)),
        qualification='Actual arena capacity, directory and shared calibration are charged; scan workspace, original input and output tiles are separate. No serving-capacity claim.')


def measure_gpu(fn,torch,device,warmup,repeats,raw_bytes):
    for _ in range(warmup):v=fn();torch.cuda.synchronize(device);del v
    samples=[];v=None
    for _ in range(repeats):
        v=None;torch.cuda.synchronize(device);baseline=torch.cuda.memory_allocated(device)
        reserved_before=torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter();start.record();v=fn();stop.record();torch.cuda.synchronize(device)
        wall=time.perf_counter()-begin
        samples.append(dict(wall_ms=wall*1000,cuda_interval_ms=float(start.elapsed_time(stop)),
            baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline,
            baseline_reserved_bytes=reserved_before,peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            incremental_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)-reserved_before))
    ms=float(np.median([x['wall_ms'] for x in samples]))
    return v,dict(warmup=warmup,repeats=repeats,samples=samples,wall_median_ms=ms,
        cuda_interval_median_ms=float(np.median([x['cuda_interval_ms'] for x in samples])),
        wall_p95_ms=float(np.quantile([x['wall_ms'] for x in samples],.95)),
        original_byte_GB_per_second=(raw_bytes/1e6/ms if ms else None),
        incremental_peak_allocated_bytes=max(x['incremental_peak_allocated_bytes'] for x in samples),
        scope='Resident input/state and preallocated arena; pack includes count, GPU scan, offset and write. Tile reads include output/status allocation. Events are stream intervals, synchronized wall includes host dispatch. On-chip registers/shared memory are not Torch allocations.')


def run_case(case,args,torch=None,device=None):
    g=case.layout;s=case.state;state_bytes=s.references.nbytes+s.widths.nbytes
    log(case.name+': independent byte oracle')
    begin=time.perf_counter();oracle=p.reference_encode(case.words,s,g,capacity_bytes=g.padded_raw_bytes)
    expected=p.original_tiles(case.words,g,np.arange(g.sectors))
    np.testing.assert_array_equal(p.reference_unpack(oracle,s,np.arange(g.sectors)),expected)
    row=dict(name=case.name,software_case=case.software_case,layout=g.description(),provenance=case.provenance,
        calibration_sha256=s.sha256,original_words=int(case.words.size),cpu_oracle_seconds=time.perf_counter()-begin,
        required_payload_bytes=int(oracle.descriptors[:,1].sum()),gpu_measured=not args.cpu_oracle,arenas=[])
    if torch is not None:
        begin=time.perf_counter();ds=p.to_cuda(s,device);source=torch.from_numpy(case.words.copy()).to(device)
        torch.cuda.synchronize(device);row['cold_state_and_source_upload_seconds']=time.perf_counter()-begin
        row['cuda_input_word_bytes']=source.numel()*source.element_size()
    for ratio in args.arena_ratios:
        capacity=max(16,int(np.ceil(g.padded_raw_bytes*ratio/16))*16)
        desc=oracle.descriptors.copy();desc[desc[:,0]+desc[:,1]>capacity,2]=-1
        stats=packet_statistics(desc,capacity,g,state_bytes)
        item=dict(arena_ratio_of_padded_raw=ratio,stats=stats,kv_to_p=None,full_sector_unpack=None,selected_tile_reads=[],
                  original_words_exact=None,byte_oracle_equal=None)
        if args.cpu_oracle:
            item['byte_oracle_equal']=None
            item['original_words_exact']=bool(stats['fit']);item['validation_scope']='NumPy byte oracle only; no kernel execution'
            row['arenas'].append(item);continue
        log(case.name+': GPU pack, arena ratio '+str(ratio))
        begin=time.perf_counter();arena=p.allocate_arena(g,ds,capacity);torch.cuda.synchronize(device)
        item['cold_arena_allocate_seconds']=time.perf_counter()-begin
        begin=time.perf_counter();p.encode_into(source,ds,arena);torch.cuda.synchronize(device)
        item['cold_first_pack_seconds_including_JIT']=time.perf_counter()-begin
        packed,item['kv_to_p']=measure_gpu(lambda:p.encode_into(source,ds,arena),torch,device,args.warmup,args.repeats,g.raw_bytes)
        del packed
        actual=arena.descriptors.cpu().numpy();np.testing.assert_array_equal(actual,desc)
        data=arena.payload.cpu().numpy()
        for sid,(off,size,mode,_) in enumerate(actual):
            if mode>=0:np.testing.assert_array_equal(data[off:off+size],oracle.payload[off:off+size])
        item['byte_oracle_equal']=True
        if not stats['fit']:
            bad=np.flatnonzero(actual[:,2]<0);ids=torch.from_numpy(bad.copy()).to(device)
            out,status=p.unpack_tiles(arena,ds,ids)
            if not np.all(status.cpu().numpy()!=0):raise AssertionError('unpublished arena sector was accepted')
            np.testing.assert_array_equal(out.cpu().numpy(),0)
            item['capacity_failure_safe']=True;item['original_words_exact']=False
            row['arenas'].append(item);del out,status,arena,ids;continue
        all_ids=torch.arange(g.sectors,device=device,dtype=torch.int64)
        result,item['full_sector_unpack']=measure_gpu(lambda:p.unpack_tiles(arena,ds,all_ids),torch,device,args.warmup,args.repeats,g.padded_raw_bytes)
        out,status=result
        np.testing.assert_array_equal(status.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),expected)
        item['original_words_exact']=True;del out,status,result
        for count in sorted(set(min(n,g.sectors) for n in args.tile_counts)):
            selected=np.linspace(0,g.sectors-1,count,dtype=np.int64)[::-1].copy()
            ids=torch.from_numpy(selected).to(device)
            result,timing=measure_gpu(lambda:p.unpack_tiles(arena,ds,ids),torch,device,args.warmup,args.repeats,count*g.sector_words*2)
            out,status=result;np.testing.assert_array_equal(status.cpu().numpy(),0)
            np.testing.assert_array_equal(out.cpu().numpy(),expected[selected])
            item['selected_tile_reads'].append(dict(sectors=count,output_tile_bytes=count*g.sector_words*2,
                id_bytes=count*8,timing=timing,original_words_exact=True,scope='Only selected head tiles returned; not an attention measurement'))
            del out,status,result
        row['arenas'].append(item);del arena,all_ids,ids
    if torch is not None:del ds
    if torch is not None and args.compare_p0 and not case.software_case:
        if g.tokens%g.tile_tokens:
            row['p0_control']=dict(measured=False,reason='original P0 requires a full token tile')
        else:
            from qwen_kv_bench import transfer_pool_gpu as old
            state0=old.make_state(s.references.copy(),s.widths.copy(),device=device)
            packet,tenc=measure_gpu(lambda:old.encode_pool(source,state0,tile_tokens=g.tile_tokens),torch,device,args.warmup,args.repeats,g.raw_bytes)
            restored,tdec=measure_gpu(lambda:old.decode_pool(packet,state0),torch,device,args.warmup,args.repeats,g.raw_bytes)
            np.testing.assert_array_equal(restored.cpu().numpy(),case.words)
            row['p0_control']=dict(measured=True,kv_to_p=tenc,full_decode=tdec,exact=True,
                stats=old.packet_stats(packet,state0,include_offsets=True),scope='Same words and calibration; different P0 layout and full output order. Selected-tile P1 times are not compared with full P0 decode.')
            del restored,packet,state0
    return row


def summary(report):
    lines=['# P1 GPU pool page and head-tile benchmark','',
        'Status: **'+report['status']+'**. GPU measured: **'+str(report['coverage']['gpu_measured'])+'**.','',
        'T encoding, T-to-P conversion, SSD I/O and compressed attention: **not implemented by this benchmark**.','']
    if report.get('error'):lines+=['Error: '+report['error'],'']
    lines+=['| Case | Arena/raw | Fits | Required payload bpv | Allocated packet bpv | KV to P wall ms | Full sector unpack wall ms | Exact |',
            '|---|---:|---|---:|---:|---:|---:|---|']
    for row in report.get('rows',[]):
        if row['software_case']:continue
        for a in row['arenas']:
            stats=a['stats'];enc=a['kv_to_p'];dec=a['full_sector_unpack']
            em='unmeasured' if enc is None else f"{enc['wall_median_ms']:.5f}"
            dm='unmeasured' if dec is None else f"{dec['wall_median_ms']:.5f}"
            exact=str(a['original_words_exact']) if stats['fit'] else 'not admitted'
            lines.append(f"| {row['name']} | {a['arena_ratio_of_padded_raw']:.2f} | {stats['fit']} | {stats['required_payload_bpv']:.5f} | {stats['allocated_packet_bpv']:.5f} | {em} | {dm} | {exact} |")
    lines+=['','Packet accounting includes the entire arena reservation and directory. CPU-oracle mode projects these byte counts; it does not measure GPU allocations. Shared calibration and GPU scan workspace are separately recorded; actual CUDA runs include original input and output/status tiles in peak allocation samples. Required payload bpv alone is not a resident-memory result.',
        '', 'Selected-tile timings, cold allocation/JIT, P0 paired controls if requested, software edge cases and full sample distributions are in results.json. Full-sector decode is a validation/throughput control, not compressed attention.',
        '', 'This is an append-batch bounded arena prototype. It has no reusable serving allocator, concurrent publication protocol or attention integration. Capacity exhaustion leaves sectors unpublished and retains the original source; it is not silently repaired on CPU.']
    return '\n'.join(lines)+'\n'


def parser():
    a=argparse.ArgumentParser(description=__doc__);s=a.add_mutually_exclusive_group()
    s.add_argument('--fixture',type=Path);s.add_argument('--captures-from',type=Path)
    a.add_argument('--cpu-oracle',action='store_true');a.add_argument('--device',default='cuda:0')
    a.add_argument('--tile-tokens',type=int,default=16);a.add_argument('--page-tokens',type=int,default=32)
    a.add_argument('--page-starts',type=lambda x:[int(i) for i in x.split(',')],default=[0,512,1024,-1])
    a.add_argument('--arena-ratios',type=lambda x:[float(i) for i in x.split(',')],default=[.8,1.])
    a.add_argument('--tile-counts',type=lambda x:[int(i) for i in x.split(',')],default=[1,8,32])
    a.add_argument('--warmup',type=int,default=3);a.add_argument('--repeats',type=int,default=20)
    a.add_argument('--compare-p0',action='store_true');a.add_argument('--output',type=Path,required=True)
    return a


def main(argv=None):
    args=parser().parse_args(argv)
    if args.output.exists():raise ValueError('use a fresh output directory')
    if args.repeats<1 or args.warmup<0 or args.page_tokens<=0 or not args.arena_ratios or any(not np.isfinite(x) or x<=0 for x in args.arena_ratios):raise ValueError('invalid benchmark bounds')
    if not args.tile_counts or any(x<=0 for x in args.tile_counts):raise ValueError('positive tile counts required')
    args.output.mkdir(parents=True)
    sources=['qwen_kv_bench/pool_p1.py','qwen_kv_bench/pool_p1_kernels.py','scripts/bench_pool_p1.py',
             'scripts/bench_team_pool_reference.py','experiments/team_transfer_original/combined.py']
    report=dict(format='pool_p1_measurement_v1',status='running',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sources=[dict(path=x,sha256=sha(ROOT/x)) for x in sources],rows=[],platform=platform.platform(),
        coverage=dict(gpu_measured=False,cpu_oracle=args.cpu_oracle,gpu_pool_pack=False,gpu_head_tile_unpack=False,
            gpu_transfer_entropy=False,gpu_T_to_P=False,compressed_attention=False,serving_capacity=False))
    write(args.output/'results.json',report)
    try:
        torch=device=None
        if args.cpu_oracle:report['device']=dict(selected='cpu_oracle',gpu=False)
        else:
            torch,device=p._cuda_torch(args.device)
            import triton
            torch.cuda.set_device(device)
            prop=torch.cuda.get_device_properties(device)
            report['device']=dict(selected=str(device),name=prop.name,capability=list(torch.cuda.get_device_capability(device)),
                total_memory_bytes=prop.total_memory,torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__)
        if args.captures_from:cases,identity=captured_cases(args.captures_from,args.tile_tokens,args.page_tokens,args.page_starts)
        else:cases,identity=fixture_cases(args.fixture or ROOT/'benchmark_fixtures/team_transfer_smol_replay_v1',args.tile_tokens)
        report['input_identity']=identity
        write(args.output/'source_manifest.json',dict(sources=report['sources'],input_identity=identity))
        for case in cases+edge_cases():
            report['rows'].append(run_case(case,args,torch,device));write(args.output/'results.json',report)
        report['status']='completed'
        if not args.cpu_oracle:report['coverage'].update(gpu_measured=True,gpu_pool_pack=True,gpu_head_tile_unpack=True)
        report['unique_model_words']=sum(x['original_words'] for x in report['rows'] if not x['software_case'])
        report['all_model_cases_have_exact_full_capacity_roundtrip']=all(any(a['stats']['fit'] and a['original_words_exact'] for a in x['arenas']) for x in report['rows'] if not x['software_case'])
        if not report['all_model_cases_have_exact_full_capacity_roundtrip']:raise AssertionError('no complete exact roundtrip at tested arena capacities')
        code=0
    except Exception as e:
        report['status']='failed';report['error']=str(e);report['traceback']=traceback.format_exc();code=1
    write(args.output/'results.json',report);(args.output/'summary.md').write_text(summary(report))
    log(report['status']+': '+str(args.output/'summary.md'))
    if code:print(report['error'],file=sys.stderr)
    return code


if __name__=='__main__':raise SystemExit(main())

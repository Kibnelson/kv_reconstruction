#!/usr/bin/env python3
"""Required compiled-GPU and fresh-receiver checks before matched PD serving."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def gpu():
    import torch
    if os.environ.get('TRITON_INTERPRET','0')!='0' or not torch.cuda.is_available():
        raise RuntimeError('actual compiled CUDA is required for the PD readiness gate')
    return torch.device('cuda',0)


def digest(words):
    return hashlib.sha256(words.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def receiver(job):
    import torch
    from kv_compare.pd.matched import MatchedT
    dev=gpu()
    state=torch.load(job['bundle'],map_location='cpu',weights_only=False)
    packet=torch.load(job['packet'],map_location='cpu',weights_only=False)
    codec=MatchedT(state,job['codec'],dev)
    parts={k:v.to(dev) for k,v in packet['parts'].items()}
    T=packet['meta']['T']
    output=torch.full((codec.L,T,codec.W),12345,dtype=torch.int16,device=dev)
    codec.decode(parts,packet['meta'],lambda l,w:output[l].copy_(w))
    actual=digest(output)
    if actual!=job['expected_sha256']:
        raise AssertionError('fresh receiver changed reconstructed BF16 words')
    return dict(codec=job['codec'],gpu_executed=True,capture_path_provided=False,
        exact=True,words_sha256=actual,tokens=T,position=packet['meta']['pos0'])


def checks(args):
    import torch
    from kv_compare.fast_dense_checks import check as fast_check
    from kv_compare.head_spread_checks import check as head_check
    from kv_compare.pd.matched import MatchedT,NAMES,clear_alignment
    from kv_compare.predict import Doc
    dev=gpu()
    previous=json.loads(Path(args.selftest).read_text())
    for name in NAMES:
        rr=[x for x in previous.get('codecs',[]) if x.get('codec')==name]
        if name in previous.get('codec_errors',{}) or {x['tokens'] for x in rr}!={8192,3000} or not all(x.get('exact') for x in rr):
            raise AssertionError('required codec did not pass both captured GPU cases: '+name)
    result=dict(fast_patterns=fast_check(dev,all_patterns=True),
        compact_patterns=head_check(dev,all_patterns=True),fresh_receivers=[],dense_stream_parity=False)
    state=torch.load(args.bundle,map_location='cpu',weights_only=False)
    doc=Doc(args.captures,'wtrain0')
    T,pos=513,37
    words=torch.stack([doc.words(l,dev,rows=pos+T)[pos:pos+T].to(torch.int16) for l in range(doc.layers)])
    expected=digest(words)
    with tempfile.TemporaryDirectory(prefix='pd_matched_receiver_',dir=Path(args.output).parent) as folder:
        folder=Path(folder)
        fast_parts,_=MatchedT(state,'t-fast',dev).encode(words,pos0=pos)
        control_parts,_=MatchedT(state,'t-dense-cached',dev).encode(words,pos0=pos)
        if not all(torch.equal(fast_parts[k],control_parts[k]) for k in fast_parts):
            raise AssertionError('fast and cached-original dense stream bytes differ')
        result['dense_stream_parity']=True
        del fast_parts,control_parts
        from kv_compare import layerwise as lw
        from kv_compare.pd.wire import compact_segments
        compact_global=MatchedT(state,'t-compact-global',dev)
        reference_codec,Tp=compact_global.codec(T,pos)
        original_codec=lw.LayerwiseOneStep(reference_codec.k,compact_global.C,compact_global.hd,
            tokens_per_program=128,mode='uniform',search=True,chunks=1)
        padded=torch.cat((words,torch.zeros(compact_global.L,Tp-T,compact_global.W,dtype=torch.int16,device=dev)),1)
        arena,starts=original_codec.encode(padded)
        clear_alignment(arena,starts,compact_global.hd)
        _,cap=original_codec.layout(Tp)
        packed=compact_segments(arena,starts,cap,compact_global.hd)
        parts,_=compact_global.encode(words,pos)
        if not torch.equal(parts['starts'],starts) or not torch.equal(parts['words'],packed):
            raise AssertionError('compact global control changed original T stream bytes')
        result['compact_global_stream_parity']=True
        del arena,starts,packed,parts,padded,reference_codec,original_codec,compact_global
        for name in NAMES:
            parts,meta=MatchedT(state,name,dev).encode(words,pos0=pos)
            packet=folder/(name+'.pt')
            torch.save(dict(parts={k:v.cpu() for k,v in parts.items()},meta=meta),packet)
            del parts
            job=dict(bundle=str(Path(args.bundle).resolve()),packet=str(packet),codec=name,expected_sha256=expected)
            jp=folder/(name+'.json');jp.write_text(json.dumps(job))
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--receiver-job',str(jp)],
                capture_output=True,text=True)
            if proc.returncode:
                raise RuntimeError('fresh receiver failed for '+name+': '+proc.stderr[-2000:])
            proof=json.loads(proc.stdout)
            if not proof.get('exact') or proof['words_sha256']!=expected:
                raise AssertionError('invalid independent receiver proof')
            result['fresh_receivers'].append(proof)
            packet.unlink()
    result['gpu_executed']=True
    result['policy_report']=state['t_matched']['report']
    result['state_sha256']=state['t_matched']['sha256']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle');p.add_argument('--captures');p.add_argument('--selftest');p.add_argument('--output')
    p.add_argument('--receiver-job')
    args=p.parse_args()
    if args.receiver_job:
        print(json.dumps(receiver(json.loads(Path(args.receiver_job).read_text())),sort_keys=True));return
    if not all((args.bundle,args.captures,args.selftest,args.output)):
        p.error('--bundle, --captures, --selftest and --output are required')
    report=dict(format='pd_matched_readiness_v1',ok=False,gpu_executed=False,serving_validated=False)
    try:
        report.update(checks(args));report['ok']=True
    except Exception:
        report['error']=traceback.format_exc()
    Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True),flush=True)
    if not report['ok']:raise SystemExit(1)


if __name__=='__main__':main()

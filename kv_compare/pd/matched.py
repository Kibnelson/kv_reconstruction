"""Fast/compact T in the common PD transport, with independent calibration state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

NAMES = ('t-fast', 't-dense-cached', 't-compact', 't-compact-global')
SCHEMA = 't_pd_matched_v1'
TB = 128
PAD = 256


def clear_alignment(arena, starts, hd):
    """Canonicalize the at-most-seven unused words after each rANS state."""
    s=starts.to(torch.int64)
    base=torch.div(s-2*hd,8,rounding_mode='floor')*8
    indices=base[:,None]+2*hd+torch.arange(8,device=arena.device)[None,:]
    arena[indices[indices<s[:,None]]]=0


class MatchedT:
    """Request-local codec objects; no mutable arena is shared across requests."""
    def __init__(self, bundle, name, device):
        from .codecs import _cache_slice_tables, _on
        from ..head_spread import Policy
        from ..head_spread_wire import policy_from_record
        if name not in NAMES or bundle.get('t_matched', {}).get('format') != SCHEMA:
            raise ValueError('matched T state is missing or incompatible')
        _cache_slice_tables()
        state = bundle['t_matched']
        self.name, self.device = name, torch.device(device)
        self.state_digest = state['sha256']
        self.compact = name.startswith('t-compact')
        self.C, self.hd, self.L, self.W = (int(bundle['meta'][k]) for k in ('C','hd','L','W'))
        source = state['grouped'] if self.compact else state['dense']
        self.k = {k: _on(v,self.device) if torch.is_tensor(v) else v for k,v in source.items()}
        self.cos, self.sin = _on(bundle['cos'],self.device), _on(bundle['sin'],self.device)
        self.device = self.k['Aq'].device
        self.policy = policy_from_record(state['policy']) if name=='t-compact' else Policy()
        self.policy_digest = self.policy.digest()

    def codec(self, tokens, pos0):
        from ..fast_dense_encode import FastDenseEncode, CachedFusedDense
        from ..head_spread import HeadSpreadCodec
        if type(tokens)!=int or tokens<1 or type(pos0)!=int or pos0<0:
            raise ValueError('positive length and nonnegative position required')
        padded = (tokens+PAD-1)//PAD*PAD
        if pos0+padded > self.cos.shape[0]:
            raise ValueError('request exceeds exported RoPE positions')
        k = dict(self.k,cos=self.cos[pos0:pos0+padded].contiguous(),sin=self.sin[pos0:pos0+padded].contiguous())
        cls = FastDenseEncode if self.name=='t-fast' else CachedFusedDense
        args = dict(tokens_per_program=TB,mode='uniform',search=True,chunks=1)
        codec = HeadSpreadCodec(k,self.C,self.hd,self.policy,**args) if self.compact else cls(k,self.C,self.hd,**args)
        return codec,padded

    def encode(self, words, pos0=0):
        from .wire import compact_segments
        if words.dtype!=torch.int16 or words.device!=self.device or words.ndim!=3 or (words.shape[0],words.shape[2])!=(self.L,self.W):
            raise ValueError('matched T word geometry/device mismatch')
        T = int(words.shape[1])
        codec,Tp = self.codec(T,pos0)
        wp = words.contiguous() if T==Tp else torch.cat((words,torch.zeros(self.L,Tp-T,self.W,dtype=torch.int16,device=self.device)),1)
        arena,starts = codec.encode(wp)
        clear_alignment(arena,starts,self.hd)
        _,cap = codec.layout(Tp)
        packed = compact_segments(arena,starts,cap,self.hd)
        return {'words':packed,'starts':starts},dict(codec=self.name,T=T,Tpad=Tp,cap=cap,pos0=pos0,
            state_sha256=self.state_digest,policy_sha256=self.policy_digest)

    def decode(self, parts, meta, sink):
        from .wire import expand_segments
        if meta.get('codec')!=self.name or meta.get('state_sha256')!=self.state_digest or meta.get('policy_sha256')!=self.policy_digest:
            raise ValueError('matched T packet requires different shared state/policy')
        T,pos0 = int(meta['T']),int(meta.get('pos0',0))
        codec,Tp = self.codec(T,pos0)
        segments,cap = codec.layout(Tp)
        if Tp!=meta['Tpad'] or cap!=meta['cap'] or parts['starts'].numel()!=segments:
            raise ValueError('matched T stream geometry differs')
        arena = expand_segments(parts['words'],parts['starts'],cap,self.hd,
            out=torch.empty(segments*cap,dtype=torch.int16,device=self.device))
        codec.decode(arena,parts['starts'],Tp,sink=lambda layer,w:sink(layer,w[:T]))


class Window:
    def __init__(self, doc, start, stop):
        self.name = doc.name
        self.K,self.V,self.ids = doc.K[:,start:stop],doc.V[:,start:stop],doc.ids[start:stop]
        self.Q = None if doc.Q is None else doc.Q[:,start:stop]
        self.layers,self.tokens,self.columns = self.K.shape
        self.position = start

    def words(self, layer, device, rows=None):
        from ..core import as_words
        stop = self.tokens if rows is None else rows
        return torch.cat((as_words(np.asarray(self.K[layer,:stop]),device),
            as_words(np.asarray(self.V[layer,:stop]),device)),1)


def policy_choice(rows):
    """DEV frame bytes, then shared-state cost and alpha; global is a fallback."""
    return min(rows,key=lambda x:(x['frame_bytes'],x['policy_bytes'],x['alpha']))


def fit_bundle(captures, train, tables, device, main, log=print):
    """Use the NUC algorithm, freshly fitted; no serving/test prompt is read."""
    from .. import fused, grouped
    from ..predict import Study
    from ..head_spread import HeadSpreadCodec, Policy, fit_head_map
    from ..head_spread_wire import consts_digest, policy_record
    from .export import _cpu
    from .wire import bundle, header_to_bytes
    study = Study(captures,device,train,tables,[],None,log=log)
    original = study.tables
    half = original.tokens//2
    if half<TB or half%TB or original.tokens%2:
        raise ValueError('tables must split into two complete 128-token halves')
    study.tables = Window(original,0,half)
    dev = Window(original,half,2*half)
    log('Matched T: TRAIN predictor, first tables half spread, second half DEV selection')
    dense = fused.prepare(study,log=log)
    heads = study.W//study.hd
    compact = grouped.prepare_groups(study,dense,heads,log=log)
    for constants in (dense,compact):
        constants['cos'],constants['sin'] = main['cos'].to(study.device),main['sin'].to(study.device)
    model = HeadSpreadCodec(compact,study.C,study.hd,Policy(),tokens_per_program=TB)
    records = []
    for doc in study.train:
        words = torch.stack([doc.words(l,study.device).to(torch.int16) for l in range(study.L)])
        features,sums = model.statistics(words)
        records.append(sums.cpu().numpy().copy())
        del words,features,sums
    learned = fit_head_map(records,study.hd)
    del records,model
    out = dict(format=SCHEMA,dense=_cpu({k:v for k,v in dense.items() if k not in ('cos','sin')}),
        grouped=_cpu({k:v for k,v in compact.items() if k not in ('cos','sin')}),
        policy=policy_record(Policy()),sha256='0'*64)
    root = dict(main,t_matched=out)
    words = torch.stack([dev.words(l,study.device).to(torch.int16) for l in range(study.L)])
    candidates = [(Policy(),None)] + [(Policy(2,a,0,learned.weights,learned.bias),None) for a in (.25,.5,.75,1.)]
    rows = []
    for policy,_ in candidates:
        out['policy'] = policy_record(policy)
        codec = MatchedT(root,'t-compact',study.device)
        parts,meta = codec.encode(words,pos0=dev.position)
        h,blob = bundle(parts,meta)
        decoded = torch.empty_like(words)
        codec.decode(parts,meta,lambda l,w:decoded[l].copy_(w))
        if not torch.equal(words,decoded):
            raise AssertionError('DEV candidate did not reconstruct every BF16 word')
        rows.append(dict(kind=policy.kind,alpha=policy.alpha,frame_bytes=blob.numel()+len(header_to_bytes(h)),
            policy_bytes=policy.array_bytes,policy_sha256=policy.digest(),exact=True))
        del parts,meta,h,blob,decoded,codec
    chosen = policy_choice(rows)
    selected = Policy() if chosen['kind']==0 else Policy(2,chosen['alpha'],0,learned.weights,learned.bias)
    out['policy'] = policy_record(selected)
    evidence = []
    for name in list(train)+[tables]:
        for suffix in ('K_bits.npy','V_bits.npy','ids.npy'):
            path = Path(captures)/(name+'_'+suffix)
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
            evidence.append(dict(name=path.name,sha256=h.hexdigest()))
    report = dict(format=SCHEMA,training=list(train),tables=tables,spread_range=[0,half],
        selection_range=[half,2*half],test_used_for_selection=False,serving_prompts_used_for_selection=False,
        selected=chosen,candidates=rows,inputs=evidence,TB=TB,chunks=1,
        state_origin='fresh matched campaign calibration; not the NUC frozen files',
        nominal_dense_sha256=consts_digest(dense,study.C,study.hd,TB,'uniform'),
        nominal_compact_sha256=consts_digest(compact,study.C,study.hd,TB,'uniform'))
    out['sha256'] = hashlib.sha256(json.dumps(report,sort_keys=True).encode()+selected.digest().encode()).hexdigest()
    out['report'] = report
    log('Matched T policy frozen: '+json.dumps(chosen,sort_keys=True))
    return out

"""AQUA-KV's released Quanto residual/cache path, with portable GPU framing.

The upstream class/method bodies are loaded unchanged from pinned sources.
Only their HF Cache base and rotary helper are supplied locally: the released
code targets Transformers 4.46, while this experiment uses 4.57/Qwen3. No HIGGS
imports, HF model patches, recent-token buffer or attention sinks are used.
This is a lossy, single-page codec adaptation, not paper serving reproduction.
Native Quanto unpack kernel bodies are unchanged; a small build adapter adds
the current CUDA stream and device guard to their host launches.
"""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, TypeVar

import numpy as np
from . import matched_transfer as matched, transfer_probability as prob
from .native_baseline_sources import AQUA_COMMIT, AQUA_FILES, QUANTO_VERSION, verify_sources


def _definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):raise ValueError('missing pinned AQUA definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def _rotary(q, k, cos, sin):
    import torch
    def one(x):
        a,b=x.chunk(2, dim=-1)
        return x*cos.unsqueeze(1)+torch.cat((-b,a),dim=-1)*sin.unsqueeze(1)
    return one(q), one(k)


def load_upstream(root):
    import torch
    from optimum import quanto
    if quanto.__version__ != QUANTO_VERSION:raise RuntimeError('install pinned optimum-quanto=='+QUANTO_VERSION)
    verify_sources(root, AQUA_FILES)
    from .quanto_unpack import ensure_cuda_unpack
    unpack_metadata=ensure_cuda_unpack()
    ns=dict(torch=torch, nn=torch.nn, TypeVar=TypeVar, Any=Any, Dict=Dict, Optional=Optional, Tuple=Tuple,
        MaxOptimizer=quanto.MaxOptimizer, qint2=quanto.qint2, qint4=quanto.qint4,
        quantize_weight=quanto.quantize_weight, apply_rotary_pos_emb=_rotary,
        transformers=SimpleNamespace(cache_utils=SimpleNamespace(Cache=object)))
    _definitions(Path(root)/'aquakv/quantizers.py', ['QuantizerBase','QuantoQuantizer'],ns)
    _definitions(Path(root)/'aquakv/cache_utils.py',
        ['SingleChunkQuantizedCacheWithPredictors','combine_heads','split_heads','apply_rotary_to_keys'],ns)
    # This small module has no optional model/quantizer dependencies.
    spec=importlib.util.spec_from_file_location('_matched_aqua_linear',Path(root)/'aquakv/linear_utils.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return SimpleNamespace(**ns, fit_linear_regression=module.fit_linear_regression,unpack_metadata=unpack_metadata)


def rope(positions, head_dim, theta, device):
    import torch
    inv=1./(theta**(torch.arange(0,head_dim,2,dtype=torch.float32,device=device)/head_dim))
    angle=torch.as_tensor(positions,dtype=torch.float32,device=device)[:,None]*inv[None,:]
    angle=torch.cat((angle,angle),dim=-1)[None]
    return angle.cos().to(torch.bfloat16), angle.sin().to(torch.bfloat16)


def fit_state(source, captures, geometry, bits, device, identity, *, damping=1e-3):
    """Released ridge solver; each next fit sees reconstructed TRAIN context."""
    import torch
    api=load_upstream(source);quantizer=api.QuantoQuantizer(bits,q_group_size=64,axis=0)
    keys=[];values=[]
    for a in captures:
        k=torch.from_numpy(np.asarray(a['K_bits']).copy()).to(device).view(torch.bfloat16)
        v=torch.from_numpy(np.asarray(a['V_bits']).copy()).to(device).view(torch.bfloat16)
        cos,sin=rope(a['positions'],geometry['head_dim'],geometry['theta'],device)
        k=k.reshape(k.shape[0],k.shape[1],geometry['heads'],geometry['head_dim']).transpose(1,2)
        # [layers, heads, tokens, dim]; broadcast the same angles per layer.
        _,k=_rotary(k,k,cos,-sin)
        keys.append(k.transpose(1,2).reshape(v.shape));values.append(v)
    k=torch.cat(keys,dim=1);v=torch.cat(values,dim=1);arrays={};old_k=k[0];old_v=v[0]
    def predictor(x,y,label):
        w,b=api.fit_linear_regression(x,y,reg_rate=damping,fit_intercept=True,
            compute_device=device,compute_dtype=torch.float32,chunk_size=4096)
        module=torch.nn.Linear(x.shape[1],y.shape[1],dtype=torch.bfloat16,device=device)
        module.weight.data.copy_(w);module.bias.data.copy_(b)
        arrays[label+'_weight']=module.weight.detach().view(torch.uint16).cpu().numpy().copy()
        arrays[label+'_bias']=module.bias.detach().view(torch.uint16).cpu().numpy().copy()
        return module(x)
    with torch.no_grad():
        for layer in range(1,k.shape[0]):
            predicted=predictor(old_k,k[layer],f'k{layer}')
            new_k=predicted+quantizer.quantize_dequantize(k[layer]-predicted)
            predicted=predictor(torch.cat((new_k,old_v),dim=-1),v[layer],f'v{layer}')
            old_v=predicted+quantizer.quantize_dequantize(v[layer]-predicted);old_k=new_k
    meta=dict(method='AQUA-KV Quanto page adaptation',upstream_commit=AQUA_COMMIT,quanto=QUANTO_VERSION,
        bits=bits,group_size=64,first_layer='raw unrotated BF16',geometry=geometry,damping=damping,
        fit_identity=identity,fit_rows=int(k.shape[1]),training_context='sequentially reconstructed TRAIN KV')
    state_hash=prob._array_hash(meta,arrays)
    return meta,arrays,state_hash


def save_state(path, meta, arrays, state_hash):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.part').open('wb') as f:
        np.savez(f,metadata=np.array(json.dumps(meta,sort_keys=True)),state_sha256=np.array(state_hash),**arrays)
    path.with_suffix('.part').replace(path)


class Aqua(matched.Adapter):
    lossless=False

    def __init__(self, device, source, state_path):
        import torch
        self.device=device;self.api=load_upstream(source)
        with np.load(state_path,allow_pickle=False) as a:
            self.meta=json.loads(str(a['metadata']));self.state_sha256=str(a['state_sha256'])
            arrays={n:a[n].copy() for n in a.files if n not in ('metadata','state_sha256')}
        if prob._array_hash(self.meta,arrays)!=self.state_sha256:raise ValueError('AQUA state hash mismatch')
        self.name=f"aquakv_quanto{self.meta['bits']}";self.g=self.meta['geometry']
        self.quantizer=self.api.QuantoQuantizer(self.meta['bits'],q_group_size=64,axis=0)
        self.kp={};self.vp={}
        for layer in range(1,self.g['layers']):
            for label,dst in [('k',self.kp),('v',self.vp)]:
                w=arrays[f'{label}{layer}_weight'];b=arrays[f'{label}{layer}_bias']
                module=torch.nn.Linear(w.shape[1],w.shape[0],dtype=torch.bfloat16,device=device)
                module.weight.data.copy_(torch.from_numpy(w).to(device).view(torch.bfloat16))
                module.bias.data.copy_(torch.from_numpy(b).to(device).view(torch.bfloat16));dst[layer]=module.eval()

    def _cache(self):
        return self.api.SingleChunkQuantizedCacheWithPredictors(quantizer=self.quantizer,
            first_layer_quantizer=None,key_predictors=self.kp,value_predictors=self.vp,move_predictors_to_devices=False)

    @staticmethod
    def _store(q, prefix, parts):
        import torch
        from optimum.quanto.tensor.weights.qbits import WeightQBitsTensor
        from optimum.quanto.tensor.packed import PackedTensor
        if type(q) is torch.Tensor:
            dtype=str(q.dtype).split('.')[-1]
            if dtype not in ('bfloat16','uint8','int32','float32'):raise ValueError('unexpected AQUA tensor dtype')
            parts[prefix]=q.detach().contiguous().view(torch.uint16) if dtype=='bfloat16' else q.detach().contiguous()
            return dict(part=prefix,dtype=dtype)
        if type(q) not in (WeightQBitsTensor,PackedTensor):raise ValueError('unexpected Quanto packing; group size 64 required')
        names,meta=q.__tensor_flatten__()
        return dict(kind=type(q).__name__,meta=meta,children={n:Aqua._store(getattr(q,n),prefix+n,parts) for n in names})

    @staticmethod
    def _restore(desc, parts):
        import torch
        from optimum.quanto.tensor.weights.qbits import WeightQBitsTensor
        from optimum.quanto.tensor.packed import PackedTensor
        if 'part' in desc:
            x=parts[desc['part']]
            return x.view(torch.bfloat16) if desc['dtype']=='bfloat16' else x
        cls={'WeightQBitsTensor':WeightQBitsTensor,'PackedTensor':PackedTensor}.get(desc['kind'])
        if cls is None:raise ValueError('unexpected Quanto tensor class')
        return cls.__tensor_unflatten__({n:Aqua._restore(d,parts) for n,d in desc['children'].items()},desc['meta'],None,None)

    def _check_shape(self, shape):
        if shape[0]!=self.g['layers'] or shape[2]!=2*self.g['heads']*self.g['head_dim']:
            raise ValueError('AQUA geometry differs from frozen predictors')

    def encode(self, words, start):
        import torch
        self._check_shape(words.shape);cache=self._cache();n=words.shape[1];c=words.shape[-1]//2
        cos,sin=rope(range(start,start+n),self.g['head_dim'],self.g['theta'],self.device)
        positions=torch.arange(start,start+n,device=self.device)
        with torch.no_grad():
            for layer in range(words.shape[0]):
                k=words[layer,:,:c].contiguous().view(torch.bfloat16).reshape(1,n,self.g['heads'],self.g['head_dim']).transpose(1,2)
                v=words[layer,:,c:].contiguous().view(torch.bfloat16).reshape(1,n,self.g['heads'],self.g['head_dim']).transpose(1,2)
                cache.update(k,v,layer,dict(cos=cos,sin=sin,cache_position=positions))
        parts={};desc={}
        for layer in range(words.shape[0]):
            for kind,states in [('k',cache.key_states_cache),('v',cache.value_states_cache)]:
                desc[f'{kind}{layer}']=self._store(states[layer],f'{kind}{layer}',parts)
        desc['cos']=self._store(cos,'cos',parts);desc['sin']=self._store(sin,'sin',parts)
        # Only compressed residuals, raw anchor and angles survive this call.
        return matched.Record(tuple(words.shape),start,parts,dict(tensors=desc,lossless=False))

    def decode(self, record):
        import torch
        matched.validate_record(record);self._check_shape(record.shape);cache=self._cache()
        desc=record.attributes['tensors'];cache.cos=self._restore(desc['cos'],record.parts);cache.sin=self._restore(desc['sin'],record.parts)
        cache.head_dim=self.g['head_dim'];empty=torch.empty((1,self.g['heads'],0,self.g['head_dim']),device=self.device,dtype=torch.bfloat16)
        for layer in range(record.shape[0]):
            cache.key_states_cache[layer]=self._restore(desc[f'k{layer}'],record.parts)
            cache.value_states_cache[layer]=self._restore(desc[f'v{layer}'],record.parts)
        output=[]
        with torch.no_grad():
            for layer in range(record.shape[0]):
                k,v=cache.update(empty,empty,layer)
                output.append(torch.cat((self.api.combine_heads(k)[0],self.api.combine_heads(v)[0]),dim=-1))
        return torch.stack(output).contiguous().view(torch.uint16)

    def statistics(self):
        n=sum(p.numel()*p.element_size() for d in (self.kp,self.vp) for m in d.values() for p in m.parameters())
        return dict(persistent_device_bytes=n,lossless=False,upstream_commit=AQUA_COMMIT,quantizer='upstream Quanto',
            native_unpack=self.api.unpack_metadata,
            residual_bits=self.meta['bits'],group_size=64,calibration=self.meta,
            integration='unchanged upstream method bodies; local HF Cache base/rotary compatibility and portable framing',
            retained_state='raw unrotated first layer, packed residuals, scales, shifts, BF16 cos/sin; all charged',
            quality_scope='KV reconstruction metrics; no end-to-end perplexity or LongBench result')

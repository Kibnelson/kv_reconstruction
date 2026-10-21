"""GPU T1 feasibility path: causal Block32 A/B parameters and binary rANS.

All KV arithmetic, probability construction and entropy coding run on the
selected CUDA device. CPU control reads the final byte count to size the
compacted output. Export/import moves compressed bytes and frames/checksums
them; it does not reconstruct KV. This is not T0/constriction compatible.
"""
from dataclasses import dataclass
import contextlib
import hashlib
import json
import math
import struct
import numpy as np
from . import transfer_reference as ref
from . import transfer_probability as prob
from .transfer_coset import BD,NTAB
from .transfer_gpu_predictor import TorchTransferPredictor,_require_torch,_copy_tensor

FORMAT='kv_T1_binary_rans_v1'
MAGIC=b'KVT1BR01'
HEADER=struct.Struct('<8sIQQ32s')


@dataclass
class Packet:
    payload:object
    directory:object
    shape:tuple
    start:int
    state_sha256:str
    raw:bool
    encoder_error:object=None
    encoder_workspace_capacity_bytes:int=0


class Engine:
    def __init__(self,base,state,device='cuda:0',*,_interpreter=False):
        t=_require_torch();d=t.device(device)
        if _interpreter:
            import os
            if d.type!='cpu' or os.environ.get('TRITON_INTERPRET')!='1':
                raise RuntimeError('CPU validation requires the explicit Triton interpreter')
        elif d.type!='cuda' or not t.cuda.is_available():
            raise RuntimeError('T1 requires CUDA; no CPU entropy fallback')
        if base.metadata_rank or base.mode!='block' or state.arm not in ('A','B') or state.hint_bits:
            raise ValueError('T1 supports metadata-free frozen block A/B only')
        actual_base=ref._state_digest(base)
        if base.state_sha256 and base.state_sha256!=actual_base:raise ValueError('base state changed after calibration')
        base.state_sha256=actual_base
        if state.base_sha256!=base.state_sha256:raise ValueError('adapter/base identity mismatch')
        if base.geometry['channels']%base.block_size or base.block_size%2:
            raise ValueError('channel groups must preserve complete key pairs and K/V boundaries')
        width=2*base.geometry['channels']
        for table in (base.cosine,base.sine):
            if table.shape!=(base.geometry['max_positions'],base.geometry['head_dim']//2) or table.dtype.kind not in 'iu' or np.any(table < -32768) or np.any(table > 32768):raise ValueError('invalid bounded integer RoPE tables')
        if state.scale_table.shape!=(33,) or not np.isfinite(state.scale_table).all() or np.any(state.scale_table<=0):raise ValueError('invalid frozen width multiplier table')
        groups=width//base.block_size
        if state.context_strength not in (0,2,4):raise ValueError('invalid context strength')
        if state.arm=='B':
            if state.coefficients.shape!=(base.geometry['layers'],groups,8) or state.coefficients.dtype.kind not in 'iu' or np.any(state.coefficients < -64*prob.Q) or np.any(state.coefficients > 64*prob.Q):raise ValueError('invalid bounded context coefficients')
            if state.thresholds.shape!=(base.geometry['layers'],groups,base.block_size,10) or state.thresholds.dtype.kind not in 'iu' or np.any(state.thresholds<0) or np.any(state.thresholds>1<<50):raise ValueError('invalid frozen bucket thresholds')
        if len(base.predictors)!=base.geometry['layers']:raise ValueError('wrong predictor layer count')
        for l,blocks in enumerate(base.predictors):
            if len(blocks)!=width//base.block_size:raise ValueError('wrong predictor group count')
            for g,block in enumerate(blocks):
                columns=np.asarray(block['columns']);available=(width if l else 0)+g*base.block_size
                if columns.ndim!=1 or columns.dtype.kind not in 'iu' or np.any(columns<0) or np.any(columns>=available):
                    raise ValueError('predictor refers to unavailable decoder context')
                if block['A'].shape!=(base.block_size,len(columns)) or block['b'].shape!=(base.block_size,) or block['inv_s'].shape!=(base.block_size,):
                    raise ValueError('invalid frozen block geometry')
                if not np.isfinite(block['inv_s']).all() or np.any(block['inv_s']<=0):raise ValueError('invalid logistic widths')
                with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                    deployed=block['inv_s'][:,None]*state.scale_table[None,:]
                if not np.isfinite(deployed).all() or np.any(deployed<=0):raise ValueError('adapted logistic widths overflow/underflow')
        self.base=base;self.adapter=state;self.interpreter=bool(_interpreter)
        self.p=TorchTransferPredictor.from_reference(base,device=d,coefficient_dtype='int32')
        self.device=self.p.device;self.t=t
        maximum=max((b['max_row_l1'] for layer in self.p.predictors for b in layer),default=0)
        limit=min(1<<40,((1<<61)-(1<<16))//max(1,2*maximum))
        if limit<65536:raise ValueError('coefficients exceed the bounded feature domain')
        self.feature_limit=1<<(limit.bit_length()-1)
        self.coefficients=_copy_tensor(state.coefficients,self.device,t.int64)
        self.thresholds=_copy_tensor(state.thresholds,self.device,t.int64)
        self.scales=_copy_tensor(state.scale_table,self.device,t.float64)
        self.boundaries=_copy_tensor(BD,self.device,t.float64)
        if base.f_tab.shape!=(NTAB,) or base.f_tab.dtype.kind not in 'iu' or base.f_tab[0]!=0 or base.f_tab[-1]!=(1<<32) or np.any(base.f_tab<0) or np.any(base.f_tab>1<<32) or np.any(np.diff(base.f_tab.astype(np.int64))<0):
            raise ValueError('invalid monotone logistic table')
        self.table=_copy_tensor(base.f_tab,self.device,t.int64)
        identity=dict(format=FORMAT,base=base.state_sha256,adapter=state.digest(),precision=12,
            feature_limit=self.feature_limit,probability='binary conditional BF16 rounding bins plus uniform 1/65536 word prior',
            lanes='one token, all layers/groups',boundaries_sha256=hashlib.sha256(BD.tobytes()).hexdigest())
        self.state_sha256=hashlib.sha256(prob.canonical(identity)).hexdigest()
        self.identity=identity

    def _shape(self,shape,start):
        layers,tokens,width=ref._shape(self.base,shape,start)
        if tokens>32:raise ValueError('T1 prototype accepts at most 32 tokens per independent chunk')
        return layers,tokens,width

    def _fixed(self,words):
        t=self.t
        value=(words.to(t.int32)<<16).contiguous().view(t.float32).to(t.float64)*65536.
        value=t.where(t.isfinite(value),value,t.zeros_like(value))
        return t.clamp(t.round(value),-self.feature_limit,self.feature_limit).to(t.int64)

    def _rope(self,value,ch,start,inverse=False):
        t=self.t;dim=self.base.geometry['head_dim']
        index=((ch+t.arange(0,value.shape[1],2,device=self.device))%dim)//2
        co=self.p.cosine[start:start+len(value)].index_select(1,index)
        si=self.p.sine[start:start+len(value)].index_select(1,index)
        if inverse:si=-si
        a,b=value[:,0::2],value[:,1::2];out=t.empty_like(value)
        out[:,0::2]=(a*co-b*si+(1<<14))>>15
        out[:,1::2]=(a*si+b*co+(1<<14))>>15
        return out

    def _mean(self,features,l,g,start):
        t=self.t;b=self.p.predictors[l][g];selected=features.index_select(1,b['columns'])
        acc=t.zeros((len(features),self.base.block_size),dtype=t.int64,device=self.device)
        for begin in range(0,selected.shape[1],64):
            acc+=(selected[:,None,begin:begin+64]*b['A'][None,:,begin:begin+64]).sum(2,dtype=t.int64)
        value=t.clamp(((acc+(1<<15))>>16)+b['b'],-self.feature_limit,self.feature_limit)
        if g*self.base.block_size<self.base.geometry['channels']:value=self._rope(value,g*self.base.block_size,start)
        return value

    def _context(self,previous,sums,last,count,tokens):
        t=self.t;x=t.zeros((tokens,8),dtype=t.int64,device=self.device);x[:,0]=16
        if previous is not None:
            w=previous.to(t.int64);c=w.shape[1]//2;e=((w>>7)&255)-127
            x[:,1]=t.div(e[:,:c].sum(1),c,rounding_mode='floor')
            x[:,2]=t.div(e[:,c:].sum(1),c,rounding_mode='floor')
            x[:,3]=t.div((1-2*((w>>15)&1)).sum(1)*16,2*c,rounding_mode='floor')
        if count:
            x[:,4]=t.div(sums[:,0],count,rounding_mode='floor')
            x[:,5]=t.div(last,self.base.block_size,rounding_mode='floor')
            x[:,6]=t.div(sums[:,1],count,rounding_mode='floor')
            x[:,7]=t.div(sums[:,2]*16,count,rounding_mode='floor')
        return t.clamp(x,-128,128)

    def _parameters(self,features,previous,sums,last,count,l,g,start):
        t=self.t;fixed=self._mean(features,l,g,start)
        if self.adapter.arm=='A':off=t.zeros(len(features),dtype=t.int64,device=self.device)
        else:
            context=self._context(previous,sums,last,count,len(features))
            acc=(context*self.coefficients[l,g]).sum(1,dtype=t.int64)*self.adapter.context_strength
            off=t.clamp(t.div(acc+2*prob.Q,4*prob.Q,rounding_mode='floor'),-prob.MAX_OFFSET,prob.MAX_OFFSET)
        inverse=(self.p.predictors[l][g]['inv_s'][None,:]*self.scales[off+prob.MAX_OFFSET,None]).contiguous()
        return fixed,fixed.to(t.float64)/65536.,inverse,off

    def _advance(self,target,fixed,l,g,start,features,offset,sums):
        t=self.t;ch=g*self.base.block_size;values=self._fixed(target)
        canonical=self._rope(values,ch,start,True) if ch<self.base.geometry['channels'] else values
        features[:,offset+ch:offset+ch+self.base.block_size]=canonical
        if self.adapter.arm=='A':return t.zeros(len(target),dtype=t.int64,device=self.device)
        difference=values-fixed;absolute=difference.abs();threshold=self.thresholds[l,g]
        buckets=((absolute[:,:,None]>0)&(absolute[:,:,None]>=threshold[None])).sum(2,dtype=t.int64)
        last=buckets.sum(1,dtype=t.int64)
        sums[:,0]+=last;sums[:,1]+=(buckets*difference.sign()).sum(1,dtype=t.int64)
        sums[:,2]+=((absolute>0)&(absolute>=threshold[None,:,5])).sum(1,dtype=t.int64)
        return last

    def _walk(self,words,start,packet=None,trace=False):
        t=self.t;layers,tokens,width=self._shape(tuple(words.shape),start);previous=t.empty((tokens,0),dtype=t.int64,device=self.device)
        records=[]
        if packet is not None:
            from . import transfer_t1_kernels as k
            xs=t.empty(tokens,dtype=t.int64,device=self.device);position=t.empty_like(xs);error=t.empty(tokens,dtype=t.int32,device=self.device)
            k.initialize_decoder[(tokens,)](packet.payload,packet.directory,xs,position,error,SIZE=packet.payload.numel(),num_warps=1)
        for l in range(layers):
            features=t.cat((previous,t.zeros((tokens,width),dtype=t.int64,device=self.device)),1)
            offset=previous.shape[1];sums=t.zeros((tokens,3),dtype=t.int64,device=self.device);last=t.zeros(tokens,dtype=t.int64,device=self.device)
            for g in range(len(self.base.predictors[l])):
                ch=g*self.base.block_size
                fixed,mu,inverse,off=self._parameters(features,None if l==0 else words[l-1],sums,last,ch,l,g,start)
                if packet is not None:
                    k.decode_group[(tokens,)](packet.payload,packet.directory,xs,position,error,mu,inverse,self.boundaries,self.table,words,
                        T=tokens,C=width,LAYER=l,CH=ch,GROUP=self.base.block_size,SIZE=packet.payload.numel(),num_warps=1,enable_fp_fusion=False)
                if packet is None or trace:records.append((l,g,mu,inverse,off))
                last=self._advance(words[l,:,ch:ch+self.base.block_size],fixed,l,g,start,features,offset,sums)
            previous=features[:,offset:]
        if packet is not None:
            k.finish_decoder[(tokens,)](xs,position,packet.directory,error,num_warps=1)
            return words,error,records
        return records

    def encode(self,words,start=0,*,raw_fallback=True):
        with contextlib.nullcontext() if self.interpreter else self.t.cuda.device(self.device):
            return self._encode(words,start,raw_fallback=raw_fallback)

    def _encode(self,words,start=0,*,raw_fallback=True):
        t=self.t
        if words.device!=self.device or words.dtype!=t.uint16 or not words.is_contiguous():raise ValueError('resident contiguous original UINT16 words required')
        self._shape(tuple(words.shape),start)
        parameters=self._walk(words,start)
        return self._encode_prepared(words,start,parameters,raw_fallback=raw_fallback)

    def _encode_prepared(self,words,start,parameters,*,raw_fallback=True):
        """Private timing control: entropy encode with source-derived parameters.

        This is never a decoder input or a standalone codec latency result.
        The normal encode path includes the preceding causal parameter walk.
        """
        t=self.t;layers,tokens,width=self._shape(tuple(words.shape),start)
        from . import transfer_t1_kernels as k
        xs=t.full((tokens,),1<<23,dtype=t.int64,device=self.device);count=t.zeros_like(xs)
        error=t.zeros(tokens,dtype=t.int32,device=self.device)
        # Each binary symbol emits at most two bytes; 16 symbols per word.
        cap=32*layers*width+16
        scratch=t.empty((tokens,cap),dtype=t.uint8,device=self.device)
        for l,g,mu,inverse,off in reversed(parameters):
            k.encode_group[(tokens,)](words,mu,inverse,self.boundaries,self.table,xs,count,scratch,error,
                T=tokens,C=width,LAYER=l,CH=g*self.base.block_size,GROUP=self.base.block_size,CAP=cap,num_warps=1,enable_fp_fusion=False)
        ends=t.cumsum(count+4,dim=0)
        # Only an allocation length crosses to the CPU. It is charged inside
        # encode timing; no KV/probability/context tensor leaves the GPU.
        size=int(ends[-1].item())
        # JSON "false" costs one more byte than "true" in the envelope.
        raw=bool(raw_fallback and size+tokens*16+1>=words.numel()*2)
        if raw:
            data=words.reshape(-1).view(t.uint8).clone();directory=t.empty((0,2),dtype=t.int64,device=self.device)
        else:
            data=t.empty(size,dtype=t.uint8,device=self.device);directory=t.empty((tokens,2),dtype=t.int64,device=self.device)
            k.compact_lanes[(tokens,)](scratch,xs,count,ends,directory,data,CAP=cap,BLOCK=256,num_warps=4)
        return Packet(data,directory,tuple(words.shape),start,self.state_sha256,raw,error,tokens*cap)

    def decode(self,packet,*,trace=False):
        with contextlib.nullcontext() if self.interpreter else self.t.cuda.device(self.device):
            return self._decode(packet,trace=trace)

    def _decode(self,packet,*,trace=False):
        t=self.t
        self._shape(packet.shape,packet.start)
        if packet.state_sha256!=self.state_sha256 or packet.payload.device!=self.device or packet.directory.device!=self.device:
            raise ValueError('wrong T1 state or device')
        if packet.payload.dtype!=t.uint8 or packet.payload.ndim!=1 or not packet.payload.is_contiguous():raise ValueError('invalid T1 payload tensor')
        if packet.directory.dtype!=t.int64 or not packet.directory.is_contiguous():raise ValueError('invalid T1 directory')
        tokens=packet.shape[1]
        if packet.raw:
            if packet.directory.numel() or packet.payload.numel()!=math.prod(packet.shape)*2:raise ValueError('invalid raw T1 length')
            return packet.payload.view(t.uint16).reshape(packet.shape).clone(),t.zeros(tokens,dtype=t.int32,device=self.device),[]
        if tuple(packet.directory.shape)!=(tokens,2):raise ValueError('invalid T1 lane geometry')
        words=t.zeros(packet.shape,dtype=t.uint16,device=self.device)
        return self._walk(words,packet.start,packet,trace)

    def statistics(self):
        base=self.p.storage_statistics()
        extra=sum(x.numel()*x.element_size() for x in [self.coefficients,self.thresholds,self.scales,self.boundaries,self.table])
        return dict(format=FORMAT,gpu=not self.interpreter,device=str(self.device),state_sha256=self.state_sha256,
            persistent_device_bytes=base['persistent_tensor_bytes']+extra,base_predictor_device_bytes=base['persistent_tensor_bytes'],
            CPU_state_numeric_bytes=sum(x.nbytes for x in ref._arrays(self.base).values())+sum(x.nbytes for x in self.adapter.arrays().values()),
            feature_limit=self.feature_limit,scope='state only; encoder parameter replay, rANS lane scratch and chunk output are additional measured peaks')


def model_identity_digest(model_identity):
    """Bind the complete canonical identity without repeating its manifest."""
    if not model_identity:raise ValueError('model identity required for external storage')
    return hashlib.sha256(prob.canonical(model_identity)).hexdigest()


def export_packet(packet,model_identity,*,identity_mode='sha256'):
    """D2H compressed-byte export and CPU envelope/checksum; no KV decode."""
    digest=model_identity_digest(model_identity)
    if identity_mode not in ('sha256','full'):raise ValueError('invalid model identity mode')
    if packet.encoder_error is not None and bool(packet.encoder_error.any().item()):raise ValueError('T1 encoder overflow/error')
    payload=packet.payload.cpu().numpy().tobytes()
    directory=packet.directory.cpu().numpy().astype('<i8',copy=False).tobytes()
    identity={'model_identity_sha256':digest} if identity_mode=='sha256' else {'model_identity':model_identity}
    meta=prob.canonical(dict(format=FORMAT,shape=list(packet.shape),start=packet.start,
        state_sha256=packet.state_sha256,raw=packet.raw,**identity))
    body=meta+directory+payload
    return HEADER.pack(MAGIC,len(meta),len(directory),len(payload),hashlib.sha256(body).digest())+body


def _read_envelope(data):
    data=bytes(data)
    if len(data)<HEADER.size:raise ValueError('truncated T1 envelope')
    magic,mn,dn,pn,digest=HEADER.unpack(data[:HEADER.size])
    body=data[HEADER.size:]
    if magic!=MAGIC or mn>65536 or mn+dn+pn!=len(body) or hashlib.sha256(body).digest()!=digest:raise ValueError('T1 envelope length/checksum mismatch')
    meta=json.loads(body[:mn])
    return meta,body,mn,dn,pn


def packet_accounting(data):
    """Verified byte counts; neither CPU KV reconstruction nor GPU work."""
    meta,body,mn,dn,pn=_read_envelope(data)
    return dict(complete_bytes=HEADER.size+len(body),payload_bytes=pn,directory_bytes=dn,
        envelope_bytes=HEADER.size+mn,identity_mode='sha256' if 'model_identity_sha256' in meta else 'full')


def import_packet(data,engine,model_identity):
    """Validate CPU envelope and upload only compressed payload/directory.

    Old full-identity envelopes remain readable. New envelopes carry the
    SHA256 of that same complete canonical identity; both forms require the
    receiver's independently supplied expected identity and saved state.
    """
    meta,body,mn,dn,pn=_read_envelope(data);shape=tuple(meta['shape'])
    full='model_identity' in meta;hashed='model_identity_sha256' in meta
    if full==hashed:raise ValueError('exactly one T1 model identity binding required')
    expected=model_identity_digest(model_identity)
    matches=meta['model_identity_sha256']==expected if hashed else meta['model_identity']==model_identity
    if meta['format']!=FORMAT or meta['state_sha256']!=engine.state_sha256 or not matches:raise ValueError('T1 model/state identity mismatch')
    engine._shape(shape,meta['start'])
    if type(meta['raw']) is not bool:raise ValueError('invalid raw mode')
    expected=0 if meta['raw'] else shape[1]*16
    if dn!=expected or (meta['raw'] and pn!=math.prod(shape)*2):raise ValueError('invalid T1 directory/raw length')
    directory=np.frombuffer(body[mn:mn+dn],dtype='<i8').copy().reshape(-1,2)
    if not meta['raw']:
        ends=directory[:,0]+directory[:,1]
        if directory[0,0]!=0 or np.any(directory[:,1]<4) or np.any(directory[:,0]<0) or np.any(ends<directory[:,0]) or ends[-1]!=pn or np.any(directory[1:,0]!=ends[:-1]):raise ValueError('invalid T1 lane extents')
    t=engine.t
    return Packet(t.from_numpy(np.frombuffer(body[mn+dn:],np.uint8).copy()).to(engine.device),
        t.from_numpy(directory).to(engine.device),shape,meta['start'],meta['state_sha256'],meta['raw'])

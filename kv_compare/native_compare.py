"""Exact native comparison adapters. No CPU substitute for a GPU codec.

Third-party kernels are imported from pinned checkouts. Their transfer records
include every value, directories, raw fallback blocks and partial-block tails.
Host packet serialization is separate from resident codec API timing.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch

from .head_spread_wire import _launch_copy, tensors_bytes

MAGIC = b"TCMPV001"
PREFIX = struct.Struct("<8sI32s")
DTYPES = {str(x): x for x in (torch.uint8, torch.uint16, torch.int16, torch.int32, torch.int64)}
METHODS = ("raw_bf16", "t_fast", "t_hbh", "splitzip", "exans")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()


def pinned_checkout(path, expected, tracked):
    path = Path(path).resolve()
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],capture_output=True,text=True,check=True)
    if result.stdout.strip() != expected:
        raise ValueError("source checkout has the wrong revision: " + str(path))
    dirty = subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--", tracked],capture_output=True)
    if dirty.returncode:
        raise ValueError("pinned source has local edits: " + str(path / tracked))
    return path


def load_splitzip(source):
    from .mixed import SPLITZIP_COMMIT
    source = pinned_checkout(source,SPLITZIP_COMMIT,"codec_gpu.py")
    name = "_t_compare_splitzip_" + SPLITZIP_COMMIT[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,source/"codec_gpu.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def block_layout(values, block_words):
    if values < 1 or block_words < 1: raise ValueError("invalid block geometry")
    blocks, tail = divmod(values,block_words)
    return dict(blocks=blocks,whole_values=blocks*block_words,tail_values=tail,tail_raw_bytes=tail*2)


def exans_paid_bytes(lengths, tail_values, block_words):
    if tail_values < 0 or tail_values >= block_words: raise ValueError("invalid raw tail")
    if any(n < 0 or n > 2*block_words for n in lengths): raise ValueError("invalid record length")
    # Zero directory entry means one raw block; nonzero is a fitted native record.
    return 4*len(lengths)+sum(n or 2*block_words for n in lengths)+2*tail_values


def pack_parts(method, shape, state_digest, words_digest, parts, aux=None):
    if method not in METHODS: raise ValueError("unknown codec")
    descriptors, buffers = [], []
    for name,t in parts:
        if str(t.dtype) not in DTYPES or t.layout != torch.strided:
            raise ValueError("unsupported packet tensor")
        data = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        descriptors.append(dict(name=name,dtype=str(t.dtype),shape=list(t.shape),bytes=len(data)))
        buffers.append(data)
    meta = dict(version=1,method=method,shape=list(shape),state_sha256=state_digest,
        words_sha256=words_digest,parts=descriptors,aux=aux or {})
    header = json.dumps(meta,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    payload = b"".join(buffers)
    body = header + payload
    packet = PREFIX.pack(MAGIC,len(header),hashlib.sha256(body).digest()) + body
    return packet,dict(packet_bytes=len(packet),header_bytes=PREFIX.size+len(header),
        payload_bytes=len(payload),component_bytes={d["name"]:d["bytes"] for d in descriptors},layout=aux or {})


def unpack_parts(packet, method, shape, state_digest, device):
    if len(packet) < PREFIX.size: raise ValueError("truncated comparison header")
    magic,hlen,checksum = PREFIX.unpack_from(packet)
    if magic != MAGIC or hlen > (1<<20) or len(packet)<PREFIX.size+hlen:
        raise ValueError("invalid comparison header")
    body = packet[PREFIX.size:]
    if hashlib.sha256(body).digest()!=checksum: raise ValueError("comparison packet checksum mismatch")
    meta=json.loads(body[:hlen])
    if (meta["version"]!=1 or meta["method"]!=method or tuple(meta["shape"])!=tuple(shape)
            or meta["state_sha256"]!=state_digest):
        raise ValueError("packet requires different codec, shape or state")
    offset=PREFIX.size+hlen;parts={}
    for d in meta["parts"]:
        dims=d["shape"]
        if d["name"] in parts or d["dtype"] not in DTYPES or any(not isinstance(x,int) or x<0 for x in dims):
            raise ValueError("invalid comparison tensor descriptor")
        dtype=DTYPES[d["dtype"]];count=math.prod(dims)
        n=count*torch.empty((),dtype=dtype).element_size()
        if n!=d["bytes"] or n<0 or offset+n>len(packet): raise ValueError("invalid comparison tensor length")
        if n:
            t=torch.frombuffer(bytearray(packet[offset:offset+n]),dtype=dtype).reshape(dims).to(device)
        else:
            t=torch.empty(dims,dtype=dtype,device=device)
        parts[d["name"]]=t;offset+=n
    if offset!=len(packet): raise ValueError("unpaid trailing packet bytes")
    return parts,meta


class Raw:
    allocation="zero-copy resident input; reusable reconstruction output"
    def __init__(self,shape,device):
        self.shape=tuple(shape);self.device=torch.device(device)
        self.out=torch.empty(shape,dtype=torch.int16,device=device)
    def encode(self,words):return words
    def decode(self,encoded):return self.out.copy_(encoded.reshape(self.shape))
    def parts(self,encoded):return [("words",encoded)],{}
    def receive(self,parts,meta):
        if set(parts)!={"words"}:raise ValueError("raw packet fields")
        if parts["words"].dtype!=torch.int16 or tuple(parts["words"].shape)!=self.shape:
            raise ValueError("raw packet geometry")
        return parts["words"]
    def shared_bytes(self):return 0
    def close(self):pass


class SplitZip:
    allocation="authors' encode/decode allocations; calibrated LUTs reused"
    def __init__(self,state,shape,device):
        self.shape=tuple(shape);self.device=torch.device(device)
        self.module=load_splitzip(state["source"])
        self.codec=self.module.ChunkLocalSplitZipGPU(device=str(device),chunk_size=1024)
        for name,t in state["tables"].items():setattr(self.codec,name,t.to(device))
    def encode(self,words):return self.codec.encode(words.view(torch.bfloat16))
    def decode(self,encoded):return self.codec.decode(encoded).view(torch.int16).reshape(self.shape)
    def parts(self,encoded):
        names=("pk","sm","chunk_id","local_pos","esc_val")
        parts=[(name,getattr(encoded,name)) for name in names]
        if sum(t.numel()*t.element_size() for _,t in parts)!=encoded.compressed_bytes:
            raise AssertionError("SplitZip native size differs from serialized components")
        return parts,dict(n=encoded.n,n_esc=encoded.n_esc,chunk_size=encoded.chunk_size)
    def receive(self,p,m):
        if set(p)!={"pk","sm","chunk_id","local_pos","esc_val"}:raise ValueError("SplitZip packet fields")
        expected={"pk":torch.uint8,"sm":torch.uint8,"chunk_id":torch.int32,"local_pos":torch.uint16,"esc_val":torch.uint8}
        if any(p[k].dtype!=v or p[k].ndim!=1 for k,v in expected.items()):raise ValueError("SplitZip packet dtypes")
        a=m["aux"];n=int(np.prod(self.shape));esc=p["esc_val"].numel()
        if (a["n"]!=n or a["n_esc"]!=esc or a["chunk_size"]!=1024
                or p["pk"].numel()!=(n+1)//2 or p["sm"].numel()!=n
                or p["chunk_id"].numel()!=esc or p["local_pos"].numel()!=esc):
            raise ValueError("SplitZip packet geometry")
        empty=torch.empty(0,dtype=torch.int32,device=self.device)
        return self.module.ChunkLocalGPUEncoded(p["pk"],p["sm"],empty,empty,
            p["chunk_id"],p["local_pos"],p["esc_val"],n,esc,1024)
    def shared_bytes(self):return tensors_bytes(vars(self.codec).values())
    def close(self):pass


@dataclass
class ExEncoded:
    records:torch.Tensor
    fits:tuple
    raw:torch.Tensor
    tail:torch.Tensor


class ExANS:
    allocation="native handle/workspaces, record arena and reconstruction output reused; raw fallback/tail copied"
    def __init__(self,state,shape,device):
        from . import exans_official as ex
        self.ex=ex;self.shape=tuple(shape);self.device=torch.device(device)
        self.layout=block_layout(int(np.prod(shape)),ex.BLOCK_WORDS)
        self.nb=self.layout["blocks"];self.lib=ex._load(state["build"]);self.handle=ctypes.c_void_p()
        self.records=torch.zeros(self.nb,ex.RECORD_BYTES,dtype=torch.uint8,device=device)
        self.out=torch.empty(shape,dtype=torch.int16,device=device)
        if self.nb:
            seg=(ctypes.c_uint32*1)(ex.BLOCK_BYTES)
            status=self.lib.openlake_expans_create(ex.BLOCK_BYTES,ex.RECORD_BYTES,seg,1,self.nb,ctypes.byref(self.handle))
            if status:self._error("create",status)
    def _error(self,operation,status):
        msg=(self.lib.openlake_expans_last_error(self.handle) or b"unknown").decode()
        raise RuntimeError(f"ExANS {operation} ({status}): {msg}")
    @staticmethod
    def u64(values):return (ctypes.c_uint64*len(values))(*values)
    @staticmethod
    def u32(values):return (ctypes.c_uint32*len(values))(*values)
    def encode(self,words):
        flat=words.reshape(-1)
        flags=(ctypes.c_uint8*self.nb)()
        if self.nb:
            addrs=self.u64([flat.data_ptr()+i*self.ex.BLOCK_BYTES for i in range(self.nb)])
            sizes=self.u32([self.ex.BLOCK_BYTES]*self.nb);offs=self.u32(list(range(self.nb+1)))
            stream=torch.cuda.current_stream(self.device)
            status=self.lib.openlake_expans_encode(self.handle,addrs,sizes,offs,self.nb,
                self.records.data_ptr(),flags,stream.cuda_stream)
            if status:self._error("encode",status)
        fits=tuple(bool(x) for x in flags)
        raw_idx=torch.tensor([i for i,fit in enumerate(fits) if not fit],dtype=torch.int64,device=self.device)
        blocks=flat[:self.layout["whole_values"]].view(self.nb,self.ex.BLOCK_WORDS)
        raw=blocks.index_select(0,raw_idx) if raw_idx.numel() else torch.empty(0,self.ex.BLOCK_WORDS,dtype=torch.int16,device=self.device)
        tail=flat[self.layout["whole_values"]:].clone()
        return ExEncoded(self.records,fits,raw,tail)
    def decode(self,encoded):
        fit_idx=[i for i,fit in enumerate(encoded.fits) if fit]
        if fit_idx:
            addrs=self.u64([self.out.data_ptr()+i*self.ex.BLOCK_BYTES for i in fit_idx])
            sizes=self.u32([self.ex.BLOCK_BYTES]*len(fit_idx));offs=self.u32(list(range(len(fit_idx)+1)))
            idx=self.u32(fit_idx);stream=torch.cuda.current_stream(self.device)
            status=self.lib.openlake_expans_decode(self.handle,encoded.records.data_ptr(),idx,len(fit_idx),
                addrs,sizes,offs,stream.cuda_stream)
            if status:self._error("decode",status)
        raw_idx=torch.tensor([i for i,fit in enumerate(encoded.fits) if not fit],dtype=torch.int64,device=self.device)
        if raw_idx.numel():
            self.out.view(-1)[:self.layout["whole_values"]].view(self.nb,self.ex.BLOCK_WORDS).index_copy_(0,raw_idx,encoded.raw)
        self.out.view(-1)[self.layout["whole_values"]:].copy_(encoded.tail)
        return self.out
    def parts(self,encoded):
        hdr=encoded.records[:,:self.ex.HEADER_BYTES].cpu().contiguous().numpy()
        lengths=np.zeros(self.nb,np.int32)
        for i,fit in enumerate(encoded.fits):
            if fit:
                magic=int.from_bytes(hdr[i,:4].tobytes(),"little")
                n=int.from_bytes(hdr[i,20:24].tobytes(),"little")
                if magic!=0x31505845 or not self.ex.FRAME_OFFSET<=n<=self.ex.RECORD_BYTES:
                    raise ValueError("invalid fitted ExANS header")
                lengths[i]=n
        ids=np.flatnonzero(lengths).astype(np.int64);sizes=lengths[ids].astype(np.int64)
        offsets=np.r_[0,np.cumsum(sizes)].astype(np.int64)
        compact=torch.empty(int(offsets[-1]),dtype=torch.uint8,device=self.device)
        if len(ids):
            og,bg,lg=[torch.from_numpy(a.copy()).to(self.device) for a in (offsets,ids*self.ex.RECORD_BYTES,sizes)]
            _launch_copy(encoded.records.reshape(-1),compact,og,bg,lg,True)
        parts=[("lengths",torch.from_numpy(lengths)),("records",compact),("raw_blocks",encoded.raw),("tail",encoded.tail)]
        paid=sum(t.numel()*t.element_size() for _,t in parts)
        if paid!=exans_paid_bytes(lengths.tolist(),encoded.tail.numel(),self.ex.BLOCK_WORDS):
            raise AssertionError("ExANS paid record/fallback/tail mismatch")
        return parts,dict(blocks=self.nb,block_words=self.ex.BLOCK_WORDS,fit_blocks=len(ids),raw_blocks=self.nb-len(ids),
            tail_values=encoded.tail.numel(),tail_raw_bytes=encoded.tail.numel()*2)
    def receive(self,p,m):
        if set(p)!={"lengths","records","raw_blocks","tail"}:raise ValueError("ExANS packet fields")
        expected={"lengths":torch.int32,"records":torch.uint8,"raw_blocks":torch.int16,"tail":torch.int16}
        if any(p[k].dtype!=v for k,v in expected.items()):raise ValueError("ExANS packet dtypes")
        lengths=p["lengths"].cpu().numpy().astype(np.int64).reshape(-1)
        if len(lengths)!=self.nb or np.any(lengths<0) or np.any(lengths>self.ex.RECORD_BYTES):
            raise ValueError("ExANS record lengths")
        if np.any((lengths>0)&(lengths<self.ex.FRAME_OFFSET)):raise ValueError("short fitted ExANS record")
        ids=np.flatnonzero(lengths).astype(np.int64);sizes=lengths[ids]
        if (int(sizes.sum())!=p["records"].numel() or p["tail"].numel()!=self.layout["tail_values"]
                or p["raw_blocks"].numel()!=(self.nb-len(ids))*self.ex.BLOCK_WORDS):
            raise ValueError("ExANS fallback or raw-tail geometry")
        records=torch.zeros_like(self.records)
        if len(ids):
            offsets=np.r_[0,np.cumsum(sizes)].astype(np.int64)
            og,bg,lg=[torch.from_numpy(a.copy()).to(self.device) for a in (offsets,ids*self.ex.RECORD_BYTES,sizes)]
            _launch_copy(p["records"],records.reshape(-1),og,bg,lg,False)
        return ExEncoded(records,tuple(bool(x) for x in lengths),p["raw_blocks"].reshape(-1,self.ex.BLOCK_WORDS),p["tail"].reshape(-1))
    def shared_bytes(self):return 0  # native opaque workspace is reported separately via device free-memory delta
    def close(self):
        if self.handle:
            self.lib.openlake_expans_destroy(self.handle);self.handle=ctypes.c_void_p()


class TransferT:
    allocation="frozen predictor, stream arena, decoder scratch and output reused"
    def __init__(self,method,state_path,shape,device):
        from .head_spread_wire import load_state,codec_for_state
        from .fast_dense_encode import FastDenseEncode
        self.method=method;self.shape=tuple(shape);self.device=torch.device(device)
        self.state=load_state(state_path,device);self.policy=self.state["loaded_policy"]
        if method=="t_fast":
            if self.policy.kind!=0 or self.state["consts"].get("groups",1)!=1:raise ValueError("fast T requires frozen dense global")
            k=dict(self.state["consts"]);k["cos"],k["sin"]=k["cos"][:shape[1]],k["sin"][:shape[1]]
            self.codec=FastDenseEncode(k,self.state["C"],self.state["hd"],tokens_per_program=self.state["TB"],mode=self.state["mode"])
            self.codec.policy=self.policy
        else:
            if self.state["consts"].get("groups",1)!=16 or self.policy.kind!=2:
                raise ValueError("compact T requires the frozen learned head-by-head state")
            self.codec=codec_for_state(self.state,shape[1])
        segments,capacity=self.codec.layout(shape[1])
        self.arena=torch.zeros(segments*capacity,dtype=torch.int16,device=device)
        self.out=torch.empty(shape,dtype=torch.int16,device=device);self.bufs=self.codec.buffers(shape[1])
    def encode(self,words):return self.codec.encode(words,out=self.arena)
    def decode(self,encoded):return self.codec.decode(*encoded,self.shape[1],out=self.out,bufs=self.bufs)
    def pack(self,encoded,words_digest):
        from .head_spread_wire import pack
        return pack(self.codec,*encoded,self.shape[1],self.state["predictor_sha256"],self.policy,words_digest)
    def receive_packet(self,packet):
        from .head_spread_wire import unpack
        arena,starts,info=unpack(self.codec,packet,self.state["predictor_sha256"])
        return (arena,starts),info
    def shared_bytes(self):return tensors_bytes(self.codec.k.values())+self.policy.array_bytes
    def close(self):pass


def make(method,state_path,shape,device):
    if method not in METHODS:raise ValueError("unknown exact method")
    if method.startswith("t_"):return TransferT(method,state_path,shape,device)
    if method=="splitzip":state=torch.load(state_path,map_location="cpu",weights_only=True)
    else:state=json.loads(Path(state_path).read_text())
    if state["method"]!=method:raise ValueError("state belongs to another codec")
    if method=="raw_bf16":return Raw(shape,device)
    if method=="exans" and sha256_file(state["build"]["library"])!=state["library_sha256"]:
        raise ValueError("native ExANS library changed")
    return {"splitzip":SplitZip,"exans":ExANS}[method](state,shape,device)

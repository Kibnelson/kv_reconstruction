"""Checked packets in reusable pinned host buffers; existing wire bytes unchanged.

An owned HostPacket may be handed to a transport accepting the buffer protocol.
This module performs no network I/O. Its view is valid until the slot is reused;
the caller must finish sending/receiving before reusing that slot. The benchmark
uses a serial host-buffer handoff and a separate receiver GPU arena.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

from . import head_spread_wire as wire
from . import native_compare as native


def _device(device, software):
    device = torch.device(device)
    if sys.byteorder != "little":
        raise RuntimeError("existing packet ABI requires a little-endian host")
    if device.type == "cuda":
        if software or os.environ.get("TRITON_INTERPRET", "0") not in ("", "0") or not torch.cuda.is_available():
            raise RuntimeError("actual compiled CUDA is required")
    elif device.type != "cpu" or not software or os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("CPU is limited to explicit interpreter software checks")
    return device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


@dataclass
class HostPacket:
    """A contiguous CPU byte tensor plus the exact transmitted byte length."""
    storage: torch.Tensor
    nbytes: int

    def __post_init__(self):
        if (self.storage.device.type != "cpu" or self.storage.dtype != torch.uint8
                or self.storage.ndim != 1 or not self.storage.is_contiguous()
                or not 0 <= self.nbytes <= self.storage.numel()):
            raise ValueError("invalid host packet buffer")

    def view(self):
        return memoryview(self.storage.numpy())[:self.nbytes].toreadonly()

    @classmethod
    def copy_from(cls, data, *, pin):
        """Cold import for files/validation, outside warm retrieval timing."""
        view = memoryview(data)
        buffer = torch.empty(len(view), dtype=torch.uint8, pin_memory=pin)
        memoryview(buffer.numpy())[:] = view
        return cls(buffer, len(view))


class _Slot:
    def __init__(self, capacity, device, software):
        self.device = _device(device, software)
        if capacity < 1:
            raise ValueError("positive packet capacity required")
        self.capacity = int(capacity)
        self.host = torch.empty(capacity, dtype=torch.uint8, pin_memory=self.device.type == "cuda")
        self.host_np = self.host.numpy()

    def _fits(self, nbytes):
        if nbytes > self.capacity:
            raise ValueError(f"packet needs {nbytes} bytes; preallocated capacity is {self.capacity}")

    def _input(self, packet):
        if not isinstance(packet, HostPacket):
            raise TypeError("receive requires an owned HostPacket")
        self._fits(packet.nbytes)
        if self.device.type == "cuda" and not packet.storage.is_pinned():
            raise ValueError("CUDA receive requires a pinned host packet")
        return packet.view()

    def poison_receiver(self):
        raise NotImplementedError


class PinnedT(_Slot):
    """Original T header/index/tails, without large temporary Python bytes."""
    def __init__(self, codec, tokens, predictor_digest, capacity, *, software=False):
        super().__init__(capacity, codec.device, software)
        self.codec, self.tokens, self.predictor = codec, int(tokens), predictor_digest
        self.segments, self.segment_capacity = codec.layout(tokens)
        self.payload_offset = wire.HEADER.size + 4*self.segments
        if capacity < self.payload_offset:
            raise ValueError("capacity cannot hold the T header and directory")
        words = (capacity-self.payload_offset)//2
        self.send_gpu = torch.empty(words, dtype=torch.int16, device=self.device)
        self.receive_gpu = torch.empty_like(self.send_gpu)
        self.receive_arena = torch.empty(self.segments*self.segment_capacity, dtype=torch.int16, device=self.device)
        self.starts_host = torch.empty(self.segments, dtype=torch.int32, pin_memory=self.device.type == "cuda")
        self.starts_np = self.starts_host.numpy()
        self.receive_starts = torch.empty(self.segments, dtype=torch.int32, device=self.device)
        self.plan_host = torch.zeros(3, self.segments+1, dtype=torch.int64, pin_memory=self.device.type == "cuda")
        self.plan_np = self.plan_host.numpy()
        self.plan_gpu = torch.empty_like(self.plan_host, device=self.device)
        self.ends = (np.arange(self.segments,dtype=np.int64)+1)*self.segment_capacity

    def _plan(self, bases, lengths):
        self.plan_np[0,0] = 0
        np.cumsum(lengths, out=self.plan_np[0,1:])
        self.plan_np[1,:self.segments] = bases
        self.plan_np[2,:self.segments] = lengths
        self.plan_gpu.copy_(self.plan_host, non_blocking=True)

    def _copy(self, source, target, pack):
        wire._launch_copy(source, target, self.plan_gpu[0],
            self.plan_gpu[1,:self.segments], self.plan_gpu[2,:self.segments], pack)

    def pack(self, encoded, words_digest):
        arena, starts = encoded
        if (arena.device != self.device or starts.device != self.device or arena.dtype != torch.int16
                or starts.dtype != torch.int32 or starts.numel()!=self.segments
                or arena.numel()!=self.segments*self.segment_capacity):
            raise ValueError("T encoder stream geometry")
        self.starts_host.copy_(starts, non_blocking=True)
        _sync(self.device)
        s = self.starts_np.astype(np.int64)
        bases = ((s-2*self.codec.hd)//8)*8
        lengths = self.ends-bases
        padding = s-bases-2*self.codec.hd
        if (np.any(bases<self.ends-self.segment_capacity) or np.any(lengths>=1<<29)
                or np.any(lengths<2*self.codec.hd+padding)
                or np.any(padding<0) or np.any(padding>7)):
            raise ValueError("invalid used stream tail")
        nwords = int(lengths.sum())
        nbytes = self.payload_offset+2*nwords
        self._fits(nbytes)
        desc = np.frombuffer(self.host_np,dtype="<u4",count=self.segments,offset=wire.HEADER.size)
        desc[:] = (lengths<<3)|padding
        self._plan(bases,lengths)
        self._copy(arena,self.send_gpu,True)
        self.host[self.payload_offset:nbytes].copy_(self.send_gpu[:nwords].view(torch.uint8),non_blocking=True)
        _sync(self.device)
        view = memoryview(self.host_np)
        checksum = hashlib.sha256(view[wire.HEADER.size:nbytes]).digest()
        c = self.codec
        wire.HEADER.pack_into(view,0,wire.MAGIC,1,c.L,self.tokens,c.W,c.C,c.hd,c.TB,c.ng,
            int(c.mode=="exact"),self.segments,nwords,0,bytes.fromhex(self.predictor),
            bytes.fromhex(c.policy.digest()),bytes.fromhex(words_digest),checksum)
        return HostPacket(self.host,nbytes)

    def receive(self, packet):
        view = self._input(packet)
        info = wire.inspect_packet(view)  # full SHA-256 retained, no payload copy
        c = self.codec
        if (info["predictor_sha256"]!=self.predictor or info["policy_sha256"]!=c.policy.digest()
                or (info["layers"],info["tokens"],info["width"],info["C"],info["hd"],info["TB"],info["groups"],info["mode"],info["position"])
                !=(c.L,self.tokens,c.W,c.C,c.hd,c.TB,c.ng,c.mode,0)):
            raise ValueError("packet requires different predictor, policy or geometry")
        d = np.frombuffer(view,dtype="<u4",count=self.segments,offset=wire.HEADER.size).astype(np.int64)
        lengths,padding = d>>3,d&7
        if (np.any(lengths>self.segment_capacity) or np.any(lengths<2*c.hd+padding)
                or int(lengths.sum())!=info["payload_words"]):
            raise ValueError("invalid used stream directory")
        bases = self.ends-lengths
        if np.any(bases%8):
            raise ValueError("unaligned rANS state")
        self.starts_np[:] = bases+2*c.hd+padding
        self._plan(bases,lengths)
        self.receive_starts.copy_(self.starts_host,non_blocking=True)
        nwords = info["payload_words"]
        self.receive_gpu[:nwords].view(torch.uint8).copy_(packet.storage[self.payload_offset:packet.nbytes],non_blocking=True)
        self._copy(self.receive_gpu,self.receive_arena,False)
        _sync(self.device)  # permits immediate safe reuse of pinned inputs/plans
        return (self.receive_arena,self.receive_starts),info["words_sha256"]

    def poison_receiver(self):
        self.receive_gpu.fill_(12345)
        self.receive_arena.fill_(12345)

    def memory(self):
        return dict(pinned_host_bytes=wire.tensors_bytes([self.host,self.starts_host,self.plan_host]),
            device_transfer_bytes=wire.tensors_bytes([self.send_gpu,self.receive_gpu,self.receive_arena,self.receive_starts,self.plan_gpu]),
            allocation="reused pinned packet, compaction and receive arena; original codec unchanged")


class PinnedParts(_Slot):
    """Raw/SplitZip/ExANS checked envelope; compact components copied once."""
    def __init__(self, adapter, method, shape, state_digest, capacity, *, software=False):
        super().__init__(capacity, adapter.device, software)
        self.adapter,self.method,self.shape,self.state = adapter,method,tuple(shape),state_digest
        self.receive_gpu = torch.empty(capacity+512,dtype=torch.uint8,device=self.device)

    def pack(self, encoded, words_digest):
        parts,aux = self.adapter.parts(encoded)
        if len(parts)>32 or len({name for name,_ in parts})!=len(parts):
            raise ValueError("invalid component directory")
        desc = []
        for name,t in parts:
            if str(t.dtype) not in native.DTYPES or t.layout!=torch.strided:
                raise ValueError("unsupported packet tensor")
            desc.append(dict(name=name,dtype=str(t.dtype),shape=list(t.shape),bytes=t.numel()*t.element_size()))
        meta = dict(version=1,method=self.method,shape=list(self.shape),state_sha256=self.state,
            words_sha256=words_digest,parts=desc,aux=aux or {})
        header = json.dumps(meta,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        nbytes = native.PREFIX.size+len(header)+sum(d["bytes"] for d in desc)
        self._fits(nbytes)
        offset = native.PREFIX.size
        memoryview(self.host_np)[offset:offset+len(header)] = header
        offset += len(header)
        for (_,t),d in zip(parts,desc):
            # Byte copies support odd component offsets without dtype alignment assumptions.
            self.host[offset:offset+d["bytes"]].copy_(t.detach().contiguous().view(torch.uint8).reshape(-1),non_blocking=True)
            offset += d["bytes"]
        _sync(self.device)
        view = memoryview(self.host_np)
        native.PREFIX.pack_into(view,0,native.MAGIC,len(header),hashlib.sha256(view[native.PREFIX.size:nbytes]).digest())
        return HostPacket(self.host,nbytes)

    def receive(self, packet):
        view = self._input(packet)
        if len(view)<native.PREFIX.size:
            raise ValueError("truncated comparison header")
        magic,hlen,checksum = native.PREFIX.unpack_from(view)
        if magic!=native.MAGIC or hlen>1<<20 or len(view)<native.PREFIX.size+hlen:
            raise ValueError("invalid comparison header")
        if hashlib.sha256(view[native.PREFIX.size:]).digest()!=checksum:
            raise ValueError("comparison packet checksum mismatch")
        meta = json.loads(bytes(view[native.PREFIX.size:native.PREFIX.size+hlen]))
        if (meta["version"]!=1 or meta["method"]!=self.method or tuple(meta["shape"])!=self.shape
                or meta["state_sha256"]!=self.state or len(meta["parts"])>32):
            raise ValueError("packet requires different codec, shape or state")
        offset,cursor,parts = native.PREFIX.size+hlen,0,{}
        for d in meta["parts"]:
            dims=d["shape"]
            if d["name"] in parts or d["dtype"] not in native.DTYPES or any(type(x)!=int or x<0 for x in dims):
                raise ValueError("invalid comparison tensor descriptor")
            dtype = native.DTYPES[d["dtype"]]
            n = math.prod(dims)*torch.empty((),dtype=dtype).element_size()
            cursor = (cursor+15)//16*16
            if n!=d["bytes"] or offset+n>len(view) or cursor+n>self.receive_gpu.numel():
                raise ValueError("invalid comparison tensor length")
            b = self.receive_gpu[cursor:cursor+n]
            b.copy_(packet.storage[offset:offset+n],non_blocking=True)
            parts[d["name"]] = b.view(dtype).reshape(dims)
            cursor += n; offset += n
        if offset!=len(view):
            raise ValueError("unpaid trailing packet bytes")
        _sync(self.device)
        return self.adapter.receive(parts,meta),meta["words_sha256"]

    def poison_receiver(self):
        self.receive_gpu.fill_(165)

    def memory(self):
        return dict(pinned_host_bytes=self.host.numel(),device_transfer_bytes=self.receive_gpu.numel(),
            allocation="reused pinned packet and component arena; native codec/record allocations unchanged")


def make(adapter, method, shape, state_digest, capacity, *, software=False):
    if method in ("t_fast","t_hbh"):
        return PinnedT(adapter.codec,shape[1],adapter.state["predictor_sha256"],capacity,software=software)
    return PinnedParts(adapter,method,shape,state_digest,capacity,software=software)

"""Transport harness calling the pinned OpenLake ExANS C ABI unchanged.

Fixed records use the author's vLLM geometry/alignment. Non-fitting blocks have
an explicitly charged raw fallback in this harness. This is not vLLM serving.
"""
import ctypes as C
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import torch
from .causal_packet import json_bytes
from .framed_baselines import frame, unframe

COMMIT="de6c38bcc5438475bd1abf1e9b3d154cef414ef6"
RAW=999424
RECORD=((RAW*10+12)//13+4095)//4096*4096  # official _expans_record_bytes
METHOD="exans_official_fixed_framed"
STATE=hashlib.sha256(json_bytes(dict(commit=COMMIT,raw=RAW,record=RECORD,version=2))).hexdigest()


class Exans:
    def __init__(self, source, library, device, blocks):
        self.device=torch.device(device)
        if self.device.type!="cuda":
            raise RuntimeError("the official ExANS codec requires actual CUDA")
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        spec=importlib.util.spec_from_file_location("exans_author_abi",Path(source)/"external/connectors/vllm/openlake_expans.py")
        self.module=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        self.lib=C.CDLL(str(library)); self.module._configure_abi(self.lib)
        self.handle=C.c_void_p(); self.blocks=blocks
        self.records=torch.empty(blocks*RECORD,dtype=torch.uint8,device=self.device)
        self.check(self.lib.openlake_expans_create(RAW,RECORD,self.arr(C.c_uint32,[RAW]),1,blocks,C.byref(self.handle)))

    def arr(self,kind,values):
        return self.module._array(kind,values)

    def check(self,status):
        if status:
            error=self.lib.openlake_expans_last_error(self.handle)
            raise RuntimeError("official ExANS: "+(error.decode() if error else str(status)))

    def close(self):
        if self.handle.value:
            self.lib.openlake_expans_destroy(self.handle); self.handle.value=None

    def encode(self,words):
        n=words.numel()*2; count=math.ceil(n/RAW)
        if count>self.blocks:
            raise ValueError("ExANS lane capacity exceeded")
        padded=torch.zeros(count*RAW,dtype=torch.uint8,device=self.device)
        padded[:n].copy_(words.contiguous().view(torch.uint8).flatten())
        fits=(C.c_uint8*count)()
        self.check(self.lib.openlake_expans_encode(self.handle,
            self.arr(C.c_uint64,[padded.data_ptr()+i*RAW for i in range(count)]),
            self.arr(C.c_uint32,[RAW]*count),self.arr(C.c_uint32,range(count+1)),count,
            self.records.data_ptr(),fits,torch.cuda.current_stream(self.device).cuda_stream))
        good=[i for i in range(count) if fits[i]]; bad=[i for i in range(count) if not fits[i]]
        gi=torch.tensor(good,dtype=torch.int64,device=self.device)
        bi=torch.tensor(bad,dtype=torch.int64,device=self.device)
        records=self.records[:count*RECORD].view(count,RECORD).index_select(0,gi).flatten()
        raw=padded.view(count,RAW).index_select(0,bi).flatten()
        return frame(METHOD,words.shape,STATE,[("fit_indices",gi.to(torch.int32)),
            ("raw_indices",bi.to(torch.int32)),("records",records),("raw",raw)],
            dict(blocks=count,raw_block_bytes=RAW,record_bytes=RECORD,input_padding_bytes=count*RAW-n))

    def decode(self,packet):
        shape,parts,extra=unframe(packet,METHOD,STATE)
        n=math.prod(shape)*2; count=math.ceil(n/RAW)
        if (extra!=dict(blocks=count,raw_block_bytes=RAW,record_bytes=RECORD,input_padding_bytes=count*RAW-n)
                or count>self.blocks or set(parts)!={"fit_indices","raw_indices","records","raw"}):
            raise ValueError("incompatible ExANS packet geometry")
        good=parts["fit_indices"].cpu().tolist(); bad=parts["raw_indices"].cpu().tolist()
        if (parts["fit_indices"].dtype!=torch.int32 or parts["raw_indices"].dtype!=torch.int32 or
                parts["records"].dtype!=torch.uint8 or parts["raw"].dtype!=torch.uint8 or
                sorted(good+bad)!=list(range(count)) or parts["records"].numel()!=len(good)*RECORD or
                parts["raw"].numel()!=len(bad)*RAW):
            raise ValueError("invalid ExANS slots/fallback directory")
        out=torch.empty(count,RAW,dtype=torch.uint8,device=self.device)
        if good:
            gi=torch.tensor(good,dtype=torch.int64,device=self.device)
            self.records[:count*RECORD].view(count,RECORD).index_copy_(0,gi,parts["records"].view(-1,RECORD))
            self.check(self.lib.openlake_expans_decode(self.handle,self.records.data_ptr(),
                self.arr(C.c_uint32,good),len(good),self.arr(C.c_uint64,[out.data_ptr()+i*RAW for i in good]),
                self.arr(C.c_uint32,[RAW]*len(good)),self.arr(C.c_uint32,range(len(good)+1)),
                torch.cuda.current_stream(self.device).cuda_stream))
        if bad:
            bi=torch.tensor(bad,dtype=torch.int64,device=self.device)
            out.index_copy_(0,bi,parts["raw"].view(-1,RAW))
        return out.flatten()[:n].view(torch.int16).reshape(shape)

    @staticmethod
    def accounting(packet):
        h=packet.header
        fields={f["name"]:f for f in h["fields"]}
        return dict(input_padding_bytes=h["extra"]["input_padding_bytes"],
            fitting_blocks=fields["fit_indices"]["count"],raw_fallback_blocks=fields["raw_indices"]["count"],
            charged_fixed_record_bytes=fields["records"]["bytes"],charged_raw_fallback_bytes=fields["raw"]["bytes"],
            native_allocation_peaks="unmeasured",raw_block_bytes=RAW,record_bytes=RECORD,
            scope="official C ABI with fixed records; local framing, physical padding and raw overflow fallback charged")

"""Native pinned OpenLake ExANS v2 through its released C ABI.

Original kernels, histogram, phase selection and nvCOMP ANS are unchanged.
A recorded one-line host allocation-error type fix is needed to compile.
This adapter pads Qwen pages to upstream's 999424-byte physical blocks,
compacts each successful native record, and charges raw fallback on overflow.
Native CUDA allocations and synchronizations are retained and disclosed.
"""
import ctypes as C
import json
import math
from pathlib import Path

import numpy as np
from . import matched_transfer as matched
from .native_baseline_sources import EXANS_COMMIT, EXANS_FILES, sha_file, verify_sources

RAW_BYTES=999424
RECORD_BYTES=770048  # exactly the released OpenLake smoke test's slot size
_SYMBOLS=122*4096
_PHASE_BYTES=(_SYMBOLS//128)//8+((122*2+31)//32)*4
FRAME_OFFSET=((64+_SYMBOLS+_PHASE_BYTES+15)//16)*16
HEADER_MAGIC=0x31505845


def block_count(raw_bytes):
    if raw_bytes<=0 or raw_bytes%2:raise ValueError('positive BF16 byte count required')
    return (raw_bytes+RAW_BYTES-1)//RAW_BYTES


def valid_native_length(header):
    import struct
    if len(header)!=64:raise ValueError('truncated ExANS header')
    magic,version,size=struct.unpack_from('<IHH',header)
    raw,actual=struct.unpack_from('<II',header,16)
    frame_offset,frame_bytes=struct.unpack_from('<II',header,40)
    if (magic!=HEADER_MAGIC or version!=2 or size!=64 or raw!=RAW_BYTES or
            not FRAME_OFFSET<actual<=RECORD_BYTES or frame_offset!=FRAME_OFFSET or frame_bytes!=actual-frame_offset):
        raise ValueError('invalid native ExANS header or compressed size')
    return actual


def _array(values,dtype):return np.ascontiguousarray(values,dtype=dtype)
def _ptr(a):return C.c_void_p(a.ctypes.data)


class Exans(matched.Adapter):
    name='exans_native'

    def __init__(self, device, source, max_raw_bytes):
        import torch
        self.device=torch.device(device);self.source=Path(source);self.max_blocks=block_count(max_raw_bytes)
        if self.device.type!='cuda':raise ValueError('native ExANS requires actual CUDA')
        verify_sources(self.source,EXANS_FILES)
        build=json.loads((self.source/'build.json').read_text())
        library=self.source/'libopenlake_expans.so'
        if build['commit']!=EXANS_COMMIT or build['library_sha256']!=sha_file(library):
            raise ValueError('ExANS build identity differs from pinned sources')
        self.build=build;self.lib=C.CDLL(str(library));self.handle=C.c_void_p()
        # All arrays in this C ABI are HOST descriptor arrays. Payloads are
        # device pointers; native gather/scatter kernels move the actual KV.
        self.lib.openlake_expans_create.argtypes=[C.c_uint64,C.c_uint64,C.c_void_p,C.c_uint32,C.c_uint32,C.POINTER(C.c_void_p)]
        self.lib.openlake_expans_create.restype=C.c_int
        self.lib.openlake_expans_destroy.argtypes=[C.c_void_p]
        self.lib.openlake_expans_destroy.restype=None
        self.lib.openlake_expans_last_error.argtypes=[C.c_void_p];self.lib.openlake_expans_last_error.restype=C.c_char_p
        self.lib.openlake_expans_encode.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32,C.c_uint64,C.c_void_p,C.c_uint64]
        self.lib.openlake_expans_encode.restype=C.c_int
        self.lib.openlake_expans_decode.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint64]
        self.lib.openlake_expans_decode.restype=C.c_int
        segment=_array([RAW_BYTES],np.uint32)
        with torch.cuda.device(self.device):
            self._call(self.lib.openlake_expans_create(RAW_BYTES,RECORD_BYTES,_ptr(segment),1,self.max_blocks,C.byref(self.handle)))
        self.state_sha256=matched.digest(dict(method=self.name,commit=EXANS_COMMIT,version=2,raw_bytes=RAW_BYTES,
            record_bytes=RECORD_BYTES,segments=[RAW_BYTES],centers='native constant 127',overflow='charged raw block'))

    def _call(self, status):
        if status:
            reason=self.lib.openlake_expans_last_error(self.handle)
            raise RuntimeError('native ExANS: '+(reason.decode() if reason else f'status {status}'))

    def close(self):
        if getattr(self,'handle',None):self.lib.openlake_expans_destroy(self.handle);self.handle=C.c_void_p()

    def __del__(self):self.close()

    def _stream(self):
        import torch
        return int(torch.cuda.current_stream(self.device).cuda_stream)

    def encode(self, words, start):
        import torch
        raw=words.contiguous().view(torch.uint8).reshape(-1);n=raw.numel();blocks=block_count(n)
        if blocks>self.max_blocks:raise ValueError('ExANS batch exceeds initialized capacity')
        padded=torch.zeros(blocks*RAW_BYTES,dtype=torch.uint8,device=self.device);padded[:n].copy_(raw)
        records=torch.empty((blocks,RECORD_BYTES),dtype=torch.uint8,device=self.device)
        addresses=_array([padded.data_ptr()+i*RAW_BYTES for i in range(blocks)],np.uint64)
        sizes=_array([RAW_BYTES]*blocks,np.uint32);offsets=_array(range(blocks+1),np.uint32);fits=np.zeros(blocks,np.uint8)
        self._call(self.lib.openlake_expans_encode(self.handle,_ptr(addresses),_ptr(sizes),_ptr(offsets),blocks,
            records.data_ptr(),_ptr(fits),self._stream()))
        parts={};directory=[];fallbacks=0
        for i in range(blocks):
            valid=min(RAW_BYTES,n-i*RAW_BYTES)
            if fits[i]:
                length=valid_native_length(records[i,:64].cpu().numpy().tobytes())
                parts[str(i)]=records[i,:length].clone();kind='exans'
            else:
                length=valid;parts[str(i)]=padded[i*RAW_BYTES:i*RAW_BYTES+valid].clone();kind='raw';fallbacks+=1
            directory.append(dict(kind=kind,original_bytes=valid,stored_bytes=length))
        return matched.Record(tuple(words.shape),start,parts,dict(blocks=directory,raw_chunks=fallbacks,
            native_block_bytes=RAW_BYTES,native_slot_bytes=RECORD_BYTES,padded_input_bytes=blocks*RAW_BYTES,
            padding_bytes=blocks*RAW_BYTES-n,physical_record_reservation_bytes=blocks*RECORD_BYTES))

    def decode(self, record):
        import torch
        matched.validate_record(record);n=math.prod(record.shape)*2;blocks=block_count(n);a=record.attributes
        if (blocks>self.max_blocks or len(a['blocks'])!=blocks or set(record.parts)!={str(i) for i in range(blocks)}
                or a['native_block_bytes']!=RAW_BYTES or a['native_slot_bytes']!=RECORD_BYTES):
            raise ValueError('invalid ExANS physical block directory')
        padded=torch.zeros(blocks*RAW_BYTES,dtype=torch.uint8,device=self.device)
        records=torch.zeros((blocks,RECORD_BYTES),dtype=torch.uint8,device=self.device);native=[];addresses=[]
        for i,m in enumerate(a['blocks']):
            data=record.parts[str(i)];valid=min(RAW_BYTES,n-i*RAW_BYTES)
            if data.dtype!=torch.uint8 or data.ndim!=1 or data.numel()!=m['stored_bytes'] or m['original_bytes']!=valid:
                raise ValueError('invalid ExANS block length')
            if m['kind']=='raw':
                if data.numel()!=valid:raise ValueError('invalid ExANS raw fallback')
                padded[i*RAW_BYTES:i*RAW_BYTES+valid].copy_(data)
            elif m['kind']=='exans':
                if data.numel()>RECORD_BYTES or data.numel()<=FRAME_OFFSET:raise ValueError('invalid ExANS native record size')
                if valid_native_length(data[:64].cpu().numpy().tobytes())!=data.numel():raise ValueError('ExANS header/part length mismatch')
                records[i,:data.numel()].copy_(data);native.append(i);addresses.append(padded.data_ptr()+i*RAW_BYTES)
            else:raise ValueError('unknown ExANS block mode')
        if native:
            indices=_array(native,np.uint32);addresses=_array(addresses,np.uint64)
            sizes=_array([RAW_BYTES]*len(native),np.uint32);offsets=_array(range(len(native)+1),np.uint32)
            self._call(self.lib.openlake_expans_decode(self.handle,records.data_ptr(),_ptr(indices),len(native),
                _ptr(addresses),_ptr(sizes),_ptr(offsets),self._stream()))
        return padded[:n].view(torch.uint16).reshape(record.shape).clone()

    def statistics(self):
        return dict(persistent_device_bytes=None,upstream_commit=EXANS_COMMIT,record_version=2,
            upstream_kernels_unmodified=True,host_build_fix=self.build['host_build_fix'],
            native_block_bytes=RAW_BYTES,native_slot_bytes=RECORD_BYTES,
            native_max_batch_blocks=self.max_blocks,build=self.build,centers='upstream constant 127; no external calibration',
            overflow='raw BF16 block stored and charged',
            synchronization='upstream encode/decode synchronize CUDA stream; included in wall time',
            allocator_scope='Torch allocations measured; upstream cudaMalloc state/scratch peaks unmeasured')

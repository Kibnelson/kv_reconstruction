"""Native CPU decoder for unchanged learned_codec LKVR v1 records.

NativeDecoder snapshots and hashes its frozen model once. The compatibility
function caches such a snapshot by model identity; callers must treat a model
as immutable after first use. This is a CPU implementation, not a GPU kernel.
The C++ shared object is built lazily with the local g++ compiler.
"""
from __future__ import annotations
import ctypes
import hashlib
from pathlib import Path
import struct
import subprocess
import threading
import numpy as np
try:
    from . import learned_codec
except ImportError:
    import learned_codec

_HEADER=struct.Struct('<4sBBHIII32sH')
_LIB=None
_LOCK=threading.Lock()
_CACHE={}

def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is None:
            source=Path(__file__).with_suffix('.cpp')
            target=source.with_name('_native_decode_cpu.so')
            if not target.exists() or target.stat().st_mtime_ns<source.stat().st_mtime_ns:
                subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(target)],check=True)
            lib=ctypes.CDLL(str(target))
            f=lib.lkv_decode_body
            f.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint,
                        ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
            f.restype=ctypes.c_int
            _LIB=lib
    return _LIB

class NativeDecoder:
    def __init__(self, model):
        # A separate immutable snapshot prevents later codebook-array changes
        # from silently changing the frozen interpretation of a cache record.
        frozen_bytes=learned_codec.model_bytes(model)
        snapshot=learned_codec.load_model(frozen_bytes)
        self.identity=hashlib.sha256(frozen_bytes).digest()
        self.qbits=(snapshot.n_centroids-1).bit_length()
        self.n_centroids=snapshot.n_centroids
        self.codebooks={k:np.ascontiguousarray(v,dtype=np.uint16) for k,v in snapshot.codebooks.items()}
        for book in self.codebooks.values(): book.flags.writeable=False
        self.lib=_library()

    def decode_matrix(self, encoded):
        data=encoded.data if isinstance(encoded,learned_codec.EncodedMatrix) else bytes(encoded)
        if len(data)<_HEADER.size: raise ValueError('truncated learned matrix')
        magic,version,qbits,flags,group,rows,columns,identity,keylen=_HEADER.unpack_from(data)
        if magic!=b'LKVR' or version!=1 or flags or group<8 or group%8:
            raise ValueError('invalid learned matrix header')
        if identity!=self.identity: raise ValueError('learned model SHA256 mismatch')
        if qbits!=self.qbits: raise ValueError('inconsistent learned index width')
        offset=_HEADER.size+keylen
        if offset>len(data): raise ValueError('truncated dictionary name')
        key=data[_HEADER.size:offset].decode('utf8')
        if key not in self.codebooks: raise ValueError('unknown learned dictionary key')
        body=data[offset:]
        total=rows*columns
        # Every compressed vector must at least retain one byte of signs.
        # This loose bound rejects malicious dimensions before allocation.
        if total>len(body)*8: raise ValueError('matrix dimensions exceed record payload')
        out=np.empty((rows,columns),dtype=np.uint16)
        book=self.codebooks[key]
        error=ctypes.create_string_buffer(256)
        body_buffer=ctypes.c_char_p(body)
        status=self.lib.lkv_decode_body(body_buffer,len(body),book.ctypes.data,self.n_centroids,qbits,
            total,group,out.ctypes.data,error,len(error))
        if status: raise ValueError(error.value.decode('utf8'))
        return out

def decode_matrix(encoded,model):
    """Compatible with learned_codec.decode_matrix for a held-fixed model."""
    key=id(model)
    cached=_CACHE.get(key)
    if cached is None or cached[0] is not model:
        cached=(model,NativeDecoder(model))
        _CACHE[key]=cached
    return cached[1].decode_matrix(encoded)

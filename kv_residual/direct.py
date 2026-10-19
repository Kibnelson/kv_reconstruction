"""Magnitude/sign control using the same lossless backend without prediction."""
from dataclasses import dataclass
import struct
import zlib
import numpy as np
from kv_residual import entropy

HEADER=struct.Struct('<4sB3xIIIII')

@dataclass
class EncodedMatrix:
    data:bytes
    stats:dict

def encode_matrix(words,strategy='best'):
    words=np.asarray(words)
    if words.dtype!=np.uint16 or words.ndim!=2:raise ValueError('expected uint16 matrix')
    lower=entropy.encode((words&32767).astype(np.int32),strategy)
    signs=np.packbits((words.reshape(-1)>>15).astype(np.uint8),bitorder='little').tobytes()
    raw=np.ascontiguousarray(words).astype('<u2').tobytes()
    header=HEADER.pack(b'KVDR',1,*words.shape,len(signs),len(lower.data),zlib.crc32(raw))
    data=header+signs+lower.data
    parts={'header':len(header),'exact_signs':len(signs),'magnitude_stream':len(lower.data)}
    return EncodedMatrix(data,dict(serialized_bytes=len(data),raw_payload_bytes=words.nbytes,component_bytes=parts,
        magnitude_modes={str(lower.stats.get('mode',lower.stats.get('strategy','unknown'))):1}))

def decode_matrix(blob):
    blob=bytes(blob)
    if len(blob)<HEADER.size:raise ValueError('truncated direct header')
    magic,version,rows,cols,nsign,nmag,crc=HEADER.unpack_from(blob)
    n=rows*cols
    if magic!=b'KVDR' or version!=1 or nsign!=(n+7)//8 or len(blob)!=HEADER.size+nsign+nmag:
        raise ValueError('invalid direct header/length')
    signs=np.unpackbits(np.frombuffer(blob[HEADER.size:HEADER.size+nsign],dtype=np.uint8),bitorder='little')
    if np.any(signs[n:]):raise ValueError('nonzero sign padding')
    magnitude=entropy.decode(blob[HEADER.size+nsign:])
    if magnitude.shape!=(rows,cols) or np.any(magnitude<0) or np.any(magnitude>32767):raise ValueError('invalid magnitude')
    words=(magnitude.astype(np.uint16)|(signs[:n].astype(np.uint16).reshape(rows,cols)<<15))
    if zlib.crc32(words.astype('<u2').tobytes())!=crc:raise ValueError('direct checksum mismatch')
    return words

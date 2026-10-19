"""Independent record-level Deflate stage; exact original bytes are retained."""
from dataclasses import dataclass
import struct,zlib
import numpy as np

HEADER=struct.Struct('<4sBII')
RAW_HEADER=struct.Struct('<4sBII')

@dataclass(frozen=True)
class EncodedMatrix:
    data:bytes
    stats:dict

def wrap(inner,level=6):
    source=inner.data if hasattr(inner,'data') else bytes(inner)
    obj=zlib.compressobj(level,zlib.DEFLATED,-15)
    compressed=obj.compress(source)+obj.flush()
    mode,payload=(1,compressed) if len(compressed)<len(source) else (0,source)
    blob=HEADER.pack(b'KVZ1',mode,len(source),zlib.crc32(source))+payload
    return EncodedMatrix(blob,dict(serialized_bytes=len(blob),inner_serialized_bytes=len(source),
        component_bytes={'wrapper_header':HEADER.size,'stored_inner_stream':len(payload)},
        wrapper_mode_counts={str(mode):1},deflate_level=level))

def unwrap(blob,max_bytes=32*1024*1024):
    data=blob.data if hasattr(blob,'data') else bytes(blob)
    if len(data)<HEADER.size:raise ValueError('truncated stack header')
    magic,mode,size,crc=HEADER.unpack_from(data)
    if magic!=b'KVZ1' or mode not in (0,1) or size>max_bytes:raise ValueError('invalid stack header')
    payload=data[HEADER.size:]
    if mode==0:out=payload
    else:
        dec=zlib.decompressobj(-15)
        try:out=dec.decompress(payload,size+1)
        except zlib.error as e:raise ValueError('invalid Deflate stream') from e
        if not dec.eof or dec.unused_data or dec.unconsumed_tail:raise ValueError('incomplete or trailing Deflate stream')
    if len(out)!=size or zlib.crc32(out)!=crc:raise ValueError('stack length or checksum mismatch')
    return out

def raw_matrix(words,shuffle=False):
    words=np.asarray(words)
    if words.dtype!=np.uint16 or words.ndim!=2:raise ValueError('expected uint16 matrix')
    flat=np.ascontiguousarray(words).reshape(-1)
    if shuffle:
        payload=(flat&255).astype(np.uint8).tobytes()+(flat>>8).astype(np.uint8).tobytes()
    else:payload=flat.astype('<u2').tobytes()
    data=RAW_HEADER.pack(b'KVRW',int(shuffle),*words.shape)+payload
    return EncodedMatrix(data,dict(serialized_bytes=len(data),
        component_bytes={'raw_matrix_header':RAW_HEADER.size,'raw_words':len(payload)}))

def decode_raw_matrix(blob):
    if len(blob)<RAW_HEADER.size:raise ValueError('truncated raw matrix')
    magic,mode,rows,columns=RAW_HEADER.unpack_from(blob)
    if magic!=b'KVRW' or mode not in (0,1):raise ValueError('invalid raw matrix header')
    n=rows*columns;payload=blob[RAW_HEADER.size:]
    if len(payload)!=2*n:raise ValueError('wrong raw matrix payload size')
    if mode==0:words=np.frombuffer(payload,dtype='<u2').copy()
    else:
        b=np.frombuffer(payload,dtype=np.uint8).astype(np.uint16)
        words=b[:n]|(b[n:]<<8)
    return words.reshape(rows,columns)

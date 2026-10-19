"""Entropy-only controls for the remaining exact sign/fraction bytes."""
from dataclasses import dataclass
import struct
import zlib
import numpy as np
from kv_stacked import huffman_codec as huffman,wrapper

HEADER=struct.Struct('<4sHII')
SPLIT_LENGTH=struct.Struct('<I')

@dataclass
class EncodedLower:
    data:bytes
    stats:dict

def encode_lower(lower,exponents,strategy='deflate'):
    lower,exponents=np.asarray(lower),np.asarray(exponents)
    if lower.dtype!=np.uint8 or exponents.dtype!=np.uint8 or lower.ndim!=2 or lower.shape!=exponents.shape:raise ValueError('invalid lower fields')
    if strategy=='deflate':return wrapper.wrap(lower.tobytes(),6)
    if strategy=='split_deflate':
        # Entropy/layout control only: preserve all original sign/fraction bits.
        flat=np.ascontiguousarray(lower).reshape(-1)
        signs=np.packbits(flat>>7,bitorder='little').tobytes()
        fraction_bits=(flat[:,None]>>np.arange(7,dtype=np.uint8))&1
        fractions=np.packbits(fraction_bits.reshape(-1),bitorder='little').tobytes()
        sign_record,fraction_record=wrapper.wrap(signs,6),wrapper.wrap(fractions,6)
        data=SPLIT_LENGTH.pack(len(sign_record.data))+sign_record.data+fraction_record.data
        parts={'split_length_header':SPLIT_LENGTH.size}
        parts.update({'sign_'+k:v for k,v in sign_record.stats['component_bytes'].items()})
        parts.update({'fraction_'+k:v for k,v in fraction_record.stats['component_bytes'].items()})
        assert sum(parts.values())==len(data)
        return EncodedLower(data,{'serialized_bytes':len(data),'raw_lower_bytes':int(lower.nbytes),
            'component_bytes':parts,'deflate_level':6,'description':'Entropy-only separate packed sign/fraction streams',
            'sign_packed_bytes':len(signs),'fraction_packed_bytes':len(fractions),
            'sign_wrapper_mode_counts':sign_record.stats['wrapper_mode_counts'],
            'fraction_wrapper_mode_counts':fraction_record.stats['wrapper_mode_counts']})
    if strategy!='huffman':raise ValueError('unknown lower strategy')
    lengths=huffman._lengths(lower.reshape(-1))
    table=bytes(v for s in sorted(lengths) for v in (s,lengths[s]))
    coded=huffman._encode_symbols(lower.reshape(-1),huffman._codes(lengths)) if lengths else b''
    header=HEADER.pack(b'KVLH',len(lengths),lower.size,zlib.crc32(lower.tobytes()))
    data=header+table+coded
    return EncodedLower(data,{'component_bytes':{'header':len(header),'table':len(table),'payload':len(coded)}})

def decode_lower(blob,exponents,strategy='deflate'):
    blob=blob.data if hasattr(blob,'data') else bytes(blob)
    exponents=np.asarray(exponents)
    if exponents.dtype!=np.uint8 or exponents.ndim!=2:raise ValueError('invalid exponent fields')
    if strategy=='deflate':
        raw=wrapper.unwrap(blob)
        if len(raw)!=exponents.size:raise ValueError('wrong lower count')
        return np.frombuffer(raw,dtype=np.uint8).copy().reshape(exponents.shape)
    if strategy=='split_deflate':
        if len(blob)<SPLIT_LENGTH.size:raise ValueError('truncated split lower header')
        first_length=SPLIT_LENGTH.unpack_from(blob)[0]
        if first_length<wrapper.HEADER.size or first_length>len(blob)-SPLIT_LENGTH.size-wrapper.HEADER.size:
            raise ValueError('invalid split stream boundary')
        at=SPLIT_LENGTH.size+first_length
        signs=wrapper.unwrap(blob[SPLIT_LENGTH.size:at])
        fractions=wrapper.unwrap(blob[at:])
        n=int(exponents.size)
        if len(signs)!=(n+7)//8 or len(fractions)!=(n*7+7)//8:
            raise ValueError('wrong packed split lengths')
        sign_bits=np.unpackbits(np.frombuffer(signs,dtype=np.uint8),bitorder='little')
        fraction_bits=np.unpackbits(np.frombuffer(fractions,dtype=np.uint8),bitorder='little')
        if np.any(sign_bits[n:]) or np.any(fraction_bits[n*7:]):
            raise ValueError('nonzero split stream padding')
        values=np.sum(fraction_bits[:n*7].reshape(n,7).astype(np.uint16)<<np.arange(7,dtype=np.uint16),axis=1).astype(np.uint8)
        values|=sign_bits[:n]<<7
        return values.reshape(exponents.shape)
    if strategy!='huffman' or len(blob)<HEADER.size:raise ValueError('invalid lower strategy/header')
    magic,nsym,n,crc=HEADER.unpack_from(blob)
    if magic!=b'KVLH' or n!=exponents.size or not 0<=nsym<=256 or len(blob)<HEADER.size+2*nsym:
        raise ValueError('invalid lower Huffman header')
    if n==0:
        if nsym!=0 or len(blob)!=HEADER.size or crc!=zlib.crc32(b''):
            raise ValueError('invalid empty lower Huffman record')
        return np.empty(exponents.shape,dtype=np.uint8)
    if nsym==0:raise ValueError('empty table for nonempty lower values')
    table=blob[HEADER.size:HEADER.size+2*nsym]
    lengths={table[i]:table[i+1] for i in range(0,len(table),2)}
    if len(lengths)!=nsym:raise ValueError('duplicate lower symbol')
    lower=huffman._decode_symbols(blob[HEADER.size+2*nsym:],n,huffman._codes(lengths)).astype(np.uint8)
    if zlib.crc32(lower.tobytes())!=crc:raise ValueError('lower checksum mismatch')
    return lower.reshape(exponents.shape)

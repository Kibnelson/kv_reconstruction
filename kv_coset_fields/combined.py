"""Keep exact exponent Huffman coding; optionally coset-code lower fields.

The old KVHU record is returned byte-for-byte whenever the complete combined
record is not smaller. A selected KVCF record has the same 24-byte outer header,
followed by its exponent table, exponent bits, and a self-contained lower codec.
"""
from dataclasses import dataclass
import struct
import time
import zlib
import numpy as np
from kv_stacked import huffman_codec as huffman

HEADER=struct.Struct('<4sBBHIIII')
MAGIC=b'KVCF'

@dataclass
class EncodedMatrix:
    data: bytes
    stats: dict

def fields(words):
    words=np.asarray(words)
    if words.dtype!=np.uint16 or words.ndim!=2:
        raise ValueError('expected uint16 matrix')
    return (((words>>8)&128)|(words&127)).astype(np.uint8),((words>>7)&255).astype(np.uint8)

def encode_matrix(words,lower_encoder,mode=2,audit_decoder=None,allow_expansion=False):
    if mode not in (2,3):raise ValueError('invalid combined lower mode')
    lower,exponents=fields(words)
    baseline=huffman.encode_matrix(words)
    encoded=lower_encoder(lower,exponents)
    audit_seconds=0.
    if audit_decoder is not None:
        started=time.perf_counter()
        np.testing.assert_array_equal(lower,audit_decoder(encoded.data,exponents))
        audit_seconds=time.perf_counter()-started
    _,_,base_mode,nsym,_,_,_,_=HEADER.unpack_from(baseline.data)
    if base_mode:
        start=HEADER.size+lower.size
        table=baseline.data[start:start+2*nsym]
        expbits=baseline.data[start+2*nsym:]
    else:
        lengths=huffman._lengths(exponents.reshape(-1));nsym=len(lengths)
        table=bytes(v for s in sorted(lengths) for v in (s,lengths[s]))
        expbits=huffman._encode_symbols(exponents.reshape(-1),huffman._codes(lengths)) if lengths else b''
    raw=np.ascontiguousarray(words).astype('<u2').tobytes()
    header=HEADER.pack(MAGIC,1,mode,nsym,*words.shape,len(expbits),zlib.crc32(raw))
    candidate=header+table+expbits+encoded.data
    accepted=(len(candidate)<len(baseline.data) or bool(allow_expansion)) and bool(nsym)
    stats=dict(baseline.stats)
    stats['record_modes']={'combined_lower' if accepted else 'original_huffman_fallback':1}
    stats['trial_bytes']={'lower_verbatim':lower.size,'lower_encoded':len(encoded.data),'combined_candidate':len(candidate)}
    stats['audit']={'records_verified':int(audit_decoder is not None),'decode_seconds':audit_seconds}
    stats['forced_expansion']={'records':int(allow_expansion and len(candidate)>=len(baseline.data))}
    for k,v in encoded.stats.items():
        if isinstance(v,dict) and k!='component_bytes' and all(isinstance(x,(int,float)) for x in v.values()):
            stats['trial_lower_'+k]=v
    # Trial statistics are diagnostics; only component_bytes count stored bytes.
    if accepted:
        parts={'header':len(header),'exponent_table':len(table),'exponent_payload':len(expbits)}
        lp=encoded.stats.get('component_bytes')
        if lp is None:parts['lower_record']=len(encoded.data)
        else:
            assert sum(lp.values())==len(encoded.data)
            parts.update({'lower_'+k:v for k,v in lp.items()})
        assert sum(parts.values())==len(candidate)
        stats.update(serialized_bytes=len(candidate),component_bytes=parts,mode='exponents_plus_cosets')
        for k,v in encoded.stats.items():
            if isinstance(v,dict) and k!='component_bytes' and all(isinstance(x,(int,float)) for x in v.values()):
                stats['lower_'+k]=v
        return EncodedMatrix(candidate,stats)
    return EncodedMatrix(baseline.data,stats)

def decode_matrix(blob,lower_decoder,expected_mode=None):
    blob=bytes(blob)
    if blob[:4]==b'KVHU':return huffman.decode_matrix(blob)
    if len(blob)<HEADER.size:raise ValueError('truncated combined record')
    magic,version,mode,nsym,rows,cols,exp_len,checksum=HEADER.unpack_from(blob)
    if magic!=MAGIC or version!=1 or mode not in (2,3) or not 1<=nsym<=256:
        raise ValueError('invalid combined record header')
    if expected_mode is not None and mode!=expected_mode:raise ValueError('wrong lower codec')
    start=HEADER.size+2*nsym
    if len(blob)<start+exp_len:raise ValueError('truncated combined exponent stream')
    table=blob[HEADER.size:start]
    lengths={table[i]:table[i+1] for i in range(0,len(table),2)}
    if len(lengths)!=nsym:raise ValueError('duplicate exponent symbol')
    exponents=huffman._decode_symbols(blob[start:start+exp_len],rows*cols,huffman._codes(lengths)).astype(np.uint8).reshape(rows,cols)
    lower=lower_decoder(blob[start+exp_len:],exponents)
    if lower.dtype!=np.uint8 or lower.shape!=(rows,cols):raise ValueError('invalid lower field shape')
    lower=lower.astype(np.uint16)
    words=(((lower&128)<<8)|(lower&127)|(exponents.astype(np.uint16)<<7)).astype(np.uint16)
    if zlib.crc32(words.astype('<u2').tobytes())!=checksum:raise ValueError('combined checksum mismatch')
    return words

"""Same temporal random-access contract for heterogeneous exact KV codecs."""
from dataclasses import dataclass
from collections import Counter
import struct
import numpy as np

HEADER=struct.Struct('<4s7I')
MAGIC=b'KVR2'

@dataclass
class EncodedCache:
    data: bytes
    stats: dict

def encode_cache(matrices, encoder, tile_tokens=16):
    nlayers=len(matrices)
    tokens,channels=matrices[0][0].shape
    ntiles=(tokens+tile_tokens-1)//tile_tokens
    nrecords=nlayers*2*ntiles
    header=HEADER.pack(MAGIC,1,nlayers,tokens,channels,tile_tokens,ntiles,nrecords)
    cursor=HEADER.size+8*(nrecords+1)
    offsets=[cursor]; payloads=[]
    components=Counter(cache_header=HEADER.size,record_offset_table=8*(nrecords+1))
    diagnostics={}; per_layer=[]
    for layer,pair in enumerate(matrices):
        for kind,words in zip(('K','V'),pair):
            assert words.shape==(tokens,channels) and words.dtype==np.uint16
            layer_bytes=0; counts=Counter()
            for start in range(0,tokens,tile_tokens):
                encoded=encoder(words[start:start+tile_tokens],layer,kind)
                payloads.append(encoded.data); cursor+=len(encoded.data); offsets.append(cursor)
                layer_bytes+=len(encoded.data)
                parts=encoded.stats.get('component_bytes')
                if parts is None:
                    components['matrix_serialized_bytes']+=len(encoded.data)
                else:
                    assert sum(parts.values())==len(encoded.data),encoded.stats
                    components.update({'matrix_'+k:v for k,v in parts.items()})
                for key,val in encoded.stats.items():
                    if isinstance(val,dict) and all(isinstance(x,(int,float)) for x in val.values()):
                        diagnostics.setdefault(key,Counter()).update(val)
                        if key=='accepted_correction_width_histogram': counts.update(val)
            per_layer.append(dict(layer=layer,kind=kind,serialized_bytes=layer_bytes,
                accepted_correction_width_histogram=dict(counts)))
    blob=header+struct.pack('<'+'Q'*len(offsets),*offsets)+b''.join(payloads)
    assert len(blob)==cursor and sum(components.values())==len(blob)
    raw=nlayers*2*tokens*channels*2
    stats=dict(raw_payload_bytes=raw,serialized_bytes=len(blob),
        ratio_vs_raw_payload=raw/len(blob),space_saving_percent=100*(1-len(blob)/raw),
        component_bytes=dict(components),per_layer=per_layer,
        diagnostics={k:dict(v) for k,v in diagnostics.items()})
    return EncodedCache(blob,stats)

def shape(blob):
    if len(blob)<HEADER.size: raise ValueError('truncated cache header')
    magic,version,layers,tokens,channels,tile,ntiles,nrecords=HEADER.unpack_from(blob)
    if magic!=MAGIC or version!=1 or min(layers,tokens,channels,tile)<1:
        raise ValueError('invalid cache header')
    if ntiles!=(tokens+tile-1)//tile or nrecords!=layers*2*ntiles:
        raise ValueError('inconsistent cache shape')
    payload_start=HEADER.size+8*(nrecords+1)
    if len(blob)<payload_start: raise ValueError('truncated offset table')
    if struct.unpack_from('<Q',blob,HEADER.size)[0]!=payload_start or struct.unpack_from('<Q',blob,HEADER.size+8*nrecords)[0]!=len(blob):
        raise ValueError('invalid endpoint offsets')
    return dict(layers=layers,tokens=tokens,channels=channels,tile_tokens=tile,
                ntiles=ntiles,nrecords=nrecords,payload_start=payload_start)

def decode_record(blob,layer,kind,tile_index,decoder):
    s=shape(blob)
    if not 0<=layer<s['layers'] or kind not in ('K','V') or not 0<=tile_index<s['ntiles']:
        raise IndexError('record outside cache')
    index=(layer*2+(kind=='V'))*s['ntiles']+tile_index
    a,b=struct.unpack_from('<QQ',blob,HEADER.size+8*index)
    if not s['payload_start']<=a<b<=len(blob): raise ValueError('invalid record offsets')
    w=decoder(blob[a:b],layer,kind)
    rows=min(s['tile_tokens'],s['tokens']-tile_index*s['tile_tokens'])
    if w.dtype!=np.uint16 or w.shape!=(rows,s['channels']):
        raise ValueError('decoded matrix shape/dtype mismatch')
    return w

def decode_cache(blob,decoder):
    s=shape(blob)
    return [tuple(np.concatenate([decode_record(blob,l,k,t,decoder) for t in range(s['ntiles'])])
        for k in ('K','V')) for l in range(s['layers'])]

def decode_tile(blob,layer,tile_index,decoder):
    return tuple(decode_record(blob,layer,k,tile_index,decoder) for k in ('K','V'))

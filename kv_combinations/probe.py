"""Compare tile selection and group-level hybrid on existing real KV sample."""
import argparse,collections,hashlib,json,pathlib,struct,time
import numpy as np
from kv_research import container,learned_codec,e8_codec
from kv_research.native_decode import NativeDecoder
from kv_smoke.codec import encode_matrix,decode_matrix
from kv_combinations import hybrid

def record(blob,index):
    a,b=struct.unpack_from('<QQ',blob,container.HEADER.size+8*index)
    return blob[a:b]

def pack(header,records):
    cursor=container.HEADER.size+8*(len(records)+1)
    offsets=[cursor]
    for payload in records:cursor+=len(payload);offsets.append(cursor)
    return header+struct.pack('<'+'Q'*len(offsets),*offsets)+b''.join(records)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native',action='store_true')
    p.add_argument('--output',default='kv_combinations/results/probe.json')
    args=p.parse_args()
    root=pathlib.Path('kv_research');out=pathlib.Path(args.output);out.parent.mkdir(exist_ok=True,parents=True)
    learned=(root/'results/learned64_sample.bin').read_bytes()
    e8=(root/'results/e8_symbol_sample.bin').read_bytes()
    s=container.shape(learned)
    model=learned_codec.load_model(root/'models/learned64.bin')
    native=NativeDecoder(model) if args.native else None
    with np.load(root/'results/sample_raw.npz') as f:
        raw=[tuple(f[f'layer_{l:02d}_{k}'].copy() for k in ('K','V')) for l in range(s['layers'])]
    tile_records=[];group_records=[];tile_counts=collections.Counter();group_counts=collections.Counter();i=0
    component=collections.Counter(cache_header=container.HEADER.size,offset_table=8*(s['nrecords']+1))
    started=time.perf_counter()
    for layer,pair in enumerate(raw):
        for kind,words in zip(('K','V'),pair):
            for start in range(0,s['tokens'],s['tile_tokens']):
                tile=words[start:start+s['tile_tokens']]
                lr=record(learned,i);er=encode_matrix(tile,'exponent').data
                choices={0:tile.astype('<u2').tobytes(),1:lr,2:er,3:record(e8,i)}
                selected=min(choices,key=lambda k:(len(choices[k]),k))
                tile_counts[str(selected)]+=1;tile_records.append(bytes([selected])+choices[selected])
                obj=hybrid.encode_matrix(tile,lr,er)
                group_records.append(obj.data)
                group_counts.update(obj.stats['mode_counts'])
                component.update({'matrix_'+k:v for k,v in obj.stats['component_bytes'].items()})
                i+=1
    tile_blob=pack(learned[:container.HEADER.size],tile_records)
    group_blob=pack(learned[:container.HEADER.size],group_records)
    assert len(group_blob)==sum(component.values())
    def tile_decode(b,l,k):
        mode=b[0];payload=b[1:]
        if mode==0:return np.frombuffer(payload,dtype='<u2').reshape(-1,s['channels']).copy()
        if mode==1:return native.decode_matrix(payload) if native else learned_codec.decode_matrix(payload,model)
        if mode==2:return decode_matrix(payload)
        if mode==3:return e8_codec.decode_matrix(payload)
        raise ValueError('unknown tile mode')
    decoded={
      'tile_hybrid':container.decode_cache(tile_blob,tile_decode),
      'group_hybrid':container.decode_cache(group_blob,lambda b,l,k:hybrid.decode_matrix(b,model,native_decoder=native))}
    for name,cache in decoded.items():
        assert all(np.array_equal(a,b) for x,y in zip(raw,cache) for a,b in zip(x,y)),name
    rawbytes=sum(w.nbytes for pair in raw for w in pair)
    state=(root/'models/learned64.bin').stat().st_size
    rows={name:{'encoded_bytes':len(blob),'state_bytes':state,
       'space_saving_percent_with_state':100*(1-(len(blob)+state)/rawbytes),
       'additional_bytes_saved_vs_learned':len(learned)-len(blob),'all_words_exact':True,
       'encoded_sha256':hashlib.sha256(blob).hexdigest()}
       for name,blob in [('learned',learned),('tile_hybrid',tile_blob),('group_hybrid',group_blob)]}
    results=dict(scope='Exploratory existing512-token Antarctic final sample; not new heldout validation. No fitting/weight changes. Every byte and complete frozen dictionary are charged.',
       tokens=s['tokens'],raw_payload_bytes=rawbytes,records=s['nrecords'],
       tile_modes={'0':'raw','1':'learned','2':'exponent','3':'E8 symbol'},tile_mode_counts=dict(tile_counts),
       group_modes={'0':'raw','1':'learned','2':'exponent'},group_mode_counts=dict(group_counts),
       group_component_bytes=dict(component),measurements=rows,probe_seconds=time.perf_counter()-started,
       input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
           root/'results/sample_raw.npz',root/'results/learned64_sample.bin',
           root/'results/e8_symbol_sample.bin',root/'models/learned64.bin']},
       performance_scope='Probe wall time includes encoding and equality checks; no reconstruction speed or GPU claim.')
    out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()

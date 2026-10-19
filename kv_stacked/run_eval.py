"""Compare literal stacks, field refinements and strong raw controls."""
import argparse,collections,hashlib,json,pathlib,platform,struct,time,zlib
import numpy as np
from kv_research import container,learned_codec
from kv_research.native_decode import NativeDecoder
from kv_smoke import codec
from kv_stacked import wrapper,field_codec,huffman_codec

DEFAULT=['exponent','exponent_deflate','raw_deflate','byte_shuffle_deflate',
    'field_packed','field_packed_deflate','field_best','field_best_deflate',
    'learned','learned_deflate','field_huffman','field_huffman_deflate']

def matrices(data):
    return [(data[f'layer_{i:02d}_K'],data[f'layer_{i:02d}_V']) for i in range(sum(k.endswith('_K') for k in data))]

def equal(a,b):return len(a)==len(b) and all(np.array_equal(x,y) for p,q in zip(a,b) for x,y in zip(p,q))

def aggregate(cases,states,names):
    out={}
    for name in names:
        rows=[c['codecs'][name] for c in cases];raw=sum(r['raw_payload_bytes'] for r in rows);encoded=sum(r['serialized_bytes'] for r in rows)
        out[name]=dict(raw_payload_bytes=raw,serialized_bytes=encoded,
            space_saving_payload_percent=100*(1-encoded/raw),decoder_state_bytes=states[name],
            savings_including_state_percent={str(n):100*(1-(encoded+len(rows)*states[name]/n)/raw) for n in (1,4,16)},
            all_exact=all(r['roundtrip_exact'] for r in rows),
            encode_seconds=sum(r['encode_seconds'] for r in rows),decode_seconds=sum(r['decode_seconds'] for r in rows))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='kv_stacked/dev_config.json')
    p.add_argument('--manifest',default='.kv_research_data/dev_manifest.json')
    p.add_argument('--output',default='kv_stacked/results/dev.json');p.add_argument('--only-index',type=int)
    p.add_argument('--generation-winner');p.add_argument('--save-sample',action='store_true')
    args=p.parse_args();config=json.loads(pathlib.Path(args.config).read_text())
    names=config['candidates'];level=config['deflate_level'];tile=config['tile_tokens']
    manifest_path=pathlib.Path(args.manifest);manifest=json.loads(manifest_path.read_text());data_dir=manifest_path.parent
    modelpath=pathlib.Path('kv_research/models/learned64.bin');book=learned_codec.load_model(modelpath);native=NativeDecoder(book)
    states={n:modelpath.stat().st_size if n.startswith('learned') else 0 for n in names}
    result=dict(config=config,capture_manifest=manifest,cases=[],state_bytes=states,
        runtime={'python':platform.python_version(),'numpy':np.__version__,'zlib':zlib.ZLIB_RUNTIME_VERSION,'device':'cpu'},
        scope='Exact storage/recovery; CPU reference, no GPU/serving result. Tile streams reset independently. No dictionaries carried across requests or records except frozen learned state.')
    selected=enumerate(manifest['cases'])
    if args.only_index is not None:selected=[(args.only_index,manifest['cases'][args.only_index])]
    output=pathlib.Path(args.output);output.parent.mkdir(exist_ok=True,parents=True)
    for index,case in selected:
        print('case '+case['name'],flush=True)
        with np.load(data_dir/case['file']) as f:raw=matrices({k:f[k].copy() for k in f.files})
        payload=b''.join(w.astype('<u2').tobytes() for pair in raw for w in pair)
        assert len(payload)==case['raw_payload_bytes'] and hashlib.sha256(payload).hexdigest()==case['raw_sha256']
        cache={};encoders={};decoders={}
        def construct(name):
            if name in cache:return cache[name]
            dependency_seconds=0.
            if name=='exponent':
                enc=lambda w,l,k:codec.encode_matrix(w,'exponent',64)
                dec=lambda b,l,k:codec.decode_matrix(b)
            elif name=='learned':
                enc=lambda w,l,k:learned_codec.encode_matrix(w,book,f'layer_{l:02d}_{k}',64)
                dec=lambda b,l,k:native.decode_matrix(b)
            elif name in ('raw','byte_shuffle'):
                enc=lambda w,l,k:wrapper.raw_matrix(w,name=='byte_shuffle')
                dec=lambda b,l,k:wrapper.decode_raw_matrix(b)
            elif name=='field_packed':
                enc=lambda w,l,k:field_codec.encode_matrix(w,strategy='packed',level=level)
                dec=lambda b,l,k:field_codec.decode_matrix(b)
            elif name=='field_packed_deflate':
                enc=lambda w,l,k:field_codec.encode_matrix(w,strategy='packed_deflate',level=level)
                dec=lambda b,l,k:field_codec.decode_matrix(b)
            elif name=='field_best':
                enc=lambda w,l,k:field_codec.encode_matrix(w,strategy='best',level=level)
                dec=lambda b,l,k:field_codec.decode_matrix(b)
            elif name=='field_huffman':
                enc=lambda w,l,k:huffman_codec.encode_matrix(w)
                dec=lambda b,l,k:huffman_codec.decode_matrix(b)
            elif name.endswith('_deflate'):
                base=name[:-8];source,dependency_seconds=construct(base);cursor=[0]
                def enc(w,l,k):
                    i=cursor[0];cursor[0]+=1
                    a,b=struct.unpack_from('<QQ',source.data,container.HEADER.size+i*8)
                    return wrapper.wrap(source.data[a:b],level)
                inner=decoders[base]
                dec=lambda b,l,k:inner(wrapper.unwrap(b),l,k)
            else:raise ValueError('unknown candidate '+name)
            encoders[name]=enc;decoders[name]=dec
            t=time.perf_counter();encoded=container.encode_cache(raw,enc,tile_tokens=tile)
            elapsed=time.perf_counter()-t+dependency_seconds
            cache[name]=(encoded,elapsed)
            return cache[name]
        row={k:v for k,v in case.items() if k!='input_token_ids'};row['codecs']={}
        for name in names:
            encoded,seconds=construct(name)
            t=time.perf_counter();recovered=container.decode_cache(encoded.data,decoders[name]);decode_seconds=time.perf_counter()-t
            assert equal(raw,recovered),name+' exact recovery failed'
            damaged=bytearray(encoded.data);s=container.shape(damaged);damaged[s['payload_start']]=0
            tindex=s['ntiles']-1;actual=container.decode_tile(damaged,s['layers']-1,tindex,decoders[name])
            assert all(np.array_equal(a,b[tindex*tile:]) for a,b in zip(actual,raw[-1]))
            stats={**encoded.stats,'roundtrip_exact':True,'independent_later_tile':True,
                'encode_seconds':seconds,'decode_seconds':decode_seconds,
                'encoded_sha256':hashlib.sha256(encoded.data).hexdigest(),
                'saving_with_state_percent':100*(1-(len(encoded.data)+states[name])/len(payload))}
            if args.save_sample and index==0 and name==args.generation_winner:(output.parent/'winner_sample.bin').write_bytes(encoded.data)
            if args.generation_winner==name and index==0:
                import torch
                from transformers import AutoModelForCausalLM,AutoTokenizer
                from kv_research.run_eval import generation_check
                torch.set_num_threads(4);torch.set_num_interop_threads(1)
                lm=AutoModelForCausalLM.from_pretrained('.smoke_model',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
                tok=AutoTokenizer.from_pretrained('.smoke_model',local_files_only=True)
                stats['generation']=generation_check(lm,tok,raw,encoded.data,decoders[name],case['first_token'])
                del lm
            row['codecs'][name]=stats
            print(f"  {name}: {stats['saving_with_state_percent']:.3f}% saved, exact",flush=True)
        result['cases'].append(row);result['aggregate']=aggregate(result['cases'],states,names)
        output.write_text(json.dumps(result,indent=2)+'\n')
    print('complete '+str(output),flush=True)

if __name__=='__main__':main()

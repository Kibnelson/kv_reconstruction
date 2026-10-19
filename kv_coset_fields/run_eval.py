"""Exact-byte comparison of exponent Huffman with complementary lower codecs."""
import argparse
import hashlib
import json
import pathlib
import platform
import time
import zlib
import numpy as np
from kv_research import container
from kv_stacked import huffman_codec
from kv_stacked.run_eval import aggregate,matrices,equal
from kv_coset_fields import combined,controls

def factories(config):
    encoders={};decoders={};states={}
    for name,spec in config['candidates'].items():
        kind=spec['kind'];states[name]=0
        if kind=='baseline':
            encoders[name]=lambda w,l,k:huffman_codec.encode_matrix(w)
            decoders[name]=lambda b,l,k:huffman_codec.decode_matrix(b)
        elif kind=='entropy':
            strategy=spec['strategy']
            def enc(w,l,k,strategy=strategy):
                return combined.encode_matrix(w,lambda lower,e:controls.encode_lower(lower,e,strategy),2,
                    lambda payload,e:controls.decode_lower(payload,e,strategy))
            def dec(b,l,k,strategy=strategy):
                return combined.decode_matrix(b,lambda payload,e:controls.decode_lower(payload,e,strategy),2)
            encoders[name]=enc;decoders[name]=dec
        elif kind=='temporal':
            from kv_coset_fields import temporal
            options=spec['options']
            def enc(w,l,k,options=options):
                return combined.encode_matrix(w,lambda lower,e:temporal.encode_lower(lower,e,**options),2,temporal.decode_lower)
            def dec(b,l,k):
                return combined.decode_matrix(b,temporal.decode_lower,2)
            encoders[name]=enc;decoders[name]=dec
        elif kind=='learned':
            from kv_coset_fields import learned_lower
            path=pathlib.Path(spec['model']);model=learned_lower.load_model(path);states[name]=path.stat().st_size
            options=spec.get('options',{});allow_expansion=spec.get('allow_expansion',False)
            def enc(w,l,k,model=model,options=options,allow_expansion=allow_expansion):
                return combined.encode_matrix(w,lambda lower,e:learned_lower.encode_lower(lower,e,model,f'layer_{l:02d}_{k}',**options),3,
                    lambda payload,e:learned_lower.decode_lower(payload,e,model,f'layer_{l:02d}_{k}'),allow_expansion)
            def dec(b,l,k,model=model):
                return combined.decode_matrix(b,lambda payload,e:learned_lower.decode_lower(payload,e,model,f'layer_{l:02d}_{k}'),3)
            encoders[name]=enc;decoders[name]=dec
        else:raise ValueError('unknown kind '+kind)
    return encoders,decoders,states

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='kv_coset_fields/dev_config.json')
    p.add_argument('--manifest',default='.kv_research_data/dev_manifest.json')
    p.add_argument('--output',default='kv_coset_fields/results/dev.json');p.add_argument('--only-index',type=int)
    p.add_argument('--generation-winner');p.add_argument('--save-sample',action='store_true')
    args=p.parse_args();config=json.loads(pathlib.Path(args.config).read_text())
    encoders,decoders,states=factories(config);names=list(config['candidates'])
    manifest_path=pathlib.Path(args.manifest);manifest=json.loads(manifest_path.read_text())
    result=dict(config=config,capture_manifest=manifest,cases=[],state_bytes=states,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,zlib=zlib.ZLIB_RUNTIME_VERSION,device='cpu'),
        scope='Exact exponent Huffman plus modular cosets of lower sign/fraction fields; CPU storage/reference only. Full external model bytes charged per cache.')
    selected=enumerate(manifest['cases'])
    if args.only_index is not None:selected=[(args.only_index,manifest['cases'][args.only_index])]
    output=pathlib.Path(args.output);output.parent.mkdir(exist_ok=True,parents=True)
    for index,case in selected:
        print('case '+case['name'],flush=True)
        with np.load(manifest_path.parent/case['file']) as f:raw=matrices({k:f[k].copy() for k in f.files})
        payload=b''.join(w.astype('<u2').tobytes() for pair in raw for w in pair)
        assert len(payload)==case['raw_payload_bytes'] and hashlib.sha256(payload).hexdigest()==case['raw_sha256']
        row={k:v for k,v in case.items() if k!='input_token_ids'};row['codecs']={}
        for name in names:
            started=time.perf_counter();encoded=container.encode_cache(raw,encoders[name],tile_tokens=config['tile_tokens'])
            enc_seconds=time.perf_counter()-started
            started=time.perf_counter();recovered=container.decode_cache(encoded.data,decoders[name]);dec_seconds=time.perf_counter()-started
            assert equal(raw,recovered),name+' exact recovery failed'
            damaged=bytearray(encoded.data);shape=container.shape(damaged);damaged[shape['payload_start']]^=255
            tindex=shape['ntiles']-1;actual=container.decode_tile(damaged,shape['layers']-1,tindex,decoders[name])
            assert all(np.array_equal(a,b[tindex*config['tile_tokens']:]) for a,b in zip(actual,raw[-1]))
            stats={**encoded.stats,'roundtrip_exact':True,'independent_later_tile':True,
                'encode_seconds':enc_seconds,'decode_seconds':dec_seconds,
                'encoded_sha256':hashlib.sha256(encoded.data).hexdigest(),
                'saving_with_state_percent':100*(1-(len(encoded.data)+states[name])/len(payload))}
            if args.save_sample and index==0 and name==args.generation_winner:
                (output.parent/'forced_sample.bin').write_bytes(encoded.data)
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
            modes=stats.get('diagnostics',{}).get('record_modes',{})
            print(f"  {name}: {stats['saving_with_state_percent']:.4f}% saved, exact; {modes}",flush=True)
        result['cases'].append(row);result['aggregate']=aggregate(result['cases'],states,names)
        output.write_text(json.dumps(result,indent=2)+'\n')
    print('complete '+str(output),flush=True)

if __name__=='__main__':main()

"""Actual indexed bytes, exact recovery and optional model generation checks."""
import argparse, collections, hashlib, json, pathlib, time
import numpy as np
from kv_research import container, e8_codec, learned_codec
from kv_research.model_utils import matrices_from_mapping,equal_matrices
from kv_smoke import codec

def adapters(spec):
    family=spec['family']
    if family=='baseline':
        return (lambda w,l,k:codec.encode_matrix(w,spec['method'],group_size=64),
                lambda b,l,k:codec.decode_matrix(b),dict(serialized_bytes=0,runtime_parameter_array_bytes=0))
    if family=='e8':
        return (lambda w,l,k:e8_codec.encode_matrix(w,method=spec['method'],group_size=64,**spec.get('params',{})),
                lambda b,l,k:e8_codec.decode_matrix(b),dict(serialized_bytes=0,runtime_parameter_array_bytes=0))
    if family=='learned':
        model=learned_codec.load_model(spec['model_path'])
        modelbytes=pathlib.Path(spec['model_path']).read_bytes()
        state=dict(serialized_bytes=len(modelbytes),sha256=hashlib.sha256(modelbytes).hexdigest(),
             runtime_parameter_array_bytes=sum(a.nbytes for a in model.codebooks.values()),
             codebook_count=len(model.codebooks),n_centroids=model.n_centroids)
        return (lambda w,l,k:learned_codec.encode_matrix(w,model,f'layer_{l:02d}_{k}',group_size=64),
                lambda b,l,k:learned_codec.decode_matrix(b,model),state)
    raise ValueError('unknown family')

def aggregate(cases,specs,states):
    out={}
    for spec in specs:
        name=spec['name']; rows=[c['codecs'][name] for c in cases]
        raw=sum(x['raw_payload_bytes'] for x in rows); enc=sum(x['serialized_bytes'] for x in rows)
        # Repeat each evaluated case N times with one loaded decoder. This
        # charges state separately for each case in the aggregate, avoiding
        # treating unrelated evaluation runs as concurrent caches.
        state=states[name]['serialized_bytes']
        row=dict(raw_payload_bytes=raw,serialized_bytes=enc,
            payload_space_saving_percent=100*(1-enc/raw),
            all_roundtrips_exact=all(x['roundtrip_exact'] for x in rows),
            encode_seconds_total=sum(x['encode_seconds'] for x in rows),
            decode_seconds_total=sum(x['decode_seconds'] for x in rows),decoder_state=states[name],
            savings_including_state_percent={str(n):100*(1-(enc+len(rows)*state/n)/raw) for n in (1,4,16)})
        components=collections.Counter()
        for x in rows: components.update(x['component_bytes'])
        row['component_bytes']=dict(components)
        out[name]=row
    return out

def generation_check(model,tokenizer,matrices,blob,decoder,first_token,steps=8):
    import torch
    from kv_research.model_utils import step,cache_matrices
    reference=matrices
    recent=[(np.empty((0,k.shape[1]),np.uint16),np.empty((0,v.shape[1]),np.uint16)) for k,v in matrices]
    tokens=[]; errors=[]; seconds=0.
    token=first_token
    for _ in range(steps):
        ref=step(model,token,reference)
        t=time.perf_counter(); recovered=container.decode_cache(blob,decoder)
        recovered=[(np.concatenate((k,rk)),np.concatenate((v,rv))) for (k,v),(rk,rv) in zip(recovered,recent)]
        seconds+=time.perf_counter()-t
        observed=step(model,token,recovered)
        error=float(torch.max(torch.abs(ref.logits.float()-observed.logits.float())))
        if not torch.equal(ref.logits,observed.logits): raise AssertionError(f'logits mismatch {error}')
        errors.append(error)
        refkv=cache_matrices(ref.past_key_values); testkv=cache_matrices(observed.past_key_values)
        assert equal_matrices(refkv,testkv),'new KV differs'
        token=int(ref.logits[0,-1].argmax()); tokens.append(token)
        reference=refkv
        recent=[(np.concatenate((rk,k[-1:])),np.concatenate((rv,v[-1:]))) for (rk,rv),(k,v) in zip(recent,testkv)]
    return dict(steps=steps,all_step_logits_bitwise_equal=True,max_abs_logit_difference=max(errors),
        output_token_ids=tokens,output_text=tokenizer.decode(tokens),reconstruction_seconds=seconds,
        note='Full prefix restored on CPU each step; output list begins after token chosen by prefill. No compact-resident model/GPU performance claim.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='.kv_research_data')
    p.add_argument('--split',choices=('dev','test'),required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model-dir',default='.smoke_model')
    p.add_argument('--generation',action='store_true')
    p.add_argument('--save-sample',action='store_true')
    args=p.parse_args()
    root=pathlib.Path(args.data_dir); output=pathlib.Path(args.output); output.parent.mkdir(exist_ok=True,parents=True)
    specs=json.loads(pathlib.Path(args.config).read_text())['candidates']
    manifest=json.loads((root/(args.split+'_manifest.json')).read_text())
    loaded={s['name']:adapters(s) for s in specs}
    states={name:triplet[2] for name,triplet in loaded.items()}
    result=dict(split=args.split,candidate_config=json.loads(pathlib.Path(args.config).read_text()),
        capture_manifest=manifest,cases=[],decoder_states=states,
        scope='CPU exact-storage and full-restore model check. No GPU kernel, concurrency run, E8P12 reproduction, neural autoencoder or representative benchmark.')
    if args.generation:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    for index,case in enumerate(manifest['cases']):
        print('case '+case['name'],flush=True)
        matrices=matrices_from_mapping(dict(np.load(root/case['file'])))
        raw=b''.join(w.astype('<u2').tobytes() for pair in matrices for w in pair)
        assert len(raw)==case['raw_payload_bytes'],'capture byte count mismatch'
        assert hashlib.sha256(raw).hexdigest()==case['raw_sha256'],'capture hash mismatch'
        row={k:v for k,v in case.items() if k!='input_token_ids'};row['codecs']={}
        for spec in specs:
            name=spec['name'];encoder,decoder,state=loaded[name]
            t=time.perf_counter();encoded=container.encode_cache(matrices,encoder)
            encode_seconds=time.perf_counter()-t
            t=time.perf_counter();recovered=container.decode_cache(encoded.data,decoder)
            decode_seconds=time.perf_counter()-t
            assert equal_matrices(matrices,recovered),name+' recovery mismatch'
            # Test a later temporal tile after corrupting an unrelated first payload.
            damaged=bytearray(encoded.data);s=container.shape(damaged);damaged[s['payload_start']]=0
            last=s['ntiles']-1
            tile=container.decode_tile(damaged,s['layers']-1,last,decoder)
            start=last*s['tile_tokens']
            assert all(np.array_equal(w,raw[start:]) for w,raw in zip(tile,matrices[-1]))
            stats={**encoded.stats,'encode_seconds':encode_seconds,'decode_seconds':decode_seconds,
                'roundtrip_exact':True,'later_tile_independence_passed':True,
                'encoded_sha256':hashlib.sha256(encoded.data).hexdigest(),
                'savings_including_state_percent':{str(n):100*(1-(len(encoded.data)+state['serialized_bytes']/n)/encoded.stats['raw_payload_bytes']) for n in (1,4,16)}}
            if args.generation and index==0 and spec['family'] in ('e8','learned') and spec.get('generation',True):
                stats['generation']=generation_check(model,tokenizer,matrices,encoded.data,decoder,case['first_token'])
            if args.save_sample and index==0 and spec['family'] in ('e8','learned'):
                sample=output.parent/(name+'_sample.bin');sample.write_bytes(encoded.data)
            row['codecs'][name]=stats
            print(f"  {name}: {len(encoded.data)/2**20:.3f}MiB saving={stats['space_saving_percent']:.2f}% state-inclusive={stats['savings_including_state_percent']['1']:.2f}% exact",flush=True)
        result['cases'].append(row)
        result['aggregate']=aggregate(result['cases'],specs,states)
        output.write_text(json.dumps(result,indent=2)+'\n')
    print('complete '+str(output),flush=True)

if __name__=='__main__':main()

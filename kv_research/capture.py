"""Generate KV samples from pinned pretrained SmolLM2 on explicit split fixtures."""
import argparse, hashlib, json, pathlib, platform, time
import numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_research.fixtures import SPLITS, build_input_ids
from kv_research.model_utils import cache_matrices, mapping
from kv_smoke.download_model import MODEL_ID, REVISION

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',default='.smoke_model')
    p.add_argument('--data-dir',default='.kv_research_data')
    p.add_argument('--split',choices=SPLITS,required=True)
    p.add_argument('--lengths',default='1024')
    args=p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(19)
    root=pathlib.Path(args.data_dir); root.mkdir(exist_ok=True,parents=True)
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    manifest={'split':args.split,'model':MODEL_ID,'revision':REVISION,
      'runtime':dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,device='cpu',dtype='bfloat16',threads=4),
      'cases':[]}
    for f in SPLITS[args.split]:
        for n in map(int,args.lengths.split(',')):
            ids=build_input_ids(tok,f,n)
            t=time.perf_counter()
            with torch.inference_mode():
                out=model(torch.tensor([ids],dtype=torch.long),use_cache=True,logits_to_keep=1)
                matrices=cache_matrices(out.past_key_values)
            name=f'{f["name"]}_{n}'
            target=root/(name+'.npz')
            np.savez(target,**mapping(matrices))
            raw=b''.join(w.astype('<u2').tobytes() for pair in matrices for w in pair)
            row=dict(name=name,fixture=f['name'],kind=f['kind'],tokens=n,file=target.name,
              input_token_ids=ids,first_token=int(out.logits[0,-1].argmax()),
              raw_payload_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),prefill_seconds=time.perf_counter()-t)
            manifest['cases'].append(row)
            print(f'captured {name}: {len(raw)/2**20:.2f} MiB BF16 KV',flush=True)
    (root/(args.split+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()

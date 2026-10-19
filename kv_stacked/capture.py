"""Capture fresh final BF16 KV with the unchanged pinned checkpoint."""
import argparse,hashlib,json,pathlib,time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from kv_research.model_utils import cache_matrices,mapping
from kv_research.fixtures import build_input_ids
from kv_smoke.download_model import MODEL_ID,REVISION
from kv_stacked.fixtures import FINAL

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',default='.smoke_model');p.add_argument('--output',default='.kv_stacked_data')
    args=p.parse_args();out=pathlib.Path(args.output);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    manifest=dict(model=MODEL_ID,revision=REVISION,split='fresh_stacked_final',cases=[])
    for f in FINAL:
        for n in (512,2048):
            ids=build_input_ids(tok,f,n);started=time.perf_counter()
            with torch.inference_mode():
                output=model(torch.tensor([ids]),use_cache=True,logits_to_keep=1)
                matrices=cache_matrices(output.past_key_values)
            name=f'{f["name"]}_{n}';file=name+'.npz';np.savez(out/file,**mapping(matrices))
            raw=b''.join(w.astype('<u2').tobytes() for pair in matrices for w in pair)
            manifest['cases'].append(dict(name=name,kind=f['kind'],tokens=n,file=file,input_token_ids=ids,
                first_token=int(output.logits[0,-1].argmax()),raw_payload_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),prefill_seconds=time.perf_counter()-started))
            print('captured '+name,flush=True)
    (out/'final_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()

"""Eight model steps using the unchanged-format native learned decoder."""
import json,pathlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from kv_research import learned_codec
from kv_research.native_decode import NativeDecoder
from kv_research.model_utils import matrices_from_mapping
from kv_research.run_eval import generation_check

def main():
    root=pathlib.Path(__file__).resolve().parent;project=root.parent
    final=json.loads((root/'results/final.json').read_text())
    case=final['capture_manifest']['cases'][0]
    raw=root/'results/sample_raw.npz'
    if not raw.exists():raw=project/'.kv_research_data'/case['file']
    matrices=matrices_from_mapping(dict(np.load(raw)))
    book=learned_codec.load_model(root/'models/learned64.bin')
    native=NativeDecoder(book)
    blob=(root/'results/learned64_sample.bin').read_bytes()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    model=AutoModelForCausalLM.from_pretrained(project/'.smoke_model',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    tokenizer=AutoTokenizer.from_pretrained(project/'.smoke_model',local_files_only=True)
    result=generation_check(model,tokenizer,matrices,blob,lambda b,l,k:native.decode_matrix(b),case['first_token'])
    result['implementation']='Native CPU integer decoder, identical frozen records/codebook; whole-prefix restore, eight steps; no GPU or compact-resident attention.'
    (root/'results/native_generation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

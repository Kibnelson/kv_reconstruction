"""Native-BF16 model/cache conversion, preserving original stored word bits."""
import numpy as np
import torch
from transformers import DynamicCache

def cache_matrices(cache):
    result=[]
    for k,v in cache.to_legacy_cache():
        pair=[]
        for t in (k,v):
            assert t.dtype==torch.bfloat16
            matrix=t[0].permute(1,0,2).contiguous().reshape(t.shape[2],-1)
            pair.append(matrix.view(torch.uint16).cpu().numpy().copy())
        result.append(tuple(pair))
    return result

def tensor_cache(matrices,config):
    cache=DynamicCache(config=config)
    h=config.num_key_value_heads
    d=config.hidden_size//config.num_attention_heads
    for layer,pair in enumerate(matrices):
        tensors=[]
        for w in pair:
            t=torch.from_numpy(np.ascontiguousarray(w)).view(torch.bfloat16)
            tensors.append(t.reshape(w.shape[0],h,d).permute(1,0,2).unsqueeze(0).contiguous())
        cache.update(*tensors,layer)
    return cache

def mapping(matrices):
    return {f'layer_{layer:02d}_{kind}':w for layer,pair in enumerate(matrices)
            for kind,w in zip(('K','V'),pair)}

def matrices_from_mapping(data):
    n=sum(key.endswith('_K') for key in data)
    return [(data[f'layer_{i:02d}_K'],data[f'layer_{i:02d}_V']) for i in range(n)]

def equal_matrices(a,b):
    return len(a)==len(b) and all(np.array_equal(x,y) for pa,pb in zip(a,b) for x,y in zip(pa,pb))

@torch.inference_mode()
def step(model,token,matrices):
    n=matrices[0][0].shape[0]
    return model(input_ids=torch.tensor([[token]],dtype=torch.long),
        attention_mask=torch.ones((1,n+1),dtype=torch.long),
        past_key_values=tensor_cache(matrices,model.config),use_cache=True,logits_to_keep=1)

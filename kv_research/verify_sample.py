"""Verify bundled real KV records with NumPy alone, without model weights."""
import hashlib,json,pathlib
import numpy as np
from kv_research import container,e8_codec,learned_codec

def main():
    root=pathlib.Path(__file__).resolve().parent
    meta=json.loads((root/'results/sample_metadata.json').read_text())
    with np.load(root/'results/sample_raw.npz') as f:
        raw={key:f[key] for key in f.files}
    checked=0
    for spec in meta['candidates']:
        name=spec['name']; blob=(root/'results'/f'{name}_sample.bin').read_bytes()
        assert hashlib.sha256(blob).hexdigest()==spec['encoded_sha256']
        if spec['family']=='e8':
            decoder=lambda b,l,k:e8_codec.decode_matrix(b)
        else:
            model=learned_codec.load_model(root/'models'/pathlib.Path(spec['model_path']).name)
            decoder=lambda b,l,k:learned_codec.decode_matrix(b,model)
        recovered=container.decode_cache(blob,decoder)
        for layer,pair in enumerate(recovered):
            for kind,w in zip(('K','V'),pair):
                assert np.array_equal(w,raw[f'layer_{layer:02d}_{kind}'])
                checked+=w.size
        print(name+': all original BF16 words recovered exactly')
    print(f'PASS: {checked:,} BF16 word comparisons across {len(meta["candidates"])} records')

if __name__=='__main__':main()

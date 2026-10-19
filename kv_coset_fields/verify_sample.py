"""NumPy-only exact verification that executes actual learned modular cosets."""
import hashlib
import json
import pathlib
import struct
import numpy as np
from kv_research import container
from kv_coset_fields import combined,learned_lower

def main():
    root=pathlib.Path(__file__).resolve().parent
    info=json.loads((root/'results/sample_metadata.json').read_text())
    blob=(root/'results/forced_sample.bin').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==info['encoded_sha256']
    model_path=root/'models/lower16.bin'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==info['model_sha256']
    model=learned_lower.load_model(model_path)
    shape=container.shape(blob)
    for i in range(shape['nrecords']):
        a=struct.unpack_from('<Q',blob,container.HEADER.size+8*i)[0]
        assert blob[a:a+4]==combined.MAGIC,'sample unexpectedly used a baseline fallback'
    decoder=lambda b,l,k:combined.decode_matrix(b,lambda payload,e:learned_lower.decode_lower(payload,e,model,f'layer_{l:02d}_{k}'),3)
    recovered=container.decode_cache(blob,decoder)
    digest=hashlib.sha256()
    with np.load(root/'results/sample_raw.npz') as source:
        for layer,pair in enumerate(recovered):
            for kind,words in zip(('K','V'),pair):
                np.testing.assert_array_equal(words,source[f'layer_{layer:02d}_{kind}'])
                digest.update(words.astype('<u2').tobytes())
    assert digest.hexdigest()==info['raw_sha256']
    print(f"PASS: {shape['nrecords']:,} actual coset records recovered all {info['raw_payload_bytes']//2:,} original BF16 words exactly")

if __name__=='__main__':main()

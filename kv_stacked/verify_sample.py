"""NumPy-only verification of the bundled canonical-Huffman real KV sample."""
import hashlib,json,pathlib
import numpy as np
from kv_research import container
from kv_stacked import huffman_codec

def main():
    root=pathlib.Path(__file__).resolve().parent
    info=json.loads((root/'results/sample_metadata.json').read_text())
    blob=(root/'results/winner_sample.bin').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==info['encoded_sha256']
    recovered=container.decode_cache(blob,lambda b,l,k:huffman_codec.decode_matrix(b))
    with np.load(root/'results/sample_raw.npz') as raw:
        for layer,pair in enumerate(recovered):
            for kind,w in zip(('K','V'),pair):np.testing.assert_array_equal(w,raw[f'layer_{layer:02d}_{kind}'])
    print(f"PASS: all {info['raw_payload_bytes']//2:,} original BF16 words recovered exactly")

if __name__=='__main__':main()

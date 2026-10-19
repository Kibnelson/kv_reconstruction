"""NumPy-only exact verification of both actual learned reconstruction paths."""
import hashlib
import json
import pathlib
import struct
import numpy as np
from kv_research import container,learned_codec
from kv_residual import learned,learned_e8

def main():
    root=pathlib.Path(__file__).resolve().parent;project=root.parent
    metadata=json.loads((root/'results/sample_metadata.json').read_text())
    model_path=project/metadata['model_file']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==metadata['model_sha256']
    model=learned_codec.load_model(model_path)
    for sample in metadata['samples']:
        blob=(root/'results'/sample['file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==sample['encoded_sha256']
        module=learned_e8 if sample['kind']=='learned_e8' else learned
        shape=container.shape(blob)
        for i in range(shape['nrecords']):
            a,b=struct.unpack_from('<QQ',blob,container.HEADER.size+8*i)
            record=blob[a:b]
            if module is learned:
                assert learned._HEADER.unpack_from(record)[2]==2,'sample skipped signed corrections'
            else:
                header=learned_e8._HEADER.unpack_from(record)
                assert header[0]==b'LKE8' and header[4]&1 and not header[4]&2
                at=learned_e8._HEADER.size+header[-1]
                assert learned_e8._GROUP.unpack_from(record,at)[0]==1,'sample skipped E8'
        recovered=container.decode_cache(blob,lambda payload,l,k:module.decode_matrix(payload,model))
        digest=hashlib.sha256()
        with np.load(root/'results/sample_raw.npz') as raw:
            for layer,pair in enumerate(recovered):
                for kind,words in zip(('K','V'),pair):
                    np.testing.assert_array_equal(words,raw[f'layer_{layer:02d}_{kind}'])
                    digest.update(words.astype('<u2').tobytes())
        assert digest.hexdigest()==metadata['raw_sha256']
        print(f"PASS {sample['candidate']}: {shape['nrecords']:,} actual records recovered all {metadata['raw_payload_bytes']//2:,} BF16 words exactly")

if __name__=='__main__':main()

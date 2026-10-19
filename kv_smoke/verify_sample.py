"""Verify the supplied real-model cache sample using NumPy only."""
import pathlib
import numpy as np
from cache_container import decode_cache, cache_shape
root=pathlib.Path(__file__).parent/"results"
blob=(root/"sample_encoded_cache.bin").read_bytes()
decoded=decode_cache(blob)
reference=np.load(root/"sample_raw_cache.npz")
words=0
for layer,pair in enumerate(decoded):
    for kind,value in zip(("K","V"),pair):
        original=reference[f"layer_{layer:02d}_{kind}"]
        np.testing.assert_array_equal(value,original)
        words+=value.size
print({"sample_roundtrip_exact":True,"bf16_words":words,"shape":cache_shape(blob)})


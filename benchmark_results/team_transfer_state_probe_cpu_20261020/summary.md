# Exact transfer predictor state storage probe

Frozen INT64 predictor coefficients were copied to INT32 after checking their entire range. The original state was retained unchanged. This changes shared-state storage, not KV stream bitrate.

| Quantity | Original | INT32 coefficients |
| --- | ---: | ---: |
| Coefficient bytes | 50,429,952 | 25,214,976 |
| All frozen array bytes | 51,795,464 | 26,580,488 |

All **184,320 predictions** in **360 blocks** across **30 layers** matched bit for bit on 16 real TRAIN tokens. INT64 feature inputs and INT32 coefficient storage produced INT64 matrix products.

Frozen arrays shrink by **48.68%**, saving **25,214,976 bytes**.

No GPU performance, cross-platform reproducibility, or additional KV-stream savings were measured.

An original held-out TEST range-coded packet was also decoded using the compact state. Only its `state_sha256` header field was replaced; the entropy payload and decoded-word checksum were preserved. All **368,640 decoded BF16 words** matched the original decoder output. The **444,216-byte entropy stream** was unchanged.

A further **storage arithmetic candidate** selects INT16 per block when every coefficient fits, otherwise INT32: **14,702,592 coefficient bytes** and **16,068,104 total frozen-array bytes**. This mixed-dtype candidate has not been validated through packet decoding.

# Frozen KV device component replay

Device: `cuda:0`. GPU measured: **True**. Transfer entropy coding remains CPU-only; predictor replay reads known KV.

| Page | Pool payload bpv | Complete wire bpv | Pool exact | Predictor exact |
|---|---:|---:|---|---|
| page_000 | 11.65269 | 11.66693 | True | True |
| page_001 | 11.33826 | 11.35250 | True | True |
| page_002 | 11.60323 | 11.61747 | True | True |
| page_003 | 11.29475 | 11.30898 | True | True |

Compared 1,474,560 original BF16 words across 4 pages. CPU transfer packets: **9.430599 bits/value**. This packet rate is separate from the device pool format.

Synchronized wall medians in milliseconds (warm state):

| Page | Pool encode | Pool decode | Raw H2D | Packed H2D + unpack | Predictor replay |
|---|---:|---:|---:|---:|---:|
| page_000 | 15.4325 | 14.8996 | 0.0493 | 14.8572 | 174.1569 |
| page_001 | 15.4522 | 14.7490 | 0.0479 | 14.8321 | 173.9332 |
| page_002 | 15.5391 | 14.8868 | 0.0479 | 14.8771 | 173.7948 |
| page_003 | 15.5685 | 14.8079 | 0.0478 | 14.8592 | 174.2766 |

Persistent device state: pool **23,040 bytes**; predictor **27,237,376 bytes**. The full CPU decoder state, cold setup, CUDA event intervals and per-call incremental peaks (including output and scratch) are recorded in `results.json`.

Pool payload includes each original record's 2-byte escape header. Complete wire also includes a serialized geometry header and the INT64 block offset directory. Shared pool calibration and predictor tensors are reported separately. CUDA samples report events and synchronized wall time with warm shared state; cold initialization is separate. CPU runs leave H2D and CUDA timings unmeasured.

These components do not demonstrate a GPU transfer entropy decoder, compressed-KV attention, or serving/concurrency speedup. CPU transfer bytes are reported separately from device component coverage.

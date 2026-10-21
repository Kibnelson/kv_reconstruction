# GPU KV path component batch

Status: **completed**. GPU measured: **True**.

| Attention case | Tail | Dense wall ms | P1 wall ms | Max output difference |
|---|---:|---:|---:|---:|
| frozen_smol/page_000 | 0 | 1.23743 | 1.25720 | 0 |
| frozen_smol/page_000 | 3 | 1.20731 | 1.26235 | 0 |
| frozen_smol/page_001 | 0 | 1.20450 | 1.27085 | 0 |
| frozen_smol/page_001 | 3 | 1.23045 | 1.26938 | 0 |
| frozen_smol/page_002 | 0 | 1.20899 | 1.26502 | 0 |
| frozen_smol/page_002 | 3 | 1.21425 | 1.25964 | 0 |
| frozen_smol/page_003 | 0 | 1.21335 | 1.26429 | 0 |
| frozen_smol/page_003 | 3 | 1.21675 | 1.25522 | 0 |
| qwen_reuse/test_pottery_kiln_diary_0 | 0 | 0.21000 | 0.21344 | 2.38e-07 |
| qwen_reuse/test_pottery_kiln_diary_0 | 3 | 0.20833 | 0.21078 | 1.51e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 0 | 0.20379 | 0.21403 | 1.56e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 3 | 0.20629 | 0.21283 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 0 | 0.20771 | 0.21253 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 3 | 0.20857 | 0.21110 | 1.64e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 0 | 0.20456 | 0.21895 | 1.49e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 3 | 0.20653 | 0.21374 | 1.19e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 0 | 0.20842 | 0.21355 | 2.01e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 3 | 0.20752 | 0.21359 | 1.51e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 0 | 0.20651 | 0.22004 | 1.97e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 3 | 0.20900 | 0.21246 | 2.27e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 0 | 0.21064 | 0.22042 | 1.45e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 3 | 0.20979 | 0.21331 | 1.75e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 0 | 0.20779 | 0.22217 | 1.79e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 3 | 0.20808 | 0.21265 | 2.09e-07 |

| Transfer case | Arm | T complete bpv | Raw fallback | KV to T ms | T to KV ms | P to T ms | T to P ms | Exact |
|---|---|---:|---|---:|---:|---:|---:|---|
| frozen_smol/page_000 | A | 9.66055 | False | 354.521 | 243.440 | 355.267 | 241.013 | True |
| frozen_smol/page_001 | A | 9.11254 | False | 397.519 | 240.622 | 397.204 | 242.034 | True |
| frozen_smol/page_002 | A | 9.64627 | False | 395.951 | 241.297 | 396.668 | 242.659 | True |
| frozen_smol/page_003 | A | 9.34631 | False | 398.234 | 240.946 | 397.374 | 241.074 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | 9.17834 | False | 454.328 | 317.184 | 452.926 | 317.538 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | 10.06049 | False | 454.656 | 317.355 | 453.378 | 317.214 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | 9.99316 | False | 455.500 | 316.806 | 453.546 | 317.207 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | 9.78412 | False | 455.156 | 318.895 | 454.108 | 317.041 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | 9.12720 | False | 456.511 | 317.998 | 453.389 | 317.814 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | 9.59076 | False | 455.155 | 317.301 | 454.368 | 317.418 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | 9.44614 | False | 454.705 | 317.871 | 453.412 | 318.253 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | 9.45724 | False | 454.418 | 317.654 | 453.421 | 318.703 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | 7.74146 | False | 544.411 | 416.442 | 542.988 | 415.493 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | 9.71503 | False | 545.764 | 415.736 | 543.877 | 414.090 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | 9.61575 | False | 544.851 | 415.009 | 546.036 | 415.921 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | 9.36581 | False | 544.831 | 416.004 | 547.096 | 416.362 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | 7.69046 | False | 544.132 | 415.475 | 544.459 | 416.023 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | 9.07468 | False | 543.882 | 415.694 | 545.058 | 415.542 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | 8.80435 | False | 545.415 | 414.174 | 545.763 | 413.543 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | 8.82047 | False | 545.141 | 414.570 | 544.145 | 415.293 | True |

| Group | Complete bpv | Payload bpv | Directory + envelope bpv | Legacy full-identity bpv |
|---|---:|---:|---:|---:|
| frozen_smol/A | 9.441417 | 9.423600 | 0.017817 | 9.447559 |
| qwen_reuse/A | 9.579681 | 9.554596 | 0.025085 | 9.749481 |
| qwen_reuse/B | 8.853500 | 8.828415 | 0.025085 | 9.023300 |

| Diagnostic encoder control | Parameters ms | Prepared entropy ms | Encode device events | Decode device events |
|---|---:|---:|---:|---:|
| frozen_smol/A/page_000 | 146.730 | 192.521 | 27453 | 27447 |
| frozen_smol/A/page_003 | 190.543 | 193.483 | 27448 | 27441 |
| qwen_reuse/A/test_pottery_kiln_diary_0 | 304.651 | 142.472 | 43409 | 43402 |
| qwen_reuse/A/test_python_dependency_walk_2016 | 305.682 | 143.959 | 43393 | 43386 |
| qwen_reuse/B/test_pottery_kiln_diary_0 | 395.360 | 142.195 | 56024 | 56017 |
| qwen_reuse/B/test_python_dependency_walk_2016 | 396.780 | 139.582 | 56005 | 55997 |

Prepared entropy uses parameters computed from known source KV; it is an encoder timing control. Actual causal decode remains packet-only. Profiled event totals are separate from the uninstrumented medians; missing CUDA profiler activity is explicit.

T1 is a new 12-bit binary-rANS stream with per-token restarts. Complete external bytes include the envelope, model/state identity digests, directory, checksum and payload. The independently expected full model identity is hashed and verified, and legacy full-identity envelopes remain readable. Shared state and measured scratch/peaks are separate. Raw fallback is explicitly identified.

T rates do not inherit the earlier CPU ~44% result. P rates and T rates are separate representations. GPU-body codec times exclude separately reported compressed-byte export/import and CPU envelope/checksum. The actual filesystem cycle is measured directly.

Attention uses seeded component queries, not model logits. The bridge temporarily materializes one bounded chunk; it is not a fused two-layer-scratch converter. Filesystem reads may hit the page cache. Native cache-writer/generation integration, concurrency, cold SSD and real network throughput remain unimplemented.

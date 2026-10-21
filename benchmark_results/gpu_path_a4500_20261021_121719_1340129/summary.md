# GPU KV path component batch

Status: **completed**. GPU measured: **True**.

| Attention case | Tail | Dense wall ms | P1 wall ms | Max output difference |
|---|---:|---:|---:|---:|
| frozen_smol/page_000 | 0 | 1.21699 | 1.25233 | 0 |
| frozen_smol/page_000 | 3 | 1.21129 | 1.25505 | 0 |
| frozen_smol/page_001 | 0 | 1.19730 | 1.26166 | 0 |
| frozen_smol/page_001 | 3 | 1.22316 | 1.26448 | 0 |
| frozen_smol/page_002 | 0 | 1.21067 | 1.25957 | 0 |
| frozen_smol/page_002 | 3 | 1.21440 | 1.26311 | 0 |
| frozen_smol/page_003 | 0 | 1.21653 | 1.26423 | 0 |
| frozen_smol/page_003 | 3 | 1.21456 | 1.26188 | 0 |
| qwen_reuse/test_pottery_kiln_diary_0 | 0 | 0.20581 | 0.21247 | 2.38e-07 |
| qwen_reuse/test_pottery_kiln_diary_0 | 3 | 0.20381 | 0.20920 | 1.51e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 0 | 0.20434 | 0.21692 | 1.56e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 3 | 0.20588 | 0.21287 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 0 | 0.20204 | 0.21588 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 3 | 0.20929 | 0.21216 | 1.64e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 0 | 0.20590 | 0.21651 | 1.49e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 3 | 0.20973 | 0.21139 | 1.19e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 0 | 0.20777 | 0.22043 | 2.01e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 3 | 0.20420 | 0.21236 | 1.51e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 0 | 0.20371 | 0.21283 | 1.97e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 3 | 0.20610 | 0.21243 | 2.27e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 0 | 0.20849 | 0.21979 | 1.45e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 3 | 0.21443 | 0.21705 | 1.75e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 0 | 0.20375 | 0.21379 | 1.79e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 3 | 0.20940 | 0.20770 | 2.09e-07 |

| Transfer case | Arm | Input chunk | Entropy | T complete bpv | Raw fallback | KV to T ms | T to KV ms | P to T ms | T to P ms | Exact |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| frozen_smol/page_000 | A | 512 | scalar | 9.66055 | False | 306.570 | 231.542 | 308.616 | 225.837 | True |
| frozen_smol/page_000 | A | 512 | cached | 9.66055 | False | 139.141 | 192.912 | 140.123 | 194.153 | True |
| frozen_smol/page_001 | A | 512 | scalar | 9.11254 | False | 306.763 | 220.518 | 309.076 | 222.188 | True |
| frozen_smol/page_001 | A | 512 | cached | 9.11254 | False | 138.679 | 194.174 | 138.709 | 195.635 | True |
| frozen_smol/page_002 | A | 512 | scalar | 9.64627 | False | 309.020 | 220.737 | 309.066 | 220.863 | True |
| frozen_smol/page_002 | A | 512 | cached | 9.64627 | False | 138.110 | 193.018 | 138.323 | 194.544 | True |
| frozen_smol/page_003 | A | 512 | scalar | 9.34631 | False | 308.725 | 222.554 | 308.888 | 220.731 | True |
| frozen_smol/page_003 | A | 512 | cached | 9.34631 | False | 139.252 | 195.786 | 139.270 | 195.584 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | 512 | scalar | 9.17834 | False | 236.579 | 181.800 | 236.459 | 180.934 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | 512 | cached | 9.17834 | False | 115.884 | 159.706 | 115.568 | 162.158 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | 512 | scalar | 10.06049 | False | 236.548 | 179.608 | 237.128 | 182.921 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | 512 | cached | 10.06049 | False | 115.640 | 161.181 | 116.686 | 162.491 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | 512 | scalar | 9.99316 | False | 236.602 | 182.067 | 236.421 | 179.640 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | 512 | cached | 9.99316 | False | 115.394 | 161.390 | 116.034 | 161.453 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | 512 | scalar | 9.78412 | False | 236.552 | 182.989 | 238.050 | 180.086 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | 512 | cached | 9.78412 | False | 115.613 | 162.475 | 115.527 | 158.684 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | 512 | scalar | 9.12720 | False | 235.697 | 179.919 | 235.700 | 181.539 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | 512 | cached | 9.12720 | False | 115.351 | 161.688 | 115.929 | 158.749 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | 512 | scalar | 9.59076 | False | 236.481 | 177.808 | 235.552 | 183.190 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | 512 | cached | 9.59076 | False | 115.850 | 162.238 | 115.191 | 159.434 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | 512 | scalar | 9.44614 | False | 236.676 | 178.463 | 236.890 | 182.070 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | 512 | cached | 9.44614 | False | 115.845 | 161.097 | 115.093 | 162.462 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | 512 | scalar | 9.45724 | False | 235.940 | 183.554 | 235.887 | 179.905 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | 512 | cached | 9.45724 | False | 115.333 | 161.847 | 115.092 | 159.901 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | 512 | scalar | 7.74146 | False | 306.691 | 208.501 | 307.053 | 207.803 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | 512 | cached | 7.74146 | False | 186.330 | 191.300 | 187.617 | 190.753 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | 512 | scalar | 9.71503 | False | 307.750 | 209.707 | 307.826 | 208.595 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | 512 | cached | 9.71503 | False | 186.567 | 186.953 | 187.152 | 190.203 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | 512 | scalar | 9.61575 | False | 306.382 | 210.170 | 307.673 | 207.926 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | 512 | cached | 9.61575 | False | 186.798 | 192.470 | 185.975 | 188.466 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | 512 | scalar | 9.36581 | False | 307.900 | 207.834 | 308.050 | 210.211 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | 512 | cached | 9.36581 | False | 186.423 | 190.045 | 187.679 | 191.990 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | 512 | scalar | 7.69046 | False | 307.497 | 210.185 | 306.389 | 208.276 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | 512 | cached | 7.69046 | False | 186.709 | 192.379 | 186.527 | 188.789 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | 512 | scalar | 9.07468 | False | 307.853 | 207.693 | 307.890 | 210.496 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | 512 | cached | 9.07468 | False | 186.448 | 190.260 | 187.721 | 189.614 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | 512 | scalar | 8.80435 | False | 307.385 | 210.265 | 307.439 | 206.820 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | 512 | cached | 8.80435 | False | 187.486 | 192.716 | 186.819 | 188.974 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | 512 | scalar | 8.82047 | False | 308.038 | 207.633 | 308.915 | 210.338 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | 512 | cached | 8.82047 | False | 185.912 | 191.856 | 186.062 | 190.570 | True |

| Group | Complete bpv | Payload bpv | Directory + envelope bpv | Legacy full-identity bpv |
|---|---:|---:|---:|---:|
| frozen_smol/A/chunk512/cached | 9.441417 | 9.423600 | 0.017817 | 9.447559 |
| frozen_smol/A/chunk512/scalar | 9.441417 | 9.423600 | 0.017817 | 9.447559 |
| qwen_reuse/A/chunk512/cached | 9.579681 | 9.554596 | 0.025085 | 9.749481 |
| qwen_reuse/A/chunk512/scalar | 9.579681 | 9.554596 | 0.025085 | 9.749481 |
| qwen_reuse/B/chunk512/cached | 8.853500 | 8.828415 | 0.025085 | 9.023300 |
| qwen_reuse/B/chunk512/scalar | 8.853500 | 8.828415 | 0.025085 | 9.023300 |

| Diagnostic encoder control | Parameters ms | Prepared entropy ms | Encode device events | Decode device events |
|---|---:|---:|---:|---:|
| frozen_smol/A/chunk512/scalar/page_000 | 109.787 | 189.081 | 18553 | 18548 |
| frozen_smol/A/chunk512/cached/page_000 | 108.180 | 37.447 | 18913 | 18548 |
| frozen_smol/A/chunk512/scalar/page_003 | 102.465 | 194.752 | 18553 | 18548 |
| frozen_smol/A/chunk512/cached/page_003 | 107.558 | 37.427 | 18913 | 18548 |
| qwen_reuse/A/chunk512/scalar/test_pottery_kiln_diary_0 | 95.199 | 139.599 | 16103 | 16098 |
| qwen_reuse/A/chunk512/cached/test_pottery_kiln_diary_0 | 89.367 | 26.699 | 16359 | 16098 |
| qwen_reuse/A/chunk512/scalar/test_python_dependency_walk_2016 | 94.014 | 139.155 | 16103 | 16098 |
| qwen_reuse/A/chunk512/cached/test_python_dependency_walk_2016 | 94.583 | 26.687 | 16359 | 16098 |
| qwen_reuse/B/chunk512/scalar/test_pottery_kiln_diary_0 | 161.913 | 141.843 | 28483 | 28478 |
| qwen_reuse/B/chunk512/cached/test_pottery_kiln_diary_0 | 159.443 | 26.636 | 28739 | 28478 |
| qwen_reuse/B/chunk512/scalar/test_python_dependency_walk_2016 | 158.421 | 143.834 | 28483 | 28478 |
| qwen_reuse/B/chunk512/cached/test_python_dependency_walk_2016 | 157.710 | 26.652 | 28739 | 28478 |

Prepared entropy is an encoder control. Actual causal decode remains packet-only. Instrumented totals are separate from normal medians; missing CUDA activity is explicit.
Diagnostics run only in fresh child processes; parent timings never enable profiling. Leaf CUDA activity excludes user annotation ranges.

| Entropy comparison vs scalar | Encode speedup | Decode speedup | Complete T bytes identical |
|---|---:|---:|---|
| frozen_smol/A/chunk512/cached | 2.215x | 1.140x | True |
| qwen_reuse/A/chunk512/cached | 2.044x | 1.120x | True |
| qwen_reuse/B/chunk512/cached | 1.648x | 1.092x | True |

Cached encoding evaluates the same conditional frequencies in parallel, using a reusable UINT16 cache. Cached decoding reuses CDF interval endpoints while decoding causally. Speedups are medians of paired per-page ratios; all temporary memory and extra kernel launches are charged.

T1 is a new 12-bit binary-rANS stream with per-token restarts. Complete external bytes include the envelope, model/state identity digests, directory, checksum and payload. The independently expected full model identity is hashed and verified, and legacy full-identity envelopes remain readable. Shared state and measured scratch/peaks are separate. Raw fallback is explicitly identified.

T rates do not inherit the earlier CPU ~44% result. P rates and T rates are separate representations. GPU-body codec times exclude separately reported compressed-byte export/import and CPU envelope/checksum. The actual filesystem cycle is measured directly.

Attention uses seeded component queries, not model logits. The bridge temporarily materializes one bounded chunk; it is not a fused two-layer-scratch converter. Filesystem reads may hit the page cache. Native cache-writer/generation integration, concurrency, cold SSD and real network throughput remain unimplemented.

# GPU KV path component batch

Status: **completed**. GPU measured: **True**.

| Attention case | Tail | Dense wall ms | P1 wall ms | Max output difference |
|---|---:|---:|---:|---:|
| frozen_smol/page_000 | 0 | 1.33733 | 1.35301 | 0 |
| frozen_smol/page_000 | 3 | 1.32071 | 1.35315 | 0 |
| frozen_smol/page_001 | 0 | 1.31553 | 1.36812 | 0 |
| frozen_smol/page_001 | 3 | 1.35649 | 1.34987 | 0 |
| frozen_smol/page_002 | 0 | 1.32643 | 1.35416 | 0 |
| frozen_smol/page_002 | 3 | 1.32077 | 1.34469 | 0 |
| frozen_smol/page_003 | 0 | 1.31569 | 1.35138 | 0 |
| frozen_smol/page_003 | 3 | 1.32377 | 1.34691 | 0 |
| qwen_reuse/test_pottery_kiln_diary_0 | 0 | 0.21816 | 0.22579 | 2.38e-07 |
| qwen_reuse/test_pottery_kiln_diary_0 | 3 | 0.21885 | 0.22312 | 1.51e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 0 | 0.21803 | 0.22306 | 1.56e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 3 | 0.21725 | 0.21941 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 0 | 0.21780 | 0.22098 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 3 | 0.21621 | 0.22008 | 1.64e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 0 | 0.21801 | 0.22846 | 1.49e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 3 | 0.22216 | 0.22551 | 1.19e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 0 | 0.22242 | 0.22637 | 2.01e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 3 | 0.22232 | 0.22281 | 1.51e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 0 | 0.22199 | 0.23183 | 1.97e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 3 | 0.22282 | 0.22267 | 2.27e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 0 | 0.21994 | 0.22832 | 1.45e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 3 | 0.22035 | 0.22252 | 1.75e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 0 | 0.21697 | 0.22352 | 1.79e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 3 | 0.21805 | 0.21884 | 2.09e-07 |

| Transfer case | Arm | Input chunk | T complete bpv | Raw fallback | KV to T ms | T to KV ms | P to T ms | T to P ms | Exact |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| frozen_smol/page_000 | A | 64 | 9.66055 | False | 362.523 | 244.038 | 360.901 | 251.022 | True |
| frozen_smol/page_000 | A | 512 | 9.66055 | False | 315.284 | 221.422 | 315.159 | 223.868 | True |
| frozen_smol/page_000 | A | all | 9.66055 | False | 310.021 | 221.765 | 311.735 | 221.797 | True |
| frozen_smol/page_001 | A | 64 | 9.11254 | False | 362.196 | 241.552 | 361.863 | 240.427 | True |
| frozen_smol/page_001 | A | 512 | 9.11254 | False | 314.934 | 220.100 | 315.422 | 220.235 | True |
| frozen_smol/page_001 | A | all | 9.11254 | False | 311.124 | 221.593 | 310.392 | 220.532 | True |
| frozen_smol/page_002 | A | 64 | 9.64627 | False | 361.368 | 241.859 | 361.614 | 243.411 | True |
| frozen_smol/page_002 | A | 512 | 9.64627 | False | 314.543 | 222.309 | 315.415 | 222.906 | True |
| frozen_smol/page_002 | A | all | 9.64627 | False | 310.084 | 223.840 | 311.235 | 222.247 | True |
| frozen_smol/page_003 | A | 64 | 9.34631 | False | 361.260 | 241.690 | 360.483 | 241.758 | True |
| frozen_smol/page_003 | A | 512 | 9.34631 | False | 314.705 | 221.113 | 314.688 | 221.363 | True |
| frozen_smol/page_003 | A | all | 9.34631 | False | 310.444 | 221.004 | 311.275 | 222.396 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | 64 | 9.17834 | False | 393.080 | 257.811 | 392.327 | 258.413 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | 512 | 9.17834 | False | 240.755 | 179.915 | 241.193 | 180.876 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | all | 9.17834 | False | 220.370 | 180.062 | 220.821 | 179.857 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | 64 | 10.06049 | False | 392.762 | 257.898 | 393.721 | 260.492 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | 512 | 10.06049 | False | 240.676 | 180.310 | 240.984 | 183.209 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | all | 10.06049 | False | 220.595 | 178.083 | 220.629 | 178.210 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | 64 | 9.99316 | False | 394.332 | 260.432 | 392.944 | 260.162 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | 512 | 9.99316 | False | 241.517 | 183.780 | 241.383 | 180.249 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | all | 9.99316 | False | 221.175 | 177.808 | 219.391 | 178.225 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | 64 | 9.78412 | False | 393.805 | 260.335 | 393.304 | 258.995 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | 512 | 9.78412 | False | 240.906 | 181.010 | 241.629 | 183.677 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | all | 9.78412 | False | 220.089 | 176.770 | 220.742 | 176.930 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | 64 | 9.12720 | False | 394.289 | 258.476 | 393.185 | 259.403 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | 512 | 9.12720 | False | 241.085 | 178.996 | 240.820 | 181.224 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | all | 9.12720 | False | 220.766 | 176.429 | 221.063 | 176.453 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | 64 | 9.59076 | False | 393.776 | 259.718 | 393.606 | 260.805 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | 512 | 9.59076 | False | 241.178 | 184.484 | 241.098 | 179.650 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | all | 9.59076 | False | 220.749 | 179.520 | 213.187 | 179.835 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | 64 | 9.44614 | False | 393.093 | 262.060 | 392.733 | 259.760 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | 512 | 9.44614 | False | 241.325 | 181.852 | 241.727 | 180.382 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | all | 9.44614 | False | 220.617 | 181.132 | 220.954 | 179.707 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | 64 | 9.45724 | False | 393.447 | 258.473 | 392.690 | 260.054 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | 512 | 9.45724 | False | 240.978 | 182.086 | 241.501 | 178.874 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | all | 9.45724 | False | 220.625 | 177.942 | 221.313 | 178.118 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | 64 | 7.74146 | False | 467.220 | 333.209 | 467.686 | 333.777 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | 512 | 7.74146 | False | 313.820 | 209.717 | 314.413 | 207.804 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | all | 7.74146 | False | 293.587 | 207.658 | 293.900 | 206.568 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | 64 | 9.71503 | False | 465.412 | 331.586 | 466.176 | 332.938 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | 512 | 9.71503 | False | 314.628 | 209.984 | 314.582 | 207.868 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | all | 9.71503 | False | 294.404 | 204.287 | 295.506 | 209.212 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | 64 | 9.61575 | False | 465.836 | 331.751 | 466.817 | 333.854 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | 512 | 9.61575 | False | 315.822 | 207.840 | 315.008 | 208.211 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | all | 9.61575 | False | 294.295 | 204.933 | 293.820 | 204.139 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | 64 | 9.36581 | False | 465.465 | 332.654 | 466.080 | 332.632 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | 512 | 9.36581 | False | 316.127 | 208.952 | 314.986 | 210.395 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | all | 9.36581 | False | 294.053 | 206.094 | 294.016 | 206.392 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | 64 | 7.69046 | False | 466.313 | 332.177 | 466.388 | 334.112 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | 512 | 7.69046 | False | 314.522 | 207.331 | 314.325 | 210.422 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | all | 7.69046 | False | 294.546 | 206.875 | 294.549 | 205.729 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | 64 | 9.07468 | False | 466.091 | 332.596 | 465.846 | 333.065 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | 512 | 9.07468 | False | 314.150 | 209.241 | 314.378 | 210.666 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | all | 9.07468 | False | 294.171 | 206.539 | 294.334 | 205.495 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | 64 | 8.80435 | False | 465.205 | 334.656 | 466.694 | 332.489 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | 512 | 8.80435 | False | 315.395 | 209.632 | 315.340 | 210.691 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | all | 8.80435 | False | 294.157 | 206.406 | 294.392 | 206.629 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | 64 | 8.82047 | False | 466.233 | 333.292 | 467.203 | 334.089 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | 512 | 8.82047 | False | 314.157 | 209.503 | 314.724 | 210.684 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | all | 8.82047 | False | 294.082 | 208.221 | 294.456 | 208.219 | True |

| Group | Complete bpv | Payload bpv | Directory + envelope bpv | Legacy full-identity bpv |
|---|---:|---:|---:|---:|
| frozen_smol/A/chunkall | 9.441417 | 9.423600 | 0.017817 | 9.447559 |
| frozen_smol/A/chunk64 | 9.441417 | 9.423600 | 0.017817 | 9.447559 |
| frozen_smol/A/chunk512 | 9.441417 | 9.423600 | 0.017817 | 9.447559 |
| qwen_reuse/A/chunkall | 9.579681 | 9.554596 | 0.025085 | 9.749481 |
| qwen_reuse/A/chunk64 | 9.579681 | 9.554596 | 0.025085 | 9.749481 |
| qwen_reuse/A/chunk512 | 9.579681 | 9.554596 | 0.025085 | 9.749481 |
| qwen_reuse/B/chunkall | 8.853500 | 8.828415 | 0.025085 | 9.023300 |
| qwen_reuse/B/chunk64 | 8.853500 | 8.828415 | 0.025085 | 9.023300 |
| qwen_reuse/B/chunk512 | 8.853500 | 8.828415 | 0.025085 | 9.023300 |

| Diagnostic encoder control | Parameters ms | Prepared entropy ms | Encode device events | Decode device events |
|---|---:|---:|---:|---:|
| frozen_smol/A/chunk64/page_000 | 146.601 | 191.048 | 26371 | 26366 |
| frozen_smol/A/chunk512/page_000 | 102.207 | 187.636 | 18553 | 18548 |
| frozen_smol/A/chunkall/page_000 | 103.015 | 189.187 | 17944 | 17939 |
| frozen_smol/A/chunk64/page_003 | 145.236 | 187.411 | 26371 | 26366 |
| frozen_smol/A/chunk512/page_003 | 106.646 | 194.851 | 18553 | 18548 |
| frozen_smol/A/chunkall/page_003 | 103.378 | 189.545 | 17944 | 17939 |
| qwen_reuse/A/chunk64/test_pottery_kiln_diary_0 | 239.353 | 139.051 | 42647 | 42641 |
| qwen_reuse/A/chunk512/test_pottery_kiln_diary_0 | 93.689 | 138.613 | 16103 | 16098 |
| qwen_reuse/A/chunkall/test_pottery_kiln_diary_0 | 67.244 | 140.837 | 12692 | 12687 |
| qwen_reuse/A/chunk64/test_python_dependency_walk_2016 | 246.481 | 139.634 | 42647 | 42641 |
| qwen_reuse/A/chunk512/test_python_dependency_walk_2016 | 86.980 | 138.539 | 16103 | 16098 |
| qwen_reuse/A/chunkall/test_python_dependency_walk_2016 | 68.603 | 139.500 | 12692 | 12687 |
| qwen_reuse/B/chunk64/test_pottery_kiln_diary_0 | 320.203 | 139.403 | 55027 | 55021 |
| qwen_reuse/B/chunk512/test_pottery_kiln_diary_0 | 166.679 | 142.196 | 28483 | 28478 |
| qwen_reuse/B/chunkall/test_pottery_kiln_diary_0 | 146.746 | 139.514 | 25072 | 25067 |
| qwen_reuse/B/chunk64/test_python_dependency_walk_2016 | 311.938 | 138.411 | 55027 | 55021 |
| qwen_reuse/B/chunk512/test_python_dependency_walk_2016 | 156.789 | 141.017 | 28483 | 28478 |
| qwen_reuse/B/chunkall/test_python_dependency_walk_2016 | 147.880 | 139.485 | 25072 | 25067 |

Prepared entropy is an encoder control. Actual causal decode remains packet-only. Instrumented totals are separate from normal medians; missing CUDA activity is explicit.
Diagnostics run only in fresh child processes; parent timings never enable profiling. Leaf CUDA activity excludes user annotation ranges.

| Input-chunk comparison | Encode speedup | Decode speedup | Complete T bytes identical |
|---|---:|---:|---|
| frozen_smol/A/chunkall | 1.165x | 1.092x | True |
| frozen_smol/A/chunk512 | 1.149x | 1.095x | True |
| qwen_reuse/A/chunkall | 1.784x | 1.450x | True |
| qwen_reuse/A/chunk512 | 1.633x | 1.432x | True |
| qwen_reuse/B/chunkall | 1.583x | 1.612x | True |
| qwen_reuse/B/chunk512 | 1.481x | 1.591x | True |

Comparison rows are medians of paired per-page speedup ratios. Larger prediction input chunks may need more transient GPU memory; full timing samples and peaks remain in results.json.

T1 is a new 12-bit binary-rANS stream with per-token restarts. Complete external bytes include the envelope, model/state identity digests, directory, checksum and payload. The independently expected full model identity is hashed and verified, and legacy full-identity envelopes remain readable. Shared state and measured scratch/peaks are separate. Raw fallback is explicitly identified.

T rates do not inherit the earlier CPU ~44% result. P rates and T rates are separate representations. GPU-body codec times exclude separately reported compressed-byte export/import and CPU envelope/checksum. The actual filesystem cycle is measured directly.

Attention uses seeded component queries, not model logits. The bridge temporarily materializes one bounded chunk; it is not a fused two-layer-scratch converter. Filesystem reads may hit the page cache. Native cache-writer/generation integration, concurrency, cold SSD and real network throughput remain unimplemented.

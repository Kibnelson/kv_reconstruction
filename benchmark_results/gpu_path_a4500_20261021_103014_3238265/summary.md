# GPU KV path component batch

Status: **completed**. GPU measured: **True**.

| Attention case | Tail | Dense wall ms | P1 wall ms | Max output difference |
|---|---:|---:|---:|---:|
| frozen_smol/page_000 | 0 | 1.22414 | 1.27771 | 0 |
| frozen_smol/page_000 | 3 | 1.21406 | 1.25374 | 0 |
| frozen_smol/page_001 | 0 | 1.22143 | 1.25124 | 0 |
| frozen_smol/page_001 | 3 | 1.20899 | 1.25882 | 0 |
| frozen_smol/page_002 | 0 | 1.21553 | 1.26361 | 0 |
| frozen_smol/page_002 | 3 | 1.21442 | 1.25655 | 0 |
| frozen_smol/page_003 | 0 | 1.21918 | 1.25935 | 0 |
| frozen_smol/page_003 | 3 | 1.21691 | 1.25688 | 0 |
| qwen_reuse/test_pottery_kiln_diary_0 | 0 | 0.21331 | 0.21093 | 2.38e-07 |
| qwen_reuse/test_pottery_kiln_diary_0 | 3 | 0.21249 | 0.21045 | 1.51e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 0 | 0.21523 | 0.22433 | 1.56e-07 |
| qwen_reuse/test_pottery_kiln_diary_512 | 3 | 0.21879 | 0.21305 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 0 | 0.20825 | 0.21636 | 1.19e-07 |
| qwen_reuse/test_pottery_kiln_diary_1024 | 3 | 0.21936 | 0.22437 | 1.64e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 0 | 0.21187 | 0.21637 | 1.49e-07 |
| qwen_reuse/test_pottery_kiln_diary_2016 | 3 | 0.20634 | 0.21341 | 1.19e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 0 | 0.20387 | 0.21577 | 2.01e-07 |
| qwen_reuse/test_python_dependency_walk_0 | 3 | 0.20259 | 0.21025 | 1.51e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 0 | 0.20887 | 0.21764 | 1.97e-07 |
| qwen_reuse/test_python_dependency_walk_512 | 3 | 0.20657 | 0.21358 | 2.27e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 0 | 0.20488 | 0.21569 | 1.45e-07 |
| qwen_reuse/test_python_dependency_walk_1024 | 3 | 0.20673 | 0.21056 | 1.75e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 0 | 0.20348 | 0.21758 | 1.79e-07 |
| qwen_reuse/test_python_dependency_walk_2016 | 3 | 0.21256 | 0.21425 | 2.09e-07 |

| Transfer case | Arm | T complete bpv | Raw fallback | KV to T ms | T to KV ms | P to T ms | T to P ms | Exact |
|---|---|---:|---|---:|---:|---:|---:|---|
| frozen_smol/page_000 | A | 9.66669 | False | 356.267 | 245.425 | 356.619 | 241.588 | True |
| frozen_smol/page_001 | A | 9.11868 | False | 355.739 | 240.073 | 355.187 | 240.153 | True |
| frozen_smol/page_002 | A | 9.65241 | False | 355.163 | 240.314 | 355.452 | 240.134 | True |
| frozen_smol/page_003 | A | 9.35245 | False | 355.285 | 242.625 | 356.316 | 241.677 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | A | 9.34814 | False | 390.876 | 257.514 | 390.020 | 256.779 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | A | 10.23029 | False | 389.641 | 254.593 | 388.984 | 255.587 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | A | 10.16296 | False | 389.181 | 255.895 | 388.433 | 257.294 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | A | 9.95392 | False | 390.095 | 256.669 | 389.876 | 256.837 | True |
| qwen_reuse/test_python_dependency_walk_0 | A | 9.29700 | False | 388.972 | 254.875 | 389.453 | 255.407 | True |
| qwen_reuse/test_python_dependency_walk_512 | A | 9.76056 | False | 390.444 | 257.552 | 389.792 | 256.506 | True |
| qwen_reuse/test_python_dependency_walk_1024 | A | 9.61594 | False | 390.245 | 256.131 | 389.256 | 257.926 | True |
| qwen_reuse/test_python_dependency_walk_2016 | A | 9.62704 | False | 389.550 | 255.826 | 389.081 | 259.015 | True |
| qwen_reuse/test_pottery_kiln_diary_0 | B | 7.91125 | False | 460.413 | 328.294 | 460.844 | 327.590 | True |
| qwen_reuse/test_pottery_kiln_diary_512 | B | 9.88483 | False | 461.525 | 329.245 | 463.218 | 330.537 | True |
| qwen_reuse/test_pottery_kiln_diary_1024 | B | 9.78555 | False | 461.785 | 329.206 | 464.009 | 328.807 | True |
| qwen_reuse/test_pottery_kiln_diary_2016 | B | 9.53561 | False | 460.805 | 328.186 | 461.957 | 328.390 | True |
| qwen_reuse/test_python_dependency_walk_0 | B | 7.86026 | False | 460.629 | 329.270 | 463.665 | 330.085 | True |
| qwen_reuse/test_python_dependency_walk_512 | B | 9.24448 | False | 462.257 | 329.274 | 461.852 | 329.164 | True |
| qwen_reuse/test_python_dependency_walk_1024 | B | 8.97415 | False | 461.829 | 329.401 | 462.893 | 330.556 | True |
| qwen_reuse/test_python_dependency_walk_2016 | B | 8.99026 | False | 461.947 | 328.771 | 463.396 | 329.575 | True |

T1 is a new 12-bit binary-rANS stream with per-token restarts. Complete external bytes include the envelope, model/state identities, directory, checksum and payload. Shared state and measured scratch/peaks are separate. Raw fallback is explicitly identified.

T rates do not inherit the earlier CPU ~44% result. P rates and T rates are separate representations. GPU-body codec times exclude separately reported compressed-byte export/import and CPU envelope/checksum. The actual filesystem cycle is measured directly.

Attention uses seeded component queries, not model logits. The bridge temporarily materializes one bounded chunk; it is not a fused two-layer-scratch converter. Filesystem reads may hit the page cache. Native cache-writer/generation integration, concurrency, cold SSD and real network throughput remain unimplemented.

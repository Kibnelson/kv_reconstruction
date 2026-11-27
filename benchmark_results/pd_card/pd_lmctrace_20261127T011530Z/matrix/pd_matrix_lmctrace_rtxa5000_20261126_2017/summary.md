# LMCache lifecycle: the runs side by side

| | disk, NIXL, per-engine, carry-forward | disk, NIXL, per-engine, chat RAG | disk, NIXL, shared, carry-forward | disk, NIXL, shared, chat RAG | memory, NIXL, per-engine, carry-forward | memory, NIXL, per-engine, chat RAG | memory, NIXL, shared, carry-forward | memory, NIXL, shared, chat RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache server lasted the run | True | True | True | True | True | True | True | True |
| LMCache server errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P to D | nixl | nixl | nixl | nixl | nixl | nixl | nixl | nixl |
| LMCache servers | per_engine | per_engine | shared | shared | per_engine | per_engine | shared | shared |
| prompt layout | interleaved | history_first | interleaved | history_first | interleaved | history_first | interleaved | history_first |
| passages | all | all | all | all | all | all | all | all |
| conversations | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| disk reads | retain | retain | retain | retain | retain | retain | retain | retain |
| chunks bound by the engines' stores | 40 | 26 | 12 | 12 | 40 | 26 | 12 | 12 |
| chunks stored by | {'P': 12, 'D': 28} | {'P': 12, 'D': 14} | {'P': 12} | {'P': 12} | {'P': 12, 'D': 28} | {'P': 12, 'D': 14} | {'P': 12} | {'P': 12} |
| chunks written to memory | 40 | 26 | 12 | 12 | 40 | 26 | 12 | 12 |
| stored MB | 1174.400 | 763.400 | 352.300 | 352.300 | 1174.400 | 763.400 | 352.300 | 352.300 |
| chunks to disk | 40 | 26 | 12 | 12 | 0 | 0 | 0 | 0 |
| chunks removed from memory | 40 | 26 | 12 | 12 | 0 | 0 | 0 | 0 |
| pulls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pulled from | {} | {} | {} | {} | {} | {} | {} | {} |
| never pulled | 40 | 26 | 12 | 12 | 40 | 26 | 12 | 12 |
| D tokens from own GPU | 1888 | 176 | 1888 | 176 | 1888 | 176 | 1888 | 176 |
| D tokens loaded from LMCache | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D tokens read from P over NIXL | 7341 | 4062 | 7341 | 4062 | 7341 | 4062 | 7341 | 4062 |
| NIXL read ms (median) | 351.370 | 175.350 | 351.980 | 176.470 | 352.140 | 173.240 | 355.350 | 176.220 |
| NIXL read ms (p90) | 364.910 | 187.370 | 365.650 | 188.680 | 379.500 | 187.740 | 366.620 | 186.790 |
| NIXL read GB/s (median) | 0.597 | 0.588 | 0.599 | 0.584 | 0.591 | 0.595 | 0.600 | 0.585 |
| D's answers = one engine's | 3 of 5 | 4 of 5 | 3 of 5 | 4 of 5 | 3 of 5 | 4 of 5 | 3 of 5 | 4 of 5 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 378.090 | 260.690 | 427.630 | 256.700 | 394.860 | 271.100 | 382.890 | 294.130 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 70.470 | 64.410 | 69.680 | 64.750 | 69.880 | 65.080 | 69.420 | 65.350 |
| memory to disk ms (median) | 265.960 | 110.010 | 149.590 | 109.740 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 40 of 40; 0 of 0; 40 of 40; 3 of 5; 40 of 40; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 9 files, 0.26 GB; {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 7 files, 0.21 GB | 26 of 26; 0 of 0; 26 of 26; 4 of 5; 26 of 26; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 5 files, 0.15 GB; {'_default': '0 objects, 0 bytes'}; 6 files, 0.18 GB; {'_default': '0 objects, 0 bytes'}; 3 files, 0.09 GB | 12 of 12; 0 of 0; 12 of 12; 3 of 5; 12 of 12; 0 of 0; 0; 9 of 9 disk reads still in memory at the end (prefetch policy retain); {'_default': '9 objects, 264241152 bytes'}; 12 files, 0.35 GB | 12 of 12; 0 of 0; 12 of 12; 4 of 5; 12 of 12; 0 of 0; 0; 1 of 1 disk reads still in memory at the end (prefetch policy retain); {'_default': '1 objects, 29360128 bytes'}; 12 files, 0.35 GB | 40 of 40; 0 of 0; 40 of 40; 3 of 5 | 26 of 26; 0 of 0; 26 of 26; 4 of 5 | 12 of 12; 0 of 0; 12 of 12; 3 of 5 | 12 of 12; 0 of 0; 12 of 12; 4 of 5 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_203227, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_202827, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_202424, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_202019, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_203026, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_202628, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_202221, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_201728

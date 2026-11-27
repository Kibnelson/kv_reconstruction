# LMCache lifecycle: the runs side by side

| | disk, NIXL, per-engine, carry-forward | disk, NIXL, per-engine, chat RAG | disk, NIXL, shared, carry-forward | disk, NIXL, shared, chat RAG | memory, NIXL, per-engine, carry-forward | memory, NIXL, per-engine, chat RAG | memory, NIXL, shared, carry-forward | memory, NIXL, shared, chat RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache server lasted the run | True | True | True | True | True | True | True | True |
| LMCache server errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P to D | nixl | nixl | nixl | nixl | nixl | nixl | nixl | nixl |
| LMCache servers | per_engine | per_engine | shared | shared | per_engine | per_engine | shared | shared |
| prompt layout | interleaved | history_first | interleaved | history_first | interleaved | history_first | interleaved | history_first |
| passages | all | all | all | all | all | all | all | all |
| conversations | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| disk reads | retain | retain | retain | retain | retain | retain | retain | retain |
| chunks bound by the engines' stores | 391 | 249 | 113 | 114 | 391 | 249 | 113 | 114 |
| chunks stored by | {'P': 113, 'D': 278} | {'P': 114, 'D': 135} | {'P': 113} | {'P': 114} | {'P': 113, 'D': 278} | {'P': 114, 'D': 135} | {'P': 113} | {'P': 114} |
| chunks written to memory | 391 | 249 | 113 | 114 | 391 | 249 | 113 | 114 |
| stored MB | 11479.800 | 7310.700 | 3317.700 | 3347.100 | 11479.800 | 7310.700 | 3317.700 | 3347.100 |
| chunks to disk | 391 | 249 | 113 | 114 | 0 | 0 | 0 | 0 |
| chunks removed from memory | 391 | 249 | 113 | 114 | 173 | 58 | 0 | 0 |
| pulls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pulled from | {} | {} | {} | {} | {} | {} | {} | {} |
| never pulled | 391 | 249 | 113 | 114 | 391 | 249 | 113 | 114 |
| D tokens from own GPU | 45024 | 4080 | 45024 | 4080 | 45024 | 4080 | 45024 | 4080 |
| D tokens loaded from LMCache | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D tokens read from P over NIXL | 72366 | 36824 | 72366 | 36824 | 72366 | 36824 | 72366 | 36824 |
| NIXL read ms (median) | 409.740 | 199.820 | 411.050 | 201.640 | 415.840 | 200.800 | 414.830 | 200.620 |
| NIXL read ms (p90) | 658.300 | 286.330 | 659.680 | 293.820 | 661.630 | 281.900 | 662.970 | 281.520 |
| NIXL read GB/s (median) | 0.619 | 0.613 | 0.618 | 0.610 | 0.614 | 0.616 | 0.614 | 0.615 |
| NIXL blocks = P's bytes | 220 of 4541 | 132 of 2318 | 220 of 4541 | 132 of 2318 | 220 of 4541 | 132 of 2318 | 220 of 4541 | 132 of 2318 |
| D's answers = one engine's | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 |
| D's answers = one engine's, NIXL's layout | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 462.200 | 235.140 | 525.700 | 252.340 | 460.670 | 234.000 | 460.940 | 231.630 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 93.160 | 82.920 | 93.080 | 82.590 | 92.550 | 81.740 | 91.890 | 82.490 |
| memory to disk ms (median) | 325.130 | 168.590 | 228.390 | 169.430 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 391 of 391; 0 of 0; 391 of 391; 26 of 33; 24 of 33; 391 of 391; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 113 files, 3.32 GB; {'_default': '0 objects, 0 bytes'}; 76 files, 2.23 GB; {'_default': '0 objects, 0 bytes'}; 95 files, 2.79 GB; {'_default': '0 objects, 0 bytes'}; 107 files, 3.14 GB | 249 of 249; 0 of 0; 249 of 249; 26 of 33; 24 of 33; 249 of 249; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 114 files, 3.35 GB; {'_default': '0 objects, 0 bytes'}; 43 files, 1.26 GB; {'_default': '0 objects, 0 bytes'}; 44 files, 1.29 GB; {'_default': '0 objects, 0 bytes'}; 48 files, 1.41 GB | 113 of 113; 0 of 0; 113 of 113; 26 of 33; 24 of 33; 113 of 113; 0 of 0; 0; 93 of 93 disk reads still in memory at the end (prefetch policy retain); {'_default': '93 objects, 2730491904 bytes'}; 113 files, 3.32 GB | 114 of 114; 0 of 0; 114 of 114; 26 of 33; 24 of 33; 114 of 114; 0 of 0; 0; 14 of 14 disk reads still in memory at the end (prefetch policy retain); {'_default': '14 objects, 411041792 bytes'}; 114 files, 3.35 GB | 391 of 391; 0 of 0; 391 of 391; 26 of 33; 24 of 33 | 249 of 249; 0 of 0; 249 of 249; 26 of 33; 24 of 33 | 113 of 113; 0 of 0; 113 of 113; 26 of 33; 24 of 33 | 114 of 114; 0 of 0; 114 of 114; 26 of 33; 24 of 33 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_222259, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_221252, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_220243, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_215232, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_221747, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_220758, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_215729, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_214637

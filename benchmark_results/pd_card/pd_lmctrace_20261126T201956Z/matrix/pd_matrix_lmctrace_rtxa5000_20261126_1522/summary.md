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
| NIXL read ms (median) | 353.110 | 173.910 | 351.060 | 178.740 | 350.280 | 174.250 | 349.970 | 177.920 |
| NIXL read ms (p90) | 377.930 | 187.870 | 362.190 | 189.710 | 366.320 | 186.510 | 363.870 | 187.560 |
| P held its blocks for D, ms (median) | 396.790 | 267.880 | 424.660 | 278.060 | 382.350 | 272.420 | 382.810 | 288.960 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 69.970 | 65.230 | 69.980 | 63.580 | 69.340 | 64.950 | 69.190 | 65.080 |
| memory to disk ms (median) | 266.780 | 109.430 | 149.550 | 108.680 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 40 of 40; 0 of 0; 40 of 40; 40 of 40; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 9 files, 0.26 GB; {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 7 files, 0.21 GB | 26 of 26; 0 of 0; 26 of 26; 26 of 26; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 5 files, 0.15 GB; {'_default': '0 objects, 0 bytes'}; 6 files, 0.18 GB; {'_default': '0 objects, 0 bytes'}; 3 files, 0.09 GB | 12 of 12; 0 of 0; 12 of 12; 12 of 12; 0 of 0; 0; 9 of 9 disk reads still in memory at the end (prefetch policy retain); {'_default': '9 objects, 264241152 bytes'}; 12 files, 0.35 GB | 12 of 12; 0 of 0; 12 of 12; 12 of 12; 0 of 0; 0; 1 of 1 disk reads still in memory at the end (prefetch policy retain); {'_default': '1 objects, 29360128 bytes'}; 12 files, 0.35 GB | 40 of 40; 0 of 0; 40 of 40 | 26 of 26; 0 of 0; 26 of 26 | 12 of 12; 0 of 0; 12 of 12 | 12 of 12; 0 of 0; 12 of 12 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_153205, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_152940, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_152713, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_152443, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_153052, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_152828, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_152557, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_152252

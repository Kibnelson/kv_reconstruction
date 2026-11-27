# LMCache lifecycle: the runs side by side

| | disk, NIXL, per-engine, carry-forward | disk, NIXL, per-engine, chat RAG | disk, NIXL, shared, carry-forward | disk, NIXL, shared, chat RAG | memory, NIXL, per-engine, carry-forward | memory, NIXL, per-engine, chat RAG | memory, NIXL, shared, carry-forward | memory, NIXL, shared, chat RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache server lasted the run | True | True | True | True | True | True | True | True |
| LMCache server errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P to D | nixl | nixl | nixl | nixl | nixl | nixl | nixl | nixl |
| LMCache servers | per_engine | per_engine | shared | shared | per_engine | per_engine | shared | shared |
| prompt layout | interleaved | history_first | interleaved | history_first | interleaved | history_first | interleaved | history_first |
| passages | all | all | all | all | all | all | all | all |
| conversations | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| disk reads | retain | retain | retain | retain | retain | retain | retain | retain |
| chunks bound by the engines' stores | 851 | 529 | 242 | 243 | 851 | 529 | 242 | 243 |
| chunks stored by | {'P': 242, 'D': 609} | {'P': 243, 'D': 286} | {'P': 242} | {'P': 243} | {'P': 242, 'D': 609} | {'P': 243, 'D': 286} | {'P': 242} | {'P': 243} |
| chunks written to memory | 851 | 529 | 242 | 243 | 851 | 529 | 242 | 243 |
| stored MB | 24985.500 | 15531.500 | 7105.200 | 7134.500 | 24985.500 | 15531.500 | 7105.200 | 7134.500 |
| chunks to disk | 851 | 529 | 242 | 243 | 0 | 0 | 0 | 0 |
| chunks removed from memory | 851 | 529 | 242 | 243 | 638 | 318 | 47 | 47 |
| pulls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pulled from | {} | {} | {} | {} | {} | {} | {} | {} |
| never pulled | 851 | 529 | 242 | 243 | 851 | 529 | 242 | 243 |
| D tokens from own GPU | 101344 | 10096 | 101344 | 10096 | 101344 | 10096 | 101344 | 10096 |
| D tokens loaded from LMCache | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D tokens read from P over NIXL | 157500 | 77668 | 157500 | 77668 | 157500 | 77668 | 157500 | 77668 |
| NIXL read ms (median) | 449.470 | 211.100 | 446.130 | 213.020 | 441.670 | 207.150 | 445.590 | 209.220 |
| NIXL read ms (p90) | 646.750 | 288.440 | 632.480 | 286.530 | 626.900 | 280.750 | 626.580 | 286.470 |
| NIXL read GB/s (median) | 0.607 | 0.604 | 0.612 | 0.607 | 0.619 | 0.615 | 0.618 | 0.610 |
| NIXL blocks = P's bytes | 9879 of 9879 | 4883 of 4883 | 9879 of 9879 | 4883 of 4883 | 9879 of 9879 | 4883 of 4883 | 9879 of 9879 | 4883 of 4883 |
| D's answers = one engine's | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 |
| D's answers = one engine's, NIXL's layout | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 476.550 | 236.360 | 540.490 | 244.860 | 464.640 | 232.770 | 466.970 | 234.690 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 94.120 | 83.000 | 94.380 | 82.820 | 94.530 | 82.240 | 94.120 | 83.060 |
| memory to disk ms (median) | 356.270 | 163.230 | 212.690 | 149.370 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 851 of 851; 0 of 0; 851 of 851; 62 of 69; 62 of 69; 851 of 851; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 242 files, 7.11 GB; {'_default': '0 objects, 0 bytes'}; 180 files, 5.28 GB; {'_default': '0 objects, 0 bytes'}; 206 files, 6.05 GB; {'_default': '0 objects, 0 bytes'}; 223 files, 6.55 GB | 529 of 529; 0 of 0; 529 of 529; 52 of 69; 52 of 69; 529 of 529; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 243 files, 7.13 GB; {'_default': '0 objects, 0 bytes'}; 94 files, 2.76 GB; {'_default': '0 objects, 0 bytes'}; 97 files, 2.85 GB; {'_default': '0 objects, 0 bytes'}; 95 files, 2.79 GB | 242 of 242; 0 of 0; 242 of 242; 62 of 69; 62 of 69; 242 of 242; 0 of 0; 0; 206 of 206 disk reads still in memory at the end (prefetch policy retain); {'_default': '206 objects, 6048186368 bytes'}; 242 files, 7.11 GB | 243 of 243; 0 of 0; 243 of 243; 52 of 69; 52 of 69; 243 of 243; 0 of 0; 0; 26 of 26 disk reads still in memory at the end (prefetch policy retain); {'_default': '26 objects, 763363328 bytes'}; 243 files, 7.13 GB | 851 of 851; 0 of 0; 851 of 851; 62 of 69; 62 of 69 | 529 of 529; 0 of 0; 529 of 529; 52 of 69; 52 of 69 | 242 of 242; 0 of 0; 242 of 242; 62 of 69; 62 of 69 | 243 of 243; 0 of 0; 243 of 243; 52 of 69; 52 of 69 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_234104, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_232500, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_230851, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_225243, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_233246, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_231716, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_230030, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_224358

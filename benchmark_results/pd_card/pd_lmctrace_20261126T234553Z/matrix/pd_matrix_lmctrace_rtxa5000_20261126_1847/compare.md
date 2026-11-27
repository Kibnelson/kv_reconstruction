# LMCache lifecycle: the runs side by side

| | memory, NIXL, shared, carry-forward | memory, NIXL, shared, carry-forward |
| --- | ---: | ---: |
| LMCache server lasted the run | True | True |
| LMCache server errors | 0 | 0 |
| P to D | nixl | nixl |
| LMCache servers | shared | shared |
| prompt layout | interleaved | interleaved |
| passages | all | all |
| conversations | 1 | 1 |
| disk reads | retain | retain |
| chunks bound by the engines' stores | 12 | 12 |
| chunks stored by | {'P': 12} | {'P': 12} |
| chunks written to memory | 12 | 12 |
| stored MB | 352.300 | 352.300 |
| chunks to disk | 0 | 0 |
| chunks removed from memory | 0 | 0 |
| pulls | 0 | 0 |
| pulled from | {} | {} |
| never pulled | 12 | 12 |
| D tokens from own GPU | 1888 | 1888 |
| D tokens loaded from LMCache | 0 | 0 |
| D tokens read from P over NIXL | 7341 | 7341 |
| NIXL read ms (median) | 216746.060 | 216832.950 |
| NIXL read ms (p90) | 228660.420 | 228955.130 |
| NIXL read GB/s (median) | 0.001 | 0.001 |
| UCX transports | cma, cuda_copy, cuda_ipc, tcp | cma, cuda_copy, cuda_ipc, tcp |
| P held its blocks for D, ms (median) | 216920.550 | 217020.260 |
| D tokens computed | 0 | 0 |
| pull ms, memory (median) | – | – |
| pull ms, disk (median) | – | – |
| pull ms, disk (p90) | – | – |
| lookup and prefetch ms, memory (median) | – | – |
| lookup and prefetch ms, disk (median) | – | – |
| retrieve ms (median) | – | – |
| P store ms (median) | 88.310 | 91.190 |
| memory to disk ms (median) | – | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 12 of 12; 0 of 0; 12 of 12 | 12 of 12; 0 of 0; 12 of 12 |

Runs: llama32_3b_lmc_memory_nixl_shared_carry_all_allgpus_getzcopy-on_rtxa5000_20261126_190404, llama32_3b_lmc_memory_nixl_shared_carry_all_getzcopy-on_rtxa5000_20261126_184749

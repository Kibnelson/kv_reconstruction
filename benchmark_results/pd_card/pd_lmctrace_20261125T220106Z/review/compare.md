# LMCache lifecycle: the modes side by side

| | disk | memory |
| --- | ---: | ---: |
| LMCache server lasted the run | True | True |
| stored chunks | 114 | 114 |
| stored MB | 0.000 | 0.000 |
| chunks to disk | 0 | 0 |
| chunks removed from memory | 0 | 0 |
| pulls | 0 | 0 |
| pulled from | {} | {} |
| never pulled | 114 | 114 |
| D tokens from own GPU | 101648 | 101648 |
| D tokens loaded | 0 | 0 |
| D tokens computed | 26305 | 26305 |
| pull ms, memory (median) | – | – |
| pull ms, disk (median) | – | – |
| pull ms, disk (p90) | – | – |
| lookup and prefetch ms, memory (median) | – | – |
| lookup and prefetch ms, disk (median) | – | – |
| retrieve ms (median) | – | – |
| P store ms (median) | 120.330 | 123.930 |
| memory to disk ms (median) | – | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 114 of 114; 0 of 0; 0 of 114 | 114 of 114; 0 of 0; 0 of 114 |

Runs: llama32_3b_lmc_disk_rtxa5000_20261125_170432, llama32_3b_lmc_memory_rtxa5000_20261125_170227

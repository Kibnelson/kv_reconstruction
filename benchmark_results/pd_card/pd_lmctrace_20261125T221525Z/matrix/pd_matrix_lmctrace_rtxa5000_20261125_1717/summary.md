# LMCache lifecycle: the modes side by side

| | disk | memory |
| --- | ---: | ---: |
| LMCache server lasted the run | True | True |
| LMCache server errors | 0 | 0 |
| chunks bound by P's stores | 114 | 114 |
| chunks written to memory | 114 | 114 |
| stored MB | 3347.100 | 3347.100 |
| chunks to disk | 114 | 0 |
| chunks removed from memory | 114 | 0 |
| pulls | 102 | 102 |
| pulled from | {'L2': 102} | {'L1': 102} |
| never pulled | 12 | 12 |
| D tokens from own GPU | 101632 | 101632 |
| D tokens loaded | 23040 | 23040 |
| D tokens computed | 3281 | 3281 |
| pull ms, memory (median) | – | 13.930 |
| pull ms, disk (median) | 70.940 | – |
| pull ms, disk (p90) | 97.660 | – |
| lookup and prefetch ms, memory (median) | – | 2.190 |
| lookup and prefetch ms, disk (median) | 43.990 | – |
| retrieve ms (median) | 4.770 | 5.020 |
| P store ms (median) | 115.940 | 128.630 |
| memory to disk ms (median) | 188.290 | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 114 of 114; 1632 of 1632; 114 of 114 | 114 of 114; 1632 of 1632; 114 of 114 |

Runs: llama32_3b_lmc_disk_rtxa5000_20261125_171915, llama32_3b_lmc_memory_rtxa5000_20261125_171716

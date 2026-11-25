# LMCache lifecycle: the modes side by side

| | disk | memory |
| --- | ---: | ---: |
| stored chunks | 10 | 10 |
| stored MB | 293.600 | 293.600 |
| chunks to disk | 10 | 0 |
| chunks removed from memory | 10 | 0 |
| pulls | 10 | 10 |
| pulled from | {'L2': 10} | {'L1': 10} |
| never pulled | 0 | 0 |
| D tokens from own GPU | 100624 | 100624 |
| D tokens loaded | 2864 | 2864 |
| D tokens computed | 23825 | 23825 |
| pull ms, memory (median) | – | 71.600 |
| pull ms, disk (median) | 66.340 | – |
| pull ms, disk (p90) | 76.790 | – |
| lookup and prefetch ms, memory (median) | – | 1.600 |
| lookup and prefetch ms, disk (median) | 40.740 | – |
| retrieve ms (median) | 21.530 | 64.410 |
| P store ms (median) | 232.180 | 319.460 |
| memory to disk ms (median) | 331.360 | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 10 of 10; 160 of 160; 10 of 10 | 10 of 10; 160 of 160; 10 of 10 |

Runs: llama32_3b_lmc_disk_rtxa5000_20261125_153456, llama32_3b_lmc_memory_rtxa5000_20261125_153237

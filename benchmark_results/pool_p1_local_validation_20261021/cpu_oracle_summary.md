# P1 GPU pool page and head-tile benchmark

Status: **completed**. GPU measured: **False**.

T encoding, T-to-P conversion, SSD I/O and compressed attention: **not implemented by this benchmark**.

| Case | Arena/raw | Fits | Required payload bpv | Allocated packet bpv | KV to P wall ms | Full sector unpack wall ms | Exact |
|---|---:|---|---:|---:|---:|---:|---|
| page_000 | 0.80 | True | 11.71667 | 13.05000 | unmeasured | unmeasured | True |
| page_000 | 1.00 | True | 11.71667 | 16.25000 | unmeasured | unmeasured | True |
| page_001 | 0.80 | True | 11.39757 | 13.05000 | unmeasured | unmeasured | True |
| page_001 | 1.00 | True | 11.39757 | 16.25000 | unmeasured | unmeasured | True |
| page_002 | 0.80 | True | 11.66528 | 13.05000 | unmeasured | unmeasured | True |
| page_002 | 1.00 | True | 11.66528 | 16.25000 | unmeasured | unmeasured | True |
| page_003 | 0.80 | True | 11.36076 | 13.05000 | unmeasured | unmeasured | True |
| page_003 | 1.00 | True | 11.36076 | 16.25000 | unmeasured | unmeasured | True |

Packet accounting includes the entire arena reservation and directory. CPU-oracle mode projects these byte counts; it does not measure GPU allocations. Shared calibration and GPU scan workspace are separately recorded; actual CUDA runs include original input and output/status tiles in peak allocation samples. Required payload bpv alone is not a resident-memory result.

Selected-tile timings, cold allocation/JIT, P0 paired controls if requested, software edge cases and full sample distributions are in results.json. Full-sector decode is a validation/throughput control, not compressed attention.

This is an append-batch bounded arena prototype. It has no reusable serving allocator, concurrent publication protocol or attention integration. Capacity exhaustion leaves sectors unpublished and retains the original source; it is not silently repaired on CPU.

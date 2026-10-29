# Official SplitZip native procedure on the NUC

**Status: completed.** GPU performance measured: **True**.

Unchanged released SplitZip native procedure on Llama-3.2-3B / A4500; model/hardware variation from Qwen3-32B / H200. Native codec bytes and original-byte API throughput, not framed transfer, NIC, SSD or resident attention.

| Native bpv | Saved | Encode GB/s | Decode GB/s | Original-word exact |
|---:|---:|---:|---:|---|
| 12.045640 | 24.71% | 140.967 | 313.036 | True |

Official defaults: WikiText2 raw TRAIN/TEST, four usable prompts, 4,194,304 calibration values, 268,435,456 TEST values (512 MiB BF16), 1,024-value chunks, 10 warmups, 50 iterations and five repeats.

The author repeats/truncates the collected flat KV prefix to the requested value count. This is not the structured workload of our previous cross-layer predictor study; no cross-workload speed ranking is made.

Native compressed bytes include seven bytes per escape. Shared calibration, framing and installed receiver state are outside this native metric. Link scenarios, compressed attention and T-to-P conversion are not measured.

AQUA-KV remains a separate lossy-quality experiment. ExANS serving/paper reproduction is not established by this run.

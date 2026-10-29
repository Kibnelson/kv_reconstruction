# Official baseline readiness inventory

**Readiness report only. No benchmark or GPU performance measurement was run.**

Host RAM: 62.37 GiB total; 59.55 GiB available. Swap: 2.00 GiB total; 0.50 GiB free.
Effective host limit: 62.37 GiB; effective currently available: 59.55 GiB.

| Baseline | Readiness | Blockers |
|---|---|---|
| splitzip | eligible_for_verified_preflight | None found by metadata inventory; verified preflight still required |
| exans | eligible_for_verified_preflight | None found by metadata inventory; verified preflight still required |
| aqua | blocked | original RedPajama calibration provenance is not verified by this inventory; a matching complete official predictor receipt is not verified by this inventory |

AQUA host activation arrays alone require at least **24 GiB**: `2 × 256 × 8192 × 3072 × 2` bytes. Model weights, regression/covariance state, Python and other allocations are additional. Passing this lower bound does not guarantee training fits.

Original calibration data and a matching complete predictor receipt remain unverified. Existing backup calibration is a disclosed provenance variation. This script does not retry AQUA training.

The next feasible step is the official SplitZip preflight with the pinned source/environment, then its documented benchmark entrypoint using a verified local model snapshot. Bind the discovered snapshot and model/token/data receipts before replacing the placeholders:

```text
runs/_llama_official/envs/splitzip/bin/python runs/_llama_official/sources/splitzip/bench_codec_throughput.py --device cuda:0 --model <VERIFIED_ABSOLUTE_CACHED_SNAPSHOT> --local-files-only --output <NEW_OFFICIAL_RUN_DIR>/splitzip_native.json
```

Keep the official defaults: four prompts, 4,194,304 calibration values, 268,435,456 TEST values, chunk size1,024, 10 warmups, 50 iterations and five repeats. Using supported Llama-3.2-3B on the NUC is a disclosed model/hardware variation, not a reproduction of the paper's Qwen3-32B/H200 performance. See `docs/nuc_baseline_reproduction_plan.md`.

ExANS readiness refers only to its released synthetic codec correctness preflight, which needs no model. Its serving geometry and end-to-end vLLM/RDMA procedure remain unverified.

Model cache checks inspect exact revision paths and file presence/sizes; large weights are neither hashed nor loaded. Environment checks query package metadata without importing Torch. GPU visibility comes from nvidia-smi.

PCIe and interface link speeds are inventory, not transfer measurements. The 1–400 Gb/s comparisons remain calculated scenarios. The A4500 is not a measured 400 Gb/s NUC ingress path.

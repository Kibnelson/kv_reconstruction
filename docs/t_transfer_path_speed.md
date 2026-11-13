# T in basic terms, and the next speed test

The aim is to move less KV and have the receiving GPU ready sooner, with every original BF16 bit preserved.

1. Sender and receiver share a prediction rule fitted earlier on TRAIN data.
2. The sender uses earlier KV to predict later KV. The head-by-head variant also uses heads already reconstructed in the current layer.
3. Those predictions make likely original BF16 values cheaper to encode. The sender transmits an exact entropy-coded representation, including the required base data, stream indexes and headers.
4. The receiver reconstructs the base, repeats the same predictions from reconstructed KV, and decodes the original words exactly.

The predictions can be imperfect. The encoded values still reconstruct exactly. T is not a lossy approximation or a floating-point subtraction of residuals with missing bits. No per-token prediction hint or query tensor travels with the packet. Frozen shared predictor/policy state is a separate cost.

The completed A4500 run established 44.0484% saving for Fast T and 48.5682% for Compact T on four 8,192-token WikiText2 windows from Llama-3.2-3B. The speed stage below preserves those predictors, policies and packet bytes; it makes no additional compression claim.

## Why this is the next step

The full codec comparison measured approximately 98 ms for Fast T's warm encode/decode pair. The separately measured checked host path used about 1,040 ms for packing/D2H and 605 ms for restoration/H2D per 896 MiB window. Those steps must be included when assessing the local handoff. The former codec-plus-link model did not include them.

The original checked serializers allocate Python byte strings for large payloads, concatenate them, hash/copy them, and allocate receive buffers again. The new experimental path:

- Stores the same complete packet in a reusable pinned CPU byte buffer.
- Copies GPU components directly into that buffer, avoiding full-payload `.tobytes()` and concatenation copies.
- Keeps the same SHA-256 checksums and runs both sender hashing and receiver verification inside the measured path.
- Reuses GPU compaction and receive buffers for T, with a receive arena separate from the encoder arena. It reads the stream directory from the received packet each time.
- Gives raw BF16, SplitZip and ExANS the same pinned component-buffer treatment. Authors' encoder/decoder calls remain unchanged. Their native allocations, including ExANS' reconstructed record arena, remain part of the original API behavior.

This stage changes the packet path only. It does not alter the original T sources, the entropy coding arithmetic, frozen state, native baseline libraries, the original serializers or old result files. ExANS remains the pinned native API used in the preceding comparison, not a reproduction of the OpenLake H100 blog benchmark.

## Two waiting times

| Scenario | Starting point | Timed local stages |
|---|---|---|
| Fresh handoff | Original KV already resident on GPU | Encode -> compact/check/copy to host -> verify/restore on GPU -> decode |
| Stored-packet retrieval | Compressed host packet already prepared | Verify/restore on GPU -> decode |

Fresh handoff includes preparation because the current request needs it now. Stored-packet retrieval reports preparation separately because it occurred when the KV was stored. Neither scenario discards that system cost. Both keep full checksum verification.

This is a **local GPU -> compressed host packet -> GPU** test on the NUC. Host-buffer ownership is handed directly to the receiver code. It does not measure NIC/socket traffic, SSD/DRAM storage read latency, two GPUs, P1 conversion, cold shared-state loading or stage overlap. Real distributed transfer remains a subsequent measurement. Ideal link times at 3–400 Gbps are separately modeled from the measured complete local path plus paid packet bytes / bandwidth.

The returned `HostPacket` owns a view of reusable storage, not an immutable copy. Sending/receiving must finish before the slot is reused. A future asynchronous transport must use completion tracking or multiple slots; this benchmark deliberately uses serial boundaries. Pins and extra device buffers are reported as setup costs and memory, rather than counted as compressed payload.

## Measurement and validation

`scripts/bench_t_transfer_path.py` compares `legacy_checked` and `pinned_checked` for every selected method. It reuses states and native libraries retained by `t_native_baselines_v1_20261113T183344Z`; no calibration, fitting, package installation or rebuild occurs.

Defaults: all five methods, all four frozen windows, three rounds, one warmup and three measured samples for **each complete scenario**. Codec order rotates between rounds and path order alternates between cases/rounds. Each codec/round has a worker process. Large temporary buffers are released before independent receiver processes run. Warm component methods, setup, capture input staging, pinned host allocation and additional device transfer buffers are recorded separately. Torch peak increments do not cover ExANS' opaque native allocations.

Wall time covers all host work and actual compiled CUDA work, with synchronization at each serial stage boundary. Fresh total is directly timed from resident input to decoded output; retrieval total is directly timed from prepared compressed host bytes to decoded output. The report also records each constituent stage. Aggregation takes the median of samples, then the median across rounds per window, then the mean across windows. It does not add independent medians to invent the total.

Every timed sample reconstructs every BF16 word; the full word comparison runs outside the timer, while the packet protocol's SHA-256 stays inside. Before timing, each new packet must match the original checked serializer byte for byte for the same encoded input. All paid packet sizes must match the completed reference. T packets must additionally match the frozen reference SHA-256. Pinned receivers are overwritten with test values before first validation to detect dependence on cached encoder/receiver contents.

For every method/window in round zero, both the original receiver and the new pinned receiver decode the pinned packet in fresh processes. They receive packet/state paths and an expected digest only, with no original capture path. Missing or failed proofs prevent a review gate pass. Full coverage, actual GPU execution, at least three rounds, exact reconstruction, unchanged bytes and at least a 5% fresh local-path speed gain are required for a per-method review candidate. No automatic merge is performed.

Slot capacity is caller-provisioned as the largest paid packet for that method in the already frozen reference windows. This only provisions storage; it supplies no symbol hint and changes no coded bytes. The packet directory is generated/read on every call. Exceeding capacity fails explicitly; broader data or shapes need a suitable capacity and separate validation. This stage supports the frozen zero-position T window geometry, not a general serving transport protocol.

## NUC command

Run the smoke stage first:

```bash
cd ~/kv_t_head_spread_v1 &&
git pull --ff-only origin experiment/t-head-aware-spread-v1 &&
unset TRITON_INTERPRET &&
KV_HOME=~/kv_a5000 SMOKE=1 bash scripts/run_t_transfer_path_nuc.sh
```

The smoke uses one complete 896 MiB reference window, one round, one warmup and two measured samples per scenario for all five methods. It cannot pass the full-run gate. After review, the full command is:

```bash
KV_HOME=~/kv_a5000 bash scripts/run_t_transfer_path_nuc.sh
```

`METHODS=t_fast` can isolate T for diagnosis; full defaults retain raw BF16, SplitZip and ExANS controls. `T_PATH_OUTPUT`, `T_PATH_VENV`, `REFERENCE_RUN` and `CAPTURES` override the new output directory, existing pinned environment and read-only inputs. No existing output is overwritten. The GPU CLI rejects the CPU interpreter before creating its results directory.

Reports go to a new `t_checked_transfer_v1_<UTC timestamp>` directory with `summary.md`, sample-level `results.json`, worker/proof logs and `share.zip`. Packets are deleted after successful proof checks and are excluded from the share archive. GPU gains are pending execution of this new test on the NUC.

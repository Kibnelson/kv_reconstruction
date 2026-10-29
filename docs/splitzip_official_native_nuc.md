# Official SplitZip native procedure on the A4500

The readiness inventory at `7f36b5b` found a clean pinned author checkout,
the existing environment and cached Llama snapshot. It did not execute CUDA,
verify large weight contents or measure performance. The next run checks those
prerequisites and then executes the unchanged author entrypoint.

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_splitzip_official_native_a4500.sh
```

The wrapper writes a fresh run directory and publishes its small receipts,
including failures. It does not install dependencies, download weights, retry
AQUA or modify author source. Model/data caches must already be present.

## Procedure and scope

Source is `Intelligent-Microsystems-Lab/SplitZip` at
`42c743118a29db3270ae6a921a4a4eecb20cad04`. The official benchmark is
`bench_codec_throughput.py`. The only arguments supplied are the CUDA device,
verified local model directory, local-only model loading and output filename.
The model is `unsloth/Llama-3.2-3B` revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`.

All author workload and timing defaults stay in place:

| Setting | Value |
|---|---|
| Dataset | WikiText2 raw, TRAIN calibration and TEST measurement |
| Prompt selection | First four usable prompts per split; normalize whitespace; at least 80 characters |
| Calibration | 4,194,304 BF16 values |
| TEST tensor | 268,435,456 BF16 values, 512 MiB raw |
| Chunk size | 1,024 values |
| Timing | 10 warmups, 50 calls between synchronizations, five repeats |
| Exactness | Original/restored INT16 word equality before timing |
| Throughput numerator | Original uncompressed BF16 bytes |

The author code repeats or truncates the collected flat tensor to reach the
requested count. This is the released native procedure on a supported smaller
model and different GPU, not reproduction of Qwen3-32B/H200 paper numbers.
Native compressed bytes charge seven bytes per escape. Deployment framing,
shared calibration, file/network transfer and GPU allocation peaks are outside
this native metric.

Before the timed child process, a separate untimed preflight checks real CUDA
codec correctness and records source/environment/model/data/token identities.
The original benchmark performs its own model forwards and bitwise check; it
does not consume substituted preflight tensors. The runner hashes complete
local weight files and verifies Hugging Face blob digests where available.
Local file hashes establish a receipt, not independent remote model attestation.

Source is checked before and after execution. The native JSON is retained
unchanged alongside the exact command, logs, preflight and model receipts.
The model and dataset cache are used offline, so a missing cache produces a
failed report instead of a silently different workload.

## Comparison after this run

Keep this result separate from the frozen `810b516` matched-input T comparison,
which uses PEP/CPython documents and different packet/timing boundaries.
Do not rank native WikiText throughput against the old 21.71-ms decode number.
Next use complete identical layer/token/channel TEST words for the matched
comparison, with separate TRAIN calibration and the declared packet accounting.

Lossless SplitZip/ExANS and our exact T remain one track. Official AQUA HIGGS
and our lossy quality/transport results remain another. Resident P attention
and T-to-P conversion require their own measurements.

The NUC inventory reports 62.37 GiB host RAM, with 59.55 GiB available at that
instant. This exceeds AQUA's 24-GiB activation-array lower bound but does not
verify a complete training fit. Original RedPajama provenance and matching
official predictors remain unresolved. ExANS can next run its released
synthetic correctness test; that alone supplies no official serving throughput.

See [the full plan](nuc_baseline_reproduction_plan.md) for baseline provenance,
lossy/physical-byte distinctions, deployment state and the T/P evaluation order.

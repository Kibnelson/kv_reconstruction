# Official baseline progress on the NUC

## SplitZip: native procedure completed

Report commit: [10771af](https://github.com/Kibnelson/kv_reconstruction/commit/10771af7e2bff74b4d14c3a170de2187f82e6268).
Report directory:
`benchmark_results/splitzip_official_native_a4500_20261028_212622_1967778/`.

The pinned author script ran unchanged on the existing BF16
`unsloth/Llama-3.2-3B` snapshot and RTX A4500. It kept all workload and timing
defaults. This is a supported model/hardware variation of the released native
procedure, not reproduction of the Qwen3-32B/H200 paper numbers.

| Native metric | Measured result |
|---|---:|
| Original BF16 bytes | 536,870,912 (512 MiB) |
| Native compressed bytes | 404,184,609 |
| Native bits/value | 12.045640 |
| Bytes saved | 24.71475% |
| Encode | 140.967 GB/s; mean 3.80874 ms |
| Decode | 313.036 GB/s; mean 1.71505 ms |
| Exact original words | Yes, author INT16 equality gate before timing |

Audit: all eight packaged file sizes/SHA256 digests match. Source hashes before
and after equal the reviewed pin. Both complete weight-shard digests and the
tokenizer digest match their Hugging Face content-addressed cache blobs. The
untimed CUDA preflight also reconstructed all 65,536 BF16 bit patterns exactly.
WikiText raw TRAIN/TEST text, selected prompt/token, and dataset cache identities
were recorded. The unchanged native script does not expose its final repeated
KV tensor digest or actual module placement; those remain unobserved.

The author measures a repeated/truncated flat tensor from four TEST prompts,
with separate TRAIN calibration. Native bytes include seven-byte escapes and
exclude deployment framing/shared calibration. The old `810b516` test instead
uses structured PEP/CPython KV and framed restore calls. Their latencies must not
share a speed ranking. An identical structured-KV comparison is still required.

## ExANS: released correctness test completed

Report commit: `7e65a98`. Report directory:
`benchmark_results/exans_official_preflight_a4500_20261028_213554_2171122/`.
All eleven packaged artifact digests match, the eight checked source files
stayed unchanged and all seven commands succeeded. The fresh native test passed
exact round-trip and overflow checks. GPU performance remains unmeasured by
this test; it supplies no replacement for the prior matched-adapter latency.

The public release has an author synthetic round-trip/overflow test, not an
identified real-KV performance harness that reproduces the research post.
The completed run used the unchanged test from source pin
`de6c38bcc5438475bd1abf1e9b3d154cef414ef6`. Its reproducible command is:

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_exans_official_preflight_a4500.sh
```

This uses the existing cached CUDA 12.8 toolkit and nvCOMP 5.3.0.16, with no
package installation or download. Inventory found no `nvcc` on PATH; the prior
build receipt points to the project-local toolkit. The runner checks that cache
before configuring a fresh build directory for SM86. If prerequisites are
missing, it preserves and publishes a failed report.

Author source stays unchanged. The same disclosed compiler-injected compatibility
header forwards a unique_ptr allocation-error path to the author's existing
handler. The build configuration is therefore a compatibility variation. New
binary/library hashes and command/link/source receipts are recorded; the old
receipt authenticated only the library, not the test executable.

The official test checks exact reconstruction of a compressible synthetic
999,424-byte block and rejection of an overflowing fixture. It exits zero
silently on success. Missing shared libraries must fail the preflight, even
when `ldd` itself returns zero. Neither build duration nor test duration is codec
throughput; this step produces no new model-specific compression/performance row.
Existing framed C-ABI transfer results retain their separate scope. Official
serving geometry and research-post performance reproduction remain unverified.

## Remaining comparisons

| Track | State |
|---|---|
| Lossless T | SplitZip native and ExANS released correctness completed; frozen identical-KV full encode/decode timing is next |
| Lossy quality/T | Official AQUA calibration provenance and complete matching predictors unresolved; our full-protocol quality integration pending |
| Resident P | Earlier P1 pack/unpack exists; Llama compressed attention and T-to-P conversion remain unmeasured |

Preserve `810b516` as the previous matched-input transfer reference. The next
[frozen transfer run](frozen_transfer_evaluation.md) reuses its exact words,
state and codec implementations, with both common author-style batch timing and
isolated-call latency. Do not change codec/predictor kernels during this audit. See
[the protocol/T/P plan](nuc_baseline_reproduction_plan.md) for evaluation scopes
and [the SplitZip runner notes](splitzip_official_native_nuc.md) for its defaults.

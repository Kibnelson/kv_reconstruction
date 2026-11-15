# Fast and Compact T in our independent PD campaign

Our isolated source branch is `experiment/t-pd-matched-campaign-v1`. It starts from the other team's `pd-baselines` commit `313bc6de96c750a4a297a1a2e7610b9ac464cd90`. The original NUC T results, T source branch, RTX setup branch, `pd-baselines` and `main` stay separate. This is a new GPU experiment; no live serving result is claimed before its campaign runs.

## What is matched

The fork includes their engine-startup, NIXL import, nvCOMP stream, benchmark timeout and failed-log retention fixes through that pin. Their earlier job 30163 did not complete a serving comparison, so it is not a successful performance reference.

The campaign keeps their Slurm card, six-stage create/run/cleanup/verify-cleanup/review/publish lifecycle, node-local runtime, managed Python/uv cache, model revision, vLLM stack, original codecs, requests, metrics and transport implementations. Each run uses four allocated RTX A5000s on `gpu-grad-01` by default, 16 CPUs and 192 GB RAM. Other jobs can share CPU memory bandwidth and PCIe switches. The same-allocation node survey records the actual hardware and measured copy paths.

The inherited methods are raw BF16, SplitZip, ExANS, AdaCode, exactkv, nvCOMP, original T and T2, with NIXL and LMCache references. IBP remains conditional on a successful build and available disk. Missing methods remain visible as skips and invalidate complete declared coverage. The smoke covers 1P+3D and 2P+2D host transfers, 2P+2D GPU IPC copies, a CPU tier, native connector references and eager T/T2. The full matrices retain both layouts and all methods.

## Our added codecs and controls

| Name | Behavior |
|---|---|
| `t-fast` | The NUC FastDenseEncode forward pass with original T coding/decoding; 128-token programs, one stream chunk. |
| `t-dense-cached` | Original fused dense forward with cached static tables, using exactly the same predictor and stream geometry as `t-fast`. |
| `t-compact` | Head-by-head prediction with a TRAIN-fitted head-width map and DEV-selected blend. |
| `t-compact-global` | The same grouped predictor and stream geometry with the global prior; isolates the learned policy. |

The six experiment modules are copied unchanged from T source commit `19b32be6744e2d12f39e615ff806da8fb7dc0744`. The common PD adapter handles page order, nonzero positions, padding, tails and the shared-state identity. It zeroes the unused alignment words after rANS states so that byte-parity checks compare canonical frames. It uses the other team's GPU blob/transport format for every method. The THSPRV01 host packet and the NUC pinned-buffer serializer are not the serving wire format of this experiment.

The added family gets freshly calibrated states, not the NUC's frozen state files. Its predictor uses the same six WikiText-2 TRAIN captures, nominal spread uses the first half of the tables capture, and the second half selects the policy blend. Original baseline/T2 export is unchanged. Serving/test prompts never select the policy. The global prior is an eligible fallback. The policy arrays are shared state, and their byte count and frozen digest are recorded. The NUC's 44.05% and 48.57% savings are previous measurements; this campaign measures its own complete serving frames and latency.

## Comparison checks

- Fresh PD engines explicitly disable automatic prefix caching. This prevents an earlier codec's requests from populating a cache that bypasses later handoffs. CPU-tier runs retain prefix caching and reset it per point.
- Tier workload seeds and source texts are fixed by client count and repeat for every codec. Fresh PD arrival seeds are fixed per repeat. Prompt/text hashes and model revision are recorded. Benchmark environment overrides are frozen at campaign creation.
- Smoke uses one repeat and eight engine runs. Matrices default to one repeat, matching the other team's allocation budget; `PD_MATRIX_ROUNDS=3` requests three repeats after smoke timing establishes a scope that fits the allocation. The expanded run list is frozen at create and replayed once. Run names include seconds and process ID to avoid overwrite. A full repeated comparison requires at least three repeats; one-pass results remain preliminary. Full campaign durations are unmeasured, and all codecs/links at three repeats can exceed the 12-hour job limit.
- Setup requires captured exact-word checks at 8,192 and 3,000 tokens for all four new methods, compiled all-BF16-pattern kernel checks, fast/cached and compact/global stream checks, and four fresh GPU receiver processes using only frame and shared state. Missing or failed added methods stop setup.
- Correctness transfers verify reconstructed checksums and compare every installed paged-cache word against the decoded words. Timed serving sweeps retain the inherited verification setting; correctness checks and receiver proofs are separate evidence. The transfer logs also report aligned codec blob plus JSON header bytes, alongside the inherited payload-only bits. Control handshakes are separate from that codec frame.
- `review/matched_comparison.json` checks required methods, actual installs, completed versus requested counts, failed requests, skips, metric fields and repeat coverage. Its companion `matched_comparison.md` aggregates the available repeats by median and keeps shared-codec and nvCOMP engine groups separate. Use this report for repeated comparisons; the inherited matrix renderer was designed for single-pass runs. A published failure remains a failure. Smoke never sets `full_repeated_comparison_validated` true. No result triggers a merge.

## Our own login, tmux and checkout

Start from your computer using your own SSH account. Replace `YOUR_USER@YOUR_LOGIN_HOST` with your cluster login address. This address has not been supplied here; no account or SSH login has been created or tested by this change. The account needs GitHub SSH access and permission to submit four-GPU Slurm jobs. The login host needs Git, tmux, Slurm and your own uv CLI. The default Slurm project is `oana2026` on `gpu-grad-01`; set `T_PD_ACCOUNT`, `T_PD_NODE`, `T_PD_PARTITION` and `T_PD_QOS` for your account if needed.

```bash
ssh -t YOUR_USER@YOUR_LOGIN_HOST
```

On that login, create our dedicated tmux server and session. Its socket is `kvpd-t-matched` and its session is `smoke`. GPU jobs keep running there when you disconnect.

```bash
tmux -L kvpd-t-matched -f /dev/null new-session -s smoke
```

Inside this new session, make our standalone clone and run our launcher. The sparse checkout keeps historical benchmark bundles out of the home directory. This is a fresh clone with its own Git metadata. The source tree is pinned before any allocation.

```bash
git clone --filter=blob:none --single-branch --no-checkout \
    --branch experiment/t-pd-matched-campaign-v1 \
    git@github.com:Kibnelson/kv_reconstruction.git "$HOME/kv_t_pd_independent" &&
cd "$HOME/kv_t_pd_independent" &&
git sparse-checkout init --no-cone &&
git sparse-checkout set --no-cone '/*' '!/benchmark_results/' \
    '!/benchmark_fixtures/team_transfer_smol_replay_v1/' &&
git switch --detach HEAD &&
bash scripts/run_t_pd_independent.sh smoke
```

The launcher checks your Git author identity before allocating GPUs. If it is missing, configure it in this clone with `git config user.name "Your Name"` and `git config user.email "your-real-email"`, then rerun `bash scripts/run_t_pd_independent.sh smoke`. If your uv executable is outside PATH and `~/.local/bin/uv`, set `T_PD_UV` to its absolute path. Models and Python environments are installed by our own allocated setup.

| Component | Our location |
|---|---|
| Source clone | `~/kv_t_pd_independent` |
| tmux server / session | `kvpd-t-matched` / `smoke` |
| Node-local code, runtime, package/model cache and evidence | `/tmp/$USER/kvpd-t-matched-$USER/` |
| Saved state, login log and stage status | `~/.kvpd_t_matched/campaigns/<tag>/` |
| Published results | `evidence/experiment/t-pd-matched-campaign-v1/...` |

The launcher discards inherited campaign hooks, paths and benchmark overrides, then selects our matrix, root and settings explicitly. Code, model and package cache, runtime, logs, campaign tags and GPU allocation belong to this campaign. A physical cluster and its scheduler can still be shared. The smoke run requests four GPUs, 16 CPUs and 192 GB RAM for up to six hours; the other stages request their own small allocations. A first setup downloads into our own cache, and disk preflight still applies.

For user `nbore`, the default root is `/tmp/nbore/kvpd-t-matched-nbore`. `T_PD_ROOT` explicitly overrides it. Older campaigns retain the root in their saved `state.sh`; close them out using that recorded location. The default change applies to new campaigns and does not relocate existing records.

To detach, press `Ctrl-b`, then `d`. To reconnect from your computer:

```bash
ssh -t YOUR_USER@YOUR_LOGIN_HOST \
    'tmux -L kvpd-t-matched attach-session -t smoke'
```

The launcher prints `STATE=...` and `LOG=...`, and keeps a `status.txt` beside them. If a stage stops, those files remain. For closeout, source the printed state file in our session and use the saved pin, for example `bash scripts/pd_card.sh cleanup "$KVPD_PIN" "$KVPD_TAG"`, then `verify-cleanup`, `review` and `publish`. A run tag is single-use. A later full comparison uses a new tag and one selected matrix: `ring1p3d`, `ring2p2d`, `direct`, `tier` or `t2eager`; choose `T_PD_ROUNDS=3` only after smoke establishes an allocation that fits.

Send the `PUBLISHED evidence/experiment/t-pd-matched-campaign-v1/...` line for review. Publication adds only the campaign evidence in a new results branch. The source branch is unchanged by measured runs.

## Preflight stops and older filtered clones

A `KVPD_RUN_STATUS=stopped` result precedes model setup and serving. The exact reason is saved in the campaign's `run.json`; current code also prints `KVPD_STOP_REASON=...`. Driver, uv and disk preflights remain enforced. Inspect the recorded reason before choosing a new run tag.

The node-local code and evidence clones now inherit the home clone's promisor configuration before checkout. A real filtered-clone regression reproduces the earlier missing-object error and exercises create, a deliberate disk stop, cleanup, review and publication. Git's [partial-clone documentation](https://git-scm.com/docs/partial-clone) explains why omitted objects need a configured remote. The home checkout remains sparse.

For a campaign already stopped at an older pin, keep that checkout at its saved pin while closing it out. Source its printed `state.sh`. In a small CPU allocation on its `KVPD_NODE`, set `remote.origin.promisor=true` and `remote.origin.partialclonefilter=blob:none` in its two existing repositories: `$KVPD_ROOT/kvpd-tools/$KVPD_PIN` and `$KVPD_ROOT/kvpd-evidence/$KVPD_TAG/repo`. This repairs repository metadata without editing the pinned source. Read `$KVPD_ROOT/kvpd-evidence/$KVPD_TAG/repo/benchmark_results/pd_card/$KVPD_TAG/run.json` and `disk_plan.json` if present. Then use the saved pin for `cleanup`, `verify-cleanup`, `review` and `publish`. Update the home checkout only after closeout; the next campaign uses a new tag and the updated branch head.

## Engine build tools and failed link surveys

Job 30196 reached the link survey and engine startup but did not produce a valid serving comparison. Its survey child reported a CUDA launch failure, and its engines reported `FileNotFoundError: ninja`. These are separate failures; the CUDA traceback alone does not establish the failing copy or prove a codec bug.

Setup now explicitly installs Ninja in both the vLLM and LMCache environments. The runner puts the selected environment's `bin` first on PATH and requires its Ninja executable to run successfully before launching engines. The executable path and version are recorded in `env.json`.

Failed survey children and incorrect copy patterns return a failure after saving `specs.json` and its summary. The controller records `specs failed`, collects the evidence and ends the run stage before starting the matrix; normal closeout can then proceed. A failed survey cannot pass the comparison gate.

The pair probe now prints the active GPU pair and route before timing it. Serial host staging explicitly selects and synchronizes the source and destination devices. Both pair and concurrent-pattern checks compare the complete transferred buffer outside the timer. A new `--paths host`, `--paths direct` or `--paths host_pipe` option allows each route to be diagnosed in a fresh process. The original observed CUDA failure still needs a real GPU retest; CPU contracts do not establish that it is fixed.

Within a GPU allocation with an available codec-environment interpreter, a focused diagnostic can use `CUDA_LAUNCH_BLOCKING=1` and `--mib 64 --pairs 0-1 --paths host --reps 2`. Start the other routes in separate processes. These runs are diagnostic only: their output is flagged and cannot pass the performance comparison gate. The campaign launcher clears inherited `CUDA_LAUNCH_BLOCKING` for measured runs. Begin a new full campaign only after the focused GPU check resolves the copy failure.

## Interpretation limits

The ring link is paced shared host memory, not a measured NIC. Its configured rate is per prefill: 2P has two links, so equal per-link rates do not mean equal aggregate network budgets across layouts. GPU IPC measures the local PCIe path; the peer-access survey identifies hardware support. The CPU tier measures DRAM storage/reload, not SSD. Codecs sharing one engine also share its configured cache capacity and loaded constant sets; NIXL/nvCOMP/LMCache have separate engine runs. Record actual memory and allocation differences before drawing serving-capacity conclusions. Cross-branch numeric claims require matching hardware, model revision, prompt/text hashes, runtime versions, cache policy, seeds, load and measured link capacity; rerunning the original methods inside this campaign supplies the primary controls.

## After the cancelled job 30196

The published evidence is commit `90012974da8fe60ae67f20d4cb4b1f47760cb835` on
`evidence/experiment/t-pd-matched-campaign-v1/90112640bc59/smoke/job30196`.
Cleanup job 30204 was verified separately. Its captured self-test records exact
round trips for `t-fast`, `t-dense-cached`, `t-compact`, and `t-compact-global`
at 8,192 and 3,000 tokens in the A5000 vLLM environment. That is useful codec
evidence, not a completed serving comparison or the missing independent-receiver
readiness proof. Non-eager engines failed on Ninja; LMCache also failed on its
missing `lmcache.integration.vllm.vllm_v1_adapter` module. Eager engines started,
but no completed serving sweep was published.

Slurm cancellation bypassed the controller's normal record collection. Cleanup
now preserves the setup readiness records, package versions and run tail before
deleting a verified runtime. This does not recover files already removed from
job 30196, and does not rewrite its published evidence.

Run the focused copy diagnostic first, in our own login/tmux and standalone
checkout after old-campaign closeout:

```bash
cd "$HOME/kv_t_pd_independent" &&
git fetch --no-tags origin experiment/t-pd-matched-campaign-v1 &&
git merge --ff-only FETCH_HEAD &&
bash scripts/run_t_pd_diagnostics.sh
```

It requests four GPUs, four CPUs and 16 GiB for up to 45 minutes, copies your
account's uv into a private runtime and uses the campaign's own package cache
to make a minimal pinned Torch 2.11.0+cu128 environment. It runs no model
download, calibration or vLLM engines. Source
files are extracted from the recorded Git pin inside the allocation. Its runtime
is separate under `/tmp/$USER/kvpd-t-matched-$USER/kvpd-diagnostics/<tag>` and is
kept for inspection; it does not delete old runs or the reusable cache.

Defaults are 64 MiB, two repetitions, logical GPU pairs 0↔1 and 1↔2, with
`host`, `host_pipe` and `direct` each in a fresh Torch process, followed by an
independent `cudart_peer` process. Each Torch child first exercises all four
GPUs' host legs. The runtime-only child uses the local controls described
below on its two selected devices. Complete pair buffers are compared outside
the timers. `CUDA_LAUNCH_BLOCKING=1` makes this diagnostic only; no latency can
pass a performance or serving gate. Host-leg and route markers locate failures.
Use `T_PD_DIAG_MIB=896` for a later document-sized check after the small check
passes; `T_PD_DIAG_PAIRS` selects directed logical indexes from the allocation.
These selected-pair checks do not replace the full node survey.

The printed `OUTPUT` directory is under
`~/.kvpd_t_matched/diagnostics/links_<tag>/`. It retains `summary.md`,
`diagnostics.json`, per-route logs/results, GPU UUIDs, topology, package versions
and `share.zip`. A failed copy stays failed; the script does not silently select
a different transport or start the matrix. Review the diagnostic report before
the engine-startup check and the next matched smoke campaign. The LMCache adapter
failure remains a separate required follow-up.

### Diagnostic exits before a copy test

The attempt `links_20261115T155840Z_694088` at source `b2e2886` returned
Slurm exit 1 without a `DIAG_JOB` or copy marker. The old launcher printed its
job marker only after environment setup, so that output does not identify
the failed setup step. It supplies no new GPU-copy result.

The launcher now prints the job, node and root immediately on allocation entry,
then `DIAG_STAGE` before each setup step. Node errors report
`DIAG_NODE_STOP stage=... line=... exit=... job=...`; directory and cached-uv
checks also name the rejected path. Environment creation and package
installation keep their normal error output. `node_job.txt` and
`node_status.txt` retain the allocation identity and shell failure status in the
printed home `OUTPUT` directory, beside `login.log`.

Update this standalone checkout and rerun the same diagnostic command above.
Each attempt gets a new tag. If it stops before `DIAG_BEGIN`, share the output
including `DIAG_STAGE` and `DIAG_NODE_STOP` so the actual setup failure can be
resolved. If it reaches a copy failure, inspect its per-route report instead.
This logging change does not establish that the earlier setup or CUDA failure
is fixed and does not start the serving matrix.

### Missing cached uv in job 30222

The next attempt, `links_20261115T162649Z_705458` at source `c1a7a2c`,
identified the setup failure: job 30222 stopped at `cached_uv` because
`/tmp/nbore/kvpd-t-matched-nbore/kvpd-cache/bin/uv` was unavailable.
It reached no GPU-copy test. This does not explain the separate CUDA launch
failure in job 30196.

The diagnostic no longer requires a tool left by an earlier campaign. Before
requesting GPUs, it selects `T_PD_UV`, then an executable on PATH, then
`~/.local/bin/uv`, verifies the selected executable runs, and makes a private
temporary snapshot on shared home. The allocation copies that snapshot into
`<diagnostic-runtime>/bin/uv` and verifies its SHA-256 before running it. The
original source may be a symlink or disappear after submission without
breaking the allocated job. Only the temporary home snapshot is removed on
launcher exit; the node runtime and reports remain.

Missing package/Python cache directories are created under our own root.
An empty cache still needs the managed Python and pinned packages downloaded.
The source path, hash and version are saved as `uv_provenance.txt` and
`uv_version.txt`; executables are outside the report bundle. If no usable
account tool exists, the launcher stops before allocating GPUs and requests
an account install or an explicit `T_PD_UV=/absolute/path/to/uv` setting.
Update the standalone checkout and rerun the same diagnostic command above;
there is no need to edit or repopulate an old campaign by hand.

### Direct-copy parity failures in job 30223

The provided tail from `links_20261115T163810Z_708926`, job 30223, reports
direct-copy parity failures for 0→1, 1→0, 1→2 and 2→1 at 64 MiB. It also
shows passing `host` and `host_pipe` cases for 1→0, 1→2 and 2→1. The report
ended `DIAG_STATUS=failed`; it is not a serving or compression result. These
tests transfer ordinary bytes without invoking a T/T2 codec. The underlying
cause is not established by the traceback or by advertised peer support.

The diagnostic now adds a fourth case, `cudart_peer`, in a separate process
for each selected pair. It imports NumPy and package metadata, but never
imports Torch. Its installed Torch version identifies the pinned environment;
it does not describe the implementation of this copy. The child resolves the
CUDA 12 runtime from the environment's package files, requires runtime 12.8,
allocates buffers with `cudaMalloc`, enables peer access in both directions,
and uses `cudaMemcpyPeer` with explicit source and destination device IDs.

Every repetition uses a new random CPU reference and its bytewise inverse as
the destination's initial contents. Both selected GPUs must first pass full
H2D, local D2D and D2H controls. After the peer copy, the child compares the
entire destination to the reference, verifies the source is unchanged, and
saves mismatch counts, the first mismatch, hashes and untouched poison bytes.
Failed comparisons retain their JSON as well as logs. A missing peer capability,
failed local control, bad return code, corrupted byte or incomplete repetition
fails the case. It never retries through host memory as a substitute.

`platform_context.json` also records read-only peer-read/write capabilities,
relevant IOMMU boot flags and visible GPU IOMMU groups. Group presence or
capability output alone does not establish the cause or validate a working
peer transfer. No driver, BIOS, IOMMU, ACS or scheduler setting is changed.

Update the standalone checkout and run `bash scripts/run_t_pd_diagnostics.sh`
again. Allocation size and package pins are unchanged; the default selected
pairs now produce 16 cases. The original `direct` failures remain failures
even if the runtime case passes. Comparing the two implementations and the
local controls helps decide whether to investigate framework/allocator/copy
differences or the lower-level node peer path. Neither outcome alone proves
a particular driver or IOMMU fault, and the full serving matrix remains pending.

Relevant primary sources:

- [PyTorch 2.11 CUDA copy implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Copy.cu)
  already selects devices and inserts cross-device stream dependencies.
- [CUDA 12.8 programming guide](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-c-programming-guide/index.html#iommu-on-linux)
  describes the Linux bare-metal IOMMU limitation for PCIe peer transfers.
- [CUDA 12.8 memory API reference](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-runtime-api/group__CUDART__MEMORY.html)
  defines `cudaMemcpyPeer`. Its name alone is not proof of the physical path;
  this diagnostic explicitly enables peer access and records the controls.

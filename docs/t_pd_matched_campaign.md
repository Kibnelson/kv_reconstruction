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
| Node-local code, runtime, package/model cache and evidence | `/tmp/kvpd-t-matched-$USER/` |
| Saved state, login log and stage status | `~/.kvpd_t_matched/campaigns/<tag>/` |
| Published results | `evidence/experiment/t-pd-matched-campaign-v1/...` |

The launcher discards inherited campaign hooks, paths and benchmark overrides, then selects our matrix, root and settings explicitly. Code, model and package cache, runtime, logs, campaign tags and GPU allocation belong to this campaign. A physical cluster and its scheduler can still be shared. The smoke run requests four GPUs, 16 CPUs and 192 GB RAM for up to six hours; the other stages request their own small allocations. A first setup downloads into our own cache, and disk preflight still applies.

To detach, press `Ctrl-b`, then `d`. To reconnect from your computer:

```bash
ssh -t YOUR_USER@YOUR_LOGIN_HOST \
    'tmux -L kvpd-t-matched attach-session -t smoke'
```

The launcher prints `STATE=...` and `LOG=...`, and keeps a `status.txt` beside them. If a stage stops, those files remain. For closeout, source the printed state file in our session and use the saved pin, for example `bash scripts/pd_card.sh cleanup "$KVPD_PIN" "$KVPD_TAG"`, then `verify-cleanup`, `review` and `publish`. A run tag is single-use. A later full comparison uses a new tag and one selected matrix: `ring1p3d`, `ring2p2d`, `direct`, `tier` or `t2eager`; choose `T_PD_ROUNDS=3` only after smoke establishes an allocation that fits.

Send the `PUBLISHED evidence/experiment/t-pd-matched-campaign-v1/...` line for review. Publication adds only the campaign evidence in a new results branch. The source branch is unchanged by measured runs.

## Interpretation limits

The ring link is paced shared host memory, not a measured NIC. Its configured rate is per prefill: 2P has two links, so equal per-link rates do not mean equal aggregate network budgets across layouts. GPU IPC measures the local PCIe path; the peer-access survey identifies hardware support. The CPU tier measures DRAM storage/reload, not SSD. Codecs sharing one engine also share its configured cache capacity and loaded constant sets; NIXL/nvCOMP/LMCache have separate engine runs. Record actual memory and allocation differences before drawing serving-capacity conclusions. Cross-branch numeric claims require matching hardware, model revision, prompt/text hashes, runtime versions, cache policy, seeds, load and measured link capacity; rerunning the original methods inside this campaign supplies the primary controls.

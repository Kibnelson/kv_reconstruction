# Our copies of vLLM and LMCache

These are our own copies, detached from upstream: they were imported once, unchanged, and are never pulled from
upstream again. Every change made to them since is a trace hook (below), so the git history shows exactly where the
trace was added.

| Copy | Upstream | Tag | Commit | Files |
| --- | --- | --- | --- | --- |
| `vllm/` | https://github.com/vllm-project/vllm | `v0.32.0` | `5df63a7203cdb9328d824a27b682e77c7b9928be` | 4,164 |
| `lmcache/` | https://github.com/LMCache/LMCache | `v0.5.7.post1` | `1181bea5c65a6f477e3680acd14ffac988b13dc3` | 1,067 |

**What was imported.** The package itself, the C++/CUDA and Rust sources, and the build files (vLLM: `vllm/`,
`csrc/`, `cmake/`, `rust/`, `requirements/`, `CMakeLists.txt`, `setup.py`, `pyproject.toml`, `MANIFEST.in`,
`LICENSE`, `README.md`, `rust-toolchain.toml`; LMCache: `lmcache/`, `csrc/`, `rust/`, `setup_extensions/`,
`requirements/`, `CMakeLists.txt`, `setup.py`, `pyproject.toml`, `pyproject_cli.toml`, `MANIFEST.in`, `LICENSE`,
`README.md`). Left out: tests, docs, examples, benchmarks, tools, Docker files, and the upstream notes for coding
agents (`AGENTS.md`, `CLAUDE.md`). `vllm.sha256` and `lmcache.sha256` hold the hash of every imported file as
imported.

**Checked against the release wheels** (`vllm==0.32.0`, `lmcache==0.5.7.post1` from PyPI): every Python file of both
copies is byte-identical to the wheel's (vLLM 2,641 of 2,641, LMCache 904 of 904). The wheels add only what their
build generates: `_version.py`, generated protobuf modules, vendored kernels' Python (vLLM's `third_party/deep_gemm`),
and the compiled `.so` files.

**How they run.** The job's setup installs the release wheels of exactly these versions for their compiled parts,
then lays our copies' Python over them in a per-job tree (`scripts/lmc_overlay.py`). Before that it checks that every
file we did not hook is byte-identical to the wheel's, so the only Python that runs differently is our hooks. LMCache's
compiled modules are built from these copies: all four the wheel has (`cuda_ops`, `lmcache_native`, `lmcache_fs`,
`lmcache_redis`, from `lmcache/csrc`, built together so they share their types,
`scripts/lmc_build_ops.py`, with vLLM's environment's CUDA 13.0 headers and the CUDA 13.0 compiler from pip), because the wheel's were built against another torch and do not load beside vLLM
0.32's torch 2.13 (LMCache's torch fallback then crashed its server: job 30694). vLLM's kernels are the wheel's.

**Trace hooks.** A hook is a whole added line ending in `# KVTRACE`, calling our module `kvtrace` (in the repository
root). Removing every line ending in `# KVTRACE` gives back the imported file exactly; `scripts/lmc_overlay.py check`
verifies this against the hashes above. To see every hook:

```bash
git grep -n "# KVTRACE" -- third_party/
git diff b2189bd -- third_party/          # b2189bd: the import, unchanged
python3 scripts/lmc_overlay.py hooks      # file, line and event of each
```

The hooks (41 lines in 10 files: vLLM's scheduler, KV cache manager and block pool, and its NIXL connector's pull
scheduler and workers; LMCache's vLLM connector, event bus, file-system adapter and storage manager), the records they write and how the lifecycle report joins them: `docs/lmcache_lifecycle.md`.

# Isolated RTX P/D readiness

This test branch starts from `pd-vllm-protocol` commit `66b35e495b9b803c8e31cbaa1bfd56322e2a684d`. The working Fast T experiment stays on its own branch and environment. This stage adds only an environment installer and readiness check. It fits no predictor, exports no codec, downloads no model and starts no serving engines.

## Observed RTX machine

The user reported one RTX A4500, 20,470 MiB total and 19,880 MiB free, driver 580.126.20, 62 GiB host RAM and 58 GiB available. The separate `~/kv_a5000/venv_vllm` environment was missing. Disk space has not yet been reported; the setup checks its selected 20 GiB of free headroom before installation.

A short 1P+1D correctness test may fit with two Llama-3.2-3B engines, small contexts and eager execution. The engines' simultaneous startup has not been measured. Shared-GPU results establish function and resource behavior; independently assigned 1P+2D and 2P+2D performance tests use three and four GPUs respectively.

## Stage 1: environment only

From a new worktree on `experiment/t-pd-rtx-smoke-v1`, run:

```bash
unset TRITON_INTERPRET
KV_HOME="$HOME/kv_a5000" bash scripts/setup_pd_rtx_env.sh
```

The installer creates or updates only the separate `venv_vllm` below KV_HOME, installs `vllm==0.32.0` and the proxy's aiohttp dependency, records the resolved package versions, and runs `pip check`. Python 3.12 is selected explicitly; `PD_BOOTSTRAP_PYTHON` can point to that interpreter. It rejects a target overlapping the existing codec venv. It uses the selected venv's absolute Python for all package operations with pip's isolated option.

The runtime target follows the existing connector's pinned vLLM version. See the [vLLM 0.32 GPU installation guide](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/) for its wheel and environment requirements. This installer checks the R580 driver prerequisite, then verifies actual CUDA execution rather than assuming a compatible version string is sufficient.

Runtime diagnostics compile a small Triton CUDA kernel and copy all 65,536 possible BF16 word patterns exactly. They import the actual P/D connector against the installed vLLM API and construct a fail-on-transfer-error configuration. Neither a successful import nor this synthetic kernel establishes a real serving handoff. The report keeps `PD_handoff_validated`, `real_vllm_engine_started`, `T_integration_validated` and `network_performance_validated` false.

Outputs go to a new `KV_HOME/runs/pd_rtx_env_v1_*` directory: setup log, package freeze, GPU inventory, codec-environment fingerprints and `environment.json`. Success ends with `PD ENV READY`. A readiness failure exits nonzero and leaves its diagnostic report. This script performs no Git checkout, commit, merge or push and changes no frozen benchmark artifacts. The existing `pd_setup.sh` and `run_pd_vllm.sh` are not called.

## Stage 2: raw serving handoff, pending

After the environment report passes, prepare a short 1-prefill/1-decode smoke with raw KV, the pinned Llama model revision, batch size one and short prompts. Confirm actual external KV matching, successful installation checksums, and visible transfer failures; generated text alone is insufficient. Check real cache layout and both engines' total memory before selecting memory shares. Preserve the existing T codec environment and reuse model files already cached.

The raw smoke launcher and its measured result are not part of this environment-only stage. The latest frozen Fast T/pinned packet path is also not integrated into the inherited P/D connector. That integration needs separate packet/word validation in the newer vLLM runtime, page-layout handling, arbitrary lengths and buffer-lifetime checks. The 685 ms checked local path remains a one-GPU host-buffer result, not a P/D TTFT measurement.

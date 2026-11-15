# Focused PD GPU diagnostics: software validation

The added login launcher uses our standalone experiment checkout, a separate
diagnostic runtime under our own `/tmp/$USER/kvpd-t-matched-$USER` root, the
campaign's reusable uv cache, and a short four-GPU Slurm allocation. The probe
environment is pinned to Torch 2.11.0+cu128 and NumPy 2.3.5. No model capture,
calibration, vLLM engine or serving matrix is launched by this diagnostic.

Each selected pair/route runs in a fresh process with CUDA_LAUNCH_BLOCKING=1.
The manager requires the declared route, payload, four visible GPUs, pinned
Torch/CUDA versions and complete-buffer parity before a case passes. Failed
children and timeouts keep their logs. Reports and share.zip remain in the
user's home evidence directory. Diagnostic timings never validate performance.

Cleanup now collects setup proofs, package records and the run tail before
deleting an inactive runtime, including when Slurm killed its controller.

Validation: 39 software tests passed in 171.927 seconds; Python compilation and
Bash syntax checks passed. Fixtures exercise real subprocess isolation and the
login/archive/cache/report path, but **no actual GPU was executed here**. The
GPU copy failure and compiled engine startup still require retesting on the node.
See job30196_review.md for the separately reviewed published GPU evidence.

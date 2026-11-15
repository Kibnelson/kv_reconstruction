# Independent uv bootstrap for the focused PD diagnostic

User-run job 30222 (`links_20261115T162649Z_705458`, source `c1a7a2c`)
stopped at `cached_uv`: its expected node-local cached uv executable was
unavailable. It reached no GPU copy test. This does not explain or resolve
the earlier CUDA launch failure in job 30196.

The launcher now selects the account's uv from `T_PD_UV`, PATH or
`~/.local/bin/uv` before requesting GPUs. It snapshots the executable on
shared home, records its hash and version, copies it into the private node
runtime and verifies that copy before execution. The allocated job needs no
old cached executable or original source path. The temporary home snapshot
is removed on launcher exit and is outside the report bundle. Missing own
package and managed-Python cache directories are created, with ownership
checks. Python and GPU package pins and diagnostic allocation resources
remain unchanged. A missing account tool stops before allocation.

Seven CPU orchestration tests passed, including fresh-node bootstrap,
explicit/PATH/home tool selection, disappearance of a symlink target,
snapshot corruption, missing/nonexecutable tools, setup failures, snapshot
cleanup and report contents. Fixtures use fake Slurm and GPU commands.
Bash syntax and Python compilation passed. A separate real-uv CLI check
verified snapshot execution, a copied executable after snapshot removal,
and offline creation of an environment using an explicitly supplied local
Python. It did not install the managed GPU environment or Torch packages.

No GPU retest or serving/performance result is claimed. Published old
campaign evidence and earlier software-validation records are unchanged.

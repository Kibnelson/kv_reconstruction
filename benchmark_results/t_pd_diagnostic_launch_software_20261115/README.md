# Diagnostic launcher failure reporting (software checks)

The user-run diagnostic `links_20261115T155840Z_694088` at `b2e2886`
returned Slurm exit 1 before the old launcher printed its job marker. The
observed output does not identify the failed setup command. No GPU-copy
result was produced by that attempt.

The launcher now prints allocation identity before setup, reports each stage,
retains node job/status records in the home output directory, names rejected
private-directory/cache paths, and exposes environment/package installer
output. It preserves the failing shell exit status. These changes improve
failure reporting; the original remote setup and CUDA issues still require
actual node evidence.

Seven CPU orchestration tests passed. Their subprocess fixtures exercise a
successful launch, silent environment-creation and package-install failures,
an unavailable cached uv executable, and an unsafe root directory. Failure
cases cannot reach the copy worker. The tests use fake Slurm/GPU commands;
they are not GPU measurements. Bash syntax and Python compilation also passed.

No GPU algorithms, package pins, allocation resources, serving matrix or
compression settings were changed. The original 39-test software record and
published job 30196 evidence remain separate and unchanged.

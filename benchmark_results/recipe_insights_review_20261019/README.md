# Recipe diagnostic reports

These files preserve a completed recipe diagnostic. The complete `results.json`,
including every method and per-record trace, is stored losslessly in
`results.json.gz`; no result filtering or tuning is performed by this utility.

Read `summary.md` and `recipe_analysis.json` for findings. Consult the reports'
environment, configuration and scope fields to distinguish CPU functional
measurements from GPU measurements. These diagnostics alone do not establish
long-context, concurrent-serving performance or compact-resident attention.

Restore the original JSON bytes with:

```bash
gzip -dc results.json.gz > results.json
```

`artifact_manifest.json` records SHA256 and byte counts for every other delivered
file, plus the uncompressed results SHA256 and byte count. Gzip decompression was
verified against the complete original JSON before this folder was created.

Raw KV arrays and calibration model binaries are not included. Full calibration
provenance, artifact identities and accounting remain in the original reports;
the source and frozen manifests are copied unchanged.

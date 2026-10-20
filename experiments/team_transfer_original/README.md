# Original team transfer and pool experiment

The six files listed in `source_manifest.json` are byte-for-byte copies from
`kv_combined_test.zip`, supplied on 20 October 2026. The manifest records the
archive SHA256 and the hash of each extracted file.

`TEAM_NOTE.md` is the separately supplied proposal, also copied unchanged. It
records the team's claims and historical measurements; the source audit and
new validation report distinguish those claims from checks executed here.

`combined.py` contains the end-to-end transfer and pool reference;
`combined_results.txt` is its historical output. `exact_ideas.py`,
`deploy_checks.py` and `pool_variants.py` contain the component studies.
`smol_dump_act.py` produces activation captures from a supplied GGUF model.

The archive does not include the original GGUF weights, activation captures or
fitted state. The historical rates are therefore recorded evidence, rather
than a newly reproduced measurement on those exact inputs.

For the independently tested, packet-only decoder, paid prefill-metadata and
coset comparisons, see [the validation guide](../../docs/transfer_metadata_coset_validation.md).
The original scripts are retained unchanged; fixes and experiment extensions
live in `qwen_kv_bench/` and `scripts/`.

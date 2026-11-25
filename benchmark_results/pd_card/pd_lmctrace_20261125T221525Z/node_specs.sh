set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/f159ef6cab8469acf4e771bc16f305a724bdd91a'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261125T221525Z/repo/benchmark_results/pd_card/pd_lmctrace_20261125T221525Z/node_specs' --kv-home "$KV_HOME"

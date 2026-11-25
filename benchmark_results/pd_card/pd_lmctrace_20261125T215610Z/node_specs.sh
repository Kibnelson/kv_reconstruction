set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/f0dac4a8ccfab35de20dacf1e2ba5e1f75db65de'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261125T215610Z/repo/benchmark_results/pd_card/pd_lmctrace_20261125T215610Z/node_specs' --kv-home "$KV_HOME"

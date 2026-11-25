set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/29a277bf1ed9698132a057e66b78a478ec0b429e'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261125T203203Z/repo/benchmark_results/pd_card/pd_lmctrace_20261125T203203Z/node_specs' --kv-home "$KV_HOME"

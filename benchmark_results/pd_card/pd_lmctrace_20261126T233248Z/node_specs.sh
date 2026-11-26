set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/99536dce0cd93dacfc3875e7a7ea08195ea1692c'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261126T233248Z/repo/benchmark_results/pd_card/pd_lmctrace_20261126T233248Z/node_specs' --kv-home "$KV_HOME"

set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/bbc499a6626004f1bd5aeaa023a266cfd0084c6b'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261126T201956Z/repo/benchmark_results/pd_card/pd_lmctrace_20261126T201956Z/node_specs' --kv-home "$KV_HOME"

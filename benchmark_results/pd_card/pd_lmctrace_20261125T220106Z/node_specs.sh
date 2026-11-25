set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/4f9fc89248f917655fd5270fb97ce562f2005917'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261125T220106Z/repo/benchmark_results/pd_card/pd_lmctrace_20261125T220106Z/node_specs' --kv-home "$KV_HOME"

set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/277bea7047fee1e574492b867216e6b3c726d9b2'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_capture_20261125T141553Z/repo/benchmark_results/pd_card/pd_capture_20261125T141553Z/node_specs' --kv-home "$KV_HOME"

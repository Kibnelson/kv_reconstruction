set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/badb38842369fb8445d4c86500212169bbee0ed0'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_smoke_20261115T005605Z/repo/benchmark_results/pd_card/pd_smoke_20261115T005605Z/node_specs' --kv-home "$KV_HOME"

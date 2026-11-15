set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-t-matched-nbore/kvpd-tools/90112640bc59bac6ea1a82f79b8c53f2b22aa996'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-t-matched-nbore/kvpd-evidence/pd_smoke_tmatched_20261115T112147Z_594875/repo/benchmark_results/pd_card/pd_smoke_tmatched_20261115T112147Z_594875/node_specs' --kv-home "$KV_HOME"

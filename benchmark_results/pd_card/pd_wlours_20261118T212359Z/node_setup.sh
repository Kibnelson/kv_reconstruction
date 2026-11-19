set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/9595205729f52e5f430f6c32488b9935e949b543'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

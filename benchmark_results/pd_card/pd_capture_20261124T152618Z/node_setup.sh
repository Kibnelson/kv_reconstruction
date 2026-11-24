set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/c56fa39ebaeaffdc8b1e8b52cf0c98ae890be62a'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

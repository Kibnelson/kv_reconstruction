set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/41b13a14d2d158655bddffd00b5f2fc3c42b4843'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

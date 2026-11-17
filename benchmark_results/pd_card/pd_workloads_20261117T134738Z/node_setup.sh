set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/f7d06243f43578c514419d0e8c87421d2a029ba0'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

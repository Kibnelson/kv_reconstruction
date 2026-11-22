set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/d4987666a39e915bb42160d8f565fd9f2558a944'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

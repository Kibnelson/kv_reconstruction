set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/313bc6de96c750a4a297a1a2e7610b9ac464cd90'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

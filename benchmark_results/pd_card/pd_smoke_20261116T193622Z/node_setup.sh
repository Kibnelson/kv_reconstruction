set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/ac08292270523ce01c864149bbf4beab772d229d'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

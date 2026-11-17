set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/2dc34476b0093e650b1bdad5b01e1363ae9b9555'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

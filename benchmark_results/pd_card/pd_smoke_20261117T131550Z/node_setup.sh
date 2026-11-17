set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/91b86f2f175f5762448416db771de797bdc52cdf'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

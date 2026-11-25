set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/277bea7047fee1e574492b867216e6b3c726d9b2'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh

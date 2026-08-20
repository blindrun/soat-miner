#!/usr/bin/env bash
# Ergo -> WoolyPooly. Source: https://woolypooly.com/en/coin/erg
# Edit WALLET, then run. This is plain Stratum V1/TCP (no TLS).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=9YOUR_ERGO_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --pool pool.woolypooly.com:3100 --wallet "${WALLET}" \
     --worker "$WORKER" --pass x "$@"

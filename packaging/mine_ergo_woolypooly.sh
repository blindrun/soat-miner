#!/usr/bin/env bash
# Ergo -> WoolyPooly. Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=9YOUR_ERGO_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --pool pool.woolypooly.com:3100 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

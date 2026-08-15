#!/usr/bin/env bash
# Ergo -> WoolyPooly. Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=9ea2QrXbTTmEhRA92qcnDVD98aeENUc4oJdDGxF7GKMBZ47wLTR
WORKER=$(hostname -s)
exec ./soat-miner.sh --pool pool.woolypooly.com:3100 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

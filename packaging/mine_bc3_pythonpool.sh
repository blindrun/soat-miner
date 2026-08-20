#!/usr/bin/env bash
# Bitcoin III -> PythonPool. Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool stratum.pythonpool.dev:3357 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

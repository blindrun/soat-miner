#!/usr/bin/env bash
# Bitcoin III -> Hashbay PROP. 0.5% fee, proportional.
# Their solo pool is port 3345, same host and same fee.
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool stratum.hashbay.io:3344 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

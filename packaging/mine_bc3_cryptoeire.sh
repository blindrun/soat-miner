#!/usr/bin/env bash
# Bitcoin III -> Crypto-Eire. 1% fee, solo.
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool stratum.crypto-eire.com:3362 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

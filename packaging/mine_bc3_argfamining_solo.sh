#!/usr/bin/env bash
# Bitcoin III -> ArgfaMining SOLO. 1% fee, solo: you find the block, you keep it.
# 24152 is the GPU port. 24052 is their CPU port - same pool, do not use it.
# Other regions, same port: stratum.argfamining.com (Venezuela),
# stratum-eu.argfamining.com (Frankfurt), stratum-in.argfamining.com (Mumbai).
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool stratum-us.argfamining.com:24152 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

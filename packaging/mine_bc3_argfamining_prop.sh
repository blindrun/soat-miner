#!/usr/bin/env bash
# Bitcoin III -> ArgfaMining PROP. 1% fee, proportional: every block is split by shares in the round.
# 24153 is the GPU port. 24053 is their CPU port - same pool, do not use it.
# Other regions, same port: stratum.argfamining.com (Venezuela),
# stratum-eu.argfamining.com (Frankfurt), stratum-in.argfamining.com (Mumbai).
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool stratum-us.argfamining.com:24153 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

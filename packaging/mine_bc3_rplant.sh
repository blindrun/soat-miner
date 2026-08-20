#!/usr/bin/env bash
# Bitcoin III -> rplant.xyz. 1% fee, proportional (2% if you use their solo prefix).
# Other regions, same port: stratum-eu, stratum-ru, stratum-asia.rplant.xyz.
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool stratum-na.rplant.xyz:7157 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

#!/usr/bin/env bash
# Bitcoin III -> AxeHub SOLO. 0% fee, solo - 100% of the block goes to the finder.
# Their PPLNS pool is port 4338 and charges 1%.
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool pool.axehub.app:3338 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

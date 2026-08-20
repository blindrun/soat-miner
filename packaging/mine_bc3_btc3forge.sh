#!/usr/bin/env bash
# Bitcoin III -> BTC3 Forge. 0.5% fee, proportional.
# Their solo endpoint solo.btc3forge.com:3333 is advertised but refuses
# connections, so there is no launcher for it.
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool btc3forge.com:3337 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

#!/usr/bin/env bash
# Bitcoin III -> Vexta PPLNS. 0.5% fee, PPLNS.
# Same host also runs solo on 7334 and proportional on 7335.
# Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=YOUR_BC3_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo sha3-256t --pool vexta-pool.co.uk:7333 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

#!/usr/bin/env bash
# Pearl -> HeroMiners. Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool pearl.herominers.com:1200 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

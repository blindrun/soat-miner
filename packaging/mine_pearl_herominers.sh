#!/usr/bin/env bash
# Pearl -> HeroMiners. Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool pearl.herominers.com:1200 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

#!/usr/bin/env bash
# Pearl -> LuckyPool. 1% fee, proportional.
# Three difficulty tiers on the same host, all vardiff:
#   3360 starts at 2M   3361 starts at 4M   3362 starts at 8M
# Observed 888888 on connect, so the pool moves it below the tier floor
# for a new worker.
# Other regions, same ports: pearl-eu1, pearl-eu2, pearl-pl, pearl-tr,
# pearl-ru, pearl-us-west, pearl-us-central, pearl-us-ord, pearl-br,
# pearl-ca1, pearl-ca2, pearl-sg1, pearl-sg2, pearl-id, pearl-hk,
# pearl-in, pearl-jp, pearl-au .luckypool.io
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool pearl-us-east.luckypool.io:3360 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

#!/usr/bin/env bash
# Pearl -> mkpool SOLO. 2% fee. You find the block, you keep it.
# Handed out a fixed difficulty of 2097184, essentially HeroMiners'.
# 3411 is the working port. 3413 is advertised on the same host but
# closes the connection without answering, so there is no launcher.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool pearl.mkpool.com:3411 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

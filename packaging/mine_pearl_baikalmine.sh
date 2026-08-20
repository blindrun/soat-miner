#!/usr/bin/env bash
# Pearl -> BaikalMine. 0.5% fee, PPLNS - the cheapest verified pool
# that is not free.
# Handed out a fixed difficulty of 262144, 8x easier than HeroMiners.
# Other region, same port: pearl-ru2.baikalmine.com (Moscow).
# The pool also publishes pearl.baikalmine.com:2010, but that host did
# not answer a TCP connect from here; pearl-eu did, so use pearl-eu.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool pearl-eu.baikalmine.com:2010 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

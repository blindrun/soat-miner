#!/usr/bin/env bash
# Pearl -> K1Pool SOLO. 1% fee. You find the block, you keep it.
# Handed out a fixed difficulty of 1310720.
# Other regions, same port: eu.pearlsolo.k1pool.com, and
# ru.pearlsolo.k1pool.org (note .org, not .com).
# Their PPLNS pool is mine_pearl_k1pool.sh and charges nothing.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool us.pearlsolo.k1pool.com:3362 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

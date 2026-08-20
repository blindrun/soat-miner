#!/usr/bin/env bash
# Pearl -> K1Pool PPLNS. 0% fee.
# Handed out a fixed difficulty of 1966080.
# Other regions, same port: eu.pearl.k1pool.com, cn.pearl.k1pool.com
# Their solo pool is a separate host and port - see
# mine_pearl_k1pool_solo.sh.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool us.pearl.k1pool.com:3360 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

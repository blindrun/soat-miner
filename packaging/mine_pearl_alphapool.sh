#!/usr/bin/env bash
# Pearl -> AlphaPool. 0% fee, PPLNS.
# Handed out 50000 on connect - 42x easier than HeroMiners' fixed
# 2097152, and the easiest bound of any pool with real hashrate.
# USE PORT 5571. It is their plain-stratum port. 5566 (their shim) and
# 5573 (solo/lottery) both open with a pearl.challenge proof-of-work
# handshake this miner does not implement, and never authorize.
# Other regions, same port 5571: us1, eu1, ru1, sg1 .alphapool.tech
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool us2.alphapool.tech:5571 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

#!/usr/bin/env bash
# Pearl -> RabbitMiner. 1% fee, proportional.
# Vardiff; handed out 232827 on connect, the lowest of the big pools
# apart from AlphaPool and Suprnova.
# Other region, same port: fi.rabbitminer.cc (Finland).
# Their web builder offers a static difficulty through a d=VALUE in the
# password field. This miner always sends pass "x" on the Pearl path,
# so that knob is not reachable from here yet.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool nl.rabbitminer.cc:1902 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

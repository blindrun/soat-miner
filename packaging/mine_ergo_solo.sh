#!/usr/bin/env bash
# Ergo solo, against your own node.
#
# The payout address is configured on the NODE, not here - the node hands out
# work and decides who gets paid. In ergo.conf:
#   ergo { node { mining = true, useExternalMiner = true } }
cd "$(dirname "${BASH_SOURCE[0]}")"
NODE=127.0.0.1
exec ./soat-miner.sh --node "$NODE" --port 9053 "$@"

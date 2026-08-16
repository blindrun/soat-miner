#!/usr/bin/env bash
# Ergo -> Lithos, the decentralised pool protocol.
#
# Lithos is not a coin and not an algorithm: it is a pool protocol whose
# reference client runs a stratum server on your own machine. You mine ordinary
# Autolykos v2 into it, and it settles rewards on-chain from Non-Interactive
# Share Proofs.
#
# Before this will work you need, on this machine or a reachable one:
#   1. a fully synced Ergo node
#   2. Java 11
#   3. the Lithos client (github.com/Lithos-Protocol/Lithos-Client), configured
#      in conf/application.conf to point at your node, and running
#
# The payout identity comes from the NODE the Lithos client is attached to, not
# from an address given here - which is why no --wallet is needed. --worker is
# just a label in the client's log.
cd "$(dirname "${BASH_SOURCE[0]}")"
LITHOS_HOST=127.0.0.1
LITHOS_PORT=4444
WORKER=$(hostname -s)
exec ./soat-miner.sh --lithos --pool "$LITHOS_HOST:$LITHOS_PORT" \
     --worker "$WORKER" "$@"

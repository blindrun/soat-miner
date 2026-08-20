#!/usr/bin/env bash
# Ravencoin -> HeroMiners. Edit WALLET, then run.
#
# Port 1140 is HeroMiners' plain stratum port. They also run 1150, which speaks
# the EthereumStratum/1.0.0 flavour - the miner detects which from the
# subscribe reply, so either works, but 1140 is what their docs point at.
#
# NOT YET PROVEN END TO END: this endpoint has been verified to answer and to
# speak a protocol this miner implements, but no share from this miner has been
# accepted by any KawPow pool yet. Treat the first run as a test.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=RYOUR_RAVENCOIN_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo kawpow --pool ravencoin.herominers.com:1140 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

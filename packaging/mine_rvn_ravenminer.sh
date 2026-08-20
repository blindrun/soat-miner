#!/usr/bin/env bash
# Ravencoin -> RavenMiner. Edit WALLET, then run.
#
# NOT YET PROVEN END TO END: this endpoint has been verified to answer and to
# speak a protocol this miner implements, but no share from this miner has been
# accepted by any KawPow pool yet. Treat the first run as a test.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=RYOUR_RAVENCOIN_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo kawpow --pool stratum.ravenminer.com:3838 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

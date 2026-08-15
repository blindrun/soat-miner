#!/usr/bin/env bash
# Ergo -> HeroMiners. Edit WALLET, then run.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=9ea2QrXbTTmEhRA92qcnDVD98aeENUc4oJdDGxF7GKMBZ47wLTR
WORKER=$(hostname -s)
exec ./soat-miner.sh --pool ergo.herominers.com:1180 \
     --wallet "$WALLET" --worker "$WORKER" --pass x "$@"

#!/usr/bin/env bash
# Pearl -> JetSkiPool (PearlSki). 1% pool fee, proportional.
# They also charge a 2% miner fee on their own miner; this one is ours,
# so only the pool fee applies.
# Handed out a fixed difficulty of 2000000 on connect.
# France only - the pool advertises no other region.
# pearl.jetskipool.ai redirects to pearlski.jetskipool.ai for the site;
# the stratum host is the pearlski one.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool pearlski.jetskipool.ai:6970 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

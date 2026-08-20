#!/usr/bin/env bash
# Pearl -> Suprnova. PPLNS, currently 0% on a launch promotion.
# 3373 is the vardiff port and the one to use. It opened at difficulty
# 244 - four orders of magnitude below HeroMiners' fixed 2097152, which
# is what a slow card needs to produce a testable share in minutes
# rather than hours.
# The fixed ports are lower still on connect. Measured, not advertised:
#   3370 opened at 1.22    3371 opened at 488    3372 opened at 1953
# 3374 is the same as 3373 over SSL/TLS, which this miner does not speak.
# Suprnova labels difficulty in units 16384x ours, so its own job id
# suffix reads 4000000 where the target decodes to 244. The target on
# the wire is what the miner mines against, so ignore the label.
# Other regions, same ports: stratum-eu2, stratum-us, stratum-apac
# .suprnova.cc
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool prl.suprnova.cc:3373 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

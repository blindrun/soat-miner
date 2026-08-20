#!/usr/bin/env bash
# Pearl -> Kryptex. 1% fee, PROP or SOLO (their PPS+ tier is 2%).
# Handed out difficulty 2097120 on connect, the same fixed bound
# HeroMiners uses.
# 8048 is the same pool over SSL/TLS, which this miner does not speak.
# Kryptex publishes eight regional servers; prl.kryptex.network routes
# you to the nearest one, so there is no per-region launcher.
# Edit WALLET, then run.
#
# WALLET here is a PEARL address, not the Ergo one in config.txt. Pearl is
# a different chain: an Ergo address cannot be paid by a Pearl pool, and
# the pool answers one with "Invalid Pearl address". A Pearl address
# starts with prl1 and is about 63 characters.
cd "$(dirname "${BASH_SOURCE[0]}")"
WALLET=prl1YOUR_PEARL_ADDRESS_HERE
WORKER=$(hostname -s)
exec ./soat-miner.sh --algo pearl-pow --pool prl.kryptex.network:7048 \
     --wallet "$WALLET" --worker "$WORKER" "$@"

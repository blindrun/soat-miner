#!/usr/bin/env bash
# BENCHMARK ONLY - measures hashrate. It does NOT mine: no pool, no wallet,
# no shares, no payouts. That is why no address is needed here.
# To actually mine, edit WALLET in mine_ergo_herominers.sh and run that.
#
# Hashrate depends heavily on the dataset size, which grows with chain height.
# Pass a height to compare against older figures:
#   ./benchmark.sh --bench-height 900000    # 2022-era dataset (~2.9 GB)
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./soat-miner.sh --bench "$@"

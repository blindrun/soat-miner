#!/usr/bin/env bash
# Benchmark without a pool or node.
#
# Hashrate depends heavily on the dataset size, which grows with chain height.
# Pass a height to compare against older figures:
#   ./benchmark.sh --bench-height 900000    # 2022-era dataset (~2.9 GB)
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./soat-miner.sh --bench "$@"

#!/usr/bin/env bash
# Reproduce the CMake/CTest form of the offline BC3 protocol gate.
#
# This script has no network, wallet, pool, or GPU-runtime action.  CMake must
# still find an installed CUDA compiler because the project has a CUDA target;
# the built and executed targets are only host-side test binaries.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cmake_bin=${CMAKE_BIN:-cmake}
ctest_bin=${CTEST_BIN:-ctest}
build_dir=${BC3_CMAKE_BUILD_DIR:-"$repo_root/build/bc3-ctest"}

if ! command -v "$cmake_bin" >/dev/null 2>&1; then
    echo "error: CMake is required but '$cmake_bin' is not on PATH." >&2
    echo "hint: use a CMake-equipped build host; do not install packages from this script." >&2
    exit 127
fi
if ! command -v "$ctest_bin" >/dev/null 2>&1; then
    echo "error: CTest is required but '$ctest_bin' is not on PATH." >&2
    echo "hint: set CTEST_BIN to the CTest paired with CMAKE_BIN." >&2
    exit 127
fi

cmake_args=(
    -S "$repo_root"
    -B "$build_dir"
    -DBUILD_TESTING=ON
    -DCMAKE_BUILD_TYPE=Release
)
if [[ -n ${BC3_CMAKE_GENERATOR:-} ]]; then
    cmake_args+=(-G "$BC3_CMAKE_GENERATOR")
fi
if [[ -n ${BC3_CMAKE_ARCHITECTURES:-} ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${BC3_CMAKE_ARCHITECTURES}")
fi

echo "+ $cmake_bin ${cmake_args[*]}"
"$cmake_bin" "${cmake_args[@]}"
echo "+ $cmake_bin --build $build_dir --target test-btc-stratum test-bc3-destination --config Release"
"$cmake_bin" --build "$build_dir" --target test-btc-stratum test-bc3-destination --config Release
echo "+ $ctest_bin --test-dir $build_dir --output-on-failure -R ^(btc-stratum-offline|bc3-destination-schema)$ -C Release"
"$ctest_bin" --test-dir "$build_dir" --output-on-failure \
    -R '^(btc-stratum-offline|bc3-destination-schema)$' -C Release

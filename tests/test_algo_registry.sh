#!/usr/bin/env bash
# Do the four lists that describe an algorithm still agree?
#
# Adding an algorithm means editing four separate places, and nothing checked
# that you did all four:
#
#   src/core/registry.cu        which algorithms the CUDA binary has
#   src/core/vk_registry.cpp    which the Vulkan binary has
#   packaging/soat-miner.sh     its coin name, address placeholder, scripts
#   packaging/soat-miner.sh     whether it is CUDA-only
#
# Each is an explicit table on purpose - the registries document why, and it is
# a good reason - but "explicit" and "unchecked" are different things. Forget
# the Vulkan one and AMD users are told the algorithm does not exist. Forget the
# coin metadata and the launcher falls through to `*)` and tells a Ravencoin
# miner to set their ERGO address, which is a wrong message and not an error.
#
# The binaries are the authority here: whatever `--list-algos` reports is what
# the miner really has, and everything else is checked against that.
#
# No GPU: --list-algos exits before any device is opened.
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_BIN="${1:-$HERE/soat-miner}"
VK_BIN="${2:-$HERE/soat-miner-vk}"
fail=0

check() {
    if [ "$1" = "0" ]; then
        printf '  ok    %s\n' "$2"
    else
        printf '  FAIL  %s\n' "$2"
        [ -n "${3:-}" ] && printf '        %s\n' "$3"
        fail=$((fail + 1))
    fi
}

[ -x "$CUDA_BIN" ] || { echo "  SKIP  $CUDA_BIN not built"; exit 0; }
[ -x "$VK_BIN" ]   || { echo "  SKIP  $VK_BIN not built"; exit 0; }

cuda_algos="$("$CUDA_BIN" --list-algos 2>/dev/null | sort)"
vk_algos="$("$VK_BIN" --list-algos 2>/dev/null | sort)"

[ -n "$cuda_algos" ] || { check 1 "the CUDA binary lists its algorithms"; exit 1; }

# Anything the Vulkan build has must exist in the CUDA build too. The reverse is
# allowed - that is what CUDA-only means - but a Vulkan-only algorithm would be
# a registry that drifted rather than a decision anyone made.
extra_vk="$(comm -13 <(echo "$cuda_algos") <(echo "$vk_algos"))"
check "$([ -z "$extra_vk" ] && echo 0 || echo 1)" \
      "every Vulkan algorithm also exists in the CUDA build" "$extra_vk"

# An algorithm the CUDA build has and the Vulkan build does not must be
# DECLARED CUDA-only in the launcher, not merely missing.
for a in $(comm -23 <(echo "$cuda_algos") <(echo "$vk_algos")); do
    if grep -qE "^[[:space:]]*$a\)[[:space:]]*CUDA_ONLY_ALGO=" "$HERE/packaging/soat-miner.sh"; then
        check 0 "$a is missing from Vulkan and the launcher says so"
    else
        check 1 "$a is missing from the Vulkan build and nothing declares it CUDA-only" \
              "an AMD user would be told the algorithm does not exist"
    fi
done

# Every algorithm must have real coin metadata rather than falling through to
# the Ergo default. Tested through the launcher's own output, because that is
# what a user actually reads.
for a in $cuda_algos; do
    dir="$(mktemp -d)"
    cp "$HERE/packaging/soat-miner.sh" "$dir/"
    cp "$HERE/packaging/config.txt" "$dir/"
    # POOL must come from config.txt, NOT from --pool. Passing it on the
    # command line puts the launcher into "using command-line settings" mode,
    # which skips the config wallet check entirely - so the first version of
    # this test saw no message and reported three false failures.
    sed -i "s/^ALGO=.*/ALGO=$a/; s/^WALLET=.*/WALLET=/; s/^POOL=.*/POOL=example:1234/" \
        "$dir/config.txt"
    printf '#!/bin/sh\necho ran\n' > "$dir/soat-miner"
    cp "$dir/soat-miner" "$dir/soat-miner-vk"
    chmod +x "$dir/soat-miner" "$dir/soat-miner-vk" "$dir/soat-miner.sh"
    msg="$(cd "$dir" && ./soat-miner.sh 2>&1 | grep -m1 'Set WALLET')"
    rm -rf "$dir"

    if [ "$a" = "autolykos2" ]; then
        continue   # Ergo IS the default, so it cannot be distinguished here
    fi
    case "$msg" in
        *Ergo*) check 1 "$a has its own coin metadata in the launcher" \
                      "fell through to the Ergo default: $msg" ;;
        "")     check 1 "$a reaches the wallet check at all" "no message" ;;
        *)      check 0 "$a has its own coin metadata in the launcher" ;;
    esac
done

# --- every algorithm names its own launcher scripts, and they exist -------
# Benchmark mode tells the user which scripts mine what they just benchmarked.
# That used to be an if-chain over algorithm names in run.cpp: it told BC3
# users to go and mine Ergo, was fixed by adding a branch, and then told
# KawPow users to go and mine Ergo. It is now Algorithm::launcherGlob().
#
# Checked at source level rather than by running --bench, because benchmarking
# KawPow allocates a 6.11 GB DAG before it prints anything useful.
for f in $(grep -rl 'const char \*name() const override' "$HERE/src/algos/"); do
    if ! grep -q 'launcherGlob() const override' "$f"; then
        check 1 "$(basename "$(dirname "$f")") names its launcher scripts" \
              "no launcherGlob() in $f - benchmarking it would name no script"
        continue
    fi
    glob="$(sed -n 's/.*launcherGlob() const override { return "\([^"]*\)".*/\1/p' "$f")"
    # The glob has to match real files, or it points users at nothing.
    n=$(ls "$HERE"/packaging/$glob 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        check 1 "$(basename "$(dirname "$f")") launcher glob '$glob' matches real scripts" \
              "matched no files in packaging/"
    else
        check 0 "$(basename "$(dirname "$f")")/$(basename "$f") points at $glob ($n file(s))"
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "algorithm registries and launcher metadata agree ($(echo "$cuda_algos" | wc -l) algorithms)"
fi
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)

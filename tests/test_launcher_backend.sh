#!/usr/bin/env bash
# Which backend does the launcher actually pick?
#
# The backend choice is per algorithm and the rules disagree on purpose:
# Autolykos prefers Vulkan on Blackwell, BC3 and KawPow prefer CUDA on NVIDIA,
# Pearl is CUDA only. That is four rules in one script and nothing checked any
# of them, which is how BC3 quietly picked the slower backend for a while after
# it stopped being CUDA-only.
#
# It runs the real launcher against stub binaries in a temp directory, so it
# tests behaviour rather than the shape of the script. No GPU needed.
#
#   tests/test_launcher_backend.sh
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="$HERE/packaging/soat-miner.sh"
fail=0

# Runs the launcher in a sandbox and echoes which stub it invoked.
#
# Settings go into config.txt, NOT the environment. The launcher does
# `set -a; source ./config.txt`, so config.txt WINS over any exported variable
# - which means an env-based version of this test silently exercises the
# defaults instead of what it asked for. The first draft of this file did
# exactly that and two checks passed for the wrong reason.
#
# nvidia-smi is STUBBED, reporting whichever compute capability the case wants.
# Without that this whole file is vacuous on an Ada card: the general auto rule
# already sends Ada to CUDA, so "kawpow picks CUDA" passes with the KawPow rule
# deleted. Verified - removing the rule did not fail the suite until the stub
# existed. It also means the suite runs on a machine with no NVIDIA GPU at all.
#
# $1 = which stubs to place: "both" | "vk-only"
# $2 = ALGO
# $3 = BACKEND
# $4 = compute capability nvidia-smi should report, or "none" for no GPU
# rest = extra command-line arguments
# Builds a fake mining rig in a temp dir: a stubbed nvidia-smi, both miner
# binaries, and a config.txt. Defined ONCE and used by both pick() and
# pick_output(); it started as two copies that had to agree and immediately
# did not - the multi-GPU stub was added to one of them and the warning cases
# silently kept testing a single-GPU machine.
#
# $1 = dir, $2 = stubs ("both"|"vk-only"), $3 = ALGO, $4 = BACKEND
# $5 = rig: "none" for no NVIDIA GPU, "8.9" for one card, or a
#      comma-separated list of memMiB:cap pairs for several.
build_rig() {
    local dir="$1" stubs="$2" algo="$3" backend="$4" cap="$5"
    mkdir -p "$dir/bin"
    if [ "$cap" = "none" ]; then
        printf '#!/bin/sh\nexit 1\n' > "$dir/bin/nvidia-smi"
    else
        local spec="$cap"
        case "$spec" in
            *:*) : ;;
            *)   spec="16376:$cap" ;;
        esac
        cat > "$dir/bin/nvidia-smi" <<STUB
#!/bin/sh
# The two-field query must be matched BEFORE the one-field query, or the
# caps-only call returns "mem, cap" and every capability parses as VRAM.
spec="$spec"
emit() {
  printf '%s\n' "\$spec" | tr ',' '\n' | while IFS=: read -r mem c; do
    case "\$1" in
      pair) echo "\$mem, \$c" ;;
      cap)  echo "\$c" ;;
      list) echo "GPU: STUB (UUID: GPU-x)" ;;
    esac
  done
}
case "\$*" in
  *-L*)                       emit list ;;
  *memory.total,compute_cap*) emit pair ;;
  *compute_cap*)              emit cap ;;
  *)                          echo "STUB" ;;
esac
STUB
    fi
    chmod +x "$dir/bin/nvidia-smi"
    cp "$LAUNCHER" "$dir/soat-miner.sh"
    cp "$HERE/packaging/config.txt" "$dir/"
    sed -i "s/^ALGO=.*/ALGO=$algo/; s/^BACKEND=.*/BACKEND=$backend/" "$dir/config.txt"
    printf '#!/bin/sh\necho CUDA\n' > "$dir/soat-miner"
    printf '#!/bin/sh\necho VULKAN\n' > "$dir/soat-miner-vk"
    chmod +x "$dir/soat-miner" "$dir/soat-miner-vk" "$dir/soat-miner.sh"
    [ "$stubs" = "vk-only" ] && rm -f "$dir/soat-miner"
    return 0
}

pick() {
    local stubs="$1" algo="$2" backend="$3" cap="$4"; shift 4
    local dir
    dir="$(mktemp -d)"
    build_rig "$dir" "$stubs" "$algo" "$backend" "$cap"
    ( cd "$dir" && PATH="$dir/bin:$PATH" ./soat-miner.sh --bench "$@" 2>&1 ) \
        | grep -Eo '^(CUDA|VULKAN)$' | head -1
    rm -rf "$dir"
}

# Guard against the whole suite passing vacuously: if the sandbox cannot even
# run the launcher, every check would compare empty strings.
if [ -z "$(pick both autolykos2 cuda 8.9)" ]; then
    echo "  FAIL  the sandbox does not run the launcher at all; every check "
    echo "        below would be meaningless"
    exit 1
fi

pick_output() {
    local stubs="$1" algo="$2" backend="$3" cap="$4"; shift 4
    local dir
    dir="$(mktemp -d)"
    build_rig "$dir" "$stubs" "$algo" "$backend" "$cap"
    ( cd "$dir" && PATH="$dir/bin:$PATH" ./soat-miner.sh --bench "$@" 2>&1 )
    rm -rf "$dir"
}

check() {
    local what="$1" want="$2" got="$3"
    if [ "$want" = "$got" ]; then
        printf '  ok    %s\n' "$what"
    else
        printf '  FAIL  %s\n        want %s, got %s\n' "$what" "$want" "${got:-<nothing>}"
        fail=$((fail + 1))
    fi
}

# The case that matters. On BLACKWELL the general rule prefers Vulkan, because
# Autolykos is ~22% faster there. KawPow must override that: measured 4.29 MH/s
# on Vulkan against 7.38 on CUDA on a 5080.
check "kawpow on Blackwell picks CUDA, against the general rule" CUDA \
      "$(pick both kawpow auto 12.0)"
check "autolykos2 on Blackwell still picks Vulkan" VULKAN \
      "$(pick both autolykos2 auto 12.0)"
check "sha3-256t on Blackwell picks CUDA" CUDA \
      "$(pick both sha3-256t auto 12.0)"
check "kawpow on Ada picks CUDA" CUDA "$(pick both kawpow auto 8.9)"

# An explicit backend is the user asserting they know. Neither launcher had a
# test for this and the batch one got it wrong on the first attempt.
check "an explicit BACKEND=vulkan is not overridden" VULKAN \
      "$(pick both kawpow vulkan 12.0)"
check "an explicit BACKEND=cuda is not overridden" CUDA \
      "$(pick both autolykos2 cuda 12.0)"

# "kawpow" appears in real pool hostnames. Matching it as a substring of the
# command line would force CUDA for an Autolykos user mining at one of them.
check "a pool host containing 'kawpow' does not force CUDA for autolykos2" \
      VULKAN \
      "$(pick both autolykos2 auto 12.0 --pool kawpow.unmineable.com:3333)"

# On a machine with no NVIDIA GPU the NVIDIA rules must not fire at all.
check "kawpow on a non-NVIDIA machine picks Vulkan" VULKAN \
      "$(pick both kawpow auto none)"

# What must NOT happen is KawPow silently running on the slow backend because
# the CUDA binary is missing. The two launchers answer differently and both are
# acceptable: the shell one refuses with a message naming the binary, the batch
# one falls back and warns loudly that this is the slow path, not a free one.
# The contract under test is only "does not quietly run Vulkan".
warned="$(pick_output vk-only kawpow auto 12.0 | grep -ci 'slow path')"
check "kawpow warns rather than silently running Vulkan with no CUDA binary" \
      1 "$warned"

# --- mixed rigs ---------------------------------------------------------
# A homogeneous multi-GPU rig is not ambiguous and must behave exactly like a
# single card of that generation - no warning, same backend.
check "two identical Blackwells still pick Vulkan" VULKAN \
      "$(pick both autolykos2 auto 32000:12.0,32000:12.0)"
check "a homogeneous rig does not warn about mixed capabilities" 0 \
      "$(pick_output both autolykos2 auto 32000:12.0,32000:12.0 |
         grep -ci 'different compute')"

# The one that used to be wrong. Largest VRAM is the Ada card, so the launcher
# read 8.9 and chose CUDA - but the CUDA binary mines on device 0, which here
# is the Blackwell. Whichever card ran, one of the two rules was wrong about
# it, and nothing said so.
check "a mixed Blackwell/Ada rig does not silently guess a backend" 1 \
      "$(pick_output both autolykos2 auto 16376:12.0,24564:8.9 |
         grep -ci 'not well defined')"
check "and it names both capabilities so the user can tell which cards" 1 \
      "$(pick_output both autolykos2 auto 16376:12.0,24564:8.9 |
         grep -c '8.9' | head -1)"
# CUDA is the backend that works on every NVIDIA generation present, so an
# ambiguous rig takes it rather than the one that is merely faster on some.
check "an ambiguous rig falls back to the backend that works on all of them" \
      CUDA "$(pick both autolykos2 auto 16376:12.0,24564:8.9)"

if [ "$fail" -eq 0 ]; then
    echo "launcher backend selection: all checks passed"
fi
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)

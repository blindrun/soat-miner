#!/usr/bin/env bash
# SOAT Miner launcher. Reads config.txt, picks a backend, starts mining.
#
# Anything passed on the command line overrides config.txt, so the
# mine_ergo_*.sh wrappers can supply their own --pool/--wallet without having
# to satisfy the config file as well.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[[ -f config.txt ]] || { echo "config.txt not found"; exit 1; }
# shellcheck disable=SC1091
set -a; source ./config.txt; set +a

# Did the caller already say where work comes from (or that it needs none)?
# --bench belongs here too: a benchmark needs no pool, node or wallet, and
# leaving it out made benchmark.sh fail with "set WALLET in config.txt".
EXPLICIT_SOURCE=0
for a in "$@"; do
  case "$a" in
    --pool|--node|--bench|--lithos|--list-devices|--list-algos|--help|-h) EXPLICIT_SOURCE=1 ;;
  esac
done

# Backend choice is per GPU architecture, not per vendor. Measured at 7.27 GB:
#
#   RTX 5080 (Blackwell, cc 12.0)   Vulkan 267.6   CUDA 219.6   -> Vulkan +22%
#   RTX 4090 (Ada, cc 8.9)          Vulkan 162.5   CUDA 217.5   -> CUDA   +34%
#   RX 6700 XT                      Vulkan  82.9   (no CUDA)
#
# and it holds on both operating systems - on Windows the 5080 is Vulkan 259.0
# against a natively compiled sm_120 CUDA 219.6, so this is not a JIT artifact.
# Blackwell therefore wants Vulkan even though it is an NVIDIA card.
BACKEND="${BACKEND:-auto}"

# Pearl is CUDA only. Without this the Blackwell rule below sends it to the
# Vulkan binary, which does not have it, and the run fails as a bad ERGO
# address - which is a baffling thing to be told when you asked for Pearl.
# Non-NVIDIA hits the same path, so this is not only a Blackwell problem.
# Kept as a list rather than a hardcoded test: a hardcoded pearl-pow check is
# what let BC3 inherit this bug when it was added.
# Checks config.txt's ALGO too, not only the command line. Setting an ALGO
# there and running this with no arguments used to fall through to the
# Blackwell rule and pick a binary that could not mine it.
#
# BC3 (sha3-256t) is NOT on this list any more: it has a Vulkan backend and a
# .comp shader of its own, verified against the host reference on an RX 7900 XT
# as well as on NVIDIA. Leaving it here was the thing that made BC3 CUDA-only
# in practice no matter what the binary supported - an AMD user was told the
# algorithm did not exist for them.
CUDA_ONLY_ALGO=""
for a in "$@" "${ALGO:-}"; do
  case "$a" in
    pearl-pow) CUDA_ONLY_ALGO="Pearl" ;;
  esac
done
if [[ -n "$CUDA_ONLY_ALGO" ]]; then
  if [[ ! -x ./soat-miner ]]; then
    echo "$CUDA_ONLY_ALGO needs the CUDA binary (./soat-miner) and it is not here."
    echo "  $CUDA_ONLY_ALGO is NVIDIA only. There is no Vulkan or AMD build of it yet."
    exit 1
  fi
  if [[ "$BACKEND" == "vulkan" ]]; then
    echo "note: $CUDA_ONLY_ALGO is CUDA only, using the CUDA binary despite BACKEND=vulkan"
  fi
  BACKEND=cuda
fi

# Which backend wins is PER ALGORITHM, not one rule for the whole miner. The
# Blackwell-prefers-Vulkan rule below was measured on Autolykos and does not
# carry: BC3 is faster on CUDA on NVIDIA (4090: CUDA 1543 MH/s, Vulkan 1086).
# BC3 got this wrong the moment it stopped being CUDA-only and fell through to
# the Autolykos rule, and an NVIDIA card quietly picked the slower backend.
#
# KawPow is the second one, and it is not a small margin. NVIDIA's Vulkan
# driver falls off a cliff once the randomly-touched working set passes roughly
# 3 GB, and Ravencoin's DAG is 6.11 GB and growing about 8 MB every five days:
#
#   RTX 5080   Vulkan  4.29 MH/s at 100 W    CUDA  7.38 MH/s at 360 W
#   RTX 4090   Vulkan  1.68 MH/s at 118 W    CUDA 11.9  MH/s
#   RX 7900XT  Vulkan 34.0  MH/s at 290 W    (AMD is unaffected and fine)
#
# The 100 W on a 5080 is the tell: stalled, not working. This is an NVIDIA
# driver property, not a shader problem - the same shader beats CUDA when the
# working set fits - so KawPow keeps its Vulkan backend for AMD and simply must
# not be auto-selected onto it on NVIDIA, Blackwell included.
PREFERS_CUDA_ON_NVIDIA=""
for a in "$@" "${ALGO:-}"; do
  case "$a" in
    sha3-256t) PREFERS_CUDA_ON_NVIDIA="BC3" ;;
    kawpow)    PREFERS_CUDA_ON_NVIDIA="KawPow" ;;
  esac
done
if [[ -n "$PREFERS_CUDA_ON_NVIDIA" && "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if [[ -x ./soat-miner ]]; then
      BACKEND=cuda
      echo "auto: $PREFERS_CUDA_ON_NVIDIA on NVIDIA - CUDA, it beats Vulkan here"
    else
      # Without this the Blackwell rule below quietly picks Vulkan and the user
      # mines at roughly half speed with nothing said. Declining to force CUDA
      # when the binary is absent is right; doing it silently is not.
      echo "warning: $PREFERS_CUDA_ON_NVIDIA is much faster on CUDA on this GPU,"
      echo "         but ./soat-miner is not in this directory. Continuing on"
      echo "         Vulkan, which is the SLOW path for it - get the full archive."
    fi
  fi
fi

if [[ "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    # This used to say largest VRAM wins, "the same rule the binaries use", and
    # that was not true. The Vulkan build defaults to the largest-VRAM device;
    # the CUDA build defaults to device 0. On a rig where those are different
    # cards of different generations, the launcher read one card's capability
    # and the miner then ran on the other - silently, and looking fine.
    #
    # Passing --device to force agreement is NOT the fix and must not be added
    # later: nvidia-smi enumerates by PCI bus order while CUDA defaults to
    # fastest-first, so an index handed across that boundary can name a
    # different card again. Vulkan enumerates in its own order on top of that.
    #
    # So: only decide automatically when every NVIDIA GPU present reports the
    # same capability, which is every single-GPU machine and every homogeneous
    # rig. When they differ the automatic answer is not well defined, and the
    # honest move is to say so and take the backend that works on all of them.
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
             2>/dev/null | tr -d '[:blank:]' | grep . | sort -u)
    ncaps=$(printf '%s\n' "$caps" | grep -c . || true)
    cap=$(printf '%s\n' "$caps" | head -1)
    major="${cap%%.*}"

    if [[ "$ncaps" =~ ^[0-9]+$ ]] && (( ncaps > 1 )); then
      BACKEND=cuda
      echo "warning: this machine has NVIDIA GPUs of different compute"
      echo "         capabilities ($(printf '%s' "$caps" | tr '\n' ' '))."
      echo "         The CUDA build would mine on device 0 and the Vulkan"
      echo "         build on the largest-VRAM card, which may not be the"
      echo "         same GPU, so the automatic backend is not well defined."
      echo "         Using CUDA, which is correct on all of them. Set"
      echo "         BACKEND= and DEVICE= in config.txt to choose properly."
    elif [[ "$major" =~ ^[0-9]+$ ]] && (( major >= 12 )); then
      BACKEND=vulkan
      echo "auto: compute capability $cap (Blackwell) - Vulkan, ~22% faster than CUDA here"
    else
      # Unknown capability falls through to CUDA, which is right for every
      # NVIDIA generation before Blackwell.
      BACKEND=cuda
    fi
  else
    BACKEND=vulkan
  fi
fi

case "$BACKEND" in
  cuda)   BIN=./soat-miner ;;
  vulkan) BIN=./soat-miner-vk ;;
  *) echo "unknown BACKEND '$BACKEND' (use auto|cuda|vulkan)"; exit 1 ;;
esac
[[ -x "$BIN" ]] || { echo "$BIN not found or not executable"; exit 1; }

ARGS=(--batch "${BATCH:-4194304}" --interval "${INTERVAL:-5}")
ARGS+=(--mclk-offset "${MCLK_OFFSET:-0}")

# Building the next block's table ahead is CUDA only so far.
if [[ "$BIN" == "./soat-miner" ]]; then
  ARGS+=(--cache-dag "${CACHE_DAG:-auto}")
fi

if [[ "$EXPLICIT_SOURCE" == "1" ]]; then
  echo "SOAT Miner [$BACKEND] - using command-line settings"
elif [[ "${LITHOS:-no}" == "yes" ]]; then
  # No WALLET check here: on Lithos the stratum address is a label, and payment
  # follows the node the Lithos client is attached to.
  ARGS+=(--lithos --pool "${LITHOS_ADDR:-127.0.0.1:4444}"
         --worker "${WORKER:-rig1}")
  echo "SOAT Miner [$BACKEND] -> Lithos client at ${LITHOS_ADDR:-127.0.0.1:4444}"
elif [[ -n "${POOL:-}" ]]; then
  # Per-algorithm, because the placeholder and the address shape differ. This
  # used to test the Ergo placeholder only, so a BC3 config with an unedited
  # wallet sailed past and the message named the wrong coin either way.
  case "${ALGO:-autolykos2}" in
    pearl-pow) COIN="Pearl";       PLACEHOLDER="prl1YOUR_PEARL_ADDRESS_HERE"; SCRIPTS="mine_pearl_*.sh" ;;
    sha3-256t) COIN="Bitcoin III"; PLACEHOLDER="YOUR_BC3_ADDRESS_HERE";       SCRIPTS="mine_bc3_*.sh" ;;
    kawpow)    COIN="Ravencoin";   PLACEHOLDER="RYOUR_RAVENCOIN_ADDRESS_HERE"; SCRIPTS="mine_rvn_*.sh" ;;
    *)         COIN="Ergo";        PLACEHOLDER="9YOUR_ERGO_ADDRESS_HERE";     SCRIPTS="mine_ergo_*.sh" ;;
  esac
  # Any coin's placeholder, not just this coin's. Switching ALGO and forgetting
  # to switch WALLET leaves the previous coin's placeholder sitting there, and
  # testing only the current one let that through to a confusing pool error.
  #
  # Matched by SHAPE, not by listing them. The list version needed every new
  # coin added here as well as above, and a coin whose placeholder was
  # forgotten would be caught only when it was the coin being mined - which
  # is the one case that was already covered. No real address contains an
  # underscore, so this cannot reject one.
  case "$WALLET" in
    *YOUR_*ADDRESS_HERE) WALLET="" ;;
  esac
  if [[ -z "${WALLET:-}" || "$WALLET" == "$PLACEHOLDER" ]]; then
    echo "Set WALLET in config.txt to your $COIN address before pool mining."
    echo "  (or use one of the $SCRIPTS scripts and edit WALLET there)"
    exit 1
  fi
  ARGS+=(--algo "${ALGO:-autolykos2}" --pool "$POOL" --wallet "$WALLET"
         --worker "${WORKER:-rig1}" --pass "${PASSWORD:-x}")
  echo "SOAT Miner [$BACKEND] -> $COIN on $POOL as ${WORKER:-rig1} paying $WALLET"
else
  ARGS+=(--node "${NODE:-127.0.0.1}" --port "${NODE_PORT:-9053}")
  echo "SOAT Miner [$BACKEND] -> solo via node ${NODE:-127.0.0.1}:${NODE_PORT:-9053}"
fi

exec "$BIN" "${ARGS[@]}" "$@"

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

# Nothing is CUDA-only any more, and this list is kept empty rather than
# deleted because it is load-bearing the moment a new algorithm ships on one
# backend first.
#
# The rule it exists for: without it the Blackwell rule below sends a
# CUDA-only algorithm to the Vulkan binary, which does not have it, and the run
# fails as a bad ERGO address - a baffling thing to be told when you asked for
# something else. Non-NVIDIA hits the same path, so it was never only a
# Blackwell problem. It is a list rather than a hardcoded test because a
# hardcoded pearl-pow check is what let BC3 inherit this bug when it was added.
# It checks config.txt's ALGO too, not only the command line.
#
# BC3 (sha3-256t) came off this list when it got a Vulkan backend. PEARL CAME
# OFF IT TODAY for the same reason: ten shaders, each byte-identical to the
# CUDA reference on an Ada and an RDNA3 card, a chain self-check that runs
# before any share is possible, and a mock gateway that accepted a real proof.
# Leaving an algorithm here after it works elsewhere is what made BC3
# CUDA-only in practice no matter what the binary supported - an AMD user was
# told the algorithm did not exist for them.
CUDA_ONLY_ALGO=""
for a in "$@" "${ALGO:-}"; do
  case "$a" in
    # (none)
    "") ;;
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
# The REASON is per algorithm too, not just the choice. "It beats Vulkan here"
# is a measurement and it is true for BC3 (4090: CUDA 1543 MH/s, Vulkan 1086).
# It is NOT measured for Pearl, and printing it there would state a number
# nobody has taken.
PREFERS_CUDA_ON_NVIDIA=""
PREFERS_CUDA_WHY=""
for a in "$@" "${ALGO:-}"; do
  case "$a" in
    sha3-256t)
      PREFERS_CUDA_ON_NVIDIA="BC3"
      PREFERS_CUDA_WHY="it beats Vulkan here" ;;
    # Pearl on NVIDIA stays on CUDA. Not from a measurement - there is no
    # Vulkan Pearl throughput number yet - but from what the two paths are:
    # the CUDA one tunes its shape and tile configuration per card at startup
    # and the Vulkan one has a single fixed shape and an untuned GEMM. On AMD
    # the question does not arise, since Vulkan is the only backend there.
    # Revisit this line with a number, not an assumption.
    pearl-pow)
      PREFERS_CUDA_ON_NVIDIA="Pearl"
      PREFERS_CUDA_WHY="its shape tuner has no Vulkan equivalent yet" ;;
  esac
done
if [[ -n "$PREFERS_CUDA_ON_NVIDIA" && "$BACKEND" == "auto" && -x ./soat-miner ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    BACKEND=cuda
    echo "auto: $PREFERS_CUDA_ON_NVIDIA on NVIDIA - CUDA, $PREFERS_CUDA_WHY"
  fi
fi

if [[ "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    # Largest VRAM wins, which is the same rule the binaries use to pick a
    # device - so the launcher and the miner agree which GPU this is about.
    cap=$(nvidia-smi --query-gpu=memory.total,compute_cap \
            --format=csv,noheader,nounits 2>/dev/null |
          sort -t, -k1 -n | tail -1 | cut -d, -f2 | tr -d '[:space:]')
    major="${cap%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]] && (( major >= 12 )); then
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
    *)         COIN="Ergo";        PLACEHOLDER="9YOUR_ERGO_ADDRESS_HERE";     SCRIPTS="mine_ergo_*.sh" ;;
  esac
  # Any coin's placeholder, not just this coin's. Switching ALGO and forgetting
  # to switch WALLET leaves the previous coin's placeholder sitting there, and
  # testing only the current one let that through to a confusing pool error.
  case "$WALLET" in
    9YOUR_ERGO_ADDRESS_HERE|prl1YOUR_PEARL_ADDRESS_HERE|YOUR_BC3_ADDRESS_HERE) WALLET="" ;;
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

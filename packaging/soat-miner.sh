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
    --pool|--node|--bench|--list-devices|--list-algos|--help|-h) EXPLICIT_SOURCE=1 ;;
  esac
done

BACKEND="${BACKEND:-auto}"
if [[ "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    BACKEND=cuda
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

if [[ "$EXPLICIT_SOURCE" == "1" ]]; then
  echo "SOAT Miner [$BACKEND] - using command-line settings"
elif [[ -n "${POOL:-}" ]]; then
  if [[ -z "${WALLET:-}" || "$WALLET" == 9YOUR_ERGO_ADDRESS_HERE ]]; then
    echo "Set WALLET in config.txt to your Ergo address before pool mining."
    echo "  (or use one of the mine_ergo_*.sh scripts and edit WALLET there)"
    exit 1
  fi
  ARGS+=(--pool "$POOL" --wallet "$WALLET"
         --worker "${WORKER:-rig1}" --pass "${PASSWORD:-x}")
  echo "SOAT Miner [$BACKEND] -> pool $POOL as ${WORKER:-rig1}"
else
  ARGS+=(--node "${NODE:-127.0.0.1}" --port "${NODE_PORT:-9053}")
  echo "SOAT Miner [$BACKEND] -> solo via node ${NODE:-127.0.0.1}:${NODE_PORT:-9053}"
fi

exec "$BIN" "${ARGS[@]}" "$@"

#!/usr/bin/env bash
# SOAT Miner launcher. Reads config.txt, picks a backend, starts mining.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[[ -f config.txt ]] || { echo "config.txt not found"; exit 1; }
# shellcheck disable=SC1091
set -a; source ./config.txt; set +a

BACKEND="${BACKEND:-auto}"
if [[ "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    BACKEND=cuda
  else
    BACKEND=opencl
  fi
fi

case "$BACKEND" in
  cuda)   BIN=./soat-miner ;;
  opencl) BIN=./soat-miner-cl ;;
  *) echo "unknown BACKEND '$BACKEND' (use auto|cuda|opencl)"; exit 1 ;;
esac
[[ -x "$BIN" ]] || { echo "$BIN not found or not executable"; exit 1; }

ARGS=(--batch "${BATCH:-4194304}" --interval "${INTERVAL:-5}")

if [[ -n "${POOL:-}" ]]; then
  if [[ -z "${WALLET:-}" || "$WALLET" == 9YOUR_ERGO_ADDRESS_HERE ]]; then
    echo "Set WALLET in config.txt to your Ergo address before pool mining."
    exit 1
  fi
  ARGS+=(--pool "$POOL" --wallet "$WALLET"
         --worker "${WORKER:-rig1}" --pass "${PASSWORD:-x}")
  echo "SOAT Miner -> pool $POOL as ${WORKER:-rig1} [$BACKEND]"
else
  ARGS+=(--node "${NODE:-127.0.0.1}" --port "${NODE_PORT:-9053}")
  echo "SOAT Miner -> solo via node ${NODE:-127.0.0.1}:${NODE_PORT:-9053} [$BACKEND]"
fi

exec "$BIN" "${ARGS[@]}" "$@"

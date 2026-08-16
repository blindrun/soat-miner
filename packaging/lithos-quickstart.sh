#!/usr/bin/env bash
#
# lithos-quickstart - stand up a full Lithos mining setup with one command.
#
# Lithos has no operator to point a miner at: you run your OWN Ergo node, your
# OWN Lithos client, and mine into it. Doing that by hand is a long chain of
# undocumented traps. This script does the whole thing and encodes every trap
# that cost real time to find:
#
#   * Java 11 is REQUIRED by the Lithos client, and Debian 12 dropped
#     openjdk-11 from its repos - so we add Adoptium/Temurin.
#   * On testnet the stock node jar computes the WRONG difficulty at the
#     Autolykos-v2 activation height and rejects the chain there. We detect the
#     real activation difficulty from the node's own rejection and override it.
#   * The node's version2ActivationDifficultyHex must be EVEN-length hex or the
#     node dies with a misleading "initialDifficultyVersion2" parse error.
#   * The 6.1.x node jars SIGSEGV in bundled RocksDB on some hosts; 6.0.3 uses
#     LevelDB and is stable, so testnet pins 6.0.3.
#   * The Lithos client will not even START without a wallet keystore, so we
#     create one first and wire its path in.
#
# Supports Debian/Ubuntu. Run as root (or with sudo).
#
#   ./lithos-quickstart.sh --network testnet
#   ./lithos-quickstart.sh --network mainnet
#
# It is idempotent: re-running reuses what already exists. It never deletes
# chain data or a wallet you already have.

set -euo pipefail

# ---- defaults / args --------------------------------------------------------
NETWORK="testnet"
NODE_DIR="/opt/ergo-node"
LITHOS_DIR="/opt/lithos"
NODE_API_KEY="hello"
# Hash of the API key "hello". Change API_KEY and API_KEY_HASH together.
API_KEY_HASH="324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf"
DIFF="4.0G"                # starting Lithos share difficulty
ERGO_TESTNET_VER="6.0.3"   # pinned: stable LevelDB, not the 6.1.x RocksDB crash
ERGO_MAINNET_VER="6.1.3"   # mainnet: latest stable is fine
NONINTERACTIVE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --network) NETWORK="$2"; shift 2 ;;
    --node-dir) NODE_DIR="$2"; shift 2 ;;
    --lithos-dir) LITHOS_DIR="$2"; shift 2 ;;
    --diff) DIFF="$2"; shift 2 ;;
    --yes) NONINTERACTIVE=1; shift ;;
    -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

[ "$NETWORK" = "testnet" ] || [ "$NETWORK" = "mainnet" ] || { echo "network must be testnet|mainnet"; exit 2; }
[ "$(id -u)" = 0 ] || { echo "run as root (or: sudo $0 ...)"; exit 1; }

if [ "$NETWORK" = "testnet" ]; then
  API_PORT=9052; ERGO_VER="$ERGO_TESTNET_VER"
else
  API_PORT=9053; ERGO_VER="$ERGO_MAINNET_VER"
fi
JAVA11="/usr/lib/jvm/temurin-11-jdk-amd64/bin/java"

say()  { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

cat <<BANNER

  L I T H O S   Q U I C K S T A R T
  network: $NETWORK   node: $NODE_DIR   client: $LITHOS_DIR
  This installs Java 11, an Ergo $NETWORK node, and the Lithos client,
  then creates a wallet and starts everything as systemd services.

BANNER
if [ "$NONINTERACTIVE" = 0 ]; then
  read -r -p "  proceed? [y/N] " ans; case "$ans" in y|Y) ;; *) echo "aborted"; exit 0 ;; esac
fi

# ---- 1. dependencies + Java 11 ---------------------------------------------
say "Installing dependencies and Java 11 (Temurin)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl wget jq unzip gnupg apt-transport-https >/dev/null
if [ ! -x "$JAVA11" ]; then
  install -d /etc/apt/keyrings
  wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public \
    | gpg --dearmor > /etc/apt/keyrings/adoptium.gpg
  . /etc/os-release
  echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb ${VERSION_CODENAME} main" \
    > /etc/apt/sources.list.d/adoptium.list
  apt-get update -qq
  apt-get install -y -qq temurin-11-jdk >/dev/null
fi
[ -x "$JAVA11" ] || die "Temurin 11 did not install at $JAVA11"
info "Java 11: $("$JAVA11" -version 2>&1 | head -1)"

# ---- 2. Ergo node -----------------------------------------------------------
say "Setting up the Ergo $NETWORK node (v$ERGO_VER)"
install -d "$NODE_DIR"
JAR="$NODE_DIR/ergo-node.jar"
if [ ! -s "$JAR" ]; then
  info "downloading ergo-$ERGO_VER.jar ..."
  wget -q "https://github.com/ergoplatform/ergo/releases/download/v$ERGO_VER/ergo-$ERGO_VER.jar" -O "$JAR.tmp"
  mv "$JAR.tmp" "$JAR"
fi

cat > "$NODE_DIR/ergo.conf" <<CONF
ergo {
  networkType = "$NETWORK"
  node {
    useExternalMiner = true
    offlineGeneration = false
    mining = true
    extraIndex = true
  }
}
scorex {
  restApi {
    apiKeyHash = "$API_KEY_HASH"
    bindAddress = "0.0.0.0:$API_PORT"
  }
  network { nodeName = "lithos-quickstart-node" }
}
CONF

# V2 difficulty override goes here once detected (testnet only).
V2_OPT_FILE="$NODE_DIR/v2diff.opt"
[ -f "$V2_OPT_FILE" ] || : > "$V2_OPT_FILE"

write_node_unit() {
  cat > /etc/systemd/system/lithos-ergo-node.service <<UNIT
[Unit]
Description=Ergo $NETWORK node (Lithos)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
WorkingDirectory=$NODE_DIR
ExecStart=$JAVA11 -Xmx4g $(cat "$V2_OPT_FILE") -jar $JAR --$NETWORK -c $NODE_DIR/ergo.conf
Restart=on-failure
RestartSec=15
LimitNOFILE=65536
[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
}
write_node_unit
systemctl enable --now lithos-ergo-node.service >/dev/null 2>&1 || true

# wait for the node API
info "waiting for the node API on :$API_PORT ..."
for i in $(seq 1 60); do
  curl -s -m4 "http://127.0.0.1:$API_PORT/info" | grep -q headersHeight && break
  sleep 3
done
curl -s -m4 "http://127.0.0.1:$API_PORT/info" | grep -q headersHeight || die "node API never came up - check: journalctl -u lithos-ergo-node"

# ---- 2b. testnet: detect + apply the real v2-activation difficulty ----------
if [ "$NETWORK" = "testnet" ] && [ ! -s "$V2_OPT_FILE" ]; then
  say "Checking the Autolykos-v2 activation difficulty (testnet relaunches change it)"
  info "letting the node sync toward the activation height; this can take a while..."
  # Look for the node's own rejection, which prints the correct 'Given' value.
  detected=""
  for i in $(seq 1 40); do
    line=$(journalctl -u lithos-ergo-node --no-pager -o cat 2>/dev/null \
             | grep -i "should contain correct required difficulty" | tail -1 || true)
    if [ -n "$line" ]; then
      # "... Given: <given>, expected: <expected>"
      given=$(echo "$line" | grep -oiE "Given:? *[0-9]+" | grep -oE "[0-9]+" | head -1)
      if [ -n "$given" ]; then detected="$given"; break; fi
    fi
    sleep 30
  done
  if [ -n "$detected" ]; then
    hex=$(printf '%x' "$detected")
    # Base16.decode needs even-length hex.
    [ $(( ${#hex} % 2 )) -eq 1 ] && hex="0$hex"
    info "activation difficulty is $detected (0x$hex) - applying override"
    echo "-Dergo.chain.voting.version2ActivationDifficultyHex=$hex" > "$V2_OPT_FILE"
    write_node_unit
    systemctl restart lithos-ergo-node.service
    sleep 20
  else
    info "no activation-difficulty rejection seen (node may already be past it, or"
    info "the stock value is correct). If sync later freezes at one height, re-run"
    info "this script - it will detect and apply the override then."
  fi
fi

# ---- 3. wallet --------------------------------------------------------------
say "Wallet"
KS_DIR="$NODE_DIR/.ergo/wallet/keystore"
if ls "$KS_DIR"/*.json >/dev/null 2>&1; then
  info "wallet keystore already present - keeping it"
else
  WPASS_FILE="$NODE_DIR/wallet-pass.txt"
  if [ ! -s "$WPASS_FILE" ]; then
    (umask 077; head -c24 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c32 > "$WPASS_FILE")
  fi
  WPASS=$(cat "$WPASS_FILE")
  info "creating a new wallet (password saved to $WPASS_FILE, root-only)"
  resp=$(curl -s -m20 -X POST "http://127.0.0.1:$API_PORT/wallet/init" \
           -H "Content-Type: application/json" -H "api_key: $NODE_API_KEY" \
           -d "{\"pass\":\"$WPASS\"}")
  echo "$resp" | grep -q mnemonic || die "wallet init failed: $resp"
  (umask 077; echo "$resp" > "$NODE_DIR/wallet-init.json")
  info "WALLET MNEMONIC saved to $NODE_DIR/wallet-init.json (root-only) - BACK IT UP."
fi
KEYSTORE=$(ls "$KS_DIR"/*.json 2>/dev/null | head -1)
[ -n "$KEYSTORE" ] || die "no keystore found after wallet init"

# ---- 4. Lithos client -------------------------------------------------------
say "Setting up the Lithos client"
install -d "$LITHOS_DIR"
if ! ls -d "$LITHOS_DIR"/lithos-client-* >/dev/null 2>&1; then
  info "fetching the latest Lithos client release ..."
  url=$(curl -s "https://api.github.com/repos/Lithos-Protocol/Lithos-Client/releases" \
          | jq -r '[.[] | select(.prerelease==false or (.tag_name|test("test")))][0].assets[]?.browser_download_url' \
          | grep -iE '\.zip$' | head -1)
  [ -n "$url" ] || die "could not find a Lithos client release asset"
  wget -q "$url" -O "$LITHOS_DIR/lithos.zip"
  ( cd "$LITHOS_DIR" && unzip -q -o lithos.zip )
fi
CLIENT_DIR=$(ls -d "$LITHOS_DIR"/lithos-client-* | head -1)
CFG="$CLIENT_DIR/conf/application.conf"
[ -f "$CFG" ] || die "client config not found at $CFG"

WPASS=$(cat "$NODE_DIR/wallet-pass.txt" 2>/dev/null || echo "")
SECRET=$(head -c32 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c40)
netUpper=$(echo "$NETWORK" | tr a-z A-Z)
# In-place edits: network, node key, keystore, wallet pass, play secret, diff.
sed -i \
  -e "s#networkType = \"[^\"]*\"#networkType = \"$netUpper\"#" \
  -e "s#^\( *\)key = \"[^\"]*\"#\1key = \"$NODE_API_KEY\"#" \
  -e "s#storagePath = \"[^\"]*\"#storagePath = \"$KEYSTORE\"#" \
  -e "s#pass *= \"[^\"]*\"#pass = \"$WPASS\"#" \
  -e "s#play.http.secret.key=\"[^\"]*\"#play.http.secret.key=\"$SECRET\"#" \
  -e "s#diff = \"[^\"]*\"#diff = \"$DIFF\"#" \
  "$CFG"

cat > /etc/systemd/system/lithos-client.service <<UNIT
[Unit]
Description=Lithos client (stratum :4444)
After=network-online.target lithos-ergo-node.service
Wants=network-online.target
[Service]
Type=simple
Environment=JAVA_HOME=/usr/lib/jvm/temurin-11-jdk-amd64
WorkingDirectory=$CLIENT_DIR
ExecStart=$CLIENT_DIR/bin/lithos-client -Dconfig.file=$CFG -J-Xmx3g
Restart=on-failure
RestartSec=30
LimitNOFILE=65536
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable lithos-client.service >/dev/null 2>&1 || true

# ---- 5. status config + finish ---------------------------------------------
cat > /etc/lithos-quickstart.conf <<CONF
NETWORK=$NETWORK
NODE_PORT=$API_PORT
CLIENT_PORT=$API_PORT
NODE_API_KEY=$NODE_API_KEY
CONF
if [ -f "$(dirname "$0")/lithos-status" ]; then
  install -m755 "$(dirname "$0")/lithos-status" /usr/local/bin/lithos-status
fi

cat <<DONE

==> Done. Node and client are installed.

  The Lithos client is enabled but NOT started - it needs a FULLY SYNCED node
  first (it will not serve jobs until then, and it competes for memory during
  the v2 dataset build). Watch the sync:

      lithos-status --watch

  When it shows SYNCED, start the client and point your miner at it:

      systemctl start lithos-client
      soat-miner --lithos            # or: rigel -a autolykos2 -o stratum+tcp://127.0.0.1:4444 -u x

  Then 'lithos-status' tells you whether you are producing enough super shares
  to get paid. Your wallet mnemonic is in $NODE_DIR/wallet-init.json - BACK IT UP.

DONE

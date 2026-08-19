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
# NODE_API_KEY and API_KEY_HASH are GENERATED per install, not hardcoded. A
# fixed key in a public repo plus a wallet-capable API is remote wallet control
# for anyone who can reach the port. The key is generated below, persisted
# root-only, and the node binds it to loopback only. See the node setup section.
NODE_API_KEY=""
API_KEY_HASH=""
DIFF="4.0G"                # starting Lithos share difficulty
ERGO_TESTNET_VER="6.0.3"   # pinned: stable LevelDB, not the 6.1.x RocksDB crash
ERGO_MAINNET_VER="6.1.3"   # mainnet: latest stable is fine
ERGO_TESTNET_SHA256="4802cde3550623e639a5d09f45d257922e01815c5b1fe64bdafd2ebc69ec67c7"
ERGO_MAINNET_SHA256="c497e83b0db0631ae3f7080cf1b8438b9d585fd5a406274d1ada7161cb96593d"
LITHOS_TESTNET_URL="https://github.com/Lithos-Protocol/Lithos-Client/releases/download/v4.2.0-test/lithos-client-4.2.0-SNAPSHOT.zip"
LITHOS_TESTNET_SHA256="38c2b7c500e49d03fe3ddd8c98a1b1cd4fecd09c861ec9dab758cd757fba0a45"
# No official mainnet artifact is currently published. Refuse to select an
# arbitrary release: an operator must provide a reviewed URL and SHA-256.
LITHOS_MAINNET_URL="${LITHOS_MAINNET_URL:-}"
LITHOS_MAINNET_SHA256="${LITHOS_MAINNET_SHA256:-}"
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

# These land in systemd unit fields and in sed replacements that use # as the
# delimiter. A #, quote, space or newline breaks a unit or the sed confusingly.
for v in "$NODE_DIR" "$LITHOS_DIR" "$DIFF"; do
  printf '%s' "$v" | LC_ALL=C grep -q '[^A-Za-z0-9._/-]' \
    && { echo "value has a disallowed character (allowed: A-Za-z0-9 . _ / -): $v"; exit 2; }
done

if [ "$NETWORK" = "testnet" ]; then
  API_PORT=9052; ERGO_VER="$ERGO_TESTNET_VER"; ERGO_SHA256="$ERGO_TESTNET_SHA256"
  CLIENT_PORT=9000; STRATUM_PORT=4444
else
  API_PORT=9053; ERGO_VER="$ERGO_MAINNET_VER"; ERGO_SHA256="$ERGO_MAINNET_SHA256"
  CLIENT_PORT=9001; STRATUM_PORT=4445
fi
JAVA11="/usr/lib/jvm/temurin-11-jdk-amd64/bin/java"

# Network-isolate the install dirs unless the caller set explicit paths. Testnet
# and mainnet must NOT share a jar, keystore, client config or the v2diff
# override - a testnet run followed by a mainnet run otherwise reuses the wrong
# jar (6.0.3 vs 6.1.3), the testnet keystore, and the testnet startHeight.
[ "$NODE_DIR" = "/opt/ergo-node" ] && NODE_DIR="/opt/ergo-node-$NETWORK"
[ "$LITHOS_DIR" = "/opt/lithos" ] && LITHOS_DIR="/opt/lithos-$NETWORK"

# Unit names are network-suffixed too, so a testnet and a mainnet install do not
# overwrite each other's service or share one journal.
NODE_UNIT="lithos-ergo-node-$NETWORK"
CLIENT_UNIT="lithos-client-$NETWORK"

# A dedicated unprivileged service account, PER NETWORK. A single shared account
# would fix its home at first useradd, so a later mainnet node would resolve
# ~/.ergo under the testnet home while ProtectSystem=strict only allows the
# mainnet dir - the second node could not write its data. Per-network users with
# home = that network's NODE_DIR keep the two installs genuinely isolated.
SVC_USER="lithos-$NETWORK"

say()  { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Hex uses every byte read and guarantees an exact length; filtering base64 can
# silently produce a shorter secret when '+' or '/' are removed.
random_secret() {
  local n="$1" bytes raw
  case "$n" in ''|*[!0-9]*|0) die "invalid requested secret length: $n" ;; esac
  bytes=$(( (n + 1) / 2 ))
  raw=$(od -An -N "$bytes" -tx1 /dev/urandom | tr -d ' \n')
  [ ${#raw} -eq $(( bytes * 2 )) ] || die "could not read enough randomness"
  printf '%s' "${raw:0:n}"
}

verify_sha256() {
  local file="$1" want="$2"
  printf '%s  %s\n' "$want" "$file" | sha256sum -c - >/dev/null 2>&1 ||
    die "SHA-256 verification failed for $file"
}

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
apt-get install -y -qq curl wget jq unzip gnupg apt-transport-https python3 >/dev/null
if [ ! -x "$JAVA11" ]; then
  install -d /etc/apt/keyrings
  ADOPTIUM_FPR="3B04D753C9050D9A5D343F39843C48A565F8F04B"
  ADOPTIUM_KEY=$(mktemp)
  wget -qO "$ADOPTIUM_KEY" https://packages.adoptium.net/artifactory/api/gpg/key/public
  gpg --show-keys --with-colons "$ADOPTIUM_KEY" | awk -F: '$1 == "fpr" {print $10; exit}' \
    | grep -qx "$ADOPTIUM_FPR" || die "Adoptium signing-key fingerprint did not match"
  gpg --dearmor < "$ADOPTIUM_KEY" > /etc/apt/keyrings/adoptium.gpg
  rm -f "$ADOPTIUM_KEY"
  . /etc/os-release
  echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb ${VERSION_CODENAME} main" \
    > /etc/apt/sources.list.d/adoptium.list
  apt-get update -qq
  apt-get install -y -qq temurin-11-jdk >/dev/null
fi
[ -x "$JAVA11" ] || die "Temurin 11 did not install at $JAVA11"
info "Java 11: $("$JAVA11" -version 2>&1 | head -1)"

# ---- service account --------------------------------------------------------
# The node and client run as this unprivileged user, not root. It owns the
# install dirs and the wallet material.
if ! id "$SVC_USER" >/dev/null 2>&1; then
  useradd --system --home-dir "$NODE_DIR" --shell /usr/sbin/nologin "$SVC_USER"
  info "created service account $SVC_USER"
fi

# ---- 2. Ergo node -----------------------------------------------------------
say "Setting up the Ergo $NETWORK node (v$ERGO_VER)"
install -d "$NODE_DIR"
# Recursive: a pre-existing node dir (chain data, keystore) must be owned by the
# unprivileged service user or it cannot reuse it. Root-written config below is
# re-owned to root afterwards.
chown -R "$SVC_USER:$SVC_USER" "$NODE_DIR"
JAR="$NODE_DIR/ergo-node.jar"

# Generate a per-install node API key (or reuse the persisted one) and its
# blake2b-256 hash. The key is stored root-only; the node sees only the hash and
# binds the API to loopback. 48 urandom bytes -> 64 base64 chars -> ~56 after
# filtering -> a real 40-char key (24 bytes filtered would fall short of 40).
API_KEY_FILE="$NODE_DIR/api-key.txt"
if [ -s "$API_KEY_FILE" ]; then
  NODE_API_KEY=$(cat "$API_KEY_FILE")
else
  NODE_API_KEY=$(random_secret 40)
  [ ${#NODE_API_KEY} -eq 40 ] || die "could not generate a 40-character API key"
  (umask 077; printf '%s' "$NODE_API_KEY" > "$API_KEY_FILE")
fi
# `chown -R` above is needed for old chain data, but must never leave this raw
# key readable by the service account on a rerun.
chown root:root "$API_KEY_FILE"; chmod 600 "$API_KEY_FILE"
API_KEY_HASH=$(python3 -c 'import hashlib,sys; print(hashlib.blake2b(sys.argv[1].encode(), digest_size=32).hexdigest())' "$NODE_API_KEY")
[ ${#API_KEY_HASH} -eq 64 ] || die "failed to compute API key hash (need python3)"
if [ ! -s "$JAR" ]; then
  info "downloading ergo-$ERGO_VER.jar ..."
  wget -q "https://github.com/ergoplatform/ergo/releases/download/v$ERGO_VER/ergo-$ERGO_VER.jar" -O "$JAR.tmp"
  verify_sha256 "$JAR.tmp" "$ERGO_SHA256"
  mv "$JAR.tmp" "$JAR"
fi
# Re-check on every root-run invocation: a prior service-account ownership
# change must not turn an existing JAR into trusted executable input.
verify_sha256 "$JAR" "$ERGO_SHA256"
chown root:root "$JAR"; chmod 644 "$JAR"

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
    # Loopback only. This API can spend from the wallet; node and client are both
    # local, and remote miners use --lithos --pool <host>:4444, which needs no
    # node API exposure. Do NOT widen this to 0.0.0.0.
    bindAddress = "127.0.0.1:$API_PORT"
  }
  network { nodeName = "lithos-quickstart-node" }
}
CONF

# V2 difficulty override goes here once detected (testnet only).
V2_OPT_FILE="$NODE_DIR/v2diff.opt"
[ -f "$V2_OPT_FILE" ] || : > "$V2_OPT_FILE"

write_node_unit() {
  cat > /etc/systemd/system/${NODE_UNIT}.service <<UNIT
[Unit]
Description=Ergo $NETWORK node (Lithos)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=$SVC_USER
Group=$SVC_USER
Environment=HOME=$NODE_DIR
WorkingDirectory=$NODE_DIR
ExecStart=$JAVA11 -Xmx4g -Duser.home=$NODE_DIR $(cat "$V2_OPT_FILE") -jar $JAR --$NETWORK -c $NODE_DIR/ergo.conf
Restart=on-failure
RestartSec=15
LimitNOFILE=65536
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$NODE_DIR
PrivateTmp=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
}
write_node_unit
# Timestamp before the node starts, so v2diff detection reads only THIS run's
# rejection, not a stale one from an earlier restart in the same boot.
NODE_START_TS=$(date '+%Y-%m-%d %H:%M:%S')
systemctl enable --now ${NODE_UNIT}.service >/dev/null 2>&1 || true

# wait for the node API
info "waiting for the node API on :$API_PORT ..."
for i in $(seq 1 60); do
  curl -s -m4 "http://127.0.0.1:$API_PORT/info" | grep -q headersHeight && break
  sleep 3
done
curl -s -m4 "http://127.0.0.1:$API_PORT/info" | grep -q headersHeight || die "node API never came up - check: journalctl -u ${NODE_UNIT}"

# ---- 2b. testnet: detect + apply the real v2-activation difficulty ----------
if [ "$NETWORK" = "testnet" ] && [ ! -s "$V2_OPT_FILE" ]; then
  say "Checking the Autolykos-v2 activation difficulty (testnet relaunches change it)"
  info "letting the node sync toward the activation height; this can take a while..."
  # Look for the node's own rejection, which prints the correct 'Given' value.
  detected=""
  for i in $(seq 1 40); do
    line=$(journalctl -u ${NODE_UNIT} --since "$NODE_START_TS" --no-pager -o cat 2>/dev/null \
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
    systemctl restart ${NODE_UNIT}.service
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
WPASS_FILE="$NODE_DIR/wallet-pass.txt"
if ls "$KS_DIR"/*.json >/dev/null 2>&1; then
  info "wallet keystore already present - keeping it"
  # The client needs this wallet's password to open the keystore. If we never
  # saved it (a pre-existing wallet, or a partial earlier run that got past
  # keystore creation), we cannot recover it, and handing the client an empty
  # password silently breaks it. Fail loud instead of installing a broken client.
  [ -s "$WPASS_FILE" ] || die "keystore exists at $KS_DIR but $WPASS_FILE is missing.
  This wallet's password is not recoverable here. Either put it in $WPASS_FILE
  (root-only, first line), or move the keystore aside to create a fresh wallet."
else
  if [ ! -s "$WPASS_FILE" ]; then
    (umask 077; random_secret 32 > "$WPASS_FILE")
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
chown root:root "$WPASS_FILE"; chmod 600 "$WPASS_FILE"
KEYSTORE=$(ls "$KS_DIR"/*.json 2>/dev/null | head -1)
[ -n "$KEYSTORE" ] || die "no keystore found after wallet init"

# ---- 4. Lithos client -------------------------------------------------------
say "Setting up the Lithos client"
install -d "$LITHOS_DIR"
if [ "$NETWORK" = "testnet" ]; then
  url="$LITHOS_TESTNET_URL"
  lithos_sha256="$LITHOS_TESTNET_SHA256"
else
  url="$LITHOS_MAINNET_URL"
  lithos_sha256="$LITHOS_MAINNET_SHA256"
  [ -n "$url" ] && [ -n "$lithos_sha256" ] ||
    die "no reviewed mainnet Lithos artifact is pinned; set LITHOS_MAINNET_URL and LITHOS_MAINNET_SHA256"
fi
LITHOS_VERIFY_MARKER="$LITHOS_DIR/.soat-lithos-sha256"
if ! ls -d "$LITHOS_DIR"/lithos-client-* >/dev/null 2>&1; then
  info "fetching the pinned Lithos client release ..."
  wget -q "$url" -O "$LITHOS_DIR/lithos.zip"
  verify_sha256 "$LITHOS_DIR/lithos.zip" "$lithos_sha256"
  ( cd "$LITHOS_DIR" && unzip -q -o lithos.zip )
  printf '%s\n' "$lithos_sha256" > "$LITHOS_VERIFY_MARKER"
elif [ ! -f "$LITHOS_VERIFY_MARKER" ] || [ "$(cat "$LITHOS_VERIFY_MARKER")" != "$lithos_sha256" ]; then
  die "existing Lithos client is not verified for this pinned release; move it aside and re-run"
fi
CLIENT_DIR=$(ls -d "$LITHOS_DIR"/lithos-client-* | head -1)
CFG="$CLIENT_DIR/conf/application.conf"
[ -f "$CFG" ] || die "client config not found at $CFG"

WPASS=$(cat "$WPASS_FILE" 2>/dev/null || echo "")
[ -n "$WPASS" ] || die "wallet password ($WPASS_FILE) is empty - refusing to write a client that cannot open the wallet"
SECRET=$(random_secret 40)
netUpper=$(echo "$NETWORK" | tr a-z A-Z)
# In-place edits: network, node key, keystore, wallet pass, play secret, diff,
# and startHeight (the shipped config carries a testnet height that would
# otherwise bleed into a mainnet install - this wallet is fresh, so 0 is correct).
sed -i \
  -e "s#networkType = \"[^\"]*\"#networkType = \"$netUpper\"#" \
  -e "s#^\( *\)key = \"[^\"]*\"#\1key = \"$NODE_API_KEY\"#" \
  -e "s#storagePath = \"[^\"]*\"#storagePath = \"$KEYSTORE\"#" \
  -e "s#pass *= \"[^\"]*\"#pass = \"$WPASS\"#" \
  -e "s#play.http.secret.key=\"[^\"]*\"#play.http.secret.key=\"$SECRET\"#" \
  -e "s#diff = \"[^\"]*\"#diff = \"$DIFF\"#" \
  -e "s#stratumPort *= *[0-9]*#stratumPort = $STRATUM_PORT#" \
  -e "s#startHeight *= *[0-9]*#startHeight = 0#" \
  "$CFG"
# The client config now holds the wallet password. Lock it to the service user
# only - the shipped archive leaves it world-readable (0755).
chown "$SVC_USER:$SVC_USER" "$CFG"; chmod 600 "$CFG"
# The client runs as the service account, so it must own its tree.
chown -R "$SVC_USER:$SVC_USER" "$LITHOS_DIR"

cat > /etc/systemd/system/${CLIENT_UNIT}.service <<UNIT
[Unit]
Description=Lithos client (stratum :$STRATUM_PORT)
After=network-online.target ${NODE_UNIT}.service
Wants=network-online.target
[Service]
Type=simple
User=$SVC_USER
Group=$SVC_USER
Environment=JAVA_HOME=/usr/lib/jvm/temurin-11-jdk-amd64
Environment=HOME=$LITHOS_DIR
WorkingDirectory=$CLIENT_DIR
ExecStart=$CLIENT_DIR/bin/lithos-client -Dconfig.file=$CFG -Dhttp.port=$CLIENT_PORT -J-Xmx3g
Restart=on-failure
RestartSec=30
LimitNOFILE=65536
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$LITHOS_DIR
PrivateTmp=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable ${CLIENT_UNIT}.service >/dev/null 2>&1 || true

# ---- 5. status config + finish ---------------------------------------------
cat > /etc/lithos-quickstart.conf <<CONF
NETWORK=$NETWORK
NODE_PORT=$API_PORT
# The client HTTP and Stratum ports are network-specific so testnet and mainnet
# can coexist on one host.
CLIENT_PORT=$CLIENT_PORT
STRATUM_PORT=$STRATUM_PORT
NODE_UNIT=$NODE_UNIT
CLIENT_UNIT=$CLIENT_UNIT
# The node API key is deliberately NOT stored here. This file stays readable so
# an ordinary user can run lithos-status, which only queries the public /info
# and the client's /mining - neither needs the key. If you lock the node's
# /info, pass NODE_API_KEY=... to lithos-status through root-only local
# administration; the key is deliberately absent from this file.
CONF
chmod 644 /etc/lithos-quickstart.conf   # no secret in here by design
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

      systemctl start ${CLIENT_UNIT}
      soat-miner --lithos --pool 127.0.0.1:$STRATUM_PORT

  Then 'lithos-status' tells you whether you are producing enough super shares
  to get paid. Your wallet mnemonic is in $NODE_DIR/wallet-init.json - BACK IT UP.

DONE

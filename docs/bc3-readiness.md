# Bitcoin III readiness gates

This document keeps the remaining readiness checks bounded. None of its
commands starts `soat-miner`, connects to a pool, opens a wallet, or sends a
share. Do not substitute a public endpoint or an unidentified listener.

## CMake / CTest offline gate

The portable CMake equivalent of `make test-bc3-host` is:

```bash
make test-bc3-cmake
```

It runs `scripts/test-bc3-cmake.sh`, which configures with `BUILD_TESTING=ON`,
builds only `test-btc-stratum` and `test-bc3-destination`, then runs only
`btc-stratum-offline` and `bc3-destination-schema`. The protocol test reads the checked-in
`tests/fixtures/btc_stratum_v1.jsonl`; it creates no socket and has no GPU
runtime work. CMake still needs a CUDA compiler because the top-level project
declares CUDA.

For a reproducible build directory and a faster single-architecture configure:

```bash
BC3_CMAKE_BUILD_DIR="$PWD/build/bc3-ctest" \
BC3_CMAKE_ARCHITECTURES=89 \
make test-bc3-cmake
```

The runner intentionally does not install CMake or dispatch CI. If CMake or
CTest is absent, it exits with an actionable error. A CMake-equipped host can
use `CMAKE_BIN` and `CTEST_BIN` to select a paired tool installation.

## One-GPU six-vector gate — prepared, not authorized to run

Run this only under a separate approval that names one GPU index and confirms
that no interactive workload may be interrupted. It executes only
`tests/test_sha3_algo`, which uses six checked-in BC3 header vectors. It takes
no pool, wallet, node, or miner arguments.

```bash
set -euo pipefail
GPU_INDEX='<approved-index>'
export GPULOCK_WHO="bc3-six-vector-$(date -u +%Y%m%dT%H%M%SZ)"

gpulock status
gpulock check
gpulock gpus
# Continue only if GPU_INDEX is shown idle and unclaimed by the commands above.
gpulock claim "$GPU_INDEX" 'BC3 offline SHA3-256t six-vector gate' 15
cleanup() { gpulock release "$GPU_INDEX"; }
trap cleanup EXIT INT TERM

make tests/test_sha3_algo
timeout --signal=TERM --kill-after=30s 5m ./tests/test_sha3_algo
```

Record before release of the claim: UTC start/end, Git revision and dirty
state, `gpulock status`/`check` output, selected GPU index and model, exact
build/run command, timeout, and the six-vector result. If the preflight is not
clean, the build fails, the timeout fires, or any vector fails, stop and leave
the claim cleanup to the trap. Do not retry on a different GPU without a new
approval.

## Chain-only node fixture placeholder

`tests/fixtures/bc3-chain-node.env.example` is a schema, not a live config or
credential store. A future fixture capture may read exactly
`getblockchaininfo` and `getblocktemplate` after explicit approval; it must
write only a sanitized static fixture for a host test. The BC3 miner itself is
pool-only, so node data would validate host header construction, never launch
mining.

Before any node validation, the owner must provide all of the following:

1. A named disposable **regtest or testnet** host plus its service/container
   identity and deliberately approved RPC URL. An unexplained loopback port is
   not sufficient.
2. The node daemon/version and the exact existing configuration or launch
   argument that proves wallet RPC is disabled. Do not ask this worktree to
   edit configuration or inspect a wallet.
3. An approved, non-wallet, least-privilege authentication route supplied out
   of band; no credentials, cookie paths, payout addresses, or wallet names go
   in the repository.
4. Confirmation that only the two allowlisted chain methods may be invoked and
   that the endpoint is not public/mainnet production infrastructure.
5. Fresh approval for the one bounded capture after those details are supplied.

Without every item, the only valid node-related action is none.

### Exact future chain-only command shape

After Control supplies a named instance and a credential-injecting runner (the
credential itself must remain outside this repository), the bounded capture is
exactly two JSON-RPC calls:

```bash
set -euo pipefail
: "${BC3_CHAIN_NODE_RPC_URL:?Control must supply the approved endpoint}"
curl --fail --silent --show-error --max-time 10 \
  --data-raw '{"jsonrpc":"1.0","id":"bc3-readonly","method":"getblockchaininfo","params":[]}' \
  "$BC3_CHAIN_NODE_RPC_URL"
curl --fail --silent --show-error --max-time 10 \
  --data-raw '{"jsonrpc":"1.0","id":"bc3-readonly","method":"getblocktemplate","params":[{"rules":[]}]}' \
  "$BC3_CHAIN_NODE_RPC_URL"
```

The approved runner may inject its non-wallet, read-only authorization out of
band. Do not add `-u`, cookies, headers, URLs, or responses containing secrets
to shell history, this repository, logs, or the fixture. Abort unless the
response identifies the named regtest instance and wallet RPC is independently
proven disabled. Sanitize the two responses before any fixture is written.

### Selected configuration and destination boundary

The selected configuration is a **disposable local regtest** node, documented
in `tests/fixtures/bc3-regtest-node.conf.example`. It is safer than testnet for
this purpose because regtest has no public peers. The required shape is
loopback-only RPC, `disablewallet=1`, no P2P listener/discovery/connections,
no pool configuration, and no mining command. The template is not installed
or started by this worktree.

For a future sweep-destination fixture, the code accepts exactly one explicit
specification through `parseBc3SweepDestination`:

```text
address:<BC3 payout address>
secret-ref:crypto-recovery/<control-record>
```

The `secret-ref:crypto-recovery/...` value is only an opaque pointer to the
established isolated Crypto Recovery VM procedure. It has no resolver in this
repository and is never expanded into an environment value, log, fixture, CI
value, or command line. The parser has no accepted grammar for seeds,
mnemonics, private keys, passwords, cookies, or raw unprefixed values. The
pool-mining `--wallet` option remains an address input and must never receive a
reference or recovery material.

Control/user must supply, out of band, only a Crypto Recovery control record
identifier represented as `secret-ref:crypto-recovery/<control-record>`. Bind
it later through the existing isolated-VM procedure and its approved rollout
plan; this repository must never receive its content. Until then, no node,
wallet RPC, credentials, or destination resolution is authorized.

### Required non-secret operator handoff for a later binding

Control must provide all of the following out of band, with no recovery
material attached:

1. An approval identifier that authorizes one destination-binding operation.
2. Confirmation that the target is the approved disposable local regtest
   environment with wallet RPC disabled and no peers, pool, or mining.
3. Exactly one destination specification: an address supplied out of band as
   `address:<BC3 payout address>`, or the opaque
   `secret-ref:crypto-recovery/<control-record>` identifier.
4. Confirmation that the existing isolated Crypto Recovery VM procedure is the
   authorized resolver for a reference, including its designated operator and
   change/rollback authority.
5. A completion record containing only the approval identifier, destination
   type (`address` or `secret-reference`), and success/failure status.

Do not attach recovery material, credentials, filesystem locations, amounts,
or unredacted destination data to this handoff or this repository.

#!/bin/sh
# CLI contract checks that finish before CUDA initialization or networking.

set -eu

bin=${1:?usage: test_pearl_cli.sh <soat-miner binary>}
out=$(mktemp "${TMPDIR:-/tmp}/soat-pearl-cli.XXXXXX")
trap 'rm -f "$out"' EXIT HUP INT TERM

"$bin" --help >"$out"
grep -F -- '--pool HOST:PORT  stratum pool; Pearl uses its pool protocol' "$out" >/dev/null
grep -F -- '--gateway H[:PORT] pearl-gateway for pearl-pow solo' "$out" >/dev/null
grep -F -- '--pearl-transcript FILE  test-only Pearl pool metadata log' "$out" >/dev/null

reject_gateway() {
    value=$1
    if "$bin" --algo pearl-pow --gateway "$value" >"$out" 2>&1; then
        echo "accepted invalid Pearl gateway: $value" >&2
        return 1
    fi
    grep -F -- '--gateway needs HOST[:PORT]' "$out" >/dev/null
}

reject_gateway ''
reject_gateway ':8337'
reject_gateway '127.0.0.1:0'
reject_gateway '127.0.0.1:65536'

if "$bin" --algo autolykos2 --pool example.invalid:1 --pearl-transcript "$out" >"$out" 2>&1; then
    echo 'accepted Pearl transcript outside Pearl pool mode' >&2
    exit 1
fi
grep -F -- '--pearl-transcript requires --algo pearl-pow with --pool.' "$out" >/dev/null

echo 'pearl CLI contract: ok'

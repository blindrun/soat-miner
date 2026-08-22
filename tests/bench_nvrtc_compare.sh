#!/bin/sh
# What does --nvrtc actually buy, on the shipping binary?
#
# WHY THIS EXISTS AND NOT A NUMBER IN A DOCUMENT
#
# The first speedup figure for runtime compilation came from a standalone
# harness, and it was wrong in a way that took a while to see: the harness
# launched with threads=128 while the real miner uses 512, so its "baseline"
# was 23% slower than the miner it was standing in for. The ratio was therefore
# flattering by an unknown amount. A number derived from a stand-in is a model,
# not a measurement, however carefully it was computed.
#
# This runs the real binary both ways, same machine, same algorithm, back to
# back, and reports both arms rather than just the ratio - so the next person
# can see whether the baseline is the one they expect instead of taking the
# quotient on trust.
#
# EFFICIENCY, NOT JUST THROUGHPUT
#
# Matt's stated target is efficiency: the most we can squeeze out per watt.
# A kernel that is 5x faster and 5x hungrier has bought nothing, and the only
# way to know which happened is to print watts next to the rate. 235.5 MH/s at
# 167 W and 235.5 at 450 W are the same headline describing very different
# machines.
set -u

BIN=${1:-./soat-miner}
RUN=${2:-60}
ALGO=${3:-kawpow}
INTERVAL=10

if [ ! -x "$BIN" ]; then
    echo "  SKIP  $BIN not built"
    exit 0
fi

# Claim the card and CHECK THE RESULT. `gpulock claim` returning REFUSED is not
# an error exit, so a script that only calls it runs its benchmark on a card
# somebody else is mid-measurement on - which has already happened here and
# contaminated both sides.
: "${GPULOCK_WHO:=soat-miner nvrtc compare}"
export GPULOCK_WHO
if command -v gpulock >/dev/null 2>&1; then
    CLAIM=$(gpulock claim 4090 "nvrtc vs interpreted, $ALGO" $(( (RUN * 2) / 60 + 3 )) 2>&1)
    echo "$CLAIM"
    case "$CLAIM" in
        *REFUSED*|*ERROR*)
            echo "  ABORT  card not claimed; refusing to measure on a busy GPU"
            exit 1
            ;;
    esac
    # Released explicitly at each exit below, never from a trap: a trap on EXIT
    # inside one tool call fires at the end of THAT CALL, dropping the claim
    # while later commands still run against the card.
fi

run_arm() {
    # $1: label, $2: extra flag (may be empty)
    log=$(mktemp)
    # shellcheck disable=SC2086
    timeout "$RUN" "$BIN" --algo "$ALGO" --bench --plain --interval "$INTERVAL" \
        $2 > "$log" 2>&1
    rc=$?
    case "$rc" in
        124) ;;   # bench runs until stopped; timeout is the success case
        *)
            echo "  FAIL  $1: exited $rc rather than being stopped by timeout"
            echo "        last output: $(tail -1 "$log")"
            rm -f "$log"
            return 1
            ;;
    esac
    rate=$(grep -oE '"mhs_avg":[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    watts=$(grep -oE '"watts":[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    # Derived here from mhs_avg, NOT read from the miner's eff_mh_w field.
    # That field is computed from the INSTANTANEOUS rate (telemetry.h: s.hashrate
    # / watts) while the headline this script reads is mhs_avg, so mixing them
    # compares an average against an instantaneous figure. It inflated the
    # efficiency ratio to 5.19x when throughput had only moved 4.95x - and since
    # power was equal in both arms, those two ratios must be the same number.
    # A ratio that disagrees with itself is the tell.
    if [ -n "$watts" ] && [ -n "$rate" ]; then
        eff=$(echo "scale=4; $rate / $watts" | bc -l)
    else
        eff=""
    fi
    verified=$(grep -c "runtime-compiled kernel verified" "$log")
    fellback=$(grep -c "interpreted kernel" "$log")
    if [ -z "$rate" ]; then
        echo "  FAIL  $1: no stats line in ${RUN}s"
        echo "        last output: $(tail -1 "$log")"
        rm -f "$log"
        return 1
    fi
    printf '  %-14s %8s MH/s  %6s W  %6s MH/W' "$1" "$rate" "${watts:-?}" "${eff:-?}"
    [ "$verified" -gt 0 ] && printf '  [verified against interpreted]'
    printf '\n'
    echo "$rate" > "$log.rate"
    RESULT_RATE=$rate
    RESULT_EFF=$eff
    RESULT_VERIFIED=$verified
    RESULT_FELLBACK=$fellback
    rm -f "$log" "$log.rate"
    return 0
}

echo "--- $ALGO: interpreted vs runtime-compiled, ${RUN}s each ---"

run_arm "interpreted" "" || { command -v gpulock >/dev/null 2>&1 && gpulock release 4090; exit 1; }
BASE_RATE=$RESULT_RATE
BASE_EFF=$RESULT_EFF

run_arm "--nvrtc" "--nvrtc" || { command -v gpulock >/dev/null 2>&1 && gpulock release 4090; exit 1; }
JIT_RATE=$RESULT_RATE
JIT_EFF=$RESULT_EFF
JIT_VERIFIED=$RESULT_VERIFIED

command -v gpulock >/dev/null 2>&1 && gpulock release 4090

# A --nvrtc arm that silently fell back to the interpreted kernel produces two
# nearly identical numbers and a 1.0x "speedup", which reads as "runtime
# compilation does not help" when the truth is that it never ran. Say which
# happened.
if [ "${JIT_VERIFIED:-0}" -eq 0 ]; then
    echo
    echo "  WARNING: the --nvrtc arm never reported a verified kernel."
    echo "  It fell back to the interpreted one, so the comparison below is"
    echo "  interpreted-vs-interpreted and the ratio means nothing."
fi

echo
printf 'throughput: %s -> %s MH/s  (%.2fx)\n' "$BASE_RATE" "$JIT_RATE" \
    "$(echo "$JIT_RATE / $BASE_RATE" | bc -l)"
if [ -n "${BASE_EFF:-}" ] && [ -n "${JIT_EFF:-}" ]; then
    printf 'efficiency: %s -> %s MH/W  (%.2fx)\n' "$BASE_EFF" "$JIT_EFF" \
        "$(echo "$JIT_EFF / $BASE_EFF" | bc -l)"
fi
echo
echo "Per-period compile latency is NOT measured by this script. It compiles"
echo "once per period and a bench run may cross none at all."

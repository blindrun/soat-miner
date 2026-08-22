#!/bin/sh
# `--bench` must start, reach steady state and report a hashrate — on every
# algorithm the binary claims to support.
#
# WHY THIS EXISTS
#
# `--bench` is the first command most people type, and until 2026-08-21 nothing
# in the gate ran it. A null-pointer dereference in the stats block crashed it
# on every height while the pool path stayed healthy, because benchmark mode is
# the one path with no `JobSource` at all. It was found by a human trying to
# measure a hashrate, not by CI.
#
# The general shape is worth stating, because it will recur: **removing a
# construct can remove a guard nobody wrote down.** The code that crashed had
# replaced a `dynamic_cast` chain, and that chain had been null-safe by
# accident.
#
# WHAT MAKES THIS TEST ACTUALLY CATCH IT
#
# Two independent assertions, because either alone can pass a broken binary:
#
#   1. The exit status must be TIMEOUT (124), not a signal. Benchmark mode runs
#      until stopped, so a clean finish is impossible and 124 is success. A
#      segfault is 139. Checking only "did it produce output" would pass a
#      binary that printed the banner and then died.
#   2. At least one stats line with a POSITIVE hashrate. The crash happened at
#      the first stats interval, so the banner and the dataset message both
#      appeared before it. Checking only the exit code on a binary that hangs
#      would also read as 124.
#
# Needs a GPU, which is why it is in the device gate and not the offline one.
set -u

BIN=${1:-./soat-miner}
RUN=${2:-45}
INTERVAL=10
fail=0

if [ ! -x "$BIN" ]; then
    echo "  SKIP  $BIN not built"
    exit 0
fi

# The binary is the authority on what it supports, so this test carries no
# second list of algorithms to drift from the registry.
ALGOS=$("$BIN" --list-algos 2>/dev/null)
if [ -z "$ALGOS" ]; then
    echo "  FAIL  $BIN --list-algos printed nothing; cannot choose what to bench"
    exit 1
fi

for algo in $ALGOS; do
    log=$(mktemp)
    timeout "$RUN" "$BIN" --algo "$algo" --bench --plain --interval "$INTERVAL" \
        > "$log" 2>&1
    rc=$?

    # 1. exit status
    case "$rc" in
        124)
            ;;                                  # stopped by timeout: expected
        139|134|136|132)
            echo "  FAIL  $algo: died on a signal (exit $rc) - this is the"
            echo "        crash class --bench was blind to until 2026-08-21"
            fail=$((fail + 1))
            ;;
        *)
            echo "  FAIL  $algo: exited $rc; bench mode should run until stopped"
            fail=$((fail + 1))
            ;;
    esac

    # 2. a stats line with a positive hashrate
    #
    # Efficiency is reported alongside it, never on its own line. A headline
    # hashrate with no wattage is the industry habit and it is what made this
    # project's own comparisons misleading: 235.5 MH/s at 167 W and 235.5 at
    # 450 W are the same number describing very different machines.
    rate=$(grep -oE '"mhs_avg":[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    watts=$(grep -oE '"watts":[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    eff=$(grep -oE '"eff_mh_w":[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    if [ -z "$rate" ]; then
        echo "  FAIL  $algo: no stats line in ${RUN}s - it never reached steady state"
        # Show the binary's own last words. Without this the CI output says
        # only "no stats line" and the reason needs a rerun by hand, which is
        # the same filtered-view problem this test exists to avoid.
        echo "        last output: $(tail -1 "$log")"
        fail=$((fail + 1))
    elif [ "$(echo "$rate > 0" | bc -l 2>/dev/null)" = "1" ]; then
        echo "  ok    $algo  ${rate} MH/s  ${watts:-?} W  ${eff:-?} MH/W"
    else
        echo "  FAIL  $algo: reported ${rate} MH/s, which is not mining"
        fail=$((fail + 1))
    fi
    rm -f "$log"
done

if [ "$fail" -eq 0 ]; then
    echo "bench mode: all algorithms start and report"
fi
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)

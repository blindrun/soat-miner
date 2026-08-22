#!/bin/sh
# Does the async precompile actually run, and land in time?
#
# WHY THIS EXISTS
#
# Nothing else in the suite passes --nvrtc. test_nvrtc_device drives NvrtcJit
# directly and test_bench_runs never enables it, so before this file the whole
# runtime-compilation path could have been deleted and every gate would still
# have gone green. A suite that passes around the change it is meant to cover
# is worse than no suite, because it reports confidence it has not earned.
#
# WHAT WOULD OTHERWISE HIDE
#
# A precompile that lands LATE is indistinguishable from one that works if you
# only check that the code ran: the miner falls back to compiling on demand,
# produces correct shares, and reports a healthy hashrate. The only visible
# difference is ~1600 ms of stall per period instead of ~1 ms. So this asserts
# the STALL, not the code path.
#
# Period changes are forced with --bench-epoch-secs, which advances the bench
# job's height by one each interval. KawPow's period is 3 blocks, so a period
# turns every third bump - without an epoch change, so no DAG rebuild.
set -u

BIN=${1:-./soat-miner}
RUN=${2:-70}
# A height nothing has compiled before, so the CUDA driver's JIT cache
# (~/.nv/ComputeCache) cannot make a cold path look warm.
HEIGHT=${3:-3100000}
checks=0
fail=0

ok()   { checks=$((checks+1)); echo "  ok    $1"; }
bad()  { checks=$((checks+1)); fail=$((fail+1)); echo "  FAIL  $1"; [ $# -gt 1 ] && echo "        $2"; }

if [ ! -x "$BIN" ]; then
    echo "  SKIP  $BIN not built"
    exit 0
fi

log=$(mktemp)
timeout "$RUN" "$BIN" --algo kawpow --nvrtc --bench --plain \
    --bench-height "$HEIGHT" --bench-epoch-secs 8 --interval 60 \
    > "$log" 2>&1
rc=$?

if grep -q "runtime kernel compilation unavailable" "$log"; then
    # Not a failure - the binary is meant to run without the CUDA toolkit. But
    # say which, because a SKIP with no reason is how a permanently-skipping
    # gate hides.
    echo "  SKIP  $(grep -m1 'unavailable' "$log")"
    rm -f "$log"
    exit 0
fi

case "$rc" in
    124) ;;
    *) bad "miner exited $rc rather than being stopped by timeout" \
           "last output: $(tail -1 "$log")" ;;
esac

# 1. The runtime-compiled kernel was actually adopted. Without this the rest
#    would be asserting things about the interpreted path.
if grep -q "runtime-compiled kernel verified against" "$log"; then
    ok "the runtime-compiled kernel was verified and adopted"
else
    bad "--nvrtc never adopted a compiled kernel" \
        "last output: $(tail -1 "$log")"
fi

# 2. It never disagreed with the interpreted kernel.
if grep -q "disagreed with the interpreted" "$log"; then
    bad "the generated kernel disagreed with the interpreted one"
else
    ok "no disagreement with the interpreted kernel"
fi

# 3. At least one period was served from the precompile. This is the check
#    that fails if the background thread is never started, or starts too late
#    every time, or stages the wrong period.
precompiled=$(grep -c "precompiled, mining stalled" "$log")
if [ "$precompiled" -ge 1 ]; then
    ok "$precompiled period(s) served by the precompile"
else
    ondemand=$(grep -c "compiled on demand" "$log")
    bad "no period was served by the precompile" \
        "$ondemand period(s) compiled on demand instead"
fi

# 4. And it landed in time. A precompile that completes after the boundary
#    still works - by falling back - so only the stall distinguishes them.
#    On-demand is ~1600 ms; precompiled is ~1 ms. 50 ms separates them by a
#    wide margin without being brittle about scheduling noise.
worst=$(grep -o "precompiled, mining stalled [0-9.]*" "$log" \
        | awk '{print $4}' | sort -rn | head -1)
if [ -n "$worst" ]; then
    if [ "$(echo "$worst < 50" | bc -l)" = "1" ]; then
        ok "worst precompiled stall ${worst} ms, well under the ~1600 ms on-demand cost"
    else
        bad "worst precompiled stall ${worst} ms" \
            "that is not a precompile landing in time"
    fi
fi

# 5. Exactly one on-demand compile is expected - the first period of the
#    process, which has nothing staged yet. Several more means the precompile
#    is not keeping up, which is a real regression even though shares stay
#    correct.
ondemand=$(grep -c "compiled on demand" "$log")
if [ "$ondemand" -le 3 ]; then
    ok "$ondemand on-demand compile(s), consistent with process start"
else
    bad "$ondemand on-demand compiles" \
        "the precompile is not keeping up with period changes"
fi

rm -f "$log"
echo "$checks check(s), $fail failure(s)"
# A gate that skipped everything also reports success.
if [ "$checks" -eq 0 ]; then
    echo "FAIL: no checks ran"
    exit 1
fi
[ "$fail" -eq 0 ] || exit 1

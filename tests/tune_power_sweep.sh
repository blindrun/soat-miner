#!/bin/bash
# Find the MH/W knee for one algorithm on this card, by sweeping the power cap.
# Restores the original cap explicitly at the end - not via a trap, because a
# trap in a single shell invocation fires at the end of that call and can
# restore before the work is done.
set -u
ALGO=${1:-kawpow}; HEIGHT=${2:-4501000}; RUN=${3:-70}
ORIG=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
echo "original power limit: ${ORIG}W"
printf "%6s %10s %10s %10s\n" "PL(W)" "MH/s" "watts" "MH/W"
for pl in 150 200 250 300 350 400 450; do
    sudo -n nvidia-smi -pl "$pl" >/dev/null 2>&1
    log=$(mktemp)
    timeout "$RUN" ./soat-miner --algo "$ALGO" --bench --bench-height "$HEIGHT" \
        --plain --interval 15 > "$log" 2>&1
    line=$(grep -oE '"mhs_avg":[0-9.]+,"watts":[0-9.]+' "$log" | tail -1)
    mhs=$(echo "$line" | sed 's/.*mhs_avg"://; s/,.*//')
    w=$(echo "$line" | sed 's/.*watts"://')
    if [ -n "$mhs" ] && [ -n "$w" ]; then
        eff=$(echo "scale=4; $mhs / $w" | bc -l)
        printf "%6s %10s %10s %10s\n" "$pl" "$mhs" "$w" "$eff"
    else
        printf "%6s %10s\n" "$pl" "NO DATA"
    fi
    rm -f "$log"
done
sudo -n nvidia-smi -pl "$ORIG" >/dev/null 2>&1
echo "restored to $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"

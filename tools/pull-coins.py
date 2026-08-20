#!/usr/bin/env python3
"""Every mineable coin miningpoolstats knows about, filtered and ranked.

    ./pull-coins.py                    live GPU coins, ranked by revenue per hash
    ./pull-coins.py --type GPU CPU     GPU and CPU, no ASIC
    ./pull-coins.py --by-algo          collapse to one row per algorithm
    ./pull-coins.py --json             the filtered rows, unranked

Companion to pull-pools.py, which answers "which pools mine coin X". This
answers the question before it: which coins are worth having an algorithm for.

## What the ranking means, and what it does not

Daily network revenue is `e24 * pr` - coins emitted in 24h times price. Divide
by network hashrate and you get **revenue per hash per day**, which is the only
cross-coin number here that is honest without knowing how fast our own cards
are.

**It is NOT a cross-ALGORITHM ranking.** Revenue per hash compares two KawPow
coins correctly and says nothing useful about KawPow against Blake3, because a
GPU does wildly different hash counts on each. Turning this into "which
algorithm should we implement next" needs our hashrate on that algorithm,
which is what hashrate.no supplies. Until a key is passed in, treat the
per-algo view as market size and surface area, not profit.

Emission is also not always emission: some chains report `e24` as block reward
only, excluding fees. It is a floor, not a total.

## The -1 trap

miningpoolstats uses -1 for "did not report", not zero. A coin whose pools all
failed to report is not a dead coin. Rows are dropped only when the value is
genuinely absent or zero, never when it is -1.
"""
import argparse, json, re, sys, urllib.request
from collections import defaultdict

UA = "Mozilla/5.0 (compatible; soat-miner-tools/1.0)"
FRONT = "https://miningpoolstats.stream/"
DATA = "https://data.miningpoolstats.stream/data/coins_data.js"


def fetch(url, referer=None):
    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


def coins():
    # The .js carries a cache-buster scraped from the front page. Without it
    # the request returns stale data or nothing at all.
    page = fetch(FRONT)
    m = re.search(r"data/coins_data\.js\?t=(\d+)", page)
    if not m:
        sys.exit("no data token on the front page - the site layout changed")
    return json.loads(fetch(f"{DATA}?t={m.group(1)}", referer=FRONT))["data"]


def num(v):
    """-1 means 'did not report'. Absent and -1 are both None, never zero."""
    return None if v in (None, -1, "") else v


def rev_per_hash(r):
    """Daily network revenue divided by network hashrate."""
    e24, pr, hr = num(r.get("e24")), num(r.get("pr")), num(r.get("hashrate"))
    if not e24 or not pr or not hr:
        return None
    return (e24 * pr) / hr


def si(v, unit="H/s"):
    if v is None:
        return "n/a"
    for suf in ("", "K", "M", "G", "T", "P", "E"):
        if abs(v) < 1000:
            return f"{v:.1f}{suf} {unit}".strip()
        v /= 1000.0
    return f"{v:.1f}Z {unit}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", nargs="+", default=["GPU"],
                    help="hardware classes to keep (GPU CPU ASIC HDD). Default GPU.")
    ap.add_argument("--all", action="store_true",
                    help="keep coins with no pools or no reported hashrate")
    ap.add_argument("--by-algo", action="store_true", help="one row per algorithm")
    ap.add_argument("--json", action="store_true", help="dump the filtered rows")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    want = {t.upper() for t in a.type}
    rows = [r for r in coins() if (r.get("typ") or "").upper() in want]
    if not a.all:
        # A coin with no pool cannot be mined by us today, and one reporting no
        # hashrate cannot be ranked. Both stay reachable behind --all.
        rows = [r for r in rows if (num(r.get("pools")) or 0) > 0
                and (num(r.get("hashrate")) or 0) > 0]

    if a.json:
        json.dump(rows, sys.stdout, indent=2)
        return

    if a.by_algo:
        by = defaultdict(list)
        for r in rows:
            by[r.get("algo") or "?"].append(r)
        print(f"{'algorithm':22} {'coins':>5} {'pools':>6} {'daily USD':>12}  top coin")
        print("-" * 78)
        agg = []
        for algo, rs in by.items():
            pools = sum(num(r.get("pools")) or 0 for r in rs)
            usd = sum((num(r.get("e24")) or 0) * (num(r.get("pr")) or 0) for r in rs)
            top = max(rs, key=lambda r: (num(r.get("e24")) or 0) * (num(r.get("pr")) or 0))
            agg.append((usd, algo, len(rs), pools, top))
        for usd, algo, n, pools, top in sorted(agg, key=lambda t: t[0], reverse=True)[:a.top]:
            print(f"{algo:22} {n:5} {pools:6} {usd:12,.0f}  {top['symbol']}")
        return

    ranked = [(rev_per_hash(r), r) for r in rows]
    ranked = [(v, r) for v, r in ranked if v is not None]
    # Sort on the value only: a tie would otherwise fall through to
    # comparing the dicts, which raises TypeError.
    ranked.sort(key=lambda t: t[0], reverse=True)
    print(f"{'coin':>7} {'algorithm':20} {'pools':>5} {'network':>12} "
          f"{'daily USD':>11} {'rev/hash/day':>14}")
    print("-" * 78)
    for v, r in ranked[:a.top]:
        usd = (num(r.get("e24")) or 0) * (num(r.get("pr")) or 0)
        print(f"{r['symbol']:>7} {(r.get('algo') or '?'):20} {num(r.get('pools')) or 0:5} "
              f"{si(num(r.get('hashrate'))):>12} {usd:11,.0f} {v:14.3e}")
    print(f"\n{len(ranked)} of {len(rows)} rows rankable. "
          "rev/hash compares coins on the SAME algorithm only.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""List the pools mining a coin, from miningpoolstats.stream.

Why this exists: the GUI and the wiki both carried hardcoded pool lists, which
go stale silently and were the reason BC3 shipped knowing about exactly one
pool while fourteen existed.

    ./pull-pools.py bitcoin3
    ./pull-pools.py pearl --json

**It does not return stratum host:port.** miningpoolstats publishes the pool's
website, fee, hashrate and miner count - not how to connect. The stratum
endpoint still has to come from each pool's own page, by hand, once. Treat the
output as "which pools exist and are they alive", not as something that can be
pasted straight into a launcher.

The data URL carries a `t=` cache-buster that has to be scraped from the coin
page first; requesting the .js without it returns stale or nothing.
"""
import json, re, sys, urllib.request

UA = "Mozilla/5.0 (compatible; soat-miner-tools/1.0)"


def fetch(url, referer=None):
    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def pools(coin):
    page_url = f"https://miningpoolstats.stream/{coin}"
    page = fetch(page_url)
    m = re.search(r"data/" + re.escape(coin) + r"\.js\?t=(\d+)", page)
    if not m:
        sys.exit(f"no data token on {page_url} - is '{coin}' the right slug?")
    raw = fetch(f"https://data.miningpoolstats.stream/data/{coin}.js?t={m.group(1)}",
                referer=page_url)
    return json.loads(raw)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    coin = sys.argv[1]
    d = pools(coin)
    if "--json" in sys.argv:
        json.dump(d, sys.stdout, indent=2)
        sys.exit(0)

    rows = d.get("data", [])
    # -1 is miningpoolstats' "pool did not report", not zero. Do not present it
    # as zero: a pool that reports nothing is not the same as a dead pool.
    def num(v):
        return None if v in (None, -1) else v

    # Hashrates span kH/s to tens of EH/s and arrive as floats, so raw printing
    # runs the columns together and is unreadable at Pearl's scale.
    def si(v):
        if v is None:
            return "-"
        for unit in ("", "k", "M", "G", "T", "P", "E"):
            if abs(v) < 1000:
                return f"{v:.1f}{unit}"
            v /= 1000.0
        return f"{v:.1f}Z"

    print(f"{coin}: {len(rows)} pool entries, network {si(d.get('hashrate'))}H/s, "
          f"height {d.get('height')}")
    print(f"{'pool':<24}{'fee%':>6}{'miners':>8}{'hashrate':>12}   url")
    for p in sorted(rows, key=lambda x: -(x.get("hashrate") or 0)):
        hr, mi = num(p.get("hashrate")), num(p.get("miners"))
        name = p.get("pool_id") or "(unnamed)"
        print(f"{name:<24}{str(p.get('fee', '?')):>6}"
              f"{('-' if mi is None else mi):>8}{si(hr):>11}H   "
              f"{p.get('url', '')}")

#!/usr/bin/env python3
"""Does this endpoint answer, and which stratum flavour does it speak?

    ./probe-stratum.py rvn.2miners.com:6060 ravencoin.herominers.com:1140

Sends mining.subscribe, reads what comes back, disconnects. No authorize, no
address, nothing submitted - this is reconnaissance, not mining, and it is the
evidence the branch model demands before a launcher ships for a pool.

`pull-pools.py` answers "which pools exist and are they alive". It does NOT
return a stratum host:port; that comes from each pool's own page, by hand,
once. This tool is what turns a guessed host:port into a proven one.

What the probe found on Ravencoin, 2026-08-20
---------------------------------------------
Three different shapes from live pools, and the flavour is a property of the
ENDPOINT, not of the pool:

    ravencoin.herominers.com:1140  {"result":[null,"36c3"]}
    ravencoin.herominers.com:1150  {"result":[["mining.notify","<uuid>",
                                    "EthereumStratum/1.0.0"],"adb4"]}

Same pool, two ports, two protocols, reproducibly. Guessing the flavour from
the pool name would have shipped one of those two broken.

    rvn.2miners.com:6060           {"result":["06fdcdfd","bf"]}
    stratum.ravenminer.com:3838    {"result":[null,"43"]}
    rvn.suprnova.cc:6273           {"result":[null,"27ff92db",4]}
                                   then mining.set_difficulty unprompted

suprnova's three-element reply is the Bitcoin shape (extranonce1 plus an
extranonce2 size), not the Ethereum one. 2miners' second element is 1 byte and
its meaning is UNRESOLVED - it cannot be settled without seeing a notify, which
needs an authorized session, which needs a real payout address. Recorded as
open rather than assumed.

The extranonce is a PREFIX of the 8-byte nonce, so its width decides how much
of the nonce space is ours: 4 hex characters leaves 48 bits, 8 leaves 32. And
the width is counted in HEX CHARACTERS, not bytes - kawpowminer's own field is
called extraNonceSizeBytes and its author left a comment saying the name is
wrong. Reading it as bytes halves or doubles the prefix and every share is
rejected as out of range.
"""
import json
import socket
import sys
import time

TIMEOUT = 8.0


def probe(host, port, timeout=TIMEOUT):
    out = {"host": host, "port": port}
    try:
        s = socket.create_connection((host, port), timeout=timeout)
    except Exception as e:
        out["error"] = f"connect: {e.__class__.__name__}: {e}"
        return out
    try:
        s.settimeout(timeout)
        req = {"id": 1, "method": "mining.subscribe",
               "params": ["soat-miner/0.3.0", "EthereumStratum/1.0.0"]}
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        deadline = time.time() + timeout
        # Several pools volunteer a set_difficulty straight after the reply, so
        # keep reading past the first line rather than stopping at it.
        while time.time() < deadline and buf.count(b"\n") < 4:
            try:
                d = s.recv(4096)
            except socket.timeout:
                break
            if not d:
                break
            buf += d
        out["lines"] = [l for l in buf.decode("utf-8", "replace").split("\n")
                        if l.strip()]
        out["flavour"] = classify(out["lines"])
    except Exception as e:
        out["error"] = f"{e.__class__.__name__}: {e}"
    finally:
        s.close()
    return out


def classify(lines):
    """Name the shape, or say it is unknown. Never guess a working one."""
    for line in lines:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("id") != 1:
            continue
        r = m.get("result")
        if not isinstance(r, list) or not r:
            return "no usable subscribe result"
        if isinstance(r[0], list):
            return "ethereum-stratum (announced)"
        if len(r) >= 3 and isinstance(r[2], int):
            return "bitcoin-shape (extranonce1 + en2 size)"
        if r[0] is None and len(r) >= 2:
            return f"ethereum-stratum (bare), extranonce {r[1]!r}"
        return f"unresolved 2-element reply {r!r}"
    return "no reply"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for spec in sys.argv[1:]:
        host, _, port = spec.rpartition(":")
        r = probe(host, int(port))
        status = r.get("error") or r.get("flavour", "?")
        print(f"{spec:38s} {status}")
        for line in r.get("lines", []):
            print(f"    {line[:200]}")

#!/usr/bin/env python3
"""A mock KawPow stratum pool, to prove the miner speaks what real pools expect.

Every wire shape below is transcribed from replies captured off LIVE pools by
tools/probe-stratum.py, not from documentation:

  * subscribe reply     {"result":[null,"<extranonce>"]}
                        HeroMiners on its documented stratum port. Element 0 is
                        a bare null; on port 1150 the SAME pool announces a
                        subscription array instead, which is why the miner
                        detects the flavour rather than being told it.
  * mining.notify       [jobId, headerHash, seedHash, target, cleanJobs,
                         height, bits]
                        The height is load bearing: KawPow's random program
                        changes every 3 blocks, so a miner that ignores it
                        hashes a wrong program at full speed forever.
  * mining.submit       [login, jobId, "0x"+nonce, "0x"+header, "0x"+mix]
                        Five params. The mix hash is the one a nonce cannot be
                        checked without, and no other algorithm in this
                        repository produces one.

The load-bearing assertion is the EXTRANONCE PREFIX. The pool assigns the top
bytes of the 8-byte nonce, and every miner on the pool gets a different one. A
miner that ignores it submits nonces outside the range it was given; the pool
rejects every share as out of range, and nothing in the miner's own log
explains why, because the hash it computed was perfectly correct.

Run standalone:  python3 tests/kawpow_mock.py --port 3333
"""

import argparse
import json
import socket
import threading


class KawpowMock:
    """Serves one connection, records what the miner sent."""

    # Height 100 is deliberate: epoch 0, so the seed hash is 32 zero bytes and
    # the dataset is the smallest real one at about 1 GB. A mainnet height
    # would mean a 6 GB dataset and a minute of build time per run.
    HEIGHT = 100
    SEED = "00" * 32
    HEADER = "8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7"
    # Deliberately easy, so a share turns up in seconds rather than years.
    TARGET = "0000ffff00000000000000000000000000000000000000000000000000000000"
    BITS = "1e0fffff"

    def __init__(self, port=0, extranonce="a1b2", flavour="stratum"):
        self.extranonce = extranonce
        self.flavour = flavour
        self.submits = []
        self.authorized = None
        self.subscribed = False
        self.errors = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def _subscribe_result(self):
        if self.flavour == "nicehash":
            return [["mining.notify", "mock-session", "EthereumStratum/1.0.0"],
                    self.extranonce]
        return [None, self.extranonce]

    def _notify(self):
        return {"id": None, "method": "mining.notify",
                "params": [self.HEIGHT and "job-1", self.HEADER, self.SEED,
                           self.TARGET, True, self.HEIGHT, self.BITS]}

    def _serve(self):
        try:
            self._srv.settimeout(20)
            conn, _ = self._srv.accept()
        except OSError:
            return
        conn.settimeout(1.0)
        buf = b""
        with conn:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._handle(conn, line.decode("utf-8", "replace"))

    def _send(self, conn, obj):
        try:
            conn.sendall((json.dumps(obj) + "\n").encode())
        except OSError:
            pass

    def _handle(self, conn, line):
        try:
            msg = json.loads(line)
        except ValueError:
            self.errors.append("not JSON: " + line[:120])
            return
        method = msg.get("method")

        if method == "mining.subscribe":
            self.subscribed = True
            self._send(conn, {"id": msg.get("id"), "error": None,
                              "result": self._subscribe_result()})
        elif method == "mining.authorize":
            self.authorized = msg.get("params", [None])[0]
            self._send(conn, {"id": msg.get("id"), "error": None, "result": True})
            self._send(conn, self._notify())
        elif method == "mining.submit":
            self.submits.append(msg.get("params", []))
            self._send(conn, {"id": msg.get("id"), "error": None, "result": True})
        else:
            self._send(conn, {"id": msg.get("id"), "error": None, "result": True})


def base58check(version, payload20):
    """Build a valid address so no address-shaped literal lives in the tree."""
    import hashlib
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    body = bytes([version]) + payload20
    body += hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4]
    n = int.from_bytes(body, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = alphabet[r] + out
    return "1" * (len(body) - len(body.lstrip(b"\0"))) + out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--flavour", default="stratum",
                    choices=["stratum", "nicehash"])
    a = ap.parse_args()
    m = KawpowMock(a.port, flavour=a.flavour).start()
    print(f"mock KawPow pool on 127.0.0.1:{m.port} ({a.flavour})")
    print(f"a usable test address: {base58check(60, bytes([0x11]) * 20)}")
    try:
        m._thread.join()
    except KeyboardInterrupt:
        m.stop()

#!/usr/bin/env python3
"""
A mock Lithos stratum server, used to prove soat-miner speaks the protocol the
real Lithos client expects.

Every wire shape below is transcribed from the Lithos reference client's own
source (github.com/Lithos-Protocol/Lithos-Client), not from documentation:

  * subscribe reply          Response.subscribe()  -> [null, extraNonce1, en2Size]
                             note element 0 is a bare null, where a conventional
                             Ergo pool sends a subscriptions array.
  * mining.notify params     BlockTemplate.getJobParams()
                             [jobId, height, msgHex, "", "", versionHex,
                              tau (DECIMAL), "", true]
                             params[6] is tau, the SHARE target - not the block
                             target b, and not a difficulty.
  * mining.submit params     MiningStratumServer MiningConnectionState
                             [worker, jobId, extraNonce2, nTime, nonceHex]
  * share validation         LithosJobManager.validateShare()
                             extraNonce1 = bytes(params[4])[0:4]
                             extraNonce2 = bytes(params[2])
                             reject 20 unless len(extraNonce2) == en2Size
                             nonce = extraNonce1 ++ extraNonce2
                             reject 20 unless len(nonce) == 8
  * extranonce rotation      Announcement.extraNonce1()
                             -> mining.set_extranonce [en1, en2Size]

The load-bearing assertion is RECONSTRUCTION: the server validates
concat(nonce[0:4], extraNonce2), NOT the nonce field it was sent. If the miner
slices extraNonce2 wrongly, the server silently PoW-checks a different nonce
than the one the GPU actually solved, and every share dies as "low difficulty"
with nothing in the miner's log to explain it.

Run standalone:  python3 tests/lithos_mock.py --port 4444
"""

import argparse
import json
import socket
import sys
import threading
import time

# Lithos: conf/application.conf stratum.extraNonce1Size = 4, and the
# ExtraNoncePlaceholder in LithosJobManager is 8 bytes, so en2Size is 4.
EN1_SIZE = 4
EN2_SIZE = 4

# A real Ergo header digest and a height with a known N-table entry.
MSG_HEX = "a4c9f0e2b7d81536ac0f5e9d3b2718aa64f3c05d9e8b1a2736d4c8ef05193b7a"
HEIGHT = 1851444
VERSION = 2

# tau is a 256-bit TARGET. A very large one makes almost every nonce a share,
# so the test finds work in seconds instead of waiting on real difficulty.
EASY_TAU = str((1 << 252) - 1)


class Failure(Exception):
    pass


class LithosMock:
    def __init__(self, port, shares_before_rotate=3, total_shares=6,
                 send_zero_target=False):
        self.port = port
        self.shares_before_rotate = shares_before_rotate
        self.total_shares = total_shares
        self.send_zero_target = send_zero_target

        self.en1 = "aabbccdd"          # first assignment
        self.en1_rotated = "11223344"  # after mining.set_extranonce
        self.current_en1 = self.en1

        self.accepted = 0
        self.rotated = False
        self.saw_post_rotation_share = False
        self.stale_after_rotation = 0
        self.rotate_time = 0.0
        self.zero_sent_time = 0.0
        # Shares already in flight when the rotation is announced are
        # legitimate; the miner only has to switch over promptly.
        self.rotation_grace = 10.0
        self.errors = []
        self.done = threading.Event()
        self.sock = None

    # ---------------------------------------------------------------- helpers
    def send(self, conn, obj):
        conn.sendall((json.dumps(obj) + "\n").encode())

    def notify(self, conn, job_id, tau):
        self.send(conn, {
            "id": None,
            "method": "mining.notify",
            "params": [
                job_id,
                HEIGHT,
                MSG_HEX,
                "",
                "",
                format(VERSION, "x"),   # Integer.toHexString(version)
                tau,                    # DECIMAL string
                "",
                True,
            ],
        })

    def fail(self, msg):
        self.errors.append(msg)

    # ------------------------------------------------------------ share check
    def check_submit(self, params):
        """Replicates LithosJobManager.validateShare's parsing exactly."""
        if len(params) != 5:
            raise Failure(
                "mining.submit sent %d params, Lithos destructures exactly 5 "
                "[worker, jobId, extraNonce2, nTime, nonce]; anything else "
                "throws in its parse handler and the share is dropped before "
                "validation" % len(params))

        _worker, _job_id, en2_hex, _ntime, nonce_hex = params

        try:
            nonce_bytes = bytes.fromhex(nonce_hex)
            en2_bytes = bytes.fromhex(en2_hex)
        except ValueError as exc:
            raise Failure("params are not valid hex: %s" % exc)

        # Lithos: Arrays.copyOfRange(Hex.decode(en1Hex), 0, 4)
        if len(nonce_bytes) < EN1_SIZE:
            raise Failure(
                "nonce field is %d bytes; Lithos slices [0:4] and would throw"
                % len(nonce_bytes))
        en1_bytes = nonce_bytes[:EN1_SIZE]

        if len(en2_bytes) != EN2_SIZE:
            raise Failure(
                "extraNonce2 is %d bytes, Lithos rejects code 20 unless it is "
                "exactly %d" % (len(en2_bytes), EN2_SIZE))

        reconstructed = en1_bytes + en2_bytes
        if len(reconstructed) != 8:
            raise Failure("reconstructed nonce is %d bytes, must be 8"
                          % len(reconstructed))

        # THE load-bearing check.
        if reconstructed != nonce_bytes:
            raise Failure(
                "reconstruction mismatch: Lithos would PoW-check %s but the "
                "GPU solved %s. Shares would all die as 'low difficulty'."
                % (reconstructed.hex(), nonce_bytes.hex()))

        # The prefix is returned rather than asserted here. Before a rotation
        # it must match; after one, shares already in flight from the previous
        # subspace are legitimate and are tolerated for a grace period - the
        # requirement is that the miner switches over, not that it switches
        # instantly.
        return en1_bytes.hex()

    # ------------------------------------------------------------------ serve
    def serve(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen(1)
        self.sock.settimeout(120)

        try:
            conn, _addr = self.sock.accept()
        except socket.timeout:
            self.fail("miner never connected")
            self.done.set()
            return

        # A pool goes quiet between jobs; never let the mock itself time out
        # the connection and then blame the miner for it.
        conn.settimeout(None)
        buf = b""

        try:
            while not self.done.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        self.handle(conn, line.decode())
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self.done.set()

    def handle(self, conn, line):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self.fail("miner sent unparseable JSON: %r" % line)
            return

        method = msg.get("method")
        rid = msg.get("id")

        if method == "mining.subscribe":
            # Lithos Response.subscribe: element 0 is null, NOT an array.
            self.send(conn, {"id": rid,
                             "result": [None, self.en1, EN2_SIZE],
                             "error": None})
            self.send(conn, {"id": None, "method": "mining.set_difficulty",
                             "params": [1]})
            self.notify(conn, "job-1", EASY_TAU)
            return

        if method == "mining.authorize":
            # Lithos always accepts; the address is not a payout identifier.
            self.send(conn, {"id": rid, "result": True, "error": None})
            return

        if method == "mining.submit":
            params = msg.get("params") or []
            try:
                prefix = self.check_submit(params)
            except Failure as exc:
                self.fail(str(exc))
                self.send(conn, {"id": rid, "result": None,
                                 "error": "32: Low difficulty share"})
                self.done.set()
                return

            self.accepted += 1
            self.send(conn, {"id": rid, "result": True, "error": None})

            if not self.rotated:
                if prefix != self.current_en1:
                    self.fail("nonce prefix is %s but the assigned "
                              "extraNonce1 is %s" % (prefix, self.current_en1))
                    self.done.set()
                    return
                if self.accepted >= self.shares_before_rotate:
                    # Announcement.extraNonce1 -> mining.set_extranonce
                    self.rotated = True
                    self.rotate_time = time.time()
                    self.current_en1 = self.en1_rotated
                    self.send(conn, {"id": None,
                                     "method": "mining.set_extranonce",
                                     "params": [self.en1_rotated, EN2_SIZE]})
                    self.notify(conn, "job-2", EASY_TAU)
                return

            # ---- after the rotation ----
            if prefix == self.current_en1:
                if not self.saw_post_rotation_share:
                    self.saw_post_rotation_share = True
                    if self.send_zero_target:
                        # BlockTemplate's 4-arg constructor leaves tau at 0.
                        self.notify(conn, "job-zero", "0")
                        self.zero_sent_time = time.time()
                    else:
                        self.done.set()
                return

            # Old prefix: in-flight work from the subspace we just took back.
            self.stale_after_rotation += 1
            if time.time() - self.rotate_time > self.rotation_grace:
                self.fail(
                    "miner still using the old extranonce %s more than %.0fs "
                    "(%d shares) after mining.set_extranonce assigned %s - it "
                    "is not adopting the new prefix"
                    % (prefix, self.rotation_grace,
                       self.stale_after_rotation, self.current_en1))
                self.done.set()
            return

        # mining.extranonce.subscribe and friends: answer, do not crash.
        if rid is not None:
            self.send(conn, {"id": rid, "result": True, "error": None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--shares", type=int, default=6)
    ap.add_argument("--rotate-after", type=int, default=3)
    ap.add_argument("--zero-target", action="store_true")
    args = ap.parse_args()

    mock = LithosMock(args.port, args.rotate_after, args.shares,
                      args.zero_target)
    mock.serve()

    for e in mock.errors:
        print("FAIL: %s" % e)
    print("accepted=%d rotated=%s post_rotation_share=%s"
          % (mock.accepted, mock.rotated, mock.saw_post_rotation_share))
    return 1 if mock.errors else 0


if __name__ == "__main__":
    sys.exit(main())

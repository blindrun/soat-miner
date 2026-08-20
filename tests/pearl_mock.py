#!/usr/bin/env python3
"""A mock Pearl pool: enough of the gateway's stratum for the real miner.

Transcribed from the contract documented in src/core/pearl_pool.h, not from a
capture, so it stays readable and so a change to the contract shows up here as
a test failure rather than as a mystery on a live pool.

Pearl's stratum is NOT the array-params stratum the other algorithms use:

    -> {"id":1,"method":"mining.authorize","params":{"wallet":"prl1...",
                                                     "worker":"rig"}}
    <- {"id":1,"result":true,"error":null}
    <- {"id":null,"method":"mining.notify",
        "params":{"job_id":"...","header":"<152 hex>","target":"<64 hex BE>",
                  "height":N,"cert_version":3}}
    -> {"id":2,"method":"mining.submit",
        "params":{"job_id":"...","plain_proof":"<base64 bincode>"}}
    <- {"id":2,"result":true,"error":null}

The target is scaled by the miner before it is used as a bound - see
MiningConfig::penalizedTarget - so a target here must leave room for a
multiply by tileSize * (k/rank) * 128 without overflowing 256 bits.
"""
import json
import socket
import threading
import time

# 2^228. Scaled by 2^19 that is a 2^247 bound, so about one candidate in 512
# wins and a share arrives in well under a second on any real card. Small
# enough that the penalty multiply cannot overflow.
EASY_TARGET = "00000010" + "00" * 28

# A synthetic 76-byte header. Never a captured one: a real header would tie
# this fixture to a chain height and to somebody's payout address.
def header_hex(salt=0):
    return "".join("%02x" % ((i * 7 + 3 + salt) & 0xFF) for i in range(76))


class PearlMock:
    """One connection, scripted. Every field a test asserts on is recorded."""

    def __init__(self, port, total_shares=1, reject_from=None,
                 renotify_before_reply=False, push_burst=0,
                 bad_notify_first=False):
        self.port = port
        self.total_shares = total_shares
        # 0-based index of the first submit to answer with an error.
        self.reject_from = reject_from
        # Push a fresh job BEFORE replying to each submit, so the notify is
        # already queued on the socket when the miner next calls
        # drainAvailable(). That makes the stale-job guard deterministic:
        # after the reply, the old job is provably superseded, so a submit
        # naming it again is a real failure rather than a race.
        self.renotify_before_reply = renotify_before_reply
        # Number of unrelated pushes to send BEFORE the reply to a submit.
        # submit() only reads 12 lines looking for its own id.
        self.push_burst = push_burst
        # Send one malformed notify first, which the miner must ignore rather
        # than mine against.
        self.bad_notify_first = bad_notify_first

        self.authorize = None
        self.submits = []          # each: the decoded mining.submit message
        # Contract violations the pool would answer with an error code. Kept
        # separate from `errors` so the mock's own discrimination is testable:
        # feeding it a bad message must record exactly one of these.
        self.violations = []
        self.credited = 0          # shares the POOL counted, to compare against
                                   # the counter the miner reports
        self.notified = []         # each: job_id, newest last
        self.newest_at_submit = []  # the newest job when each submit arrived
        self.errors = []
        self.done = threading.Event()
        self._n = 0

    # -- wire helpers ----------------------------------------------------
    def _send(self, conn, obj):
        conn.sendall((json.dumps(obj) + "\n").encode())

    def _notify(self, conn):
        self._n += 1
        jid = "mock-%08x_2097152" % self._n
        self.notified.append(jid)
        self._send(conn, {"id": None, "method": "mining.notify",
                          "params": {"job_id": jid, "header": header_hex(self._n),
                                     "target": EASY_TARGET, "height": 1000 + self._n,
                                     "cert_version": 3}})
        return jid

    # -- the server ------------------------------------------------------
    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(1)
        srv.settimeout(60)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            self.errors.append("the miner never connected")
            self.done.set()
            srv.close()
            return
        conn.settimeout(180)
        buf = b""
        try:
            while not self.done.is_set():
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    self.errors.append("timed out waiting for the miner")
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        self.errors.append("not JSON on the wire: %r" % line[:200])
                        continue
                    self._handle(conn, msg)
                    if len(self.submits) >= self.total_shares:
                        # Let the miner log the last verdict before the socket
                        # disappears underneath it.
                        time.sleep(1.5)
                        self.done.set()
                        break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.done.set()
            try:
                conn.close()
            except OSError:
                pass
            srv.close()

    # -- contract checks -------------------------------------------------
    #
    # These are the three failures pearl_pool.h documents as looking like
    # something else at a real pool, so the mock answers them the way the pool
    # does instead of quietly tolerating them.
    def _check_params(self, conn, msg, want):
        p = msg.get("params")
        if not isinstance(p, dict):
            self.violations.append(
                "%s params are %s, not an object - a real gateway answers "
                "code 20 and the miner sees 'Invalid params'"
                % (msg.get("method"), type(p).__name__))
            self._send(conn, {"id": msg.get("id"), "result": None,
                              "error": {"code": 20, "message": "Invalid params"}})
            return None
        for k in want:
            if k not in p:
                self.violations.append("%s params are missing %r (have %s)"
                                       % (msg.get("method"), k, sorted(p)))
                self._send(conn, {"id": msg.get("id"), "result": None,
                                  "error": {"code": 24, "message": "Wallet is missing"}})
                return None
        return p

    def _handle(self, conn, msg):
        method = msg.get("method", "")
        if method == "mining.authorize":
            self.authorize = msg
            p = self._check_params(conn, msg, ("wallet",))
            if p is None:
                return
            w = p.get("wallet", "")
            if not isinstance(w, str) or not w.startswith("prl1"):
                # The real gateway's exact wording, and the reason a user with
                # an Ergo address sees "Invalid Pearl address".
                self.violations.append("wallet %r is not a Pearl address" % w[:12])
                self._send(conn, {"id": msg.get("id"), "result": None,
                                  "error": {"code": 24, "message": "Invalid Pearl address"}})
                return
            self._send(conn, {"id": msg.get("id"), "result": True, "error": None})
            if self.bad_notify_first:
                # Right shape, unusable contents: a short header. The miner
                # must refuse it, not mine 76 bytes of whatever follows.
                self._send(conn, {"id": None, "method": "mining.notify",
                                  "params": {"job_id": "mock-bad", "header": "dead",
                                             "target": EASY_TARGET, "height": 1,
                                             "cert_version": 3}})
            self._notify(conn)
            return

        if method == "mining.submit":
            p = self._check_params(conn, msg, ("job_id", "plain_proof"))
            if p is None:
                return
            import base64
            try:
                raw = base64.b64decode(p["plain_proof"], validate=True)
            except Exception:                        # noqa: BLE001
                raw = b""
            if len(raw) < 32:
                self.violations.append(
                    "plain_proof decodes to %d bytes; the pool reads m/n/k/rank "
                    "out of the first 32" % len(raw))
                self._send(conn, {"id": msg.get("id"), "result": None,
                                  "error": {"code": 23, "message": "Invalid proof"}})
                return
            idx = len(self.submits)
            self.submits.append(msg)
            # What was newest when this submit was sent, recorded before any
            # push below changes it.
            self.newest_at_submit.append(self.notified[-1])
            for _ in range(self.push_burst):
                self._notify(conn)
            if self.renotify_before_reply:
                self._notify(conn)
            accepted = self.reject_from is None or idx < self.reject_from
            if accepted:
                self.credited += 1
            reply = {"id": msg.get("id"), "result": accepted}
            reply["error"] = None if accepted else \
                {"code": 23, "message": "mock rejection, share not counted"}
            self._send(conn, reply)
            return

        # Anything else is a contract surprise worth failing on.
        self.errors.append("unexpected method %r" % method)

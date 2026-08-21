#!/usr/bin/env python3
"""End-to-end Pearl pool test: the real miner binary against a mock gateway.

WHY. Pearl had nine device gates, byte-identical to the CUDA reference on two
vendors, and until this test nothing had ever run the binary against a server.
That is the shape of coverage that hid a share-counting bug in two other
algorithms on this fleet for weeks: vectors passed, device tests passed, and
the first mock-pool run found it.

`--bench` cannot substitute. It opens no socket, so authorize, notify parsing,
submit encoding, the accept/reject verdict and the counters are all unexercised
by it.

Assertions are made against --pearl-transcript, not stdout. The transcript is
JSONL and its submit record already carries header_b3, the m/n/k/rank read back
OUT OF THE SERIALISED PROOF rather than from the miner's own variables, and
digest_be against bound_be - so this checks the arithmetic the pool is about to
redo, not merely that a submit happened. Counters are still compared against
what the pool actually credited, because a transcript cannot see a counter
wired to nothing.

Usage: python3 tests/test_pearl_pool_e2e.py [path-to-soat-miner]
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pearl_mock import PearlMock, EASY_TARGET  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 180

# Synthetic, and the right shape for the local wallet check in run.cpp. Never a
# real address in a fixture: this file is committed.
WALLET = "prl1qq" + "q" * 56

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print("   FAIL: %s" % msg)


def ok(msg):
    print("   ok: %s" % msg)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# --------------------------------------------------------------- phase 0
def prove_mock_discriminates():
    """A mock that cannot fail proves nothing. Feed it each bad message and
    confirm it records exactly one violation, and none for a good one."""
    print("== phase 0: the mock discriminates (no GPU needed) ==")
    cases = [
        ("array params",
         {"id": 1, "method": "mining.authorize", "params": [WALLET, "x"]}, 1),
        ("a `login` field instead of `wallet`",
         {"id": 1, "method": "mining.authorize",
          "params": {"login": WALLET}}, 1),
        ("an Ergo address",
         {"id": 1, "method": "mining.authorize",
          "params": {"wallet": "9f4QF8AD1nQ3nJahQVkMj8hFSVVzVom77b52JU7EW71Zexg6N8v"}}, 1),
        ("a proof that does not decode",
         {"id": 2, "method": "mining.submit",
          "params": {"job_id": "j", "plain_proof": "not base64!!"}}, 1),
        ("a well-formed authorize", 
         {"id": 1, "method": "mining.authorize",
          "params": {"wallet": WALLET, "worker": "rig"}}, 0),
    ]
    for name, msg, want in cases:
        port = free_port()
        m = PearlMock(port, total_shares=99)
        t = threading.Thread(target=m.serve, daemon=True)
        t.start()
        time.sleep(0.2)
        c = socket.create_connection(("127.0.0.1", port), timeout=10)
        c.sendall((json.dumps(msg) + "\n").encode())
        time.sleep(0.4)
        m.done.set()
        c.close()
        t.join(timeout=5)
        got = len(m.violations)
        if got != want:
            fail("mock recorded %d violation(s) for %s, expected %d: %s"
                 % (got, name, want, m.violations))
        else:
            ok("%s -> %d violation(s)" % (name, got))


# --------------------------------------------------------------- running
def pick_binary(argv):
    """Ask the binary which algorithms it has, rather than guessing from its
    name. This keeps working unchanged the day the Vulkan registry line is
    uncommented and soat-miner-vk grows pearl-pow too."""
    if len(argv) > 1:
        cands = [argv[1]]
    else:
        cands = [os.path.join(ROOT, n) for n in ("soat-miner", "soat-miner-vk")]
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            out = subprocess.run([c, "--list-algos"], capture_output=True,
                                 text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "pearl-pow" in out.split():
            return c
    return None


def run_case(binary, mock, transcript, extra_args=()):
    # DELETE THE TRANSCRIPT FIRST, AND REQUIRE A NEW ONE.
    #
    # Without this a run reads whatever the last one left at the same path.
    # That is not hypothetical: a Vulkan binary that silently ignored
    # --pearl-transcript produced no file, the reader found the previous CUDA
    # run's, and every transcript assertion passed while testing a backend that
    # had not written a byte. A test passing against stale state is the same
    # failure as a mutation test passing against a stale binary.
    if os.path.exists(transcript):
        os.remove(transcript)
    t = threading.Thread(target=mock.serve, daemon=True)
    t.start()
    time.sleep(0.3)
    proc = subprocess.Popen(
        [binary, "--algo", "pearl-pow", "--pool", "127.0.0.1:%d" % mock.port,
         "--wallet", WALLET, "--worker", "mocktest",
         "--pearl-transcript", transcript, "--plain", "--interval", "1",
         *extra_args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    lines = []

    def drain():
        for ln in proc.stdout:
            lines.append(ln.rstrip())

    d = threading.Thread(target=drain, daemon=True)
    d.start()

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if mock.done.wait(timeout=0.5):
            break
        if proc.poll() is not None:
            break
    # One more telemetry interval, so a counter printed after the last verdict
    # is in the output we assert on.
    time.sleep(2.5)
    mock.done.set()
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    d.join(timeout=5)

    events = []
    if not os.path.exists(transcript):
        fail("the miner wrote no transcript at %s - does this binary support "
             "--pearl-transcript?" % transcript)
    if os.path.exists(transcript):
        for ln in open(transcript):
            ln = ln.strip()
            if ln:
                try:
                    events.append(json.loads(ln))
                except ValueError:
                    fail("transcript line is not JSON: %r" % ln[:120])
    return events, "\n".join(lines)


def reported_accepted(text):
    """The last accepted/rejected pair the miner printed."""
    acc = rej = None
    for m in re.finditer(r'"accepted":(\d+),"rejected":(\d+)', text):
        acc, rej = int(m.group(1)), int(m.group(2))
    if acc is None:
        for m in re.finditer(r'accepted[ =:]+(\d+).{0,20}?rejected[ =:]+(\d+)',
                             text, re.I):
            acc, rej = int(m.group(1)), int(m.group(2))
    return acc, rej


def of(events, ev):
    return [e for e in events if e.get("event") == ev]


# ---------------------------------------------------------------- phases
def phase_accept(binary, tmp):
    print("== phase 1: a share is submitted, verified and counted ==")
    m = PearlMock(free_port(), total_shares=1)
    tr = os.path.join(tmp, "accept.jsonl")
    ev, out = run_case(binary, m, tr)

    for e in m.errors:
        fail("mock: " + e)
    for v in m.violations:
        fail("protocol: " + v)

    if m.authorize is None:
        fail("the miner never authorized")
        return
    ok("authorized with params %s" % sorted(m.authorize.get("params", {})))

    if not m.submits:
        fail("no share submitted in %ds against a 2^247 bound; miner said:\n%s"
             % (TIMEOUT, out[-2500:]))
        return
    ok("%d share(s) submitted" % len(m.submits))

    sp = m.submits[0]["params"]
    if sp["job_id"] != m.notified[-1] and sp["job_id"] not in m.notified:
        fail("submitted job_id %r was never notified" % sp["job_id"])
    else:
        ok("submit names a job the pool actually sent")

    subs = of(ev, "mining.submit")
    nots = of(ev, "mining.notify")
    if not subs:
        fail("the transcript recorded no mining.submit")
        return
    if not nots:
        fail("the transcript recorded no mining.notify")
    s = subs[0]

    # The arithmetic the pool is about to redo.
    if s.get("digest_within_bound") is not True:
        fail("digest_within_bound=%r: we submitted a proof above our own bound "
             "(digest_be=%s bound_be=%s)"
             % (s.get("digest_within_bound"), s.get("digest_be"), s.get("bound_be")))
    else:
        ok("digest is within the penalised bound, read back from the proof")

    # Shape, read out of the serialised proof rather than our own variables.
    if not (s.get("k") and s.get("rank")):
        fail("proof shape did not decode: m=%r n=%r k=%r rank=%r"
             % (s.get("m"), s.get("n"), s.get("k"), s.get("rank")))
    else:
        ok("proof decodes to m=%s n=%s k=%s rank=%s"
           % (s.get("m"), s.get("n"), s.get("k"), s.get("rank")))
        # The shape identifies which backend really ran: the CUDA path tunes
        # across several and the Vulkan one is fixed. A run that looked like it
        # exercised Vulkan but reported CUDA's shape is what made this line
        # worth printing rather than only checking.

    # The proof was mined against the job it names.
    hb = {n.get("job_id"): n.get("header_b3") for n in nots}
    if s.get("job_id") in hb and hb[s["job_id"]] != s.get("header_b3"):
        fail("submit's header_b3 %s does not match the notify of job %s (%s) - "
             "the proof was mined against a different header"
             % (s.get("header_b3"), s.get("job_id"), hb[s["job_id"]]))
    else:
        ok("header fingerprint ties the proof to its notify")

    reps = of(ev, "mining.reply")
    if not reps or reps[0].get("accepted") is not True:
        fail("the pool accepted, the transcript says %r"
             % (reps[0] if reps else None))
    else:
        ok("the accept is recorded as an accept")

    acc, rej = reported_accepted(out)
    if acc is None:
        fail("the miner never printed a share counter:\n%s" % out[-1500:])
    elif acc != m.credited:
        fail("the miner reports accepted=%s, the pool credited %d"
             % (acc, m.credited))
    else:
        ok("reported accepted=%d matches the %d the pool credited"
           % (acc, m.credited))

    # The transcript is committed evidence in a bug report. It must not carry
    # the payout address, and pearlTranscriptRedact must have run.
    blob = open(tr).read()
    if WALLET in blob:
        fail("the wallet appears verbatim in the transcript")
    else:
        ok("no wallet in the transcript")


def phase_reject(binary, tmp):
    print("== phase 2: a rejection is not counted as an accept ==")
    m = PearlMock(free_port(), total_shares=1, reject_from=0)
    tr = os.path.join(tmp, "reject.jsonl")
    ev, out = run_case(binary, m, tr)
    if not m.submits:
        fail("no share submitted, so the rejection path was never reached")
        return
    reps = of(ev, "mining.reply")
    if not reps or reps[0].get("accepted") is not False:
        fail("a rejected share was recorded as %r" % (reps[0] if reps else None))
    else:
        ok("recorded as rejected, reason %r" % reps[0].get("reason"))
    acc, rej = reported_accepted(out)
    if acc is None:
        fail("no counter printed")
    elif acc != 0:
        fail("a rejected share incremented accepted to %s" % acc)
    else:
        ok("accepted stayed 0 (pool credited %d), rejected=%s" % (m.credited, rej))


def phase_stale(binary, tmp):
    print("== phase 3: work replaced mid-flight is not submitted stale ==")
    # A fresh job is pushed BEFORE the reply to each submit, so by the time the
    # miner has its verdict the notify is already queued on its socket. Its
    # next submit must therefore name the new job: submit() drains what has
    # arrived and refuses a proof for work that has been replaced. That guard
    # has never been exercised by any test.
    #
    # Two shares on the SAME job is not a failure and is not asserted against -
    # a pool that has not sent new work yet should get more shares on the old
    # job. What must never happen is a share for a job the miner has already
    # been told is superseded.
    m = PearlMock(free_port(), total_shares=3, renotify_before_reply=True)
    tr = os.path.join(tmp, "stale.jsonl")
    ev, out = run_case(binary, m, tr)
    if len(m.submits) < 2:
        fail("only %d submit(s); the guard needs at least two jobs"
             % len(m.submits))
        return
    ok("%d shares across %d jobs" % (len(m.submits), len(m.notified)))
    order = {j: i for i, j in enumerate(m.notified)}
    high = -1
    for n, s in enumerate(m.submits):
        jid = s["params"]["job_id"]
        if jid not in order:
            fail("submitted an unknown job_id %r" % jid)
            continue
        if order[jid] < high:
            fail("share %d named job %r, older than one already superseded "
                 "(%r) - drainAvailable did not refuse it"
                 % (n, jid, m.notified[high]))
        high = max(high, order[jid])
        if n > 0 and jid == m.newest_at_submit[n - 1]:
            fail("share %d re-used job %r after the pool had already pushed "
                 "its replacement" % (n, jid))
    ok("no share names work the miner had been told was superseded")
    acc, _ = reported_accepted(out)
    if acc is not None and acc != m.credited:
        fail("reported accepted=%s, pool credited %d" % (acc, m.credited))
    elif acc is not None:
        ok("reported accepted=%d matches the pool's %d" % (acc, m.credited))


def phase_push_burst(binary, tmp):
    print("== phase 4: a burst of pushes before the reply ==")
    # submit() reads at most 12 lines looking for its own request id
    # (pearl_pool.h). 16 pushes ahead of the reply is a legal thing for a busy
    # pool to do, and if the window is the bug it shows up as a share the pool
    # credited but the miner did not.
    m = PearlMock(free_port(), total_shares=1, push_burst=16)
    tr = os.path.join(tmp, "burst.jsonl")
    ev, out = run_case(binary, m, tr)
    if not m.submits:
        fail("no share submitted")
        return
    reps = of(ev, "mining.reply")
    verdict = reps[0] if reps else None
    print("   pool credited %d, transcript reply %r" % (m.credited, verdict))
    if verdict is None:
        fail("the pool answered but the miner recorded no reply at all")
    elif verdict.get("accepted") is not True:
        fail("the pool accepted this share; the miner recorded accepted=%r "
             "reason=%r. 16 pushes arrived before the reply and the read "
             "window in submit() is 12 lines."
             % (verdict.get("accepted"), verdict.get("reason")))
    else:
        ok("the reply was found behind 16 pushes")
    acc, _ = reported_accepted(out)
    if acc is not None and acc != m.credited:
        fail("the pool credited %d, the miner reports accepted=%s"
             % (m.credited, acc))


def main():
    tmp = os.path.join(ROOT, "build", "pearl-e2e")
    os.makedirs(tmp, exist_ok=True)

    prove_mock_discriminates()

    binary = pick_binary(sys.argv)
    if binary is None:
        print("\nSKIP: no built binary reports pearl-pow in --list-algos")
        return 1 if FAILURES else 0
    print("\nbinary: %s" % binary)
    # SAY WHICH BACKEND RAN. A phase-1 line reading n=65536 when the Vulkan
    # shape is 16384 was the only clue that a run had not exercised what it
    # looked like it had, and it took a second run to settle. The banner is
    # cheap and removes the ambiguity.
    algos = subprocess.run([binary, "--list-algos"], capture_output=True,
                           text=True, timeout=60).stdout.split()
    print("backend:  %s  (algos: %s)"
          % (os.path.basename(binary), " ".join(algos)))

    phase_accept(binary, tmp)
    phase_reject(binary, tmp)
    phase_stale(binary, tmp)
    phase_push_burst(binary, tmp)

    print()
    if FAILURES:
        print("FAIL: %d problem(s)" % len(FAILURES))
        return 1
    print("PASS: soat-miner speaks the Pearl pool protocol correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

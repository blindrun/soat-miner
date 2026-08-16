#!/usr/bin/env python3
"""
End-to-end Lithos protocol test: runs the real miner binary against a mock
Lithos stratum server transcribed from the Lithos client's own source.

This is a protocol test, not a hashrate test. It proves four things that all
fail SILENTLY in production - the miner keeps running, the panel keeps printing
a hashrate, and no shares are ever credited:

  1. mining.submit carries the 5 params Lithos destructures.
  2. extraNonce2 is exactly the low 4 bytes of the nonce, so the nonce Lithos
     reconstructs is the nonce the GPU actually solved.
  3. mining.set_extranonce is honoured, so a mid-session rotation does not
     leave the miner hashing in a subspace it no longer owns.
  4. a zero share target (tau) is refused loudly instead of being mined
     against forever.

Usage: python3 tests/test_lithos.py [path-to-miner-binary]
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lithos_mock import LithosMock  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN = os.path.join(ROOT, "soat-miner-vk")

# The dataset build dominates: ~7 GB, a few seconds on a warm card, longer cold.
TIMEOUT = 300


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run_case(binary, zero_target=False, shares=6, rotate_after=3):
    port = free_port()
    mock = LithosMock(port, shares_before_rotate=rotate_after,
                      total_shares=shares, send_zero_target=zero_target)
    t = threading.Thread(target=mock.serve, daemon=True)
    t.start()
    time.sleep(0.3)

    proc = subprocess.Popen(
        [binary, "--lithos", "--pool", "127.0.0.1:%d" % port,
         "--plain", "--interval", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    out_lines = []

    def drain():
        for line in proc.stdout:
            out_lines.append(line.rstrip())

    d = threading.Thread(target=drain, daemon=True)
    d.start()

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if mock.done.wait(timeout=0.5):
            break
        if proc.poll() is not None:
            break
        # The zero-target job is the last thing sent; give the miner a few
        # seconds to react to it before tearing the connection down.
        if mock.zero_sent_time and time.time() - mock.zero_sent_time > 6:
            break
    # Give the miner a moment to log anything triggered by the last message.
    time.sleep(2.0)

    mock.done.set()
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    d.join(timeout=5)

    return mock, "\n".join(out_lines)


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN
    if not os.path.exists(binary):
        print("SKIP: %s not built" % binary)
        return 0

    failures = []

    # ---- case 1: normal flow + mid-session extranonce rotation -------------
    print("== case 1: submit shape, nonce reconstruction, set_extranonce ==")
    mock, out = run_case(binary)
    for e in mock.errors:
        failures.append("case1: " + e)
    if mock.accepted == 0:
        failures.append(
            "case1: miner submitted no shares at all (target was ~2^252, so "
            "almost every nonce should win). Miner output:\n" + out[-3000:])
    if not mock.rotated:
        failures.append("case1: never reached the extranonce rotation")
    elif not mock.saw_post_rotation_share:
        failures.append(
            "case1: no share arrived after mining.set_extranonce - the miner "
            "is not adopting the new prefix")
    print("   accepted=%d rotated=%s post_rotation=%s"
          % (mock.accepted, mock.rotated, mock.saw_post_rotation_share))

    # ---- case 2: zero share target must be refused, loudly ----------------
    print("== case 2: zero target (tau=0) is refused with a warning ==")
    mock2, out2 = run_case(binary, zero_target=True, shares=2, rotate_after=1)
    if not re.search(r"zero target", out2, re.I):
        failures.append(
            "case2: miner did not warn about the zero target; it would hash "
            "forever finding nothing. Output:\n" + out2[-3000:])
    else:
        print("   warned as expected")

    print()
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: soat-miner speaks the Lithos protocol correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-to-end: the real miner binary against a mock KawPow pool.

Everything else in this repository tests a piece. The vectors test the
algorithm, the device gates test the kernels, test_kawpow_stratum drives the
stratum client directly. Nothing has ever run the whole path - binary, run
loop, job parsing, dataset build, search, verify, submit - against a server.

That gap matters because the failures it catches are the ones that keep the
miner looking healthy: it connects, it prints a hashrate, and no share is ever
credited. Specifically it proves:

  1. the handshake completes and a job is accepted;
  2. mining.submit carries the FIVE params a KawPow pool destructures,
     including the mix hash, which no other algorithm here produces;
  3. the submitted nonce carries the extranonce prefix the pool assigned -
     the load-bearing one, because ignoring it means every share is out of
     range while the hash itself is perfectly correct;
  4. the header echoed back is the job's, not a stale one;
  5. the miner's own accounting agrees the share was accepted.

Uses height 100 so the dataset is epoch 0 - about 1 GB, a few seconds - rather
than mainnet's 6.11 GB. Needs a GPU; skips politely without one.

Usage: python3 tests/test_kawpow_pool.py [path-to-miner-binary]
"""

import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kawpow_mock import KawpowMock, base58check  # noqa: E402

fails = []


def check(ok, what, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        if detail:
            print(f"        {detail}")
        fails.append(what)


def run_once(binary, flavour, extranonce):
    mock = KawpowMock(0, extranonce=extranonce, flavour=flavour).start()
    wallet = base58check(60, bytes([0x11]) * 20)
    cmd = [binary, "--algo", "kawpow",
           "--pool", f"127.0.0.1:{mock.port}",
           "--wallet", wallet, "--worker", "mocktest",
           "--plain", "--interval", "2"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    out = []
    deadline = time.time() + 150
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                out += proc.stdout.read().splitlines()
                break
            line = proc.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if mock.submits and any('"accepted":1' in l or '"accepted": 1' in l
                                    for l in out):
                break
            if len(mock.submits) >= 1 and time.time() > deadline - 120:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        mock.stop()
    return mock, out, wallet


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "./soat-miner"
    if not os.access(binary, os.X_OK):
        print(f"  SKIP  {binary} not built")
        return 0

    mock, out, wallet = run_once(binary, "stratum", "a1b2")
    joined = "\n".join(out)

    if "no CUDA device" in joined or "no Vulkan" in joined:
        print("  SKIP  no usable GPU here")
        return 0

    check(mock.subscribed, "the miner subscribed")
    check(mock.authorized == f"{wallet}.mocktest",
          "it authorized as wallet.worker", f"got {mock.authorized!r}")
    check("dataset ready" in joined or "building" in joined,
          "it built the dataset for the job's height",
          joined[-300:] if not mock.submits else "")

    if not mock.submits:
        check(False, "it submitted at least one share",
              "no submit arrived; last output:\n        " +
              "\n        ".join(out[-6:]))
        return 1

    p = mock.submits[0]
    check(len(p) == 5, f"mining.submit carries 5 params, got {len(p)}", repr(p))
    if len(p) != 5:
        return 1

    login, jobid, nonce, header, mix = p
    check(login == f"{wallet}.mocktest", "param 0 is wallet.worker")
    check(jobid == "job-1", "param 1 is the job id we sent", repr(jobid))

    n = nonce[2:] if nonce.lower().startswith("0x") else nonce
    check(len(n) == 16 and all(c in "0123456789abcdefABCDEF" for c in n),
          "param 2 is a 16-hex-character nonce", repr(nonce))
    check(n.lower().startswith(mock.extranonce),
          "the nonce carries the extranonce prefix the pool assigned",
          f"prefix {mock.extranonce!r}, nonce {n!r}")

    h = header[2:] if header.lower().startswith("0x") else header
    check(h.lower() == mock.HEADER, "param 3 is the job's header hash", repr(header))

    m = mix[2:] if mix.lower().startswith("0x") else mix
    check(len(m) == 64 and all(c in "0123456789abcdefABCDEF" for c in m),
          "param 4 is a 64-hex-character mix hash", repr(mix))

    accepted = re.search(r'"accepted":\s*(\d+)', joined)
    check(accepted is not None and int(accepted.group(1)) >= 1,
          "the miner counted the share as accepted",
          f"accepted={accepted.group(1) if accepted else 'absent'}")

    if not fails:
        print(f"KawPow pool protocol: all checks passed "
              f"({len(mock.submits)} share(s) submitted)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

"""Pearl's own verifier against a LARGE shape and a tile far from the origin.

tests/pearl_oracle.py proves the wire format at m=n=256 with whatever tile the
first win lands on, which in practice is near (0,0). That leaves the encoding
unexercised where a real job lives: deep Merkle trees, leaf indices in the tens
of thousands, and a tile column far down the matrix.

Run under Pearl's uv env, same as the other oracle:
  PYTHONPATH=tests <pearl venv>/bin/python tests/pearl_oracle_large.py
"""
import sys
from base64 import b64decode, b64encode

import numpy as np
import pearl_job as pj

try:
    import pearl_mining as pm
except ImportError:
    print("pearl_mining is not importable - run this under Pearl's uv env.")
    sys.exit(2)

from blake3 import blake3

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def main():
    # Big enough to be a real job in the ways that matter, small enough that a
    # pure-Python scan finishes. n is what drives deep trees and far tiles.
    m, n, k, rank = 256, 16384, 2048, 128
    cfg = pj.MiningConfiguration.contiguous(k=k, rank=rank)
    # The verifier takes the target from the header's own nbits, so the header
    # must be a real one and the mining target must be the one it encodes.
    nbits = 0x1E010000
    header = pm.IncompleteBlockHeader(
        version=536870912,
        prev_block=bytes(range(32)),
        merkle_root=bytes(range(32, 64)),
        timestamp=1786924932,
        nbits=nbits,
    )
    header_bytes = bytes(header.to_bytes())
    key = pj.job_key(header_bytes, cfg)

    A = pj.synth_matrix(m, k, 3)
    B_t = pj.synth_matrix(n, k, 17)
    a_root = pj.blake3_keyed(key, pj.pad_to_chunk_boundary(A.tobytes()))
    bt_root = pj.blake3_keyed(key, pj.pad_to_chunk_boundary(B_t.tobytes()))
    seed_a, seed_b = pj.commitments(a_root, bt_root, key, m, n, salted=True)

    a_leaves = (m * k) // 1024
    bt_leaves = (n * k) // 1024
    print(f"shape {m}x{n} k={k} rank={rank}: "
          f"A {a_leaves} leaves, B^t {bt_leaves} leaves "
          f"({bt_leaves.bit_length() - 1} tree levels)")
    check("the B^t tree is far deeper than the small oracle's",
          bt_leaves >= 32768, str(bt_leaves))

    gen = pj.NoiseGenerator(noise_rank=rank)
    E_AL, E_AR, E_BL, E_BR = gen.generate(seed_a, seed_b, m, k, n)
    B = np.ascontiguousarray(B_t.T)
    A_n = (A.astype(np.int32) + E_AL.astype(np.int32) @ E_AR.astype(np.int32)).astype(np.int8)
    B_n = (B.astype(np.int32) + E_BL.astype(np.int32) @ E_BR.astype(np.int32)).astype(np.int8)

    # An easy target so plenty of tiles win, then keep the LAST one: the point
    # is a tile far from the origin, not any tile.
    target = pj.bits_to_target(nbits)
    bound = cfg.penalized_target(target)
    check("the header's nbits decodes to 2^232", target == 2**232, hex(target))
    check("the target scales without overflowing 256 bits", bound is not None)

    tile_h = tile_w = 16
    th = tw = rank // tile_h
    last = None
    wins = 0
    for i in range(0, m - m % rank, rank):
        for j in range(0, n - n % rank, rank):
            ts = [[pj.Transcript() for _ in range(tw)] for _ in range(th)]
            block = np.zeros((rank, rank), dtype=np.int32)
            reduction = 0
            for p in range(0, k - k % rank, rank):
                block = block + (A_n[i:i + rank, p:p + rank].astype(np.int32)
                                 @ B_n[p:p + rank, j:j + rank].astype(np.int32))
                for hi in range(th):
                    for wi in range(tw):
                        tile = block[hi * tile_h:(hi + 1) * tile_h,
                                     wi * tile_w:(wi + 1) * tile_w]
                        ts[hi][wi].rotl_xor_into(
                            reduction, pj.xor_reduce_tile(np.ascontiguousarray(tile)))
                reduction += 1
            for hi in range(th):
                for wi in range(tw):
                    digest = blake3(ts[hi][wi].to_bytes(), key=seed_a).digest()
                    if int.from_bytes(digest, "little") <= bound:
                        wins += 1
                        last = (i + hi * tile_h, j + wi * tile_w, digest)

    check("the scan found winning tiles", wins > 0, str(wins))
    if last is None:
        print(f"\n{PASS} passed, {FAIL} failed")
        return 1
    t_rows, t_cols, digest = last
    print(f"opening tile at row {t_rows}, col {t_cols} "
          f"(A leaf {t_rows * k // 1024}, B^t leaf {t_cols * k // 1024}), "
          f"{wins} wins total")
    check("the chosen tile is far from the origin", t_cols >= 8192, str(t_cols))
    check("and its B^t leaf index needs more than 16 bits",
          (t_cols * k // 1024) >= 65536 or bt_leaves >= 32768,
          str(t_cols * k // 1024))

    win = pj._open(A, B_t, key, m, n, k, rank, t_rows, t_cols,
                   tile_h, tile_w, digest)

    proof = pm.PlainProof.from_base64(b64encode(win["plain_proof"]).decode())
    check("our bincode deserialises as a PlainProof at this shape", proof is not None)
    check("round-trips to the same bytes",
          b64decode(proof.to_base64()) == win["plain_proof"])

    ok, msg = pm.verify_plain_proof_v3(header, proof)
    check("PEARL'S OWN VERIFIER ACCEPTS THE LARGE-SHAPE PROOF", ok, str(msg))

    moved = pj._open(A, B_t, key, m, n, k, rank, t_rows, t_cols + 16,
                     tile_h, tile_w, digest)
    bad = pm.PlainProof.from_base64(b64encode(moved["plain_proof"]).decode())
    ok_bad, _ = pm.verify_plain_proof_v3(header, bad)
    check("and rejects the same proof moved one tile along", not ok_bad)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

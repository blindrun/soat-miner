#!/usr/bin/env python3
"""
Check pearl_job.py against Pearl's own Rust, not against itself.

Every test in `pearl_job.py` is self-consistent: the proof verifier there is a
port of the same understanding as the proof builder, so both can be wrong
together and still agree. This file removes that by putting the real
implementation on the other side of every check - `pearl_mining`, the pyo3
extension over the `zk-pow` and `pearl-blake3` crates that the network itself
runs.

The last test is the one that matters. `verify_plain_proof_v3` is the verifier
a node uses on a submitted block, minus the ZK wrapper: it rebuilds the mining
configuration from the proof, re-derives the job key, regenerates the noise
from the commitment chain, recomputes the transcript from only the sixteen rows
and sixteen columns the proof opens, hashes it and measures it against the
rank-penalised bound. If our miner has any of that wrong, this rejects it. If
it passes, the pipeline is right end to end and the only untested thing left is
the network transport.

This does not run in the ordinary test suite, because it needs Pearl's own
toolchain built. From a clone of github.com/pearl-research-labs/pearl:

    uv sync --package pearl-gateway
    PYTHONPATH=/path/to/soat-miner/tests uv run python \\
        /path/to/soat-miner/tests/pearl_oracle.py
"""
import sys
from base64 import b64decode, b64encode

import numpy as np

import pearl_job as pj

try:
    import pearl_mining as pm
except ImportError:
    print("pearl_mining is not importable - run this under Pearl's uv env.")
    print(__doc__)
    sys.exit(2)

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
    print("1. constants agree")
    check("merkle leaf size", pm.MERKLE_LEAF_SIZE == pj.CHUNK_LEN)
    check("penalty base rank", pm.PENALTY_BASE_RANK == pj.PENALTY_BASE_RANK)

    print("2. periodic patterns")
    for pattern in ([0, 1], list(range(16)), [0, 8], [0, 1, 8, 9, 16, 17, 24, 25],
                    [0, 1, 2, 3, 4, 5, 6, 7]):
        ours = pj.PeriodicPattern.from_list(pattern)
        theirs = pm.PeriodicPattern.from_list(pattern)
        check(f"{pattern[:4]}... to_bytes",
              ours.to_bytes() == bytes(theirs.to_bytes()),
              f"{ours.to_bytes().hex()} vs {bytes(theirs.to_bytes()).hex()}")

    print("3. mining configuration serialisation")
    for k, rank in ((2048, 128), (4096, 128), (8192, 256), (65536, 1024)):
        ours = pj.MiningConfiguration.contiguous(k, rank)
        theirs = pm.MiningConfiguration(
            common_dim=k, rank=rank, mma_type=pm.MMAType.Int7xInt7ToInt32,
            rows_pattern=pm.PeriodicPattern.from_list(list(range(16))),
            cols_pattern=pm.PeriodicPattern.from_list(list(range(16))),
            moe=None)
        check(f"k={k} rank={rank} 52 bytes match",
              ours.to_bytes() == bytes(theirs.to_bytes()),
              f"{ours.to_bytes().hex()} vs {bytes(theirs.to_bytes()).hex()}")

    print("4. rank-penalised target")
    for k, rank in ((2048, 128), (4096, 128), (4096, 256)):
        ours = pj.MiningConfiguration.contiguous(k, rank)
        theirs = pm.MiningConfiguration(
            common_dim=k, rank=rank, mma_type=pm.MMAType.Int7xInt7ToInt32,
            rows_pattern=pm.PeriodicPattern.from_list(list(range(16))),
            cols_pattern=pm.PeriodicPattern.from_list(list(range(16))),
            moe=None)
        t = 2**232
        check(f"k={k} rank={rank} bound matches",
              ours.penalized_target(t) == pm.penalized_target_bound(t, theirs))

    print("5. merkle tree and multi-leaf proofs")
    # MerkleProof exposes no getters, so the comparison goes through their own
    # serialiser: an identical bincode blob means identical leaf data, leaf
    # indices, total_leaves, root AND sibling list in the same order. That also
    # checks our bincode encoder against theirs on every shape below.
    key = bytes(range(32))
    for leaves in (2, 3, 5, 8, 11, 16, 64):
        data = bytes((i * 91 + leaves) & 0xFF for i in range(leaves * pj.CHUNK_LEN))
        ours = pj.MerkleTree(data, key)
        theirs = pm.MerkleTree(data=data, key=key)
        check(f"{leaves}-leaf root", ours.root == bytes(theirs.root))

        idx = sorted({0, leaves // 2, leaves - 1})
        op = ours.multileaf_proof(idx)
        rows = list(range(len(idx)))
        mine = pj.encode_plain_proof(1, 1, 1, 128, op, rows, op, rows)
        their_leaf = pm.MatrixMerkleProof(theirs.get_multileaf_proof(idx), rows)
        theirs_bytes = b64decode(
            pm.PlainProof(1, 1, 1, 128, their_leaf, their_leaf, None).to_base64())
        check(f"{leaves}-leaf proof serialises identically", mine == theirs_bytes,
              f"{len(mine)} vs {len(theirs_bytes)} bytes")

    print("6. leaf indices from rows")
    for rows, cols in (([0], 2048), (list(range(16)), 2048),
                       (list(range(100, 116)), 4096), ([7], 512)):
        check(f"{len(rows)} rows of {cols}",
              pj.leaf_indices_from_rows(rows, cols)
              == list(pm.MerkleTree.compute_leaf_indices_from_rows(rows, (max(rows) + 1, cols))))

    print("7. a real win, verified by Pearl's own verifier")
    # regtest difficulty: nbits 0x1e010000 is a target of 2^232, which after the
    # rank penalty leaves roughly one winning transcript in thirty-two. One GEMM
    # at 256x256 produces 256 of them, so this finds a block on the first try.
    nbits = 0x1E010000
    header = pm.IncompleteBlockHeader(
        version=536870912,
        prev_block=bytes(range(32)),
        merkle_root=bytes(range(32, 64)),
        timestamp=1786924932,
        nbits=nbits,
    )
    header_bytes = bytes(header.to_bytes())
    check("header serialises to 76 bytes", len(header_bytes) == 76, str(len(header_bytes)))

    target = pj.bits_to_target(nbits)
    check("nbits decodes to the template's target", target == 2**232, hex(target))

    m = n = 256
    cfg = pj.MiningConfiguration.contiguous(k=2048, rank=128)
    rng = np.random.default_rng(20260817)
    win = None
    for _ in range(4):
        win = pj.mine_once(header_bytes, target, cfg, m, n, rng, salted=True)
        if win:
            break
    check("mined a candidate", win is not None)
    if not win:
        print(f"\n{PASS} passed, {FAIL} failed")
        return 1

    proof = pm.PlainProof.from_base64(
        b64encode(win["plain_proof"]).decode())
    check("our bincode deserialises as a PlainProof", proof is not None)
    check("round-trips to the same bytes",
          b64decode(proof.to_base64()) == win["plain_proof"])

    ok, msg = pm.verify_plain_proof_v3(header, proof)
    check("PEARL'S OWN VERIFIER ACCEPTS IT", ok, str(msg))

    # Without this the acceptance above could be a verifier that accepts
    # anything. Move the opened tile one row and it must stop verifying: the
    # transcript then belongs to a different tile and misses the target.
    moved = pj._open(win["A"], win["B_t"], win["key"], m, n, cfg.common_dim,
                     cfg.rank, win["t_rows"] + 16, win["t_cols"], 16, 16,
                     win["digest"])
    bad = pm.PlainProof.from_base64(
        b64encode(moved["plain_proof"]).decode())
    ok_bad, _ = pm.verify_plain_proof_v3(header, bad)
    check("a proof opening the WRONG tile is rejected", not ok_bad)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

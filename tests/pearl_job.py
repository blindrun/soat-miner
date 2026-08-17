#!/usr/bin/env python3
"""
Pearl job pipeline reference: everything around the kernels.

`pearl_reference.py` covers the arithmetic - noise generation, the noised
product, the XOR transcript, the blake3 PoW check. It takes the noise seeds and
the target as given, which is fine for checking a kernel and useless for
mining. This file is the other half: where those seeds actually come from, what
counts as a valid choice of matrices, and what a winning tile has to be turned
into before a node will look at it.

The load-bearing facts, all read out of pearl-research-labs/pearl (ISC) rather
than inferred from watching a miner run:

  * A job is `incomplete_header_bytes`, `target`, `cert_version`. No matrices.
    The miner picks A and B, so **choosing the matrices is the nonce**.
  * job_key = blake3(header || mining_config). The mining config commits k, the
    rank, and the shape of the hash tile - so it must be decided before the
    first hash, not after a win.
  * The Merkle root of a matrix is just blake3 of it, keyed and zero-padded to
    a 1024-byte boundary, because MERKLE_LEAF_SIZE is blake3's own CHUNK_LEN
    and the tree is blake3's own tree. Only a WIN needs the real tree, and only
    to open 16 rows of it.
  * Those two roots, salted with m and n under V3, become commitment_B then
    commitment_A. commitment_A is simultaneously the A-noise seed and the PoW
    key, which is what binds the matrices to the work.
  * The target a transcript is compared against is not the block target. It is
    scaled by tile_size * (k/rank) * 128, which is why a single GEMM yielding
    thousands of transcripts is not free difficulty.
  * A win is submitted as a `PlainProof`: bincode, base64, over the gateway's
    line-delimited JSON-RPC. The miner never builds a block. The block needs a
    plonky2 certificate that only Pearl's own prover can produce.

Consensus rules a mining configuration has to satisfy (zk-pow sanity_checks.rs,
worth stating because breaking one costs a whole GEMM for nothing):

    rank        power of two in [32, 1024], divisible by 16, and >= 128
    k           multiple of 64, >= 1024, >= 16*rank, <= min(2^16, 4*rank^2)
    hash tile   h and w even, 32 <= h*w <= 256
    m, n        <= 2^24, and the winning tile must fit inside them

Rank 128 is not a default, it is the optimum: below it blocks are rejected
outright, and above it the penalty scales the target down by 128/rank for
identical work. A 16x16 hash tile is the largest the rules allow, which means
the fewest blake3 calls per unit of matmul.
"""
import argparse
import json
import socket
import struct
import sys
from base64 import b64decode, b64encode

import numpy as np

from pearl_reference import NoiseGenerator, Transcript, xor_reduce_tile

CHUNK_LEN = 1024
BLOCK_LEN = 64
OUT_LEN = 32
PENALTY_BASE_RANK = 128
U256_MAX = 2**256 - 1

# ------------------------------------------------------------------- blake3
#
# The library's `blake3` gives digests and nothing else. A multi-leaf Merkle
# proof needs the tree's INTERNAL chaining values, so the compression function
# has to be here in the open. Everything below is checked against the library
# in test 1 - if this port drifts, the roots stop matching and the check fails.

IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)

CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16

MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)

M32 = 0xFFFFFFFF


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def _g(s, a, b, c, d, mx, my):
    s[a] = (s[a] + s[b] + mx) & M32
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & M32
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & M32
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & M32
    s[b] = _rotr(s[b] ^ s[c], 7)


def compress(cv, block_words, counter, block_len, flags):
    """The blake3 compression function. Returns all 16 output words."""
    s = [cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
         IV[0], IV[1], IV[2], IV[3],
         counter & M32, (counter >> 32) & M32, block_len, flags]
    m = list(block_words)
    for r in range(7):
        _g(s, 0, 4, 8, 12, m[0], m[1])
        _g(s, 1, 5, 9, 13, m[2], m[3])
        _g(s, 2, 6, 10, 14, m[4], m[5])
        _g(s, 3, 7, 11, 15, m[6], m[7])
        _g(s, 0, 5, 10, 15, m[8], m[9])
        _g(s, 1, 6, 11, 12, m[10], m[11])
        _g(s, 2, 7, 8, 13, m[12], m[13])
        _g(s, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in MSG_PERMUTATION]
    return [s[i] ^ s[i + 8] for i in range(8)] + \
           [s[i + 8] ^ cv[i] for i in range(8)]


def _words(block):
    return list(struct.unpack("<16I", block))


def _key_words(key):
    assert len(key) == 32
    return list(struct.unpack("<8I", key))


def _cv_bytes(words):
    return struct.pack("<8I", *words[:8])


def chunk_cv(key, chunk, chunk_index, root=False):
    """Chaining value of one chunk (<= 1024 bytes), keyed."""
    cv = _key_words(key)
    blocks = [chunk[i:i + BLOCK_LEN] for i in range(0, max(len(chunk), 1), BLOCK_LEN)]
    for i, blk in enumerate(blocks):
        flags = KEYED_HASH
        if i == 0:
            flags |= CHUNK_START
        if i == len(blocks) - 1:
            flags |= CHUNK_END
            if root:
                flags |= ROOT
        n = len(blk)
        cv = compress(cv, _words(blk + b"\0" * (BLOCK_LEN - n)), chunk_index, n, flags)[:8]
    return _cv_bytes(cv)


def parent_cv(key, left, right, root=False):
    flags = KEYED_HASH | PARENT | (ROOT if root else 0)
    out = compress(_key_words(key), _words(left + right), 0, BLOCK_LEN, flags)
    return _cv_bytes(out)


def blake3_keyed(key, data):
    """Keyed blake3 of arbitrary data, built from the tree above."""
    if len(data) <= CHUNK_LEN:
        return chunk_cv(key, data, 0, root=True)
    cvs = [chunk_cv(key, data[i:i + CHUNK_LEN], i // CHUNK_LEN)
           for i in range(0, len(data), CHUNK_LEN)]
    while len(cvs) > 2:
        cvs = _combine_layer(key, cvs)
    return parent_cv(key, cvs[0], cvs[1], root=True)


def _combine_layer(key, cvs):
    """Pair a layer, carrying an odd trailing node up unchanged.

    This is not an approximation of blake3's "left subtree is the largest
    power of two" rule, it is equivalent to it for every leaf count - which is
    what lets the miner treat a plain keyed hash as the Merkle root and only
    build the real tree on a win.
    """
    out = [parent_cv(key, cvs[i], cvs[i + 1]) for i in range(0, len(cvs) - 1, 2)]
    if len(cvs) % 2:
        out.append(cvs[-1])
    return out


def blake3_unkeyed(data):
    """Unkeyed blake3, needed only for job_key and the domain salts."""
    from blake3 import blake3
    return blake3(data).digest()


# -------------------------------------------------------------- merkle tree
def pad_to_chunk_boundary(data):
    rem = len(data) % CHUNK_LEN
    return data if rem == 0 else data + b"\0" * (CHUNK_LEN - rem)


class MerkleTree:
    """blake3's own tree over 1024-byte chunks, kept so a win can be opened."""

    def __init__(self, data, key):
        assert len(data) % CHUNK_LEN == 0 and data, "data must be chunk-aligned"
        self.data = data
        self.key = key
        leaves = [chunk_cv(key, data[i:i + CHUNK_LEN], i // CHUNK_LEN)
                  for i in range(0, len(data), CHUNK_LEN)]
        self.layers = [leaves]
        while len(self.layers[-1]) > 2:
            self.layers.append(_combine_layer(key, self.layers[-1]))
        if len(self.layers[-1]) == 2:
            self.layers.append([parent_cv(key, self.layers[-1][0],
                                          self.layers[-1][1], root=True)])

    @property
    def root(self):
        return self.layers[-1][0]

    def multileaf_proof(self, leaf_indices):
        """Siblings needed to recompute the root from the given leaves.

        The walk is level by level and the sibling is skipped whenever it is
        also being opened, which is the whole point of a MULTI-leaf proof: 32
        adjacent leaves out of 4096 cost far fewer than 32 separate paths.
        """
        idxs = sorted(set(leaf_indices))
        total = len(self.layers[0])
        assert idxs and idxs[-1] < total, "leaf index out of bounds"

        leaf_data = [self.data[i * CHUNK_LEN:(i + 1) * CHUNK_LEN] for i in idxs]

        siblings = []
        current = set(idxs)
        level_len = total
        level = 0
        while level_len > 1 and current:
            nodes = self.layers[level]
            for i in sorted(current):
                if i % 2 == 1:
                    if (i - 1) not in current:
                        siblings.append(nodes[i - 1])
                elif (i + 1) not in current and (i + 1) < level_len:
                    siblings.append(nodes[i + 1])
            current = {i // 2 for i in current}
            level_len = (level_len + 1) // 2
            level += 1

        return {"leaf_data": leaf_data, "leaf_indices": idxs,
                "total_leaves": total, "root": self.root, "siblings": siblings}


def verify_multileaf_proof(proof, key):
    """Recompute the root from the opened leaves alone.

    Only here to prove the proof builder is right before a node ever sees one.
    A node rejecting a malformed proof tells you nothing about which half is
    wrong; this does.
    """
    known = {i: chunk_cv(key, d, i)
             for i, d in zip(proof["leaf_indices"], proof["leaf_data"])}
    siblings = list(proof["siblings"])
    level_len = proof["total_leaves"]

    while level_len > 1:
        nxt = {}
        for i in sorted(known):
            if i % 2 == 1:
                left = known.get(i - 1)
                if left is None:
                    if not siblings:
                        return None
                    left = siblings.pop(0)
                right = known[i]
            else:
                left = known[i]
                right = known.get(i + 1)
                if right is None:
                    if (i + 1) < level_len:
                        if not siblings:
                            return None
                        right = siblings.pop(0)
                    else:
                        nxt[i // 2] = left      # odd tail carries up unchanged
                        continue
            nxt[i // 2] = (left, right)
        level_len = (level_len + 1) // 2
        root_level = level_len == 1
        known = {}
        for i, v in nxt.items():
            known[i] = v if isinstance(v, bytes) else \
                parent_cv(key, v[0], v[1], root=root_level)

    return known.get(0)


def leaf_indices_from_rows(row_indices, cols):
    """Which 1024-byte chunks a set of matrix rows lands in."""
    out = set()
    for row in row_indices:
        first = (row * cols) // CHUNK_LEN
        last = ((row + 1) * cols - 1) // CHUNK_LEN
        out.update(range(first, last + 1))
    return sorted(out)


# --------------------------------------------------------- mining config
class PeriodicPattern:
    """A set of indices expressed as a 3D arithmetic progression, 6 bytes.

    The hash tile is described by a pattern rather than a width because the
    chain allows strided tiles. We use a contiguous run, which is the simplest
    legal pattern and the largest allowed when both sides are 16.
    """

    NUM_DIMS = 3

    def __init__(self, shape):
        self.shape = shape

    @classmethod
    def from_list(cls, pattern):
        assert pattern and pattern[0] == 0
        assert all(a < b for a, b in zip(pattern, pattern[1:])), "must be sorted, unique"
        p = list(pattern)
        shape = []
        while len(p) > 1:
            for period in range(1, len(p)):
                if len(p) % period:
                    continue
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape.append((s, len(p) // period))
                    p = p[:period]
                    break
            else:
                raise ValueError("pattern is not periodic")
        shape.reverse()
        period = shape[-1][0] * shape[-1][1] if shape else 1
        while len(shape) < cls.NUM_DIMS:
            shape.append((period, 1))
        return cls(shape)

    def to_bytes(self):
        out = bytearray(2 * self.NUM_DIMS)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            out[2 * i] = stride // min_stride - 1
            out[2 * i + 1] = length - 1
            min_stride = stride * length
        return bytes(out)

    def to_list(self):
        res = [0]
        for stride, length in self.shape:
            res = [r + i * stride for i in range(length) for r in res]
        return sorted(res)

    def size(self):
        n = 1
        for _, length in self.shape:
            n *= length
        return n


class MiningConfiguration:
    """The 52 bytes a miner commits to before hashing anything."""

    SERIALIZED_SIZE = 52

    def __init__(self, common_dim, rank, rows_pattern, cols_pattern):
        self.common_dim = common_dim
        self.rank = rank
        self.rows_pattern = rows_pattern
        self.cols_pattern = cols_pattern

    @classmethod
    def contiguous(cls, k, rank, tile_h=16, tile_w=16):
        return cls(k, rank,
                   PeriodicPattern.from_list(list(range(tile_h))),
                   PeriodicPattern.from_list(list(range(tile_w))))

    def to_bytes(self):
        return (struct.pack("<I", self.common_dim)
                + struct.pack("<H", self.rank)
                + struct.pack("<H", 0)              # MMAType::Int7xInt7ToInt32
                + self.rows_pattern.to_bytes()
                + self.cols_pattern.to_bytes()
                + b"\0" * 32)                      # MoE trailer, zero = dense

    @property
    def tile_size(self):
        return self.rows_pattern.size() * self.cols_pattern.size()

    @property
    def dot_product_length(self):
        return self.common_dim - self.common_dim % self.rank

    def check(self, m, n):
        """Every consensus rule that can be checked before mining starts.

        Worth running once per job. Each of these costs a full GEMM to
        discover by submission, and a rejected block looks identical to bad
        luck from the miner's side.
        """
        r, k = self.rank, self.common_dim
        h, w = self.rows_pattern.size(), self.cols_pattern.size()
        problems = []
        if r < PENALTY_BASE_RANK:
            problems.append(f"rank {r} < {PENALTY_BASE_RANK}: blocks are rejected outright")
        if r & (r - 1) or not 32 <= r <= 1024 or r % 16:
            problems.append(f"rank {r} must be a power of two in [32,1024], divisible by 16")
        if k % 64 or k < 1024 or k < 16 * r or k > min(1 << 16, 4 * r * r):
            problems.append(f"k {k} must be a multiple of 64 in [max(1024,16r), min(2^16,4r^2)]")
        if h % 2 or w % 2 or not 32 <= h * w <= 256:
            problems.append(f"hash tile {h}x{w}: sides must be even and 32 <= h*w <= 256")
        if m > 1 << 24 or n > 1 << 24:
            problems.append(f"m={m} n={n} must be <= 2^24")
        if max(self.rows_pattern.to_list()) >= m or max(self.cols_pattern.to_list()) >= n:
            problems.append("hash tile does not fit inside m x n")
        return problems

    def penalized_target(self, target):
        """The bound one transcript is measured against.

        The block target scaled by the work an attempt costs, then normalised
        to rank 128. Returns None when the scaling overflows 256 bits, which
        the chain treats as unusable rather than as "everything wins".
        """
        factor = self.tile_size * (self.dot_product_length // self.rank) * PENALTY_BASE_RANK
        if factor == 0 or target > U256_MAX // factor:
            return None
        return target * factor


# ------------------------------------------------------------- commitments
SEED_SALT_A = blake3_unkeyed(b"pearl/cert-v3/noise-seed/A")
SEED_SALT_B = blake3_unkeyed(b"pearl/cert-v3/noise-seed/B")


def _bind_root(root, dim, salt):
    return blake3_keyed(salt, root + struct.pack("<I", dim) + b"\0" * 28)


def job_key(incomplete_header_bytes, config):
    return blake3_unkeyed(incomplete_header_bytes + config.to_bytes())


def commitments(a_root, bt_root, key, m, n, salted=True):
    """commitment_A and commitment_B from the two matrix roots.

    commitment_A is both the A-noise seed and the PoW key, so it is the single
    value that makes the transcripts depend on the matrices. Under V3 each root
    is salted with the dimension it belongs to first; V1/V2 skip that, and
    getting it wrong produces a perfectly valid-looking proof that the node
    rejects.
    """
    if salted:
        a_root = _bind_root(a_root, m, SEED_SALT_A)
        bt_root = _bind_root(bt_root, n, SEED_SALT_B)
    commitment_b = blake3_unkeyed(key + bt_root)
    commitment_a = blake3_unkeyed(commitment_b + a_root)
    return commitment_a, commitment_b


# -------------------------------------------------------------- plain proof
def _u64(v):
    return struct.pack("<Q", v)


def _encode_merkle_proof(p):
    out = bytearray()
    out += _u64(len(p["leaf_data"]))
    for leaf in p["leaf_data"]:
        out += _u64(len(leaf)) + leaf
    out += _u64(len(p["leaf_indices"]))
    for i in p["leaf_indices"]:
        out += _u64(i)
    out += _u64(p["total_leaves"])
    out += p["root"]
    out += _u64(len(p["siblings"]))
    for s in p["siblings"]:
        out += s
    return bytes(out)


def encode_plain_proof(m, n, k, noise_rank, a_proof, a_rows, bt_proof, bt_rows):
    """bincode, exactly as `PlainProof::to_base64` writes it.

    bincode with fixint encoding: usize is u64 little-endian, a Vec is its
    length then its elements, a fixed array is raw bytes with no length, and an
    Option is a single tag byte. `moe` is None here - a dense proof - which is
    the trailing zero.
    """
    out = bytearray()
    out += _u64(m) + _u64(n) + _u64(k) + _u64(noise_rank)
    for proof, rows in ((a_proof, a_rows), (bt_proof, bt_rows)):
        out += _encode_merkle_proof(proof)
        out += _u64(len(rows))
        for r in rows:
            out += _u64(r)
    out += b"\0"                                  # Option<MoEProofParams>::None
    return bytes(out)


# ------------------------------------------------------------------ mining
def mine_once(header_bytes, target, config, m, n, rng, salted=True):
    """One attempt: pick A and B, and check every transcript they produce.

    Returns None, or a dict holding everything a submission needs. The matrices
    ARE the nonce, so a new attempt is simply a new draw here.
    """
    k, rank = config.common_dim, config.rank
    tile_h = config.rows_pattern.size()
    tile_w = config.cols_pattern.size()

    # A and B must stay inside [-64, 63] so that adding noise cannot overflow
    # int8. The chain checks this on the verifier side too.
    A = rng.integers(-64, 64, size=(m, k), dtype=np.int64).astype(np.int8)
    B = rng.integers(-64, 64, size=(k, n), dtype=np.int64).astype(np.int8)
    B_t = np.ascontiguousarray(B.T)

    key = job_key(header_bytes, config)
    a_root = blake3_keyed(key, pad_to_chunk_boundary(A.tobytes()))
    bt_root = blake3_keyed(key, pad_to_chunk_boundary(B_t.tobytes()))
    seed_a, seed_b = commitments(a_root, bt_root, key, m, n, salted=salted)

    gen = NoiseGenerator(noise_rank=rank)
    E_AL, E_AR, E_BL, E_BR = gen.generate(seed_a, seed_b, m, k, n)
    A_n = (A.astype(np.int32) + E_AL.astype(np.int32) @ E_AR.astype(np.int32)).astype(np.int8)
    B_n = (B.astype(np.int32) + E_BL.astype(np.int32) @ E_BR.astype(np.int32)).astype(np.int8)

    bound = config.penalized_target(target)
    if bound is None:
        raise ValueError("target does not scale into 256 bits for this configuration")

    from blake3 import blake3

    # Tiles accumulate independently over k, so this walks the whole output in
    # rank-sized column strips and keeps one transcript per hash tile.
    for i in range(0, m - m % rank, rank):
        for j in range(0, n - n % rank, rank):
            th, tw = rank // tile_h, rank // tile_w
            ts = [[Transcript() for _ in range(tw)] for _ in range(th)]
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
                            reduction, xor_reduce_tile(np.ascontiguousarray(tile)))
                reduction += 1

            for hi in range(th):
                for wi in range(tw):
                    digest = blake3(ts[hi][wi].to_bytes(), key=seed_a).digest()
                    if int.from_bytes(digest, "little") <= bound:
                        return _open(A, B_t, key, m, n, k, rank,
                                     i + hi * tile_h, j + wi * tile_w,
                                     tile_h, tile_w, digest)
    return None


def _open(A, B_t, key, m, n, k, rank, t_rows, t_cols, tile_h, tile_w, digest):
    """Turn a winning tile into the two Merkle openings a proof carries."""
    a_rows = list(range(t_rows, t_rows + tile_h))
    b_cols = list(range(t_cols, t_cols + tile_w))

    a_tree = MerkleTree(pad_to_chunk_boundary(A.tobytes()), key)
    bt_tree = MerkleTree(pad_to_chunk_boundary(B_t.tobytes()), key)
    a_proof = a_tree.multileaf_proof(leaf_indices_from_rows(a_rows, k))
    bt_proof = bt_tree.multileaf_proof(leaf_indices_from_rows(b_cols, k))

    return {
        "t_rows": t_rows, "t_cols": t_cols, "digest": digest,
        "a_rows": a_rows, "b_cols": b_cols,
        "plain_proof": encode_plain_proof(m, n, k, rank, a_proof, a_rows,
                                          bt_proof, b_cols),
        "a_proof": a_proof, "bt_proof": bt_proof, "key": key,
        "A": A, "B_t": B_t,
    }


def bits_to_target(bits):
    """Bitcoin's compact difficulty encoding: one exponent byte, three mantissa."""
    return (bits & 0xFFFFFF) << (8 * (((bits >> 24) & 0xFF) - 3))


# ----------------------------------------------------------- gateway client
class Gateway:
    """Line-delimited JSON-RPC to pearl-gateway.

    The gateway owns the node: it builds the coinbase, assembles the block and
    runs the plonky2 prover that turns a PlainProof into a certificate. A miner
    cannot submit a block on its own however much it wants to - the proving
    step is not optional and not reimplementable in a mining loop.
    """

    def __init__(self, host="127.0.0.1", port=8099, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.fh = self.sock.makefile("rwb")
        self.next_id = 1

    def call(self, method, params=None):
        req = {"jsonrpc": "2.0", "method": method, "params": params or {},
               "id": self.next_id}
        self.next_id += 1
        self.fh.write((json.dumps(req) + "\n").encode())
        self.fh.flush()
        line = self.fh.readline()
        if not line:
            raise RuntimeError("gateway closed the connection")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"gateway error: {resp['error']}")
        return resp["result"]

    def mining_info(self):
        r = self.call("getMiningInfo")
        return (b64decode(r["incomplete_header_bytes"]), int(r["target"]),
                int(r["cert_version"]))

    def submit(self, plain_proof, header_bytes, target, cert_version):
        return self.call("submitPlainProof", {
            "plain_proof": b64encode(plain_proof).decode(),
            "mining_job": {
                "incomplete_header_bytes": b64encode(header_bytes).decode(),
                "target": target,
                "cert_version": cert_version,
            },
        })


# ------------------------------------------------------------------- tests
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
    from blake3 import blake3

    print("1. blake3 port matches the library")
    key = bytes(range(32))
    for size in (0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 3000, 4096, 7 * 1024, 11 * 1024):
        data = bytes((i * 37 + size) & 0xFF for i in range(size))
        check(f"keyed hash of {size} bytes",
              blake3_keyed(key, data) == blake3(data, key=key).digest())
    check("a flipped byte changes the digest",
          blake3_keyed(key, b"\x00" * 2048) != blake3_keyed(key, b"\x01" + b"\x00" * 2047))

    print("2. merkle root is the plain keyed hash, and proofs verify")
    for leaves in (2, 3, 5, 8, 11, 16, 32):
        data = bytes((i * 91) & 0xFF for i in range(leaves * CHUNK_LEN))
        tree = MerkleTree(data, key)
        check(f"{leaves}-leaf root == blake3(data)",
              tree.root == blake3(data, key=key).digest())
        idx = sorted({0, leaves // 2, leaves - 1})
        proof = tree.multileaf_proof(idx)
        check(f"{leaves}-leaf proof recomputes the root",
              verify_multileaf_proof(proof, key) == tree.root)

    # Without this the proof builder could be emitting the right number of
    # siblings in the wrong order and every check above would still pass.
    data = bytes((i * 13) & 0xFF for i in range(16 * CHUNK_LEN))
    tree = MerkleTree(data, key)
    proof = tree.multileaf_proof([2, 3, 4, 5])
    check("adjacent leaves share siblings (multi-leaf, not four single paths)",
          len(proof["siblings"]) < 4 * 4, str(len(proof["siblings"])))
    bad = dict(proof, siblings=list(reversed(proof["siblings"])))
    check("a reordered sibling list does NOT verify",
          verify_multileaf_proof(bad, key) != tree.root)
    bad = dict(proof, leaf_data=[b"\x00" * CHUNK_LEN] + proof["leaf_data"][1:])
    check("a corrupted leaf does NOT verify",
          verify_multileaf_proof(bad, key) != tree.root)

    print("3. periodic patterns round-trip")
    p = PeriodicPattern.from_list(list(range(16)))
    check("contiguous 16 serialises to 6 bytes", p.to_bytes() == bytes([0, 15, 0, 0, 0, 0]),
          p.to_bytes().hex())
    check("size is 16", p.size() == 16)
    check("to_list recovers the input", p.to_list() == list(range(16)))
    prod = PeriodicPattern.from_list([0, 1, 8, 9, 16, 17, 24, 25])
    check("production-style strided pattern round-trips",
          prod.to_list() == [0, 1, 8, 9, 16, 17, 24, 25])

    print("4. mining configuration")
    cfg = MiningConfiguration.contiguous(k=2048, rank=128)
    check("serialises to 52 bytes", len(cfg.to_bytes()) == MiningConfiguration.SERIALIZED_SIZE,
          str(len(cfg.to_bytes())))
    check("header + config is exactly two blake3 blocks", 76 + 52 == 128)
    check("k=2048 rank=128 tile 16x16 is legal", cfg.check(256, 256) == [],
          str(cfg.check(256, 256)))
    check("rank 64 is rejected before any work is done",
          any("rank" in s for s in MiningConfiguration.contiguous(2048, 64).check(256, 256)))
    check("k below 16r is rejected",
          any("k " in s for s in MiningConfiguration.contiguous(1024, 128).check(256, 256)))

    print("5. the rank penalty")
    t = 2**232
    at_base = cfg.penalized_target(t)
    check("at rank 128 the penalty is exactly the work factor",
          at_base == t * cfg.tile_size * (2048 // 128) * 128)
    cfg256 = MiningConfiguration.contiguous(k=4096, rank=256)
    a = MiningConfiguration.contiguous(k=4096, rank=128).penalized_target(t)
    b = cfg256.penalized_target(t)
    check("doubling the rank halves the bound for identical work", b * 2 == a,
          f"{a} vs {b}")
    check("an unusably easy target reports None instead of accepting everything",
          cfg.penalized_target(U256_MAX) is None)

    print("6. commitment chain")
    hdr = bytes(range(76))
    jk = job_key(hdr, cfg)
    a_root, bt_root = bytes(range(32)), bytes(range(32, 64))
    ca, cb = commitments(a_root, bt_root, jk, 256, 256)
    check("commitment_B depends only on the key and B's root",
          cb == blake3(jk + _bind_root(bt_root, 256, SEED_SALT_B)).digest())
    check("commitment_A folds in A's root after B's", ca == blake3(cb + _bind_root(a_root, 256, SEED_SALT_A)).digest())
    ca2, _ = commitments(a_root, bt_root, jk, 257, 256)
    check("V3 salting binds m: changing it changes the seed", ca != ca2)
    ca3, _ = commitments(a_root, bt_root, jk, 256, 256, salted=False)
    check("the unsalted V1/V2 derivation differs from V3", ca != ca3)

    print("7. plain proof encoding")
    tree = MerkleTree(pad_to_chunk_boundary(bytes(16 * CHUNK_LEN)), key)
    pr = tree.multileaf_proof([0, 1])
    blob = encode_plain_proof(256, 256, 2048, 128, pr, [0, 1], pr, [0, 1])
    check("starts with m, n, k, rank as u64 little-endian",
          struct.unpack("<4Q", blob[:32]) == (256, 256, 2048, 128))
    check("ends with the dense Option::None tag", blob[-1] == 0)
    expect = 32 + 2 * (len(_encode_merkle_proof(pr)) + 8 + 2 * 8) + 1
    check("length is exactly the bincode layout", len(blob) == expect,
          f"{len(blob)} vs {expect}")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


# ----------------------------------------------------------------- vectors
VECTOR_MAGIC = b"PRLJ0001"


def synth_matrix(rows, cols, salt):
    """A matrix any language can regenerate from three integers.

    The vectors file stays a few hundred bytes this way instead of carrying
    megabytes of int8, and the C++ side still checks against fixed expected
    outputs rather than against its own recomputation.
    """
    idx = np.arange(rows * cols, dtype=np.int64)
    return (((idx * 37 + salt) & 0x7F) - 64).astype(np.int8).reshape(rows, cols)


def emit_vectors(path, m=64, n=48, k=2048, rank=128):
    header = bytes((i * 7 + 3) & 0xFF for i in range(76))
    cfg = MiningConfiguration.contiguous(k, rank)
    key = job_key(header, cfg)

    A = synth_matrix(m, k, 11)
    B_t = synth_matrix(n, k, 91)
    a_root = blake3_keyed(key, pad_to_chunk_boundary(A.tobytes()))
    bt_root = blake3_keyed(key, pad_to_chunk_boundary(B_t.tobytes()))
    ca, cb = commitments(a_root, bt_root, key, m, n)

    # Row 16 onwards on both sides: not row 0, so an off-by-one in the leaf
    # arithmetic shows up rather than landing on leaf 0 either way.
    a_rows = list(range(16, 32))
    b_cols = list(range(16, 32))
    a_proof = MerkleTree(pad_to_chunk_boundary(A.tobytes()), key).multileaf_proof(
        leaf_indices_from_rows(a_rows, k))
    bt_proof = MerkleTree(pad_to_chunk_boundary(B_t.tobytes()), key).multileaf_proof(
        leaf_indices_from_rows(b_cols, k))
    blob = encode_plain_proof(m, n, k, rank, a_proof, a_rows, bt_proof, b_cols)

    bound = cfg.penalized_target(2**232)
    with open(path, "wb") as fh:
        fh.write(VECTOR_MAGIC)
        fh.write(np.array([m, n, k, rank], dtype=np.int32).tobytes())
        fh.write(header)
        fh.write(cfg.to_bytes())
        fh.write(key)
        fh.write(a_root + bt_root + ca + cb)
        fh.write(bound.to_bytes(32, "little"))
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
    print(f"wrote {path}: m={m} n={n} k={k} rank={rank}, "
          f"proof {len(blob)} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-vectors", metavar="PATH",
                    help="write fixed vectors for the C++ side to check against")
    ap.add_argument("--mine", action="store_true",
                    help="mine against a running pearl-gateway")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--attempts", type=int, default=20)
    args = ap.parse_args()

    if args.emit_vectors:
        emit_vectors(args.emit_vectors)
        sys.exit(0)

    if not args.mine:
        sys.exit(main())

    gw = Gateway(args.host, args.port)
    hdr, target, cert = gw.mining_info()
    cfg = MiningConfiguration.contiguous(args.k, args.rank)
    problems = cfg.check(args.m, args.n)
    if problems:
        print("illegal mining configuration:")
        for p in problems:
            print("  -", p)
        sys.exit(2)

    print(f"height header {hdr[:4].hex()}..., target {target:#x}, cert v{cert}")
    print(f"m={args.m} n={args.n} k={args.k} rank={args.rank}, "
          f"bound {cfg.penalized_target(target):#x}")

    rng = np.random.default_rng()
    for attempt in range(args.attempts):
        win = mine_once(hdr, target, cfg, args.m, args.n, rng,
                        salted=cert >= 3)
        if win is None:
            print(f"attempt {attempt}: nothing")
            continue
        print(f"attempt {attempt}: tile at rows {win['t_rows']}, cols {win['t_cols']}")
        print(f"  digest {win['digest'].hex()}")
        assert verify_multileaf_proof(win["a_proof"], win["key"]) == win["a_proof"]["root"]
        assert verify_multileaf_proof(win["bt_proof"], win["key"]) == win["bt_proof"]["root"]
        print(f"  proof {len(win['plain_proof'])} bytes, both openings verify locally")
        print("  submitting:", gw.submit(win["plain_proof"], hdr, target, cert))
        break

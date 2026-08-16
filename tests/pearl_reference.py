#!/usr/bin/env python3
"""
Pearl NoisyGEMM reference implementation, ported from Pearl Research Labs'
miner-base (ISC licensed) to numpy + blake3, with no torch dependency.

Purpose is the same as reference.py does for Autolykos v2: prove the algorithm
is understood correctly in plain Python before any of it is written as a
kernel. A subtly wrong kernel finds nothing and reports no error, so the only
defence is a reference that is independently checkable.

What the algorithm actually is, since the name oversells it:

  * A and B are int8 matrices. Noise matrices E_AL, E_AR, E_BL, E_BR are
    derived deterministically from the block's commitment hash.
  * A_noised = A + E_AL @ E_AR, B_noised = B + E_BL @ E_BR.
  * The noised product is computed in tiles of `noise_rank`. After each full
    k-step every 16x16 output tile is XOR-reduced to one uint32 - that is the
    whole "inner hash", it is not a hash function - and folded into a 64-byte
    transcript by data[i] = rotl32(data[i], 13) ^ h.
  * PoW is blake3(transcript, key=pow_key) <= target. blake3 runs ONCE per
    transcript, not per element.
  * The noise then cancels exactly: C = C_noised - A@E_BL@E_BR - E_AL@E_AR@B.

That last identity is the strongest check available without network data, and
this file leans on it: if the port is wrong anywhere in the noise APPLICATION
path, the denoised result stops equalling A @ B exactly.

Know what it does not cover. The cancellation is algebraic, so it holds for
ANY noise matrices as long as the int8 truncation of E stays lossless. Proved
by mutation: corrupting the permutation matrix to put +1,+1 in a column
instead of +1,-1 leaves "C == A @ B" passing and is caught only by the
structural test in section 1. So the identity validates how noise is applied,
never how it is generated. Both halves need their own tests, and the
generation half is covered by the shape, range, structure and determinism
checks rather than by the identity.

Note for the kernel work: XOR is commutative and associative, so reduction
ORDER cannot make a device disagree with this reference. Any mismatch is a
real bug - wrong tile, wrong accumulation, or overflow - never an ordering
artifact.
"""
import struct
import sys
from math import ceil

import numpy as np
from blake3 import blake3

TRANSCRIPT_SIZE_U32 = 16          # 64 bytes, one blake3 message block
HASH_ACCUMULATE_ROTATION = 13
POW_TARGET_EASIEST = 2**256 - 1
BLAKE3_DIGEST_SIZE = 32


def rotl32(x: np.uint32, n: int) -> np.uint32:
    x = np.uint32(x)
    return np.uint32((np.uint32(x) << np.uint32(n)) | (np.uint32(x) >> np.uint32(32 - n)))


def mul_hi_u32(a: int, b: np.uint32) -> np.uint32:
    """High 32 bits of a 32x32 multiply, matching np_mul_hi_u32 upstream."""
    return np.uint32((np.uint64(a) * np.uint64(b)) >> np.uint64(32))


class Transcript:
    """64-byte accumulator, one per hash tile."""

    def __init__(self) -> None:
        self.data = np.zeros(TRANSCRIPT_SIZE_U32, dtype=np.uint32)

    def rotl_xor_into(self, reduction_count: int, combined_hash: np.uint32) -> None:
        # The rotation is what stops an identical hash cancelling itself out
        # when it lands on the same slot twice.
        idx = reduction_count % TRANSCRIPT_SIZE_U32
        self.data[idx] = rotl32(self.data[idx], HASH_ACCUMULATE_ROTATION) ^ np.uint32(combined_hash)

    def to_bytes(self) -> bytes:
        return b"".join(struct.pack("<I", int(w)) for w in self.data)


# --------------------------------------------------------------- noise
class NoiseGenerator:
    def __init__(self, noise_rank: int = 128, noise_range: int = 128):
        if noise_range & (noise_range - 1) or noise_range == 0:
            raise ValueError("noise_range must be a power of two")
        if noise_rank & (noise_rank - 1) or noise_rank == 0:
            raise ValueError("noise_rank must be a power of two")
        if noise_range > 128:
            raise ValueError("noise_range must fit in uint7")
        if noise_rank % BLAKE3_DIGEST_SIZE:
            raise ValueError("noise_rank must be divisible by the blake3 digest size")

        self.noise_rank = noise_rank
        # Two index draws per column, so the usable range is halved before the
        # zero-point shift. With the defaults this yields entries in [-32, 31].
        half = noise_range // 2
        self.zero_point_translation = half // 2
        self.range_mask = half - 1
        self.rank_mask = noise_rank - 1

    @staticmethod
    def _random_hash(index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
        prepend = np.zeros(8, dtype=np.int32)
        prepend[prepend_index] = 1 + index
        return blake3(prepend.tobytes() + seed, key=key).digest()

    def _uniform(self, seed: bytes, key: bytes, rows: int) -> np.ndarray:
        assert len(seed) == 32 and len(key) == 32
        cols = self.noise_rank
        draws = int(ceil(rows * cols / BLAKE3_DIGEST_SIZE))
        raw = b"".join(self._random_hash(i, seed, key, 0) for i in range(draws))
        vals = np.frombuffer(raw, dtype=np.uint8)[: rows * cols].astype(np.int16)
        out = ((vals & self.range_mask) - self.zero_point_translation).astype(np.int8)
        return out.reshape(rows, cols)

    def _permutation(self, seed: bytes, key: bytes, rows: int, cols: int,
                     assign_columns: bool) -> np.ndarray:
        assert len(seed) == 32 and len(key) == 32
        assert rows == self.noise_rank or cols == self.noise_rank
        out = np.zeros((rows, cols), dtype=np.int8)

        required = cols if assign_columns else rows
        per_line = 4
        draws = int(ceil(required * per_line / BLAKE3_DIGEST_SIZE))

        for i in range(draws):
            words = np.frombuffer(self._random_hash(i, seed, key, 1), dtype=np.uint32)
            for k in range(BLAKE3_DIGEST_SIZE // per_line):
                r = words[k]
                first = int(r & np.uint32(self.rank_mask))
                # The xor operand is always >= 1, so second != first always.
                second = first ^ int(1 + mul_hi_u32(self.noise_rank - 1, r))
                idx = i * (BLAKE3_DIGEST_SIZE // per_line) + k
                if idx >= required:
                    break
                col = np.zeros(self.noise_rank, dtype=np.int8)
                col[first] = 1
                col[second] = -1
                if assign_columns:
                    out[:, idx] = col
                else:
                    out[idx, :] = col
        return out

    def generate(self, key_A: bytes, key_B: bytes, A_rows: int, common_dim: int,
                 B_cols: int):
        for nm, v in (("A_rows", A_rows), ("common_dim", common_dim), ("B_cols", B_cols)):
            if v < self.noise_rank:
                raise ValueError(f"{nm} must be >= noise_rank ({self.noise_rank}), got {v}")
        seed_A = b"A_tensor" + b"\x00" * 24
        seed_B = b"B_tensor" + b"\x00" * 24

        E_AL = self._uniform(seed_A, key_A, A_rows)
        E_AR = self._permutation(seed_A, key_A, self.noise_rank, common_dim, True)
        E_BL = self._permutation(seed_B, key_B, common_dim, self.noise_rank, False)
        E_BR = self._uniform(seed_B, key_B, B_cols).T.copy()
        return E_AL, E_AR, E_BL, E_BR


# ----------------------------------------------------------- noisy gemm
def xor_reduce_tile(tile: np.ndarray) -> np.uint32:
    """The 'inner hash': XOR every int32 in the tile, reinterpreted as uint32."""
    assert tile.dtype == np.int32
    return np.bitwise_xor.reduce(tile.reshape(-1).view(np.uint32))


class NoisyGemm:
    def __init__(self, noise_rank=128, noise_range=128, hash_tile_h=16, hash_tile_w=16):
        self.noise_rank = noise_rank
        self.noise_range = noise_range
        self.hash_tile_h = hash_tile_h
        self.hash_tile_w = hash_tile_w
        self.opened = None

    def _check_data_range(self, m: np.ndarray, label: str) -> None:
        span = 256 - self.noise_range
        lo = -span // 2
        hi = lo + span - 1
        if m.min() < lo or m.max() > hi:
            raise ValueError(f"{label} must lie in [{lo}, {hi}] for noise_range="
                             f"{self.noise_range}, got [{m.min()}, {m.max()}]")

    def noise_A(self, A, E_AL, E_AR, E_BL):
        self._check_data_range(A, "A")
        E_A = (E_AL.astype(np.int32) @ E_AR.astype(np.int32))
        self._check_noise_range(E_A)
        A_noised = (A.astype(np.int32) + E_A).astype(np.int8)
        A_E_BL = A.astype(np.int32) @ E_BL.astype(np.int32)
        return A_noised, A_E_BL

    def noise_B(self, B, E_AR, E_BL, E_BR):
        self._check_data_range(B, "B")
        E_B = (E_BL.astype(np.int32) @ E_BR.astype(np.int32))
        self._check_noise_range(E_B)
        B_noised = (B.astype(np.int32) + E_B).astype(np.int8)
        EAR_BpEB = E_AR.astype(np.int32) @ B_noised.astype(np.int32)
        return B_noised, EAR_BpEB

    def _check_noise_range(self, E: np.ndarray) -> None:
        z = self.noise_range // 2
        if E.min() < -z or E.max() > self.noise_range - z:
            raise ValueError(f"noise outside [{-z}, {self.noise_range - z}]: "
                             f"[{E.min()}, {E.max()}]")

    def _tiled_matmul(self, A, B, pow_key, pow_target):
        m, k = A.shape
        n = B.shape[1]
        C = np.zeros((m, n), dtype=np.int32)
        found = False

        for i in range(0, m, self.noise_rank):
            i_max = min(i + self.noise_rank, m)
            for j in range(0, n, self.noise_rank):
                j_max = min(j + self.noise_rank, n)
                bh, bw = i_max - i, j_max - j

                th = bh // self.hash_tile_h
                tw = bw // self.hash_tile_w
                transcripts = [[Transcript() for _ in range(tw)] for _ in range(th)]

                C_block = np.zeros((bh, bw), dtype=np.int32)
                reduction = 0
                has_full = False
                for p in range(0, k, self.noise_rank):
                    p_max = min(p + self.noise_rank, k)
                    C_block = C_block + (A[i:i_max, p:p_max].astype(np.int32)
                                         @ B[p:p_max, j:j_max].astype(np.int32))
                    full = (bh >= self.hash_tile_h and bw >= self.hash_tile_w
                            and p_max - p == self.noise_rank)
                    if full:
                        for hi in range(th):
                            for wi in range(tw):
                                tile = C_block[hi * self.hash_tile_h:(hi + 1) * self.hash_tile_h,
                                               wi * self.hash_tile_w:(wi + 1) * self.hash_tile_w]
                                transcripts[hi][wi].rotl_xor_into(
                                    reduction, xor_reduce_tile(np.ascontiguousarray(tile)))
                        reduction += 1
                        has_full = True

                if not found and has_full:
                    for hi in range(th):
                        for wi in range(tw):
                            digest = blake3(transcripts[hi][wi].to_bytes(),
                                            key=pow_key).digest()
                            if int.from_bytes(digest, "little") <= pow_target:
                                self.opened = {
                                    "A_row_indices": list(range(i + hi * self.hash_tile_h,
                                                                i + (hi + 1) * self.hash_tile_h)),
                                    "B_column_indices": list(range(j + wi * self.hash_tile_w,
                                                                   j + (wi + 1) * self.hash_tile_w)),
                                }
                                found = True
                                break
                        if found:
                            break

                C[i:i_max, j:j_max] = C_block
        return C, found

    def noisy_gemm(self, A, B, E_AL, E_AR, E_BL, E_BR, pow_key, pow_target=0):
        A_noised, A_E_BL = self.noise_A(A, E_AL, E_AR, E_BL)
        B_noised, EAR_BpEB = self.noise_B(B, E_AR, E_BL, E_BR)

        C_noised, found = self._tiled_matmul(A_noised, B_noised, pow_key, pow_target)

        A_EB = A_E_BL @ E_BR.astype(np.int32)
        EA_BpEB = E_AL.astype(np.int32) @ EAR_BpEB
        return C_noised - A_EB - EA_BpEB, found


# ------------------------------------------------------------------ tests
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def main() -> int:
    rng = np.random.default_rng(20260816)
    rank = 128
    m = n = k = 256
    key_A = bytes(range(32))
    key_B = bytes(range(32, 64))

    print("1. noise generation")
    gen = NoiseGenerator(noise_rank=rank)
    E_AL, E_AR, E_BL, E_BR = gen.generate(key_A, key_B, m, k, n)
    check("E_AL shape m x r", E_AL.shape == (m, rank), str(E_AL.shape))
    check("E_AR shape r x k", E_AR.shape == (rank, k), str(E_AR.shape))
    check("E_BL shape k x r", E_BL.shape == (k, rank), str(E_BL.shape))
    check("E_BR shape r x n", E_BR.shape == (rank, n), str(E_BR.shape))
    check("uniform entries in [-32, 31]", E_AL.min() >= -32 and E_AL.max() <= 31,
          f"[{E_AL.min()}, {E_AL.max()}]")
    check("permutation entries in {-1,0,1}", set(np.unique(E_AR).tolist()) <= {-1, 0, 1},
          str(np.unique(E_AR)))
    check("each E_AR column has exactly one +1 and one -1",
          bool(np.all((E_AR == 1).sum(axis=0) == 1) and np.all((E_AR == -1).sum(axis=0) == 1)))
    check("deterministic for the same key",
          np.array_equal(gen.generate(key_A, key_B, m, k, n)[0], E_AL))
    check("different key gives different noise",
          not np.array_equal(gen.generate(bytes(32), key_B, m, k, n)[0], E_AL))

    print("2. transcript accumulation")
    t = Transcript()
    t.rotl_xor_into(0, np.uint32(0xDEADBEEF))
    check("first write lands verbatim", t.data[0] == np.uint32(0xDEADBEEF), hex(int(t.data[0])))
    t2 = Transcript()
    t2.rotl_xor_into(0, np.uint32(0xABCD))
    t2.rotl_xor_into(16, np.uint32(0xABCD))
    check("same hash twice does NOT cancel (this is what the rotate is for)",
          t2.data[0] != 0, hex(int(t2.data[0])))
    check("writes cycle across all 16 slots",
          all(Transcript_slot_used(i) for i in range(TRANSCRIPT_SIZE_U32)))

    print("3. inner hash is XOR reduction, not a hash function")
    tile = np.arange(256, dtype=np.int32).reshape(16, 16)
    expect = np.bitwise_xor.reduce(tile.reshape(-1).view(np.uint32))
    check("xor_reduce_tile matches a plain XOR fold", xor_reduce_tile(tile) == expect)
    shuffled = tile.reshape(-1).copy()
    rng.shuffle(shuffled)
    check("order independent (commutative/associative)",
          xor_reduce_tile(shuffled.reshape(16, 16)) == expect)

    print("4. the denoise identity - the real correctness check")
    span = 256 - 128
    lo = -span // 2
    A = rng.integers(lo, lo + span, size=(m, k), dtype=np.int64).astype(np.int8)
    B = rng.integers(lo, lo + span, size=(k, n), dtype=np.int64).astype(np.int8)
    ng = NoisyGemm(noise_rank=rank)
    C, found = ng.noisy_gemm(A, B, E_AL, E_AR, E_BL, E_BR, pow_key=key_A, pow_target=0)
    truth = A.astype(np.int32) @ B.astype(np.int32)
    check("noise cancels exactly: C == A @ B", np.array_equal(C, truth),
          f"max abs diff {np.abs(C - truth).max()}")
    check("hardest target finds nothing", not found)

    print("5. pow target behaves monotonically")
    _, found_easy = ng.noisy_gemm(A, B, E_AL, E_AR, E_BL, E_BR, key_A, POW_TARGET_EASIEST)
    check("easiest target always finds a block", found_easy)
    check("a found block records its tile coordinates",
          ng.opened is not None and len(ng.opened["A_row_indices"]) == 16)

    print("6. the paper's finding, reproduced")
    # arXiv:2606.04819 reports the verifier accepts random matrices. Nothing in
    # the PoW binds A and B to meaningful work, so random inputs mine fine.
    Ar = rng.integers(lo, lo + span, size=(m, k), dtype=np.int64).astype(np.int8)
    Br = rng.integers(lo, lo + span, size=(k, n), dtype=np.int64).astype(np.int8)
    ng2 = NoisyGemm(noise_rank=rank)
    _, found_rand = ng2.noisy_gemm(Ar, Br, E_AL, E_AR, E_BL, E_BR, key_A, POW_TARGET_EASIEST)
    check("random A and B mine exactly as well as real ones", found_rand)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


def Transcript_slot_used(i: int) -> bool:
    t = Transcript()
    t.rotl_xor_into(i, np.uint32(0xFFFF_FFFF))
    return bool(t.data[i] != 0)


# ------------------------------------------------------------- vectors
VECTOR_MAGIC = b"PRLV0002"   # 0002 adds pow_key and the expected digests


def emit_vectors(path: str, m=256, n=256, k=256, rank=128) -> None:
    """Write one test case for the device tests to check themselves against.

    Cross-language comparison needs a fixed artifact, not a re-run: if the
    device test regenerated its own inputs it could agree with itself while
    both sides drifted from the reference together.

    Layout is deliberately dumb - magic, six int32 dims, then raw little-endian
    arrays - so the C++ side needs no parser.
    """
    gen = NoiseGenerator(noise_rank=rank)
    key_A = bytes(range(32))
    key_B = bytes(range(32, 64))
    E_AL, E_AR, E_BL, E_BR = gen.generate(key_A, key_B, m, k, n)

    rng = np.random.default_rng(20260816)
    span = 256 - 128
    lo = -span // 2
    A = rng.integers(lo, lo + span, size=(m, k), dtype=np.int64).astype(np.int8)
    B = rng.integers(lo, lo + span, size=(k, n), dtype=np.int64).astype(np.int8)

    ng = NoisyGemm(noise_rank=rank)
    A_noised, _ = ng.noise_A(A, E_AL, E_AR, E_BL)
    B_noised, _ = ng.noise_B(B, E_AR, E_BL, E_BR)
    C, _ = ng.noisy_gemm(A, B, E_AL, E_AR, E_BL, E_BR, key_A, pow_target=0)

    # The noised product is what a kernel actually computes, so hand the device
    # both that and the transcripts rather than only the denoised result.
    C_noised = np.zeros((m, n), dtype=np.int32)
    transcripts = []
    for i in range(0, m, rank):
        for j in range(0, n, rank):
            blk = np.zeros((rank, rank), dtype=np.int32)
            th = rank // ng.hash_tile_h
            tw = rank // ng.hash_tile_w
            ts = [[Transcript() for _ in range(tw)] for _ in range(th)]
            red = 0
            for p in range(0, k, rank):
                blk = blk + (A_noised[i:i + rank, p:p + rank].astype(np.int32)
                             @ B_noised[p:p + rank, j:j + rank].astype(np.int32))
                for hi in range(th):
                    for wi in range(tw):
                        tile = blk[hi * 16:(hi + 1) * 16, wi * 16:(wi + 1) * 16]
                        ts[hi][wi].rotl_xor_into(
                            red, xor_reduce_tile(np.ascontiguousarray(tile)))
                red += 1
            C_noised[i:i + rank, j:j + rank] = blk
            for hi in range(th):
                for wi in range(tw):
                    transcripts.append(ts[hi][wi].data)

    # blake3 of each transcript under the pow key. This is the whole PoW
    # check, and it is what the device has to reproduce exactly.
    digests = [blake3(t.tobytes(), key=key_A).digest() for t in transcripts]

    with open(path, "wb") as fh:
        fh.write(VECTOR_MAGIC)
        fh.write(np.array([m, n, k, rank, ng.hash_tile_h, len(transcripts)],
                          dtype=np.int32).tobytes())
        fh.write(A.tobytes())
        fh.write(B.tobytes())
        fh.write(E_AL.tobytes())
        fh.write(np.ascontiguousarray(E_AR).tobytes())
        fh.write(np.ascontiguousarray(E_BL).tobytes())
        fh.write(np.ascontiguousarray(E_BR).tobytes())
        fh.write(A_noised.tobytes())
        fh.write(B_noised.tobytes())
        fh.write(C_noised.tobytes())
        fh.write(C.tobytes())
        fh.write(np.ascontiguousarray(np.array(transcripts, dtype=np.uint32)).tobytes())
        fh.write(key_A)
        fh.write(b"".join(digests))
    print(f"wrote {path}: {m}x{k} @ {k}x{n}, rank {rank}, "
          f"{len(transcripts)} transcripts, {len(digests)} digests")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--emit-vectors":
        emit_vectors(sys.argv[2])
        sys.exit(0)
    sys.exit(main())

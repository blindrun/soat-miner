// Pearl's prepare stage on the device: everything between "here is a nonce"
// and "here are two noised matrices the GEMM can eat".
//
// One attempt is:
//
//   1. generate A (m x k) and B^t (n x k) from the nonce
//   2. keyed-blake3 each into its Merkle root
//   3. fold the two roots into commitment_A and commitment_B
//   4. draw the noise from those commitments
//   5. apply it, producing A_noised (m x k) and B_noised (k x n)
//
// then noisy_gemm.cuh takes over. All five stay on the device and on one
// stream: step 3 is four hashes of a single block, which would be free on the
// host except that reading the roots back would force a synchronise per
// attempt.
//
// This stage is NOT cheap, which is worth stating plainly because the obvious
// estimate says it is. Measured on a 4090 at m=n=4096, k=2048 (see section 7
// of tests/test_pearl_prepare.cu), against a 1.05 ms GEMM:
//
//     generate A  0.027   hash A     0.124   noise draws  0.051
//     generate B  0.023   hash B^t   0.123   noise A      0.025
//     commitments 0.008                      noise B      0.121
//
// 0.50 ms, a 48% overhead. Hashing dominates, and the B noising is slow
// because it transposes.
//
// The fix is not a faster kernel, it is reading the derivation again:
//
//     commitment_B = blake3(job_key || salted B^t root)
//     commitment_A = blake3(commitment_B || salted A root)
//
// commitment_B does not depend on A. E_BL and E_BR are drawn from it, so if
// only A varies between attempts then B_noised is IDENTICAL every time -
// generating B, hashing it and noising it are once per JOB. Only A's chain and
// the PoW key recur, and A is still a free choice, so nothing is given up.
// That is 0.50 ms down to 0.196, and 42.2 to 52.6 M candidates/s.
//
// Shape then matters, and the obvious metric picks the wrong one. Hashing A
// grows as m*k and the GEMM as m*n*k, so overhead falls monotonically as the
// shape widens - but candidates per second does not (measured with the
// register-tiled kernel, section 7 of tests/test_pearl_prepare.cu):
//
//     4096x4096    32.1% overhead    83.5 M/s   115.7 TOPS
//     1024x16384   26.3%             87.4       115.8
//     2048x32768    7.5%            102.9       115.9
//     4096x32768    4.5%            106.6       116.8
//     4096x65536    1.9%             91.5        97.8
//
// The last row has the lowest overhead and is the slowest, because B^t at
// 128 MB stops being servable from L2 and the GEMM itself drops. So the shape
// is chosen on candidates per second, and 4096x32768 is the default here:
// about 180 MB of device memory, and past the point where more amortisation is
// worth anything.
//
// The one constraint all this imposes: the padded matrices must have a POWER
// OF TWO number of 1024-byte chunks, so the Merkle tree is perfectly balanced
// and the block-cooperative reduction below is exactly blake3's own tree. With
// k = 2048 that means m and n are powers of two, which costs nothing since the
// miner picks them.

#pragma once

#include <stdint.h>

#include "blake3.cuh"

namespace om {
namespace pearl {

// ------------------------------------------------------------- generation
//
// A and B are the nonce, so they have to be reproducible on the host: a win
// has to be re-derived there to build the Merkle opening. The host twin is
// om::pearl::mixSeed / fillMatrixHost in job.h, and the test checks the two
// agree byte for byte.

__device__ __forceinline__ uint64_t splitmixDev(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/**
 * Fill an int8 matrix with values in [-64, 63].
 *
 * That range is not a style choice: A and B must leave room for noise in
 * [-63, 63] without overflowing int8, and the verifier checks it.
 *
 * `count` is in bytes and must be a multiple of 8.
 */
__global__ void genMatrix(int8_t *__restrict__ out, uint64_t count, uint64_t seed) {
    const uint64_t group = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (group * 8 >= count) return;
    const uint64_t v = splitmixDev(seed + group);
    int8_t *p = out + group * 8;
#pragma unroll
    for (int i = 0; i < 8; i++)
        p[i] = (int8_t)((int)((v >> (8 * i)) & 0x7F) - 64);
}

// ----------------------------------------------------------- merkle roots

/** One thread per 1024-byte chunk. */
/**
 * A cheap order-independent fingerprint of the transcript buffer.
 *
 * Exists so the tuner can prove a configuration DID THE WORK before it is
 * allowed to win on speed. Every configuration computes the same product from
 * the same inputs, so every one of them must produce a bit-identical
 * transcript buffer; anything that fails to run, returns early, or silently
 * computes nothing leaves the zeroed buffer behind and is caught.
 */
__global__ void transcriptFingerprint(const uint32_t *__restrict__ t,
                                      uint64_t words, uint32_t *__restrict__ out) {
    uint32_t acc = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += (uint64_t)gridDim.x * blockDim.x)
        acc ^= t[i] * 2654435761u + (uint32_t)i;
    for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) atomicXor(out, acc);
}

__global__ void chunkCvs(const uint8_t *__restrict__ data, uint32_t chunks,
                         const uint32_t *__restrict__ key,
                         uint8_t *__restrict__ cvs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunks) return;
    uint32_t k[8], out[8];
#pragma unroll
    for (int j = 0; j < 8; j++) k[j] = key[j];
    b3d::chunkCv(k, data + (size_t)i * b3d::kChunkLenD, i, out);
    b3d::storeCv(out, cvs + (size_t)i * 32);
}

/**
 * Reduce 2*blockDim.x chaining values to one, per block.
 *
 * Doing several tree levels inside one launch rather than one level per launch
 * is worth caring about: a 4096x2048 matrix is 8192 chunks, so the naive way
 * is thirteen launches per matrix per attempt and the launches alone would
 * cost more than the hashing.
 *
 * Correct only for a balanced tree, which is why the chunk count is required
 * to be a power of two - see the header comment.
 */
extern __shared__ uint32_t sReduce[];

__global__ void reduceTree(const uint8_t *__restrict__ in, uint32_t count,
                           const uint32_t *__restrict__ key,
                           uint8_t *__restrict__ out) {
    const uint32_t per = blockDim.x * 2;
    const uint32_t base = blockIdx.x * per;
    if (base >= count) return;

    uint32_t k[8];
#pragma unroll
    for (int j = 0; j < 8; j++) k[j] = key[j];

    // Each slot is one 8-word chaining value.
    for (uint32_t i = threadIdx.x; i < per; i += blockDim.x) {
        uint32_t cv[8];
        b3d::loadCv(in + (size_t)(base + i) * 32, cv);
#pragma unroll
        for (int j = 0; j < 8; j++) sReduce[i * 8 + j] = cv[j];
    }
    __syncthreads();

    for (uint32_t n = per; n > 1; n >>= 1) {
        const uint32_t half = n >> 1;
        if (threadIdx.x < half) {
            uint32_t l[8], r[8], p[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                l[j] = sReduce[(threadIdx.x * 2) * 8 + j];
                r[j] = sReduce[(threadIdx.x * 2 + 1) * 8 + j];
            }
            b3d::parentCv(k, l, r, false, p);
#pragma unroll
            for (int j = 0; j < 8; j++) sReduce[threadIdx.x * 8 + j] = p[j];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint32_t cv[8];
#pragma unroll
        for (int j = 0; j < 8; j++) cv[j] = sReduce[j];
        b3d::storeCv(cv, out + (size_t)blockIdx.x * 32);
    }
}

/** The last parent, with the ROOT flag. Launch with one thread. */
__global__ void rootCv(const uint8_t *__restrict__ pair,
                       const uint32_t *__restrict__ key,
                       uint8_t *__restrict__ root) {
    uint32_t k[8], l[8], r[8], p[8];
#pragma unroll
    for (int j = 0; j < 8; j++) k[j] = key[j];
    b3d::loadCv(pair, l);
    b3d::loadCv(pair + 32, r);
    b3d::parentCv(k, l, r, true, p);
    b3d::storeCv(p, root);
}

// ------------------------------------------------------------ commitments

/**
 * Roots to commitment_A and commitment_B, in one thread.
 *
 * commitment_A is both the A-noise seed and the PoW key, so nothing downstream
 * can start until this is done - which is exactly why it runs here rather than
 * on the host: a host round trip would be a synchronise per attempt for four
 * hashes of sixty-four bytes.
 *
 * `salted` selects the V3 derivation. V1 and V2 skip it, and choosing wrong
 * yields a proof that looks well formed and is rejected by the node.
 */
__device__ inline void bindRootDev(const uint8_t *root, uint32_t dim,
                                   const uint8_t *salt, uint8_t *out) {
    // root(32) || dim u32 LE(4) || zero padding(28) is exactly one blake3 block.
    uint8_t msg[64];
    uint32_t key[8], o[8];
    for (int i = 0; i < 64; i++) msg[i] = 0;
    for (int i = 0; i < 32; i++) msg[i] = root[i];
    msg[32] = (uint8_t)dim;
    msg[33] = (uint8_t)(dim >> 8);
    msg[34] = (uint8_t)(dim >> 16);
    msg[35] = (uint8_t)(dim >> 24);
    b3d::loadCv(salt, key);
    b3d::hashBlock64(key, msg, o);
    b3d::storeCv(o, out);
}

/**
 * commitment_B, which depends on B^t and the job and NOT on A.
 *
 * Split from commitment_A on purpose: this one is per-job work. E_BL and E_BR
 * are drawn from it, so as long as only A varies between attempts, B_noised is
 * bit-identical and none of the B side has to run again. That is the
 * difference between a 48% prepare overhead and a 3% one.
 */
__global__ void deriveCommitmentB(const uint8_t *__restrict__ btRoot,
                                  const uint8_t *__restrict__ jobKey,
                                  const uint8_t *__restrict__ saltB, uint32_t n,
                                  int salted, uint8_t *__restrict__ commitB) {
    if (threadIdx.x || blockIdx.x) return;
    uint8_t b[32], msg[64];
    uint32_t out[8];
    if (salted) bindRootDev(btRoot, n, saltB, b);
    else for (int i = 0; i < 32; i++) b[i] = btRoot[i];

    for (int i = 0; i < 32; i++) { msg[i] = jobKey[i]; msg[i + 32] = b[i]; }
    b3d::hashBlock64Unkeyed(msg, out);
    b3d::storeCv(out, commitB);
}

/**
 * commitment_A, which is per-attempt: it is the A-noise seed AND the PoW key,
 * so it is the value that makes every transcript depend on the matrices.
 */
__global__ void deriveCommitmentA(const uint8_t *__restrict__ aRoot,
                                  const uint8_t *__restrict__ commitB,
                                  const uint8_t *__restrict__ saltA, uint32_t m,
                                  int salted, uint8_t *__restrict__ commitA) {
    if (threadIdx.x || blockIdx.x) return;
    uint8_t a[32], msg[64];
    uint32_t out[8];
    if (salted) bindRootDev(aRoot, m, saltA, a);
    else for (int i = 0; i < 32; i++) a[i] = aRoot[i];

    for (int i = 0; i < 32; i++) { msg[i] = commitB[i]; msg[i + 32] = a[i]; }
    b3d::hashBlock64Unkeyed(msg, out);
    b3d::storeCv(out, commitA);
}

/** Both, for tests and for the first attempt of a job. */
__global__ void deriveCommitments(const uint8_t *__restrict__ aRoot,
                                  const uint8_t *__restrict__ btRoot,
                                  const uint8_t *__restrict__ jobKey,
                                  const uint8_t *__restrict__ saltA,
                                  const uint8_t *__restrict__ saltB,
                                  uint32_t m, uint32_t n, int salted,
                                  uint8_t *__restrict__ commitA,
                                  uint8_t *__restrict__ commitB) {
    if (threadIdx.x || blockIdx.x) return;

    uint8_t a[32], b[32], msg[64];
    uint32_t out[8];
    if (salted) {
        bindRootDev(aRoot, m, saltA, a);
        bindRootDev(btRoot, n, saltB, b);
    } else {
        for (int i = 0; i < 32; i++) { a[i] = aRoot[i]; b[i] = btRoot[i]; }
    }

    for (int i = 0; i < 32; i++) { msg[i] = jobKey[i]; msg[i + 32] = b[i]; }
    b3d::hashBlock64Unkeyed(msg, out);
    b3d::storeCv(out, commitB);

    for (int i = 0; i < 32; i++) { msg[i] = commitB[i]; msg[i + 32] = a[i]; }
    b3d::hashBlock64Unkeyed(msg, out);
    b3d::storeCv(out, commitA);
}

// -------------------------------------------------------------- pow search

/**
 * Hash every transcript and compact the winners into a short list.
 *
 * The alternative - a flag per transcript, scanned on the host - means reading
 * back a megabyte per attempt to find the handful of ones that matter, or
 * usually none at all. An atomic append costs nothing when nothing wins, which
 * is the case for every attempt but one in a very long time.
 *
 * `target` is eight little-endian words, and the comparison walks from the
 * most significant down, matching the reference's
 * int.from_bytes(digest, "little") <= bound.
 */
__global__ void powScan(const uint32_t *__restrict__ transcripts, uint32_t count,
                        const uint8_t *__restrict__ key,
                        const uint32_t *__restrict__ target,
                        uint32_t *__restrict__ hitIndex,
                        uint32_t *__restrict__ hitDigest,
                        uint32_t *__restrict__ hitCount, uint32_t maxHits) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint32_t k[8], blk[16], out[8];
    b3d::loadCv(key, k);
#pragma unroll
    for (int j = 0; j < 16; j++) blk[j] = transcripts[i * 16 + j];

    uint32_t cv[8];
#pragma unroll
    for (int j = 0; j < 8; j++) cv[j] = k[j];
    b3d::compress(cv, blk, 0, 64u,
                  b3d::kKeyedHash | b3d::kChunkStart | b3d::kChunkEnd | b3d::kRoot);
#pragma unroll
    for (int j = 0; j < 8; j++) out[j] = cv[j];

    bool win = true;
    for (int j = 7; j >= 0; --j) {
        if (out[j] < target[j]) break;
        if (out[j] > target[j]) { win = false; break; }
    }
    if (!win) return;

    const uint32_t slot = atomicAdd(hitCount, 1u);
    if (slot >= maxHits) return;
    hitIndex[slot] = i;
#pragma unroll
    for (int j = 0; j < 8; j++) hitDigest[slot * 8 + j] = out[j];
}

// ------------------------------------------------------------------ noise

/**
 * `count` int8 values of uniform noise, one thread per 32-byte draw.
 *
 * Every draw is one keyed blake3 of a single block: eight int32 with a counter
 * in slot 0, then the domain seed. Embarrassingly parallel, and the whole of
 * E_AL and E_BR comes out of it.
 *
 * With noise_range 128 the mask is 63 and the shift 32, so values land in
 * [-32, 31] - half the nominal range, because two index draws share it before
 * the zero-point shift.
 */
__global__ void noiseUniform(int8_t *__restrict__ out, uint32_t count,
                             const uint8_t *__restrict__ key,
                             const uint8_t *__restrict__ seed, int mask,
                             int shift) {
    const uint32_t draw = blockIdx.x * blockDim.x + threadIdx.x;
    if (draw * 32u >= count) return;

    uint32_t k[8], digest[8];
    b3d::loadCv(key, k);

    uint8_t msg[64];
#pragma unroll
    for (int i = 0; i < 64; i++) msg[i] = 0;
    const uint32_t v = 1u + draw;
    msg[0] = (uint8_t)v; msg[1] = (uint8_t)(v >> 8);
    msg[2] = (uint8_t)(v >> 16); msg[3] = (uint8_t)(v >> 24);
    for (int i = 0; i < 32; i++) msg[32 + i] = seed[i];

    b3d::hashBlock64(k, msg, digest);

    uint8_t bytes[32];
    b3d::storeCv(digest, bytes);
    const uint32_t base = draw * 32u;
    const uint32_t n = (count - base < 32u) ? (count - base) : 32u;
    for (uint32_t i = 0; i < n; i++)
        out[base + i] = (int8_t)(((int)bytes[i] & mask) - shift);
}

/**
 * The +1 and -1 positions of a permutation matrix's lines, eight per draw.
 *
 * E_AR and E_BL are never built. E_AL @ E_AR has column j equal to
 * E_AL[:, first_j] - E_AL[:, second_j], so only the index pairs matter, and
 * the noising becomes two subtractions per element instead of a rank-long dot
 * product against a matrix of almost entirely zeros.
 *
 * The counter goes in slot 1 here rather than slot 0, which is what separates
 * the permutation draws from the uniform ones under the same key and seed.
 */
__global__ void noisePerm(uint16_t *__restrict__ first,
                          uint16_t *__restrict__ second, uint32_t count,
                          const uint8_t *__restrict__ key,
                          const uint8_t *__restrict__ seed, int rank) {
    const uint32_t draw = blockIdx.x * blockDim.x + threadIdx.x;
    if (draw * 8u >= count) return;

    uint32_t k[8], digest[8];
    b3d::loadCv(key, k);

    uint8_t msg[64];
#pragma unroll
    for (int i = 0; i < 64; i++) msg[i] = 0;
    const uint32_t v = 1u + draw;
    msg[4] = (uint8_t)v; msg[5] = (uint8_t)(v >> 8);
    msg[6] = (uint8_t)(v >> 16); msg[7] = (uint8_t)(v >> 24);
    for (int i = 0; i < 32; i++) msg[32 + i] = seed[i];

    b3d::hashBlock64(k, msg, digest);

    const uint32_t rankMask = (uint32_t)rank - 1u;
    for (int w = 0; w < 8; w++) {
        const uint32_t idx = draw * 8u + (uint32_t)w;
        if (idx >= count) break;
        const uint32_t r = digest[w];
        const uint32_t f = r & rankMask;
        // The xor operand is at least 1, so the two are never equal - which is
        // what stops a line being all zeros instead of +1/-1.
        const uint32_t hi = (uint32_t)(((uint64_t)(rank - 1) * (uint64_t)r) >> 32);
        first[idx] = (uint16_t)f;
        second[idx] = (uint16_t)(f ^ (1u + hi));
    }
}

/** A_noised[i][j] = A[i][j] + E_AL[i][first_j] - E_AL[i][second_j]. */
__global__ void applyNoiseA(const int8_t *__restrict__ A,
                            const int8_t *__restrict__ eAL,
                            const uint16_t *__restrict__ first,
                            const uint16_t *__restrict__ second,
                            int8_t *__restrict__ out, uint32_t m, uint32_t k,
                            int rank) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)m * k) return;
    const uint32_t row = (uint32_t)(idx / k);
    const uint32_t col = (uint32_t)(idx % k);
    const int32_t e = (int32_t)eAL[(size_t)row * rank + first[col]] -
                      (int32_t)eAL[(size_t)row * rank + second[col]];
    out[idx] = (int8_t)(A[idx] + e);
}

/**
 * B_noised[p][j] = B^t[j][p] + E_BR[first_p][j] - E_BR[second_p][j].
 *
 * B is held transposed everywhere else, because that is the layout its Merkle
 * root is taken over and the layout a win opens columns from. The GEMM wants
 * k x n, so the transpose happens here.
 *
 * Written the obvious way: the write and the E_BR reads coalesce, the B^t read
 * strides by k. On a 4090 that is tens of microseconds against a millisecond
 * of GEMM, so a shared-memory tile transpose is left until something measures
 * it as worth having.
 */
__global__ void applyNoiseB(const int8_t *__restrict__ Bt,
                            const int8_t *__restrict__ eBR,
                            const uint16_t *__restrict__ first,
                            const uint16_t *__restrict__ second,
                            int8_t *__restrict__ out, uint32_t n, uint32_t k) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)k * n) return;
    const uint32_t p = (uint32_t)(idx / n);
    const uint32_t j = (uint32_t)(idx % n);
    const int32_t e = (int32_t)eBR[(size_t)first[p] * n + j] -
                      (int32_t)eBR[(size_t)second[p] * n + j];
    out[idx] = (int8_t)(Bt[(size_t)j * k + p] + e);
}

/**
 * As applyNoiseB, but leaves B_noised in n-major (n x k) order.
 *
 * The WMMA kernels want B as k x n, so applyNoiseB transposes on the way in.
 * The raw-PTX kernel wants the opposite: its mma fragment holds four
 * CONSECUTIVE k for one n, which is a four-byte contiguous read out of an
 * n-major tile and four strided single-byte reads out of a k-major one.
 *
 * Not transposing is also the cheaper kernel - B^t already arrives n-major, so
 * both the read and the write are coalesced and the shared-memory tile
 * gymnastics disappear.
 */
__global__ void applyNoiseBt(const int8_t *__restrict__ Bt,
                             const int8_t *__restrict__ eBR,
                             const uint16_t *__restrict__ first,
                             const uint16_t *__restrict__ second,
                             int8_t *__restrict__ out, uint32_t n, uint32_t k) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)n * k) return;
    const uint32_t j = (uint32_t)(idx / k);       // which column of B
    const uint32_t p = (uint32_t)(idx % k);       // which k
    const int32_t e = (int32_t)eBR[(size_t)first[p] * n + j] -
                      (int32_t)eBR[(size_t)second[p] * n + j];
    out[idx] = (int8_t)(Bt[idx] + e);
}

/**
 * E_BR is drawn as n x rank and used as rank x n. Transposing it once here is
 * cheaper than striding over it for every element of B.
 */
__global__ void transposeNoiseB(const int8_t *__restrict__ in,
                                int8_t *__restrict__ out, uint32_t n, int rank) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)n * rank) return;
    const uint32_t row = (uint32_t)(idx / rank);
    const uint32_t c = (uint32_t)(idx % rank);
    out[(size_t)c * n + row] = in[idx];
}

}  // namespace pearl
}  // namespace om

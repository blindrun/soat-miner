// Autolykos v2 kernels: dataset construction and the per-nonce hit.
//
// Verified against the Ergo node's AutolykosPowScheme.scala, and against six
// real mainnet blocks via tests/reference.py before any of this was written.
//
// The shape of the algorithm, per nonce:
//
//   prei8 = H(msg || nonce)[24..32]        1 compression
//   i     = prei8 mod N
//   f     = element(i)                     1 random read
//   seed  = f || msg || nonce              (31 + 32 + 8 = 71 bytes)
//   idx   = 32 indices from H(seed)        1 compression
//   sum   = SUM element(idx[j])            32 random reads
//   hit   = H(sum as 32-byte BE)           1 compression
//
// So after the dataset exists it is 3 compressions and 33 random reads per
// nonce - memory-bandwidth bound, not compute bound. That is why building the
// dataset up front is worth it, and why the read path is where the
// performance lives.

#pragma once

#include "../../core/blake2b.cuh"

#define AL_K 32          // indices summed per nonce
#define AL_M_WORDS 1024  // M is 1024 big-endian int64s = 8192 bytes

// M's 64-bit little-endian message word at word index w is just bswap64(w):
// M stores the integer w big-endian, so reading it back little-endian is a
// byte swap. One instruction instead of 8 KB of memory traffic per element -
// and the dataset build does this 227 million times.

/** Byte-swap a 64-bit value (big-endian <-> little-endian). */
__device__ __forceinline__ uint64_t al_bswap64(uint64_t v) {
    uint32_t lo = (uint32_t)(v & 0xffffffffULL);
    uint32_t hi = (uint32_t)(v >> 32);
    uint32_t slo = __byte_perm(lo, 0, 0x0123);
    uint32_t shi = __byte_perm(hi, 0, 0x0123);
    return ((uint64_t)slo << 32) | (uint64_t)shi;
}

__device__ __forceinline__ uint32_t al_bswap32(uint32_t v) {
    return __byte_perm(v, 0, 0x0123);
}

/**
 * One dataset element: H(idx_BE32 || height_BE32 || M) with the leading byte
 * dropped, i.e. a 31-byte big-endian value.
 *
 * Input is 8200 bytes = 64 full 128-byte blocks + an 8-byte tail, so 65
 * compressions. Only the first 8 bytes differ between elements; everything
 * after is M, generated arithmetically.
 *
 * Result is returned as four 64-bit limbs, limb[0] least significant, ready
 * for 256-bit accumulation.
 */
__device__ __forceinline__ void al_gen_element(uint32_t idx, uint32_t height,
                                               uint64_t limb[4]) {
    uint64_t h[8];
    b2b256_init(h);

    uint64_t m[16];
    uint64_t t = 0;

    // --- block 0: idx(4 BE) | height(4 BE) | M[0..119] ---------------------
    m[0] = (uint64_t)al_bswap32(idx) | ((uint64_t)al_bswap32(height) << 32);
#pragma unroll
    for (int j = 1; j < 16; j++) m[j] = al_bswap64((uint64_t)(j - 1));
    t += 128;
    b2b_compress(h, m, t, false);

    // --- blocks 1..63: pure M ---------------------------------------------
    // Block b starts at M word 15 + (b-1)*16.
    for (uint32_t b = 1; b < 64; b++) {
        uint32_t base = 15u + (b - 1u) * 16u;
#pragma unroll
        for (int j = 0; j < 16; j++) m[j] = al_bswap64((uint64_t)(base + j));
        t += 128;
        b2b_compress(h, m, t, false);
    }

    // --- final block: last 8 bytes of M (word 1023) ------------------------
    m[0] = al_bswap64((uint64_t)1023);
#pragma unroll
    for (int j = 1; j < 16; j++) m[j] = 0;
    t = 8200;
    b2b_compress(h, m, t, true);

    // h holds 8 little-endian words = 32 bytes d[0..31].
    // The element is the big-endian integer over d[1..31], so:
    //   limb[3] = 0x00 d1 d2 d3 d4 d5 d6 d7
    //   limb[2] = d8..d15,  limb[1] = d16..d23,  limb[0] = d24..d31
    uint8_t d[32];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) d[i * 8 + j] = (uint8_t)(h[i] >> (8 * j));

    // d[1] is the most significant byte, d[31] the least. limb[3] therefore
    // holds d[0..7] with d[0] zeroed (that is the dropped byte), and limb[0]
    // holds d[24..31].
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const int off = (3 - l) * 8;  // limb[3] <- d[0..7], limb[0] <- d[24..31]
        uint64_t v = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t bv = (off + j == 0) ? 0 : d[off + j];  // drop d[0]
            v = (v << 8) | (uint64_t)bv;
        }
        limb[l] = v;
    }
}

/** Builds the whole dataset. One thread per element. */
__global__ void al_build_dataset(uint4 *dataset, uint32_t N, uint32_t height) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint64_t limb[4];
    al_gen_element(idx, height, limb);

    // Store as 32 aligned bytes so the mining kernel can issue one 16-byte
    // vector load pair per element instead of a byte-wise gather.
    uint4 *p = dataset + (size_t)idx * 2;
    uint4 a, b;
    a.x = (uint32_t)(limb[0] & 0xffffffffu);
    a.y = (uint32_t)(limb[0] >> 32);
    a.z = (uint32_t)(limb[1] & 0xffffffffu);
    a.w = (uint32_t)(limb[1] >> 32);
    b.x = (uint32_t)(limb[2] & 0xffffffffu);
    b.y = (uint32_t)(limb[2] >> 32);
    b.z = (uint32_t)(limb[3] & 0xffffffffu);
    b.w = (uint32_t)(limb[3] >> 32);
    p[0] = a;
    p[1] = b;
}

/** Reads element `i` back as four limbs. */
__device__ __forceinline__ void al_load_element(const uint4 *dataset,
                                                uint32_t i, uint64_t limb[4]) {
    const uint4 *p = dataset + (size_t)i * 2;
    uint4 a = __ldg(&p[0]);
    uint4 b = __ldg(&p[1]);
    limb[0] = ((uint64_t)a.y << 32) | a.x;
    limb[1] = ((uint64_t)a.w << 32) | a.z;
    limb[2] = ((uint64_t)b.y << 32) | b.x;
    limb[3] = ((uint64_t)b.w << 32) | b.z;
}

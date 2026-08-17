// blake3 on the device, general enough for Pearl's prepare stage.
//
// noisy_gemm.cuh already carries a blake3 that does exactly one thing: a keyed
// hash of a single 64-byte block, which is the whole PoW check. That one is
// deliberately narrow because it sits in the mining loop and is worth keeping
// unreadable-fast. This is the other half - chunk chaining values, parent
// nodes and a root - which the prepare stage needs to hash two multi-megabyte
// matrices into their Merkle roots.
//
// Kept separate rather than merged into that file so the verified hot kernel
// is not disturbed by work that runs once per attempt instead of once per
// tile.
//
// The host twin is om::pearl::b3 in job.h, and both are checked against the
// same vectors.

#pragma once

#include <stdint.h>

namespace om {
namespace pearl {
namespace b3d {

__constant__ uint32_t kIVd[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u,
                                 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu,
                                 0x1F83D9ABu, 0x5BE0CD19u};

constexpr uint32_t kChunkStart = 1, kChunkEnd = 2, kParent = 4, kRoot = 8,
                   kKeyedHash = 16;
constexpr int kChunkLenD = 1024;

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void g(uint32_t *s, int a, int b, int c, int d,
                                  uint32_t mx, uint32_t my) {
    s[a] += s[b] + mx;  s[d] = rotr(s[d] ^ s[a], 16);
    s[c] += s[d];       s[b] = rotr(s[b] ^ s[c], 12);
    s[a] += s[b] + my;  s[d] = rotr(s[d] ^ s[a], 8);
    s[c] += s[d];       s[b] = rotr(s[b] ^ s[c], 7);
}

/**
 * The compression function, chaining-value output only.
 *
 * The message permutation is applied with literal indices rather than a table,
 * for the same reason as in noisy_gemm.cuh: a runtime index into a register
 * array spills it to local memory.
 */
__device__ inline void compress(uint32_t cv[8], const uint32_t block[16],
                                uint64_t counter, uint32_t blockLen,
                                uint32_t flags) {
    uint32_t s[16];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = cv[i];
    s[8] = kIVd[0]; s[9] = kIVd[1]; s[10] = kIVd[2]; s[11] = kIVd[3];
    s[12] = (uint32_t)counter;
    s[13] = (uint32_t)(counter >> 32);
    s[14] = blockLen;
    s[15] = flags;

    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = block[i];

#pragma unroll
    for (int r = 0; r < 7; r++) {
        g(s, 0, 4, 8, 12, m[0], m[1]);
        g(s, 1, 5, 9, 13, m[2], m[3]);
        g(s, 2, 6, 10, 14, m[4], m[5]);
        g(s, 3, 7, 11, 15, m[6], m[7]);
        g(s, 0, 5, 10, 15, m[8], m[9]);
        g(s, 1, 6, 11, 12, m[10], m[11]);
        g(s, 2, 7, 8, 13, m[12], m[13]);
        g(s, 3, 4, 9, 14, m[14], m[15]);
        if (r < 6) {
            const uint32_t t0 = m[2],  t1 = m[6],  t2 = m[3],  t3 = m[10];
            const uint32_t t4 = m[7],  t5 = m[0],  t6 = m[4],  t7 = m[13];
            const uint32_t t8 = m[1],  t9 = m[11], t10 = m[12], t11 = m[5];
            const uint32_t t12 = m[9], t13 = m[14], t14 = m[15], t15 = m[8];
            m[0] = t0;   m[1] = t1;   m[2] = t2;   m[3] = t3;
            m[4] = t4;   m[5] = t5;   m[6] = t6;   m[7] = t7;
            m[8] = t8;   m[9] = t9;   m[10] = t10; m[11] = t11;
            m[12] = t12; m[13] = t13; m[14] = t14; m[15] = t15;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = s[i] ^ s[i + 8];
}

/** Load 64 bytes as 16 little-endian words. */
__device__ __forceinline__ void loadBlock(const uint8_t *p, uint32_t out[16]) {
#pragma unroll
    for (int i = 0; i < 16; i++)
        out[i] = (uint32_t)p[4 * i] | ((uint32_t)p[4 * i + 1] << 8) |
                 ((uint32_t)p[4 * i + 2] << 16) | ((uint32_t)p[4 * i + 3] << 24);
}

/**
 * Chaining value of a full 1024-byte chunk, keyed.
 *
 * Only the full-chunk case, because every matrix here is zero-padded to a
 * chunk boundary before it is hashed - which is the same padding the Merkle
 * leaf size implies, not an extra step.
 */
__device__ inline void chunkCv(const uint32_t key[8], const uint8_t *chunk,
                               uint64_t chunkIndex, uint32_t out[8]) {
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];

    for (int i = 0; i < 16; i++) {
        uint32_t flags = kKeyedHash;
        if (i == 0) flags |= kChunkStart;
        if (i == 15) flags |= kChunkEnd;
        uint32_t block[16];
        loadBlock(chunk + i * 64, block);
        compress(cv, block, chunkIndex, 64u, flags);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

/** Parent of two chaining values. `root` adds the ROOT flag. */
__device__ inline void parentCv(const uint32_t key[8], const uint32_t left[8],
                                const uint32_t right[8], bool root,
                                uint32_t out[8]) {
    uint32_t cv[8], block[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        cv[i] = key[i];
        block[i] = left[i];
        block[i + 8] = right[i];
    }
    compress(cv, block, 0, 64u, kKeyedHash | kParent | (root ? kRoot : 0u));
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

/** Keyed hash of exactly one 64-byte message: one chunk, one block, root. */
__device__ inline void hashBlock64(const uint32_t key[8], const uint8_t *msg,
                                   uint32_t out[8]) {
    uint32_t cv[8], block[16];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    loadBlock(msg, block);
    compress(cv, block, 0, 64u, kKeyedHash | kChunkStart | kChunkEnd | kRoot);
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

/** As hashBlock64, unkeyed - the IV stands in and the KEYED flag is dropped. */
__device__ inline void hashBlock64Unkeyed(const uint8_t *msg, uint32_t out[8]) {
    uint32_t cv[8], block[16];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = kIVd[i];
    loadBlock(msg, block);
    compress(cv, block, 0, 64u, kChunkStart | kChunkEnd | kRoot);
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

__device__ __forceinline__ void storeCv(const uint32_t cv[8], uint8_t *out) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)cv[i];
        out[4 * i + 1] = (uint8_t)(cv[i] >> 8);
        out[4 * i + 2] = (uint8_t)(cv[i] >> 16);
        out[4 * i + 3] = (uint8_t)(cv[i] >> 24);
    }
}

__device__ __forceinline__ void loadCv(const uint8_t *in, uint32_t cv[8]) {
#pragma unroll
    for (int i = 0; i < 8; i++)
        cv[i] = (uint32_t)in[4 * i] | ((uint32_t)in[4 * i + 1] << 8) |
                ((uint32_t)in[4 * i + 2] << 16) | ((uint32_t)in[4 * i + 3] << 24);
}

}  // namespace b3d
}  // namespace pearl
}  // namespace om

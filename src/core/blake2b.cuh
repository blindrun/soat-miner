// Blake2b-256 for CUDA devices.
//
// RFC 7693, unkeyed, 32-byte digest. This is the only hash Autolykos v2 uses,
// and it is called ~33 times per nonce plus 65 times per table element, so it
// is the whole performance story on the table-build side.
//
// Deliberately kept as a small explicit state struct rather than a streaming
// API: every call site here knows its message length up front, and the sizes
// that matter (40, 71, 32 and 8200 bytes) are all handled by specialised
// helpers below that avoid the generic buffering entirely.

#pragma once

#include <stdint.h>

#define B2B_ROUNDS 12

__device__ __constant__ uint64_t B2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

__device__ __constant__ uint8_t B2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

__device__ __forceinline__ uint64_t b2b_rotr64(uint64_t x, uint32_t n) {
    return (x >> n) | (x << (64 - n));
}

#define B2B_G(a, b, c, d, x, y)        \
    do {                               \
        a = a + b + (x);               \
        d = b2b_rotr64(d ^ a, 32);     \
        c = c + d;                     \
        b = b2b_rotr64(b ^ c, 24);     \
        a = a + b + (y);               \
        d = b2b_rotr64(d ^ a, 16);     \
        c = c + d;                     \
        b = b2b_rotr64(b ^ c, 63);     \
    } while (0)

/**
 * Blake2b compression. `m` is 16 little-endian 64-bit words of the block,
 * `t` the byte counter, `last` set only for the final block.
 */
__device__ __forceinline__ void b2b_compress(uint64_t h[8], const uint64_t m[16],
                                             uint64_t t, bool last) {
    uint64_t v[16];
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = h[i];
#pragma unroll
    for (int i = 0; i < 8; i++) v[8 + i] = B2B_IV[i];

    v[12] ^= t;
    // v[13] ^= t_high; always 0 here - no message exceeds 2^64 bytes.
    if (last) v[14] = ~v[14];

#pragma unroll
    for (int r = 0; r < B2B_ROUNDS; r++) {
        const uint8_t *s = B2B_SIGMA[r];
        B2B_G(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        B2B_G(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        B2B_G(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        B2B_G(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
        B2B_G(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        B2B_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        B2B_G(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        B2B_G(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }

#pragma unroll
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

/** Initial state for an unkeyed 32-byte digest. */
__device__ __forceinline__ void b2b256_init(uint64_t h[8]) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = B2B_IV[i];
    // depth=1, fanout=1, keylen=0, digest_len=32
    h[0] ^= 0x01010020ULL;
}

/**
 * Hash a short message that fits in a single 128-byte block.
 * Used for msg||nonce (40 B), seed (71 B) and the final sum (32 B).
 *
 * LEN is a template parameter so every bound below is a compile-time
 * constant. That matters more than it looks: the obvious version stages the
 * message through a uint8_t buf[128] before packing it into words, and those
 * 128 bytes live in registers. Removing the staging buffer is what takes the
 * search kernel from 250 registers down to a level where enough warps fit per
 * SM to hide the random-read latency this algorithm is built on.
 */
template <uint32_t LEN>
__device__ __forceinline__ void b2b256_short(const uint8_t *in, uint8_t out[32]) {
    static_assert(LEN <= 128, "single-block helper only");

    uint64_t h[8];
    b2b256_init(h);

    uint64_t m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint64_t w = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t idx = i * 8 + j;
            if (idx < LEN) w |= ((uint64_t)in[idx]) << (8 * j);
        }
        m[i] = w;
    }

    b2b_compress(h, m, LEN, true);

#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (uint8_t)(h[i] >> (8 * j));
}

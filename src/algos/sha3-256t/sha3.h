// SHA3-256t - the BitcoinIII (BC3) proof of work.
//
// One source of truth for the host and the CUDA device. The Vulkan shader in
// kernel.comp is unavoidably a second transcription of the same permutation,
// which is why tests/test_sha3_vulkan checks the two against each other rather
// than trusting that they were written from the same page.
//
// What BC3 actually does, from BitcoinIII-Core:
//
//   primitives/block.cpp:11   GetHash() dispatches on version bit 12
//   hash.h:164                HashWriterSHA3::GetHash() runs SHA3-256 THREE
//                             times, despite being reached through a method
//                             called GetSHA3_256dHash()
//   crypto/sha3.cpp:140       padding byte 0x06, rate 1088 bits
//
// So: hash = SHA3-256(SHA3-256(SHA3-256(header[80]))), and the 0x06 padding
// makes it true NIST SHA3, NOT Keccak-256. A Keccak-256 implementation (0x01
// padding) produces a completely different digest and every share is rejected.
//
// The digest is compared to the target as a LITTLE-endian 256-bit integer -
// arith_uint256 reads uint256's bytes low first - so limb 0 below is the least
// significant, matching Solution::hit.
//
// Verified against mainnet block 50204:
//   header 00100020fefeebfa...4636619e  ->  hash 000000000000000d86fac318...

#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define S3_HD __host__ __device__ __forceinline__
#else
#define S3_HD inline
#endif

namespace om {
namespace s3 {

/** The 80-byte header as ten little-endian lanes. Lane 9's high half is the
 *  nonce (header bytes 76..79), which is the only thing that moves. */
struct Header {
    uint64_t lane[10];
};

S3_HD uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

/** Keccak-f[1600]. The plain reference form: 24 rounds of theta, rho+pi, chi,
 *  iota. Deliberately not hand-unrolled - nvcc unrolls the inner loops itself
 *  and an unrolled version measured no faster while being far harder to check
 *  against the spec. */
S3_HD void keccakf(uint64_t st[25]) {
    static const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
    static const int ROTC[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                 27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
    static const int PILN[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                 15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

    uint64_t bc[5], t;
#pragma unroll 1
    for (int r = 0; r < 24; r++) {
        // Theta
#pragma unroll
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        // Rho and Pi
        t = st[1];
#pragma unroll
        for (int i = 0; i < 24; i++) {
            const int j = PILN[i];
            bc[0] = st[j];
            st[j] = rotl64(t, ROTC[i]);
            t = bc[0];
        }
        // Chi
#pragma unroll
        for (int j = 0; j < 25; j += 5) {
#pragma unroll
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
#pragma unroll
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // Iota
        st[0] ^= RC[r];
    }
}

/**
 * The full BC3 hash for one nonce.
 *
 * The 80-byte header is exactly ten lanes, so the first absorb needs no
 * partial-lane handling: lanes 0..9 are message, lane 10 carries the 0x06
 * pad and lane 16 the 0x80 terminator (rate 136 bytes = 17 lanes).
 */
S3_HD void hash(const Header &hdr, uint32_t nonce, uint64_t out[4]) {
    uint64_t st[25];

#pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;
#pragma unroll
    for (int i = 0; i < 9; i++) st[i] = hdr.lane[i];
    st[9] = (hdr.lane[9] & 0xffffffffULL) | ((uint64_t)nonce << 32);
    st[10] = 0x06ULL;
    st[16] = 0x8000000000000000ULL;
    keccakf(st);

    // Two more passes over the 32-byte digest. Four lanes of message, so the
    // pad byte lands in lane 4.
#pragma unroll 1
    for (int p = 0; p < 2; p++) {
        const uint64_t d0 = st[0], d1 = st[1], d2 = st[2], d3 = st[3];
#pragma unroll
        for (int i = 0; i < 25; i++) st[i] = 0;
        st[0] = d0;
        st[1] = d1;
        st[2] = d2;
        st[3] = d3;
        st[4] = 0x06ULL;
        st[16] = 0x8000000000000000ULL;
        keccakf(st);
    }

    out[0] = st[0];
    out[1] = st[1];
    out[2] = st[2];
    out[3] = st[3];
}

/** True when `hit` is at or below `target`, both little-endian limb arrays. */
S3_HD bool underTarget(const uint64_t hit[4], const uint64_t target[4]) {
#pragma unroll
    for (int i = 3; i >= 0; i--) {
        if (hit[i] != target[i]) return hit[i] < target[i];
    }
    return true;  // equal is a valid hit: consensus rejects only hash > target
}

/** Load an 80-byte serialised header into lanes. */
S3_HD void loadHeader(const uint8_t bytes[80], Header *h) {
#pragma unroll
    for (int i = 0; i < 10; i++) {
        uint64_t v = 0;
#pragma unroll
        for (int b = 7; b >= 0; b--) v = (v << 8) | (uint64_t)bytes[i * 8 + b];
        h->lane[i] = v;
    }
}

}  // namespace s3
}  // namespace om

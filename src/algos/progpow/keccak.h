// Keccak for the ProgPoW family: the f[800] permutation the ProgPoW hash is
// built on, and the f[1600] hashes the Ethash DAG underneath it is built on.
//
// Shared verbatim by KawPow and ProgPowZ. Nothing in this file differs between
// them, which is more than can be said for the layer above it.
//
// Two traps live in this file, both of which produce a miner that runs at full
// speed and has every share rejected:
//
//  * Ethash uses ORIGINAL Keccak padding (0x01), not NIST SHA3 padding (0x06).
//    The sibling file src/algos/sha3-256t/sha3.h documents the opposite choice
//    for BC3 and is deliberately not shared with this one - the permutation is
//    the same, the padding is not, and a header that quietly picked the wrong
//    one would be invisible until a pool said no.
//
//  * Keccak-f[800] here runs 22 rounds over 32-bit lanes and ProgPoW absorbs
//    into it with NO padding at all: exactly 576 bits of input are written
//    into the state and the permutation is run. It is not a hash function with
//    a sponge interface, it is a raw permutation, and treating it as SHA3-256
//    of the same bytes gives a different answer.
//
// Word convention: everything is uint32 words in little-endian byte order, so
// word i of a digest is bytes[4i..4i+3] read little-endian. That matches how
// Ethash and ProgPoW read their own hashes on every platform this miner
// supports, and it keeps the file free of byte-swap noise.

#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define KP_HD __host__ __device__ __forceinline__
#else
#define KP_HD inline
#endif

// Overridable so the mutation gate can build a binary that pads like NIST SHA3
// (0x06) instead of original Keccak (0x01), and prove the test notices.
#ifndef KAWPOW_KECCAK_PAD
#define KAWPOW_KECCAK_PAD 0x01u
#endif

namespace om {
namespace progpow {

KP_HD uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

KP_HD uint32_t rotl32(uint32_t x, uint32_t n) {
    n &= 31u;
    return n == 0 ? x : ((x << n) | (x >> (32u - n)));
}

KP_HD uint32_t rotr32(uint32_t x, uint32_t n) {
    n &= 31u;
    return n == 0 ? x : ((x >> n) | (x << (32u - n)));
}

/** Keccak-f[1600], 24 rounds. Same permutation BC3 uses; see the note above on
 *  why this is a second copy rather than a shared one. */
KP_HD void keccakf1600(uint64_t st[25]) {
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
    for (int r = 0; r < 24; r++) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        t = st[1];
        for (int i = 0; i < 24; i++) {
            const int j = PILN[i];
            bc[0] = st[j];
            st[j] = rotl64(t, ROTC[i]);
            t = bc[0];
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] = bc[i] ^ ((~bc[(i + 1) % 5]) & bc[(i + 2) % 5]);
        }
        st[0] ^= RC[r];
    }
}

/**
 * Keccak-f[800], 22 rounds over 32-bit lanes.
 *
 * The round constants are the 32 low bits of the f[1600] constants, and the
 * rotation offsets are identical - the state is the same 5x5 grid, only the
 * lane width changes.
 */
KP_HD void keccakf800(uint32_t st[25]) {
    static const uint32_t RC[22] = {
        0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u, 0x0000808bu,
        0x80000001u, 0x80008081u, 0x00008009u, 0x0000008au, 0x00000088u,
        0x80008009u, 0x8000000au, 0x8000808bu, 0x0000008bu, 0x00008089u,
        0x00008003u, 0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
        0x80008081u, 0x00008080u};
    static const int ROTC[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                 27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
    static const int PILN[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                 15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

    uint32_t bc[5], t;
    for (int r = 0; r < 22; r++) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl32(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        t = st[1];
        for (int i = 0; i < 24; i++) {
            const int j = PILN[i];
            bc[0] = st[j];
            // The f[1600] offsets are taken mod 32 for the narrower lane.
            st[j] = rotl32(t, (uint32_t)ROTC[i]);
            t = bc[0];
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] = bc[i] ^ ((~bc[(i + 1) % 5]) & bc[(i + 2) % 5]);
        }
        st[0] ^= RC[r];
    }
}

/**
 * Keccak-512 over `words32` little-endian words, original (0x01) padding.
 *
 * Only the two lengths Ethash needs are supported - 32 bytes for the epoch
 * seed and 64 bytes for every cache and dataset item - because a general
 * sponge would be dead code on the device.
 */
KP_HD void keccak512(const uint32_t *in, int words32, uint32_t out[16]) {
    // rate = (1600 - 1024) / 8 = 72 bytes = 9 lanes.
    uint64_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i < words32; i++)
        st[i >> 1] |= (uint64_t)in[i] << ((i & 1) ? 32 : 0);
    const int padByte = words32 * 4;   // absorb offset of the padding byte
    st[padByte >> 3] ^= (uint64_t)KAWPOW_KECCAK_PAD << ((padByte & 7) * 8);
    st[8] ^= 0x8000000000000000ULL;
    keccakf1600(st);
    for (int i = 0; i < 16; i++)
        out[i] = (uint32_t)(st[i >> 1] >> ((i & 1) ? 32 : 0));
}

/** Keccak-256 of exactly 32 bytes, original padding. Used only to walk the
 *  epoch seed forward, once per epoch. */
KP_HD void keccak256_32(const uint32_t in[8], uint32_t out[8]) {
    // rate = (1600 - 512) / 8 = 136 bytes = 17 lanes.
    uint64_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i < 8; i++)
        st[i >> 1] |= (uint64_t)in[i] << ((i & 1) ? 32 : 0);
    st[4] ^= (uint64_t)KAWPOW_KECCAK_PAD;
    st[16] ^= 0x8000000000000000ULL;
    keccakf1600(st);
    for (int i = 0; i < 8; i++)
        out[i] = (uint32_t)(st[i >> 1] >> ((i & 1) ? 32 : 0));
}

}  // namespace progpow
}  // namespace om

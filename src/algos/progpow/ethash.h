// The Ethash layer under every ProgPoW variant: epoch sizing, the epoch seed,
// the light cache, and the dataset items the mix reads.
//
// Shared by KawPow and ProgPowZ with exactly one difference between them, and
// it is passed in rather than compiled in: the epoch length. Ravencoin uses
// 7500 blocks, Zano uses Ethereum's 30000. Everything else here - the 16 MiB
// initial cache, the 1 GiB initial dataset, the growth rates, the prime
// rounding, RandMemoHash's three rounds, the 512 parents per item - is
// identical, because both inherited it unchanged from Ethereum.
//
// The word convention is keccak.h's: uint32 words in little-endian byte order.

#pragma once

#include <stdint.h>

#include "keccak.h"

namespace om {
namespace progpow {

/** 32-bit words per lane of one 2048-bit dataset item: 256 / (4 * 16) = 4. */
static const uint32_t kDagLoads = 4;
static const uint32_t kL1CacheBytes = 16 * 1024;
static const uint32_t kL1CacheWords = kL1CacheBytes / 4;

static const uint32_t kFnvPrime = 0x01000193u;
static const uint32_t kFnvOffsetBasis = 0x811c9dc5u;

KP_HD uint32_t fnv1(uint32_t u, uint32_t v) { return (u * kFnvPrime) ^ v; }
KP_HD uint32_t fnv1a(uint32_t u, uint32_t v) { return (u ^ v) * kFnvPrime; }

inline bool isPrime(int n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (int d = 3; (int64_t)d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}

/** Largest prime not greater than `upperBound`. Both the cache and the dataset
 *  are sized to a prime so the modulo indexing cannot fall into a short
 *  cycle. */
inline int largestPrimeNotAbove(int upperBound) {
    for (int n = upperBound; n > 1; n--)
        if (isPrime(n)) return n;
    return 0;
}

inline int epochNumber(int blockNumber, int epochLength) {
    return blockNumber > 0 ? blockNumber / epochLength : 0;
}

/** 64-byte items in the light cache. 16 MiB at epoch 0, +128 KiB per epoch. */
inline int lightCacheNumItems(int epoch) {
    const int init = (1 << 24) / 64;
    const int growth = (1 << 17) / 64;
    return largestPrimeNotAbove(init + epoch * growth);
}

/** 128-byte items in the full dataset. 1 GiB at epoch 0, +8 MiB per epoch.
 *  ProgPoW addresses it in 256-byte units, so it sees half this many. */
inline int fullDatasetNumItems(int epoch) {
    const int init = (1 << 30) / 128;
    const int growth = (1 << 23) / 128;
    return largestPrimeNotAbove(init + epoch * growth);
}

inline uint64_t fullDatasetBytes(int epoch) {
    return (uint64_t)fullDatasetNumItems(epoch) * 128ull;
}

/** Epoch seed: Keccak-256 applied to itself `epoch` times, starting at zero. */
inline void epochSeed(int epoch, uint32_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = 0;
    for (int e = 0; e < epoch; e++) {
        uint32_t next[8];
        keccak256_32(out, next);
        for (int i = 0; i < 8; i++) out[i] = next[i];
    }
}

/**
 * Builds the light cache: a chain of Keccak-512 hashes, then three rounds of
 * RandMemoHash mixing. `cache` holds numItems * 16 words.
 */
inline void buildLightCache(uint32_t *cache, int numItems, const uint32_t seed[8]) {
    keccak512(seed, 8, cache);
    for (int i = 1; i < numItems; i++)
        keccak512(cache + (size_t)(i - 1) * 16, 16, cache + (size_t)i * 16);

    for (int q = 0; q < 3; q++) {
        for (int i = 0; i < numItems; i++) {
            const uint32_t v = cache[(size_t)i * 16] % (uint32_t)numItems;
            const uint32_t w = (uint32_t)((numItems + (i - 1)) % numItems);
            uint32_t x[16];
            for (int k = 0; k < 16; k++)
                x[k] = cache[(size_t)v * 16 + k] ^ cache[(size_t)w * 16 + k];
            keccak512(x, 16, cache + (size_t)i * 16);
        }
    }
}

/** One 512-bit dataset item: 512 FNV-1 passes over pseudo-random cache items,
 *  finished with a Keccak-512. This is the whole cost of building the DAG. */
KP_HD void datasetItem512(const uint32_t *cache, int numCacheItems,
                          uint32_t index, uint32_t out[16]) {
    const uint32_t seed = index;
    uint32_t mix[16];
    const uint32_t *src = cache + (size_t)(index % (uint32_t)numCacheItems) * 16;
    for (int i = 0; i < 16; i++) mix[i] = src[i];
    mix[0] ^= seed;
    keccak512(mix, 16, mix);

    for (uint32_t j = 0; j < 512; j++) {
        const uint32_t t = fnv1(seed ^ j, mix[j % 16]);
        const uint32_t parent = t % (uint32_t)numCacheItems;
        const uint32_t *p = cache + (size_t)parent * 16;
        for (int i = 0; i < 16; i++) mix[i] = fnv1(mix[i], p[i]);
    }
    keccak512(mix, 16, out);
}

/** The 2048-bit item ProgPoW's DAG access reads: four consecutive 512-bit
 *  items. `index` counts 256-byte items, so it multiplies by four. */
KP_HD void datasetItem2048(const uint32_t *cache, int numCacheItems,
                           uint32_t index, uint32_t out[64]) {
    for (uint32_t k = 0; k < 4; k++)
        datasetItem512(cache, numCacheItems, index * 4 + k, out + k * 16);
}

// ---------------------------------------------------------------------------
// Target comparison
// ---------------------------------------------------------------------------

KP_HD uint32_t bswap32(uint32_t x) {
    return ((x & 0x000000ffu) << 24) | ((x & 0x0000ff00u) << 8) |
           ((x & 0x00ff0000u) >> 8) | ((x & 0xff000000u) >> 24);
}

/**
 * A ProgPoW hash is compared as a BIG-endian 256-bit integer: byte 0 of the
 * digest is the most significant. The core's Job::target and Solution::hit are
 * limb arrays with limb[0] least significant, so this is where the two
 * conventions meet.
 */
KP_HD void hashToLimbs(const uint32_t hash[8], uint64_t limbs[4]) {
    for (int k = 0; k < 4; k++) {
        const uint64_t be = ((uint64_t)bswap32(hash[2 * k]) << 32) |
                            (uint64_t)bswap32(hash[2 * k + 1]);
        limbs[3 - k] = be;
    }
}

KP_HD bool underTarget(const uint64_t hit[4], const uint64_t target[4]) {
    for (int i = 3; i >= 0; i--) {
        if (hit[i] < target[i]) return true;
        if (hit[i] > target[i]) return false;
    }
    return true;
}

}  // namespace progpow
}  // namespace om

// KawPow - the Ravencoin proof of work, and the shared source of truth for the
// host reference, the CUDA kernel and the correctness tests.
//
// KawPow is ProgPoW 0.9.4 driving an Ethash DAG. Two layers, and it is worth
// keeping them apart when reading this file:
//
//   Ethash layer   a light cache built from the epoch seed, from which a
//                  multi-gigabyte dataset is derived. Pure memory hardness.
//   ProgPoW layer  64 rounds of a random program - random maths, random L1
//                  cache reads, random register merges - whose shape is fixed
//                  for a period of 3 blocks and changes after it. That is the
//                  part an ASIC cannot bake in.
//
// The parameters that make it Ravencoin's rather than Ethereum's:
//
//   period_length   3       (ProgPoW's own default is 50; Ravencoin uses 3)
//   epoch_length    7500    (Ethereum uses 30000 - get this wrong and every
//                            share is computed against the wrong DAG)
//   keccak tail     the 15 words "RAVENCOINKAWPOW" fill the unused end of both
//                   Keccak-f[800] states. This is the single change that makes
//                   KawPow incompatible with plain ProgPoW, and it is not
//                   visible anywhere in the block format.
//
// Spec source: RavenProject/Ravencoin src/crypto/ethash (Apache-2.0), read as
// a specification. The code here is written from the algorithm rather than
// copied, because this repository is MIT. The published hash test vectors are
// used verbatim as facts in tests/test_kawpow_hash.
//
// Word convention throughout: uint32 words in little-endian byte order, as in
// keccak.h.

#pragma once

#include <stdint.h>

#include "keccak.h"

namespace om {
namespace kawpow {

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

// The four values below are the ones that silently produce a fast miner with
// a 100% reject rate, so each is overridable at compile time and the Makefile
// builds a deliberately broken binary for each. A correctness test that has
// never been seen to fail is not evidence; see `make test-kawpow`.
#ifndef KAWPOW_PERIOD_LENGTH
#define KAWPOW_PERIOD_LENGTH 3
#endif
#ifndef KAWPOW_EPOCH_LENGTH
#define KAWPOW_EPOCH_LENGTH 7500
#endif
#ifndef KAWPOW_RAVEN_TAIL
#define KAWPOW_RAVEN_TAIL 1
#endif

static const int kPeriodLength = KAWPOW_PERIOD_LENGTH;
static const int kEpochLength = KAWPOW_EPOCH_LENGTH;
static const uint32_t kNumRegs = 32;
static const uint32_t kNumLanes = 16;
static const int kNumCacheAccesses = 11;
static const int kNumMathOperations = 18;
static const int kNumRounds = 64;
static const uint32_t kL1CacheBytes = 16 * 1024;
static const uint32_t kL1CacheWords = kL1CacheBytes / 4;

/** 32-bit words per lane of one 2048-bit dataset item: 256 / (4 * 16) = 4. */
static const uint32_t kDagLoads = 4;

static const uint32_t kFnvPrime = 0x01000193u;
static const uint32_t kFnvOffsetBasis = 0x811c9dc5u;

/**
 * "rAVENCOINKAWPOW", one ASCII character per 32-bit word.
 *
 * The first character is lower case. Ravencoin's own source writes `0x72` and
 * comments it `//R`, so the constant and its comment disagree and the constant
 * is what consensus follows. Writing the obvious 0x52 costs two hours: the
 * KISS99 state, the register permutations and the entire Ethash layer all
 * still match a reference implementation exactly, and only the seed coming out
 * of the first Keccak-f[800] is wrong - which looks like a broken keccak
 * rather than a wrong input.
 *
 * This tail is the only thing separating KawPow from stock ProgPoW, and
 * nothing in the block format hints at it.
 */
#if KAWPOW_RAVEN_TAIL
static const uint32_t kRavencoinKawpow[15] = {
    0x00000072u, 0x00000041u, 0x00000056u, 0x00000045u, 0x0000004Eu,
    0x00000043u, 0x0000004Fu, 0x00000049u, 0x0000004Eu, 0x0000004Bu,
    0x00000041u, 0x00000057u, 0x00000050u, 0x0000004Fu, 0x00000057u};
#else
static const uint32_t kRavencoinKawpow[15] = {0};  // mutation: stock ProgPoW
#endif

// ---------------------------------------------------------------------------
// Small primitives
// ---------------------------------------------------------------------------

KP_HD uint32_t fnv1(uint32_t u, uint32_t v) { return (u * kFnvPrime) ^ v; }
KP_HD uint32_t fnv1a(uint32_t u, uint32_t v) { return (u ^ v) * kFnvPrime; }

KP_HD uint32_t clz32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return (uint32_t)__clz((int)x);
#else
    if (x == 0) return 32;
    uint32_t n = 0;
    while ((x & 0x80000000u) == 0) { x <<= 1; n++; }
    return n;
#endif
}

KP_HD uint32_t popcount32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return (uint32_t)__popc(x);
#else
    uint32_t n = 0;
    while (x) { n += x & 1u; x >>= 1; }
    return n;
#endif
}

KP_HD uint32_t mulHi32(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

/** KISS99, the 1999 specification. ProgPoW's whole program is derived from it,
 *  so a single wrong constant changes every hash. */
struct Kiss99 {
    uint32_t z, w, jsr, jcong;

    KP_HD uint32_t next() {
        z = 36969u * (z & 0xffffu) + (z >> 16);
        w = 18000u * (w & 0xffffu) + (w >> 16);
        jcong = 69069u * jcong + 1234567u;
        jsr ^= (jsr << 17);
        jsr ^= (jsr >> 13);
        jsr ^= (jsr << 5);
        return (((z << 16) + w) ^ jcong) + jsr;
    }
};

KP_HD uint32_t randomMath(uint32_t a, uint32_t b, uint32_t selector) {
    switch (selector % 11u) {
        case 0: return a + b;
        case 1: return a * b;
        case 2: return mulHi32(a, b);
        case 3: return a < b ? a : b;
        case 4: return rotl32(a, b);
        case 5: return rotr32(a, b);
        case 6: return a & b;
        case 7: return a | b;
        case 8: return a ^ b;
        case 9: return clz32(a) + clz32(b);
        default: return popcount32(a) + popcount32(b);
    }
}

/** Merge `b` into `a`, using only operations that keep `a`'s entropy even when
 *  `b` has little - which is why there is no `a & b` case here. */
KP_HD void randomMerge(uint32_t *a, uint32_t b, uint32_t selector) {
    const uint32_t x = (selector >> 16) % 31u + 1u;
    switch (selector % 4u) {
        case 0: *a = (*a * 33u) + b; break;
        case 1: *a = (*a ^ b) * 33u; break;
        case 2: *a = rotl32(*a, x) ^ b; break;
        default: *a = rotr32(*a, x) ^ b; break;
    }
}

/**
 * The per-period program state: a KISS99 generator plus two random
 * permutations of the register indexes.
 *
 * Note it is consumed by value in each of the 64 rounds, so all 64 run the
 * SAME program. That is ProgPoW's design - the program is fixed for the period
 * and repeated - and it is easy to "fix" into a much slower and entirely wrong
 * miner by passing this around by reference.
 */
struct MixRngState {
    Kiss99 rng;
    uint32_t dstSeq[32];
    uint32_t srcSeq[32];
    uint32_t dstCounter;
    uint32_t srcCounter;

    KP_HD uint32_t nextDst() { return dstSeq[(dstCounter++) % kNumRegs]; }
    KP_HD uint32_t nextSrc() { return srcSeq[(srcCounter++) % kNumRegs]; }
};

/** Builds the program state for the period containing `blockNumber`. */
KP_HD MixRngState makeMixRngState(uint64_t periodSeed) {
    const uint32_t seedLo = (uint32_t)periodSeed;
    const uint32_t seedHi = (uint32_t)(periodSeed >> 32);

    MixRngState s;
    const uint32_t z = fnv1a(kFnvOffsetBasis, seedLo);
    const uint32_t w = fnv1a(z, seedHi);
    const uint32_t jsr = fnv1a(w, seedLo);
    const uint32_t jcong = fnv1a(jsr, seedHi);
    s.rng.z = z; s.rng.w = w; s.rng.jsr = jsr; s.rng.jcong = jcong;
    s.dstCounter = 0;
    s.srcCounter = 0;

    for (uint32_t i = 0; i < kNumRegs; i++) { s.dstSeq[i] = i; s.srcSeq[i] = i; }
    // Fisher-Yates, both sequences interleaved, because the RNG is shared.
    for (uint32_t i = kNumRegs; i > 1; i--) {
        uint32_t j = s.rng.next() % i;
        uint32_t t = s.dstSeq[i - 1]; s.dstSeq[i - 1] = s.dstSeq[j]; s.dstSeq[j] = t;
        j = s.rng.next() % i;
        t = s.srcSeq[i - 1]; s.srcSeq[i - 1] = s.srcSeq[j]; s.srcSeq[j] = t;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Ethash layer: epoch sizes, seed, light cache, dataset items
// ---------------------------------------------------------------------------

inline bool isPrime(int n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (int d = 3; (int64_t)d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}

/** Largest prime not greater than `upperBound`. Both cache and dataset sizes
 *  are primed so that the modulo indexing cannot fall into a short cycle. */
inline int largestPrimeNotAbove(int upperBound) {
    for (int n = upperBound; n > 1; n--)
        if (isPrime(n)) return n;
    return 0;
}

inline int epochNumber(int blockNumber) {
    return blockNumber > 0 ? blockNumber / kEpochLength : 0;
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
// ProgPoW layer
// ---------------------------------------------------------------------------

/** Seeds the 16x32 register file from the first two words of the seed hash. */
KP_HD void initMix(const uint32_t hashSeed[2], uint32_t mix[16][32]) {
    const uint32_t z = fnv1a(kFnvOffsetBasis, hashSeed[0]);
    const uint32_t w = fnv1a(z, hashSeed[1]);
    for (uint32_t l = 0; l < kNumLanes; l++) {
        const uint32_t jsr = fnv1a(w, l);
        const uint32_t jcong = fnv1a(jsr, l);
        Kiss99 rng{z, w, jsr, jcong};
        for (uint32_t i = 0; i < kNumRegs; i++) mix[l][i] = rng.next();
    }
}

/**
 * One ProgPoW round. `state` is taken by value on purpose - see MixRngState.
 *
 * `item` is the 2048-bit dataset item for this round, already fetched, so the
 * same code serves the host (which computes items from the light cache) and
 * the device (which reads them from the DAG in VRAM).
 */
KP_HD void progpowRound(uint32_t r, uint32_t mix[16][32], MixRngState state,
                        const uint32_t *l1Cache, const uint32_t item[64]) {
    const int maxOps = kNumCacheAccesses > kNumMathOperations ? kNumCacheAccesses
                                                             : kNumMathOperations;
    for (int i = 0; i < maxOps; i++) {
        if (i < kNumCacheAccesses) {
            const uint32_t src = state.nextSrc();
            const uint32_t dst = state.nextDst();
            const uint32_t sel = state.rng.next();
            for (uint32_t l = 0; l < kNumLanes; l++) {
                const uint32_t offset = mix[l][src] % kL1CacheWords;
                randomMerge(&mix[l][dst], l1Cache[offset], sel);
            }
        }
        if (i < kNumMathOperations) {
            // Two distinct source registers, drawn from one draw so the pair
            // is uniform over ordered distinct pairs.
            const uint32_t srcRnd = state.rng.next() % (kNumRegs * (kNumRegs - 1));
            const uint32_t src1 = srcRnd % kNumRegs;
            uint32_t src2 = srcRnd / kNumRegs;
            if (src2 >= src1) ++src2;

            const uint32_t sel1 = state.rng.next();
            const uint32_t dst = state.nextDst();
            const uint32_t sel2 = state.rng.next();
            for (uint32_t l = 0; l < kNumLanes; l++) {
                const uint32_t data = randomMath(mix[l][src1], mix[l][src2], sel1);
                randomMerge(&mix[l][dst], data, sel2);
            }
        }
    }

    uint32_t dsts[kDagLoads], sels[kDagLoads];
    for (uint32_t i = 0; i < kDagLoads; i++) {
        dsts[i] = i == 0 ? 0 : state.nextDst();
        sels[i] = state.rng.next();
    }
    for (uint32_t l = 0; l < kNumLanes; l++) {
        const uint32_t offset = ((l ^ r) % kNumLanes) * kDagLoads;
        for (uint32_t i = 0; i < kDagLoads; i++)
            randomMerge(&mix[l][dsts[i]], item[offset + i], sels[i]);
    }
}

/** Which 256-byte dataset item round `r` reads, given the current mix. */
KP_HD uint32_t roundItemIndex(uint32_t r, const uint32_t mix[16][32],
                              uint32_t numItems2048) {
    return mix[r % kNumLanes][0] % numItems2048;
}

/** Folds the 16x32 register file down to the 256-bit mix hash. */
KP_HD void reduceMix(const uint32_t mix[16][32], uint32_t mixHash[8]) {
    uint32_t laneHash[16];
    for (uint32_t l = 0; l < kNumLanes; l++) {
        laneHash[l] = kFnvOffsetBasis;
        for (uint32_t i = 0; i < kNumRegs; i++)
            laneHash[l] = fnv1a(laneHash[l], mix[l][i]);
    }
    for (int i = 0; i < 8; i++) mixHash[i] = kFnvOffsetBasis;
    for (uint32_t l = 0; l < kNumLanes; l++)
        mixHash[l % 8] = fnv1a(mixHash[l % 8], laneHash[l]);
}

/**
 * The first Keccak-f[800]: header hash, nonce and the Ravencoin words go in,
 * and the first two output words seed everything downstream.
 *
 * No padding is applied. The 576-bit rate is filled exactly.
 */
KP_HD void kawpowInitialKeccak(const uint32_t headerHash[8], uint64_t nonce,
                               uint32_t state2[8]) {
    uint32_t st[25];
    for (int i = 0; i < 8; i++) st[i] = headerHash[i];
    st[8] = (uint32_t)nonce;
    st[9] = (uint32_t)(nonce >> 32);
    for (int i = 10; i < 25; i++) st[i] = kRavencoinKawpow[i - 10];
    keccakf800(st);
    for (int i = 0; i < 8; i++) state2[i] = st[i];
}

/** The final Keccak-f[800]: the carried state, the mix hash, and the Ravencoin
 *  words again. Its first eight words are the PoW hash. */
KP_HD void kawpowFinalKeccak(const uint32_t state2[8], const uint32_t mixHash[8],
                             uint32_t out[8]) {
    uint32_t st[25];
    for (int i = 0; i < 8; i++) st[i] = state2[i];
    for (int i = 8; i < 16; i++) st[i] = mixHash[i - 8];
    for (int i = 16; i < 25; i++) st[i] = kRavencoinKawpow[i - 16];
    keccakf800(st);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}

/**
 * The whole hash, host side, computing dataset items on demand from the light
 * cache. Slow by construction - it exists to verify, not to mine.
 */
inline void kawpowHashLight(const uint32_t *cache, int numCacheItems,
                            const uint32_t *l1Cache, uint32_t numItems2048,
                            int blockNumber, const uint32_t headerHash[8],
                            uint64_t nonce, uint32_t finalHash[8],
                            uint32_t mixHashOut[8]) {
    uint32_t state2[8];
    kawpowInitialKeccak(headerHash, nonce, state2);

    uint32_t mix[16][32];
    const uint32_t hashSeed[2] = {state2[0], state2[1]};
    initMix(hashSeed, mix);

    const MixRngState base =
        makeMixRngState((uint64_t)(blockNumber / kPeriodLength));

    for (uint32_t r = 0; r < (uint32_t)kNumRounds; r++) {
        const uint32_t index = roundItemIndex(r, mix, numItems2048);
        uint32_t item[64];
        datasetItem2048(cache, numCacheItems, index, item);
        progpowRound(r, mix, base, l1Cache, item);
    }

    reduceMix(mix, mixHashOut);
    kawpowFinalKeccak(state2, mixHashOut, finalHash);
}

// ---------------------------------------------------------------------------
// Target comparison
// ---------------------------------------------------------------------------

/**
 * Ravencoin compares the PoW hash as a BIG-endian 256-bit integer: byte 0 of
 * the digest is the most significant. The core's Job::target and
 * Solution::hit are limb arrays with limb[0] least significant, so this is
 * where the two conventions meet.
 */
KP_HD uint32_t bswap32(uint32_t x) {
    return ((x & 0x000000ffu) << 24) | ((x & 0x0000ff00u) << 8) |
           ((x & 0x00ff0000u) >> 8) | ((x & 0xff000000u) >> 24);
}

KP_HD void hashToLimbs(const uint32_t hash[8], uint64_t limbs[4]) {
    for (int k = 0; k < 4; k++) {
        // Bytes 8k..8k+7 are the most significant when k == 0.
        const uint32_t hi = hash[2 * k];
        const uint32_t lo = hash[2 * k + 1];
        // Each word is little-endian bytes, so reverse them to read the digest
        // as a big-endian number.
        const uint64_t be = ((uint64_t)bswap32(hi) << 32) | (uint64_t)bswap32(lo);
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

}  // namespace kawpow
}  // namespace om

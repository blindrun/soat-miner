// KawPow - the Ravencoin proof of work.
//
// Everything structural lives in src/algos/progpow/: the Ethash DAG, KISS99,
// the random program, the round, the reduction. What is genuinely Ravencoin's
// is here, and it is three things:
//
//   1. The parameters. Period 3 blocks, epoch 7500 blocks, 11 cache accesses,
//      18 math operations, ProgPoW 0.9.4's random_math labelling.
//   2. The two Keccak-f[800] wrappers. Ravencoin fills the unused end of both
//      states with the 15 words "rAVENCOINKAWPOW" and carries the first
//      state's output forward into the second, where stock ProgPoW absorbs the
//      header again. This is the only thing making KawPow incompatible with
//      ProgPoW, and nothing in the block format hints at it.
//   3. How a block header becomes the 32 bytes that go in - see
//      tests/test_kawpow_block.cpp, which is where the byte order is pinned.
//
// Spec source: RavenProject/Ravencoin src/crypto/ethash (Apache-2.0), read as
// a specification. The code is written from the algorithm rather than copied,
// because this repository is MIT. The published hash vectors are used verbatim
// as facts in tests/test_kawpow_hash, with that origin recorded.

#pragma once

#include <stdint.h>

#include "../progpow/ethash.h"
#include "../progpow/keccak.h"
#include "../progpow/progpow.h"

namespace om {
namespace kawpow {

using namespace om::progpow;

// The values below are the ones that silently produce a fast miner with a 100%
// reject rate, so each is overridable at compile time and the Makefile builds a
// deliberately broken binary for each. A correctness test that has never been
// seen to fail is not evidence; see `make test-kawpow`.
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

/** The parameters this coin hands the shared ProgPoW layer. */
KP_HD Params params() {
    Params p;
    p.periodLength = kPeriodLength;
    p.epochLength = kEpochLength;
    p.cacheAccesses = 11;
    p.mathOps = 18;
    p.mathVariant = kMath094;
    return p;
}

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
 */
// A namespace-scope array is host-only under nvcc, so the tail lives inside a
// function both sides can call rather than in two copies that could drift.
KP_HD uint32_t ravenWord(int i) {
#if KAWPOW_RAVEN_TAIL
    const uint32_t t[15] = {
        0x00000072u, 0x00000041u, 0x00000056u, 0x00000045u, 0x0000004Eu,
        0x00000043u, 0x0000004Fu, 0x00000049u, 0x0000004Eu, 0x0000004Bu,
        0x00000041u, 0x00000057u, 0x00000050u, 0x0000004Fu, 0x00000057u};
    return t[i];
#else
    (void)i;
    return 0u;  // mutation: stock ProgPoW, no private tail
#endif
}

/**
 * The first Keccak-f[800]: header hash, nonce and the Ravencoin words go in,
 * and the first two output words seed everything downstream. All eight output
 * words are carried into the final permutation.
 *
 * No padding is applied. The 576-bit rate is filled exactly, so this is a raw
 * permutation rather than a hash with a sponge interface.
 */
KP_HD void initialKeccak(const uint32_t headerHash[8], uint64_t nonce,
                         uint32_t state2[8]) {
    uint32_t st[25];
    for (int i = 0; i < 8; i++) st[i] = headerHash[i];
    st[8] = (uint32_t)nonce;
    st[9] = (uint32_t)(nonce >> 32);
    for (int i = 10; i < 25; i++) st[i] = ravenWord(i - 10);
    keccakf800(st);
    for (int i = 0; i < 8; i++) state2[i] = st[i];
}

/** The final Keccak-f[800]: the carried state, the mix hash, and the Ravencoin
 *  words again. Its first eight words are the PoW hash. */
KP_HD void finalKeccak(const uint32_t state2[8], const uint32_t mixHash[8],
                       uint32_t out[8]) {
    uint32_t st[25];
    for (int i = 0; i < 8; i++) st[i] = state2[i];
    for (int i = 8; i < 16; i++) st[i] = mixHash[i - 8];
    for (int i = 16; i < 25; i++) st[i] = ravenWord(i - 16);
    keccakf800(st);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}

inline int epochOf(int blockNumber) {
    return epochNumber(blockNumber, kEpochLength);
}

/**
 * The whole hash, host side, computing dataset items on demand from the light
 * cache. Slow by construction - it exists to verify, not to mine.
 */
inline void hashLight(const uint32_t *cache, int numCacheItems,
                      const uint32_t *l1, uint32_t numItems2048,
                      int blockNumber, const uint32_t headerHash[8],
                      uint64_t nonce, uint32_t finalHash[8],
                      uint32_t mixHashOut[8]) {
    uint32_t state2[8];
    initialKeccak(headerHash, nonce, state2);
    const uint32_t hashSeed[2] = {state2[0], state2[1]};
    hashMixLight(cache, numCacheItems, l1, numItems2048, blockNumber, params(),
                 hashSeed, mixHashOut);
    finalKeccak(state2, mixHashOut, finalHash);
}

}  // namespace kawpow
}  // namespace om

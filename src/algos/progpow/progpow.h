// The ProgPoW mix: the part that is deliberately hostile to fixed-function
// hardware, and the part KawPow and ProgPowZ genuinely share.
//
// What is actually shared, and what is not
// ----------------------------------------
// "Same family" turned out to be a weaker statement than it sounds, so this is
// checked against both references rather than assumed:
//
//                        KawPow (RVN)        ProgPowZ (ZANO)
//   ProgPoW revision     0.9.4               0.9.2
//   period length        3 blocks            50 blocks
//   Ethash epoch length  7500                30000
//   cache accesses       11                  12
//   math operations      20 -> 18            20
//   random_math mapping  see below           ROTATED BY TWO
//   keccak wrapper       Ravencoin tail      stock ProgPoW
//   everything else      identical           identical
//
// The rotation is the one that would have cost a day. ProgPoW 0.9.2 labels the
// same eleven operations in a different order, so selector%11 == 0 means
// clz(a)+clz(b) under ProgPowZ and a+b under KawPow. Nothing about a wrong
// answer here points at the cause: the DAG, the KISS99 stream and the register
// permutations all still match a reference exactly.
//
// So the round structure, KISS99, the Fisher-Yates permutations, init_mix, the
// DAG access pattern and the reduction live here once. The operation counts
// arrive as a decoded program, the maths variant as a flag, and the two keccak
// wrappers stay with the coin.

#pragma once

#include <stdint.h>

#include "ethash.h"
#include "keccak.h"

namespace om {
namespace progpow {

static const uint32_t kNumRegs = 32;
static const uint32_t kNumLanes = 16;
static const int kNumRounds = 64;

/** Upper bounds across the family, so the program tables are fixed size. */
static const int kMaxCacheAccesses = 12;
static const int kMaxMathOps = 20;

/** Which revision's random_math labelling to use. */
enum MathVariant { kMath094 = 0, kMath092 = 1 };

/** Everything that differs between one ProgPoW coin and another, except the
 *  two keccak wrappers, which are code rather than numbers. */
struct Params {
    int periodLength;
    int epochLength;
    int cacheAccesses;
    int mathOps;
    int mathVariant;
};

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

/** KISS99, the 1999 specification. The whole program derives from it, so a
 *  single wrong constant changes every hash. */
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

/**
 * The eleven operations, in ProgPoW 0.9.4's labelling.
 *
 * 0.9.2 uses the same eleven in a different order: its case 0 is 0.9.4's case
 * 9. Rotating the selector by nine is the whole difference, and doing it here
 * once beats two copies of an eleven-way switch.
 */
KP_HD uint32_t randomMath(uint32_t a, uint32_t b, uint32_t selector,
                          int variant) {
    uint32_t s = selector % 11u;
    if (variant == kMath092) s = (s + 9u) % 11u;
    switch (s) {
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
 *  `b` has little - which is why there is no `a & b` case here. Identical in
 *  0.9.2 and 0.9.4. */
KP_HD void randomMerge(uint32_t *a, uint32_t b, uint32_t selector) {
    const uint32_t x = (selector >> 16) % 31u + 1u;
    switch (selector % 4u) {
        case 0: *a = (*a * 33u) + b; break;
        case 1: *a = (*a ^ b) * 33u; break;
        case 2: *a = rotl32(*a, x) ^ b; break;
        default: *a = rotr32(*a, x) ^ b; break;
    }
}

/** KISS99 plus two random permutations of the register indexes. */
struct MixRngState {
    Kiss99 rng;
    uint32_t dstSeq[32];
    uint32_t srcSeq[32];
    uint32_t dstCounter;
    uint32_t srcCounter;

    KP_HD uint32_t nextDst() { return dstSeq[(dstCounter++) % kNumRegs]; }
    KP_HD uint32_t nextSrc() { return srcSeq[(srcCounter++) % kNumRegs]; }
};

/** Builds the program state for a period. `periodSeed` is blockNumber divided
 *  by the coin's period length. */
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
    // Fisher-Yates over both sequences, interleaved, because the RNG is shared
    // and the draw order is part of the specification.
    for (uint32_t i = kNumRegs; i > 1; i--) {
        uint32_t j = s.rng.next() % i;
        uint32_t t = s.dstSeq[i - 1]; s.dstSeq[i - 1] = s.dstSeq[j]; s.dstSeq[j] = t;
        j = s.rng.next() % i;
        t = s.srcSeq[i - 1]; s.srcSeq[i - 1] = s.srcSeq[j]; s.srcSeq[j] = t;
    }
    return s;
}

/**
 * The random program for one period, decoded once.
 *
 * ProgPoW repeats the SAME program for all 64 rounds - the reference makes
 * this happen by taking the RNG state by value in round(), which is easy to
 * "fix" into a much slower and entirely wrong miner. Decoding it into a table
 * makes the intent explicit and lets a kernel read it out of constant memory.
 */
struct Program {
    int cacheAccesses, mathOps, mathVariant;
    uint32_t cacheSrc[kMaxCacheAccesses];
    uint32_t cacheDst[kMaxCacheAccesses];
    uint32_t cacheSel[kMaxCacheAccesses];
    uint32_t mathSrc1[kMaxMathOps], mathSrc2[kMaxMathOps];
    uint32_t mathSel1[kMaxMathOps], mathDst[kMaxMathOps], mathSel2[kMaxMathOps];
    uint32_t dagDst[kDagLoads], dagSel[kDagLoads];
};

inline Program buildProgram(int blockNumber, const Params &p) {
    MixRngState s = makeMixRngState((uint64_t)(blockNumber / p.periodLength));
    Program out;
    out.cacheAccesses = p.cacheAccesses;
    out.mathOps = p.mathOps;
    out.mathVariant = p.mathVariant;

    const int maxOps = p.cacheAccesses > p.mathOps ? p.cacheAccesses : p.mathOps;
    for (int i = 0; i < maxOps; i++) {
        if (i < p.cacheAccesses) {
            out.cacheSrc[i] = s.nextSrc();
            out.cacheDst[i] = s.nextDst();
            out.cacheSel[i] = s.rng.next();
        }
        if (i < p.mathOps) {
            // Two distinct source registers from one draw, so the pair is
            // uniform over ordered distinct pairs.
            const uint32_t srcRnd = s.rng.next() % (kNumRegs * (kNumRegs - 1));
            uint32_t src1 = srcRnd % kNumRegs;
            uint32_t src2 = srcRnd / kNumRegs;
            if (src2 >= src1) ++src2;
            out.mathSrc1[i] = src1;
            out.mathSrc2[i] = src2;
            out.mathSel1[i] = s.rng.next();
            out.mathDst[i] = s.nextDst();
            out.mathSel2[i] = s.rng.next();
        }
    }
    for (uint32_t i = 0; i < kDagLoads; i++) {
        out.dagDst[i] = i == 0 ? 0 : s.nextDst();
        out.dagSel[i] = s.rng.next();
    }
    return out;
}

/** Seeds one lane's 32 registers. */
KP_HD void initLaneMix(const uint32_t hashSeed[2], uint32_t lane,
                       uint32_t mix[32]) {
    const uint32_t z = fnv1a(kFnvOffsetBasis, hashSeed[0]);
    const uint32_t w = fnv1a(z, hashSeed[1]);
    const uint32_t jsr = fnv1a(w, lane);
    const uint32_t jcong = fnv1a(jsr, lane);
    Kiss99 rng{z, w, jsr, jcong};
    for (uint32_t i = 0; i < kNumRegs; i++) mix[i] = rng.next();
}

/** Runs one round's program over one lane. `dagWords` are the four the lane
 *  owns this round; which four depends on (lane ^ round). */
KP_HD void runProgramLane(const Program &p, uint32_t *mix, const uint32_t *l1,
                          const uint32_t dagWords[kDagLoads]) {
    const int maxOps = p.cacheAccesses > p.mathOps ? p.cacheAccesses : p.mathOps;
    for (int i = 0; i < maxOps; i++) {
        if (i < p.cacheAccesses) {
            const uint32_t offset = mix[p.cacheSrc[i]] % kL1CacheWords;
            randomMerge(&mix[p.cacheDst[i]], l1[offset], p.cacheSel[i]);
        }
        if (i < p.mathOps) {
            const uint32_t data = randomMath(mix[p.mathSrc1[i]],
                                             mix[p.mathSrc2[i]],
                                             p.mathSel1[i], p.mathVariant);
            randomMerge(&mix[p.mathDst[i]], data, p.mathSel2[i]);
        }
    }
    for (uint32_t i = 0; i < kDagLoads; i++)
        randomMerge(&mix[p.dagDst[i]], dagWords[i], p.dagSel[i]);
}

/** Folds 16 lane hashes down to the 256-bit mix hash. */
KP_HD void reduceLaneHashes(const uint32_t laneHash[16], uint32_t mixHash[8]) {
    for (int i = 0; i < 8; i++) mixHash[i] = kFnvOffsetBasis;
    for (uint32_t l = 0; l < kNumLanes; l++)
        mixHash[l % 8] = fnv1a(mixHash[l % 8], laneHash[l]);
}

/**
 * The whole mix, host side, over a 16x32 register file. Used by the reference
 * and by verify(); the device splits the same work across sixteen threads.
 */
inline void hashMixLight(const uint32_t *cache, int numCacheItems,
                         const uint32_t *l1, uint32_t numItems2048,
                         int blockNumber, const Params &params,
                         const uint32_t hashSeed[2], uint32_t mixHash[8]) {
    uint32_t mix[16][32];
    for (uint32_t l = 0; l < kNumLanes; l++) initLaneMix(hashSeed, l, mix[l]);

    const Program prog = buildProgram(blockNumber, params);

    for (uint32_t r = 0; r < (uint32_t)kNumRounds; r++) {
        const uint32_t index = mix[r % kNumLanes][0] % numItems2048;
        uint32_t item[64];
        datasetItem2048(cache, numCacheItems, index, item);
        for (uint32_t l = 0; l < kNumLanes; l++) {
            const uint32_t chunk = (l ^ r) % kNumLanes;
            runProgramLane(prog, mix[l], l1, item + chunk * kDagLoads);
        }
    }

    uint32_t laneHash[16];
    for (uint32_t l = 0; l < kNumLanes; l++) {
        laneHash[l] = kFnvOffsetBasis;
        for (uint32_t i = 0; i < kNumRegs; i++)
            laneHash[l] = fnv1a(laneHash[l], mix[l][i]);
    }
    reduceLaneHashes(laneHash, mixHash);
}

}  // namespace progpow
}  // namespace om

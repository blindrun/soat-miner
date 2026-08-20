// ProgPoW device kernels: the DAG build, and the search.
//
// Shared by every ProgPoW coin. The arithmetic lives in progpow.h and ethash.h
// and is compiled for both host and device, so this file is structure only.
// That is deliberate: the host reference is what gets checked against a coin's
// published vectors, and a kernel that re-derives the maths from the same page
// is a second transcription waiting to disagree.
//
// How the search is laid out
// --------------------------
// ProgPoW is defined over 16 lanes that share one 2048-bit dataset item and
// nothing else, so a lane maps to a thread and a hash maps to sixteen
// consecutive threads. Two hashes therefore occupy one warp, and the only
// cross-lane traffic in a round is a single shuffle for the item index.
//
// The 16 KB L1 cache lives in shared memory, one copy per block, because every
// lane reads it eleven or twelve times per round at an address of its own
// choosing. Hammering it is the point of the algorithm, so bank conflicts here
// are the design, not a defect to optimise away.
//
// The one honest weakness of this version
// ---------------------------------------
// The random program is INTERPRETED: the per-period operations arrive as a
// table and the kernel reads them. Production miners generate source at
// runtime and compile it, which turns the destination register index into a
// compile-time constant and keeps the 32-register mix in registers. Here the
// index is a runtime value, so `mix[dst]` forces the array into local memory.
//
// It is correct, it is the same code on both backends, and it is slower. The
// measured cost belongs in the README rather than in a guess here. The way out
// is NVRTC on CUDA and specialisation constants on Vulkan, both recompiling
// once per period - a separate change, with its own measurement.

#pragma once

#include "ethash.h"
#include "progpow.h"

namespace om {
namespace progpow {

/** A nonce whose hash fell under target, with the mix hash a pool demands.
 *  A ProgPoW share is not just a nonce: the pool checks the mix too. */
struct Solution32 {
    uint64_t nonce;
    uint32_t hit[8];   ///< final hash, little-endian words
    uint32_t mix[8];   ///< mix hash, as submitted
};

/**
 * One thread per 512-bit dataset item.
 *
 * `start` slices the build into launches. A 6 GB DAG is 95 million items and
 * one launch of that would sit on the card for tens of seconds, which is a
 * driver reset on any machine with a display attached.
 *
 * Naive on purpose. The build is 512 pseudo-random 64-byte gathers per item
 * out of a light cache small enough to sit largely in L2, and the two
 * Keccak-512 permutations at the ends are noise beside that. Splitting an item
 * across lanes is the usual next step, and is worth doing once this one has
 * been measured rather than before.
 */
__global__ void ppBuildDag(const uint32_t *__restrict__ cache, int numCacheItems,
                           uint32_t start, uint32_t numItems512,
                           uint32_t *__restrict__ dag) {
    const uint32_t i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numItems512) return;
    uint32_t out[16];
    datasetItem512(cache, numCacheItems, i, out);
    uint32_t *dst = dag + (size_t)i * 16;
    for (int k = 0; k < 16; k++) dst[k] = out[k];
}

/**
 * The mix, minus the coin-specific keccak at either end.
 *
 * Sixteen threads per hash: `lane` is this thread's lane, `laneHash` is a
 * 16-word scratch slot shared by this hash's lanes. Every lane must call it.
 * Only lane 0's `mixHash` is filled in.
 */
__device__ __forceinline__ void ppHashMix(const uint32_t *__restrict__ dag,
                                          uint32_t numItems2048,
                                          const Program &prog,
                                          const uint32_t *l1, uint32_t lane,
                                          const uint32_t hashSeed[2],
                                          uint32_t *laneHash,
                                          uint32_t mixHash[8]) {
    uint32_t mix[32];
    initLaneMix(hashSeed, lane, mix);

    for (uint32_t r = 0; r < (uint32_t)kNumRounds; r++) {
        // The item index comes from lane (r % 16) register 0. Width 16 keeps
        // the two hashes sharing this warp independent of each other.
        const uint32_t seedReg =
            __shfl_sync(0xffffffffu, mix[0], r % kNumLanes, kNumLanes);
        const uint32_t item = seedReg % numItems2048;
        const uint32_t chunk = (lane ^ r) % kNumLanes;
        const uint32_t *src = dag + (size_t)item * 64 + chunk * kDagLoads;
        const uint32_t dagWords[kDagLoads] = {src[0], src[1], src[2], src[3]};
        runProgramLane(prog, mix, l1, dagWords);
    }

    uint32_t lh = kFnvOffsetBasis;
    for (uint32_t i = 0; i < kNumRegs; i++) lh = fnv1a(lh, mix[i]);
    laneHash[lane] = lh;
    __syncwarp();
    if (lane == 0) reduceLaneHashes(laneHash, mixHash);
}

}  // namespace progpow
}  // namespace om

// KawPow's device search: the shared ProgPoW mix with Ravencoin's keccak at
// either end. See src/algos/progpow/mine.cuh for the layout and its costs.

#pragma once

#include "../progpow/mine.cuh"
#include "kawpow.h"

namespace om {
namespace kawpow {

/**
 * Test `count` nonces from `nonceBase`. Sixteen threads per nonce.
 *
 * blockDim.x must be a multiple of 16 and the shared memory allocation must be
 * kL1CacheWords + 16 words per hash-group; algo.cu is the only caller and owns
 * both. The header hash arrives as eight scalars rather than a pointer so it
 * rides in constant memory with the rest of the launch parameters.
 */
__global__ void kpSearch(const uint32_t *__restrict__ dag,
                         uint32_t numItems2048, Program prog, uint32_t h0,
                         uint32_t h1, uint32_t h2, uint32_t h3, uint32_t h4,
                         uint32_t h5, uint32_t h6, uint32_t h7,
                         uint64_t nonceBase, uint64_t count, uint64_t t0,
                         uint64_t t1, uint64_t t2, uint64_t t3,
                         Solution32 *out, uint32_t *found, uint32_t cap) {
    extern __shared__ uint32_t shared[];
    uint32_t *l1 = shared;                          // kL1CacheWords words
    uint32_t *laneScratch = shared + kL1CacheWords; // 16 per hash group

    // The L1 cache is literally the first 16 KB of the DAG, copied in once per
    // block rather than re-read from global memory 700 times per lane.
    for (uint32_t i = threadIdx.x; i < kL1CacheWords; i += blockDim.x)
        l1[i] = dag[i];
    __syncthreads();

    const uint32_t lane = threadIdx.x % kNumLanes;
    const uint32_t group = threadIdx.x / kNumLanes;
    const uint64_t index = (uint64_t)blockIdx.x * (blockDim.x / kNumLanes) + group;
    if (index >= count) return;
    const uint64_t nonce = nonceBase + index;

    const uint32_t header[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    uint32_t state2[8];
    initialKeccak(header, nonce, state2);
    const uint32_t hashSeed[2] = {state2[0], state2[1]};

    uint32_t mixHash[8];
    ppHashMix(dag, numItems2048, prog, l1, lane, hashSeed,
              laneScratch + group * kNumLanes, mixHash);
    if (lane != 0) return;

    uint32_t final32[8];
    finalKeccak(state2, mixHash, final32);

    uint64_t hit[4];
    hashToLimbs(final32, hit);
    const uint64_t target[4] = {t0, t1, t2, t3};
    if (!underTarget(hit, target)) return;

    // Count every hit even past capacity, so the host can say how many were
    // dropped rather than reporting a full buffer as the total.
    const uint32_t slot = atomicAdd(found, 1u);
    if (slot >= cap) return;
    out[slot].nonce = nonce;
    for (int i = 0; i < 8; i++) {
        out[slot].hit[i] = final32[i];
        out[slot].mix[i] = mixHash[i];
    }
}

}  // namespace kawpow
}  // namespace om

namespace om {
namespace kawpow {

/**
 * Hash `count` consecutive nonces and write every result, under target or not.
 *
 * Not used for mining - it is what lets a test compare the device against the
 * host reference nonce by nonce, which is the only way to find a kernel that
 * is wrong for one input in a million rather than wrong for all of them.
 */
__global__ void kpHashBatch(const uint32_t *__restrict__ dag,
                            uint32_t numItems2048, Program prog, uint32_t h0,
                            uint32_t h1, uint32_t h2, uint32_t h3, uint32_t h4,
                            uint32_t h5, uint32_t h6, uint32_t h7,
                            uint64_t nonceBase, uint64_t count,
                            Solution32 *out) {
    extern __shared__ uint32_t shared[];
    uint32_t *l1 = shared;
    uint32_t *laneScratch = shared + kL1CacheWords;

    for (uint32_t i = threadIdx.x; i < kL1CacheWords; i += blockDim.x)
        l1[i] = dag[i];
    __syncthreads();

    const uint32_t lane = threadIdx.x % kNumLanes;
    const uint32_t group = threadIdx.x / kNumLanes;
    const uint64_t index = (uint64_t)blockIdx.x * (blockDim.x / kNumLanes) + group;
    if (index >= count) return;
    const uint64_t nonce = nonceBase + index;

    const uint32_t header[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    uint32_t state2[8];
    initialKeccak(header, nonce, state2);
    const uint32_t hashSeed[2] = {state2[0], state2[1]};

    uint32_t mixHash[8];
    ppHashMix(dag, numItems2048, prog, l1, lane, hashSeed,
              laneScratch + group * kNumLanes, mixHash);
    if (lane != 0) return;

    uint32_t final32[8];
    finalKeccak(state2, mixHash, final32);
    out[index].nonce = nonce;
    for (int i = 0; i < 8; i++) {
        out[index].hit[i] = final32[i];
        out[index].mix[i] = mixHash[i];
    }
}

}  // namespace kawpow
}  // namespace om

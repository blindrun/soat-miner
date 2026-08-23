// The specialised KawPow kernel, as source text.
//
// # Why this is a function and not a script
//
// A ProgPoW program is a function of the block period, so the kernel that runs
// it can be *generated with every index and selector as a literal*. That is the
// whole speedup: `mix[32]` indexed by a runtime value cannot live in registers,
// so the interpreted kernel spills it to local memory and pays a 128-byte stack
// frame per lane. With literal indices the array disappears into registers.
//
// The offline tool (`tests/gen_spec_kernel.cpp`) and the runtime NVRTC path
// both call THIS function. That is deliberate and it is the point of the file:
// two generators would drift, and the moment they drift the kernel that was
// measured is no longer the kernel that ships - a difference that shows up as
// a performance claim nobody can reproduce, or worse, as a correctness gap in
// only one of them.
//
// The generated source `#include`s the real `kawpow_mine.cuh` rather than
// restating the mix. There is one definition of this hash function in the tree
// and the generated kernel is a specialisation of it, not a copy: a second copy
// is how a miner ends up hashing confidently wrong.

#pragma once

#include <string>

#include "kawpow.h"

namespace om {
namespace kawpow {

/// The `#include` name the generated source expects for the KawPow device
/// header. NVRTC matches in-memory headers by exact name, so this constant is
/// shared by the generator and whoever registers the headers - if they disagree
/// the compile fails loudly rather than silently resolving somewhere else.
inline const char *kDeviceHeaderName() { return "kawpow_mine.cuh"; }

/// Emit the specialised lane body: one line per operation, all indices literal.
///
/// Mirrors `runProgramLane` in `progpow/mine.cuh`. The interleave of cache and
/// math operations is part of the algorithm, not a formatting choice - the two
/// sequences have different lengths and both must be emitted in step.
inline std::string emitSpecLane(const om::progpow::Program &g) {
    std::string s;
    char buf[256];
    s += "__device__ __forceinline__ void specLane(uint32_t *mix,\n";
    s += "        const uint32_t *l1, const uint32_t dagWords[kDagLoads]) {\n";
    const int maxOps = g.cacheAccesses > g.mathOps ? g.cacheAccesses : g.mathOps;
    for (int i = 0; i < maxOps; i++) {
        if (i < g.cacheAccesses) {
            snprintf(buf, sizeof(buf),
                     "    randomMerge(&mix[%u], l1[mix[%u] %% kL1CacheWords], "
                     "%uu);\n",
                     g.cacheDst[i], g.cacheSrc[i], g.cacheSel[i]);
            s += buf;
        }
        if (i < g.mathOps) {
            snprintf(buf, sizeof(buf),
                     "    randomMerge(&mix[%u], randomMath(mix[%u], mix[%u], "
                     "%uu, %d), %uu);\n",
                     g.mathDst[i], g.mathSrc1[i], g.mathSrc2[i], g.mathSel1[i],
                     g.mathVariant, g.mathSel2[i]);
            s += buf;
        }
    }
    for (uint32_t i = 0; i < om::progpow::kDagLoads; i++) {
        snprintf(buf, sizeof(buf),
                 "    randomMerge(&mix[%u], dagWords[%u], %uu);\n", g.dagDst[i],
                 i, g.dagSel[i]);
        s += buf;
    }
    s += "}\n";
    return s;
}

/// The complete translation unit NVRTC compiles for one period.
///
/// Two kernels come out of it and both are needed:
///
///   * `kpSearchSpec` - what mines.
///   * `kpHashSpec`   - writes every nonce's result whether or not it is under
///     target. This is what makes the generated kernel *checkable*. Comparing
///     solutions alone cannot distinguish a correct kernel from one that finds
///     nothing, and "finds nothing" is precisely how a broken miner presents.
///     Hashing a fixed range and comparing against the interpreted kernel
///     byte-for-byte has no such blind spot.
inline std::string specialisedSource(const om::progpow::Program &g) {
    std::string s;
    s += "// GENERATED - one period's program as literals. Do not edit.\n";
    s += "#include \"";
    s += kDeviceHeaderName();
    s += "\"\n\nnamespace om { namespace kawpow {\n\n";
    s += emitSpecLane(g);

    // The mix loop, identical to ppHashMix except for the specLane call. It is
    // restated rather than parameterised because a __device__ function pointer
    // would defeat the inlining this exists to get.
    s += R"CU(
__device__ __forceinline__ void specHashMix(const uint32_t *__restrict__ dag,
                                            uint32_t numItems2048,
                                            const uint32_t *l1, uint32_t lane,
                                            const uint32_t hashSeed[2],
                                            uint32_t *laneHash,
                                            uint32_t mixHash[8]) {
    uint32_t mix[32];
    initLaneMix(hashSeed, lane, mix);
    for (uint32_t r = 0; r < (uint32_t)kNumRounds; r++) {
        const uint32_t seedReg =
            __shfl_sync(0xffffffffu, mix[0], r % kNumLanes, kNumLanes);
        const uint32_t item = seedReg % numItems2048;
        const uint32_t chunk = (lane ^ r) % kNumLanes;
        const uint32_t *src = dag + (size_t)item * 64 + chunk * kDagLoads;
        const uint32_t dagWords[kDagLoads] = {src[0], src[1], src[2], src[3]};
        specLane(mix, l1, dagWords);
    }
    uint32_t lh = kFnvOffsetBasis;
    for (uint32_t i = 0; i < kNumRegs; i++) lh = fnv1a(lh, mix[i]);
    laneHash[lane] = lh;
    __syncwarp();
    if (lane == 0) reduceLaneHashes(laneHash, mixHash);
}

// extern "C" so cuModuleGetFunction can find them by the name written here,
// rather than by a mangled name that depends on the compiler version.
extern "C" __global__ void kpSearchSpec(
        const uint32_t *__restrict__ dag, uint32_t numItems2048, uint32_t h0,
        uint32_t h1, uint32_t h2, uint32_t h3, uint32_t h4, uint32_t h5,
        uint32_t h6, uint32_t h7, uint64_t nonceBase, uint64_t count,
        uint64_t t0, uint64_t t1, uint64_t t2, uint64_t t3, Solution32 *out,
        uint32_t *found, uint32_t cap) {
    extern __shared__ uint32_t shared[];
    uint32_t *l1 = shared;
    uint32_t *laneScratch = shared + kL1CacheWords;
    for (uint32_t i = threadIdx.x; i < kL1CacheWords; i += blockDim.x)
        l1[i] = dag[i];
    __syncthreads();

    const uint32_t lane = threadIdx.x % kNumLanes;
    const uint32_t group = threadIdx.x / kNumLanes;
    const uint64_t index =
        (uint64_t)blockIdx.x * (blockDim.x / kNumLanes) + group;
    if (index >= count) return;
    const uint64_t nonce = nonceBase + index;

    const uint32_t header[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    uint32_t state2[8];
    initialKeccak(header, nonce, state2);
    const uint32_t hashSeed[2] = {state2[0], state2[1]};

    uint32_t mixHash[8];
    specHashMix(dag, numItems2048, l1, lane, hashSeed,
                laneScratch + group * kNumLanes, mixHash);
    if (lane != 0) return;

    uint32_t final32[8];
    finalKeccak(state2, mixHash, final32);
    uint64_t hit[4];
    hashToLimbs(final32, hit);
    const uint64_t target[4] = {t0, t1, t2, t3};
    if (!underTarget(hit, target)) return;

    const uint32_t slot = atomicAdd(found, 1u);
    if (slot >= cap) return;
    out[slot].nonce = nonce;
    for (int i = 0; i < 8; i++) {
        out[slot].hit[i] = final32[i];
        out[slot].mix[i] = mixHash[i];
    }
}

extern "C" __global__ void kpHashSpec(
        const uint32_t *__restrict__ dag, uint32_t numItems2048, uint32_t h0,
        uint32_t h1, uint32_t h2, uint32_t h3, uint32_t h4, uint32_t h5,
        uint32_t h6, uint32_t h7, uint64_t nonceBase, uint64_t count,
        Solution32 *out) {
    extern __shared__ uint32_t shared[];
    uint32_t *l1 = shared;
    uint32_t *laneScratch = shared + kL1CacheWords;
    for (uint32_t i = threadIdx.x; i < kL1CacheWords; i += blockDim.x)
        l1[i] = dag[i];
    __syncthreads();

    const uint32_t lane = threadIdx.x % kNumLanes;
    const uint32_t group = threadIdx.x / kNumLanes;
    const uint64_t index =
        (uint64_t)blockIdx.x * (blockDim.x / kNumLanes) + group;
    if (index >= count) return;
    const uint64_t nonce = nonceBase + index;

    const uint32_t header[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    uint32_t state2[8];
    initialKeccak(header, nonce, state2);
    const uint32_t hashSeed[2] = {state2[0], state2[1]};

    uint32_t mixHash[8];
    specHashMix(dag, numItems2048, l1, lane, hashSeed,
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

}}  // namespace om::kawpow
)CU";
    return s;
}

}  // namespace kawpow
}  // namespace om

// Measures the specialised (NVRTC-shaped) KawPow kernel against the shipped
// interpreted one, on the REAL DAG, reporting MH/s and MH/W.
//
// The specialised kernel is what NVRTC emits at runtime; generating it with
// gen_spec_kernel and compiling with nvcc measures the same codegen without the
// runtime-compilation plumbing. What is NOT measured here is NVRTC's per-period
// compile latency, which amortises over a 3-block period.
//
// Correctness first: both kernels must agree on the mix hash before any timing
// is reported. A faster kernel that hashes differently is worthless, and the
// interpreter is the reference that makes generated code safe to trust.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <nvml.h>

#include "../src/algos/kawpow/mine.cuh"
#include "spec_kernel.cuh"

using namespace om;
using namespace om::progpow;

// The specialised search: identical to kpSearch except the mix loop calls
// specLane, whose indices and selectors are literals.
__global__ void kpSearchSpec(const uint32_t *__restrict__ dag,
                             uint32_t numItems2048, uint32_t h0, uint32_t h1,
                             uint32_t h2, uint32_t h3, uint32_t h4, uint32_t h5,
                             uint32_t h6, uint32_t h7, uint64_t nonceBase,
                             uint64_t count, uint32_t *out) {
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
    om::kawpow::initialKeccak(header, nonce, state2);
    const uint32_t hashSeed[2] = {state2[0], state2[1]};

    uint32_t mix[32];
    initLaneMix(hashSeed, lane, mix);
    for (uint32_t r = 0; r < (uint32_t)kNumRounds; r++) {
        const uint32_t seedReg =
            __shfl_sync(0xffffffffu, mix[0], r % kNumLanes, kNumLanes);
        const uint32_t item = seedReg % numItems2048;
        const uint32_t chunk = (lane ^ r) % kNumLanes;
        const uint32_t *src = dag + (size_t)item * 64 + chunk * kDagLoads;
        const uint32_t dw[kDagLoads] = {src[0], src[1], src[2], src[3]};
        om::kawpow::specLane(mix, l1, dw);
    }
    uint32_t lh = kFnvOffsetBasis;
    for (uint32_t i = 0; i < kNumRegs; i++) lh = fnv1a(lh, mix[i]);
    laneScratch[group * kNumLanes + lane] = lh;
    __syncwarp();
    if (lane != 0) return;
    uint32_t mixHash[8];
    reduceLaneHashes(laneScratch + group * kNumLanes, mixHash);

    // The same tail kpSearch runs. Without it this kernel does strictly less
    // work than the one it is compared against - 976 SASS against 1464 - and
    // the first version of this benchmark reported 6.78x on exactly that
    // asymmetry. A speedup measured against a kernel doing more work is not a
    // speedup.
    uint32_t final32[8];
    om::kawpow::finalKeccak(state2, mixHash, final32);
    uint64_t hit[4];
    hashToLimbs(final32, hit);
    const uint64_t target[4] = {0, 0, 0, 0};
    if (underTarget(hit, target)) out[0] = final32[0];   // never taken

    if (index < 64)
        for (int i = 0; i < 8; i++) out[index * 8 + i] = mixHash[i];
}

int main() {
    const int height = om::kawpow::kSpecHeight;
    const int epoch = height / 7500;
    const int nCache = lightCacheNumItems(epoch);
    const uint32_t nItems512 = (uint32_t)fullDatasetNumItems(epoch) * 2;
    const uint32_t nItems2048 = (uint32_t)(fullDatasetNumItems(epoch) / 2);
    const uint64_t dagBytes = fullDatasetBytes(epoch);
    printf("epoch %d, DAG %.2f GB\n", epoch, dagBytes / 1e9);

    uint32_t seed[8];
    epochSeed(epoch, seed);
    std::vector<uint32_t> cache((size_t)nCache * 16, 0);
    buildLightCache(cache.data(), nCache, seed);

    uint32_t *dCache = nullptr, *dDag = nullptr, *dOutA = nullptr, *dOutB = nullptr;
    cudaMalloc(&dCache, cache.size() * 4);
    cudaMemcpy(dCache, cache.data(), cache.size() * 4, cudaMemcpyHostToDevice);
    if (cudaMalloc(&dDag, dagBytes) != cudaSuccess) {
        printf("cannot allocate the %.2f GB DAG\n", dagBytes / 1e9);
        return 1;
    }
    cudaMalloc(&dOutA, 64 * 8 * 4);
    cudaMalloc(&dOutB, 64 * 8 * 4);
    const uint32_t slice = 1u << 20;
    for (uint32_t s = 0; s < nItems512; s += slice) {
        const uint32_t n = (nItems512 - s) < slice ? (nItems512 - s) : slice;
        ppBuildDag<<<(n + 255) / 256, 256>>>(dCache, nCache, s, nItems512, dDag);
    }
    cudaDeviceSynchronize();
    cudaFree(dCache);

    const uint32_t threads = 128, groups = threads / kNumLanes;
    const size_t shared = (kL1CacheWords + groups * kNumLanes) * 4;
    const uint64_t COUNT = 1u << 20;
    const unsigned blocks = (unsigned)((COUNT + groups - 1) / groups);
    uint32_t hdr[8];
    for (int i = 0; i < 8; i++) hdr[i] = 0xabababab;

    Program prog = buildProgram(height, om::kawpow::params());
    Solution32 *dSol = nullptr; uint32_t *dFound = nullptr;
    cudaMalloc(&dSol, sizeof(Solution32) * 4); cudaMalloc(&dFound, 4);
    cudaMemset(dFound, 0, 4);

    nvmlInit_v2();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex_v2(0, &dev);

    // Timing, with a warm-up launch each. Watts sampled mid-run.
    float msA = 0, msB = 0; unsigned wA = 0, wB = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int ITERS = 12;

    kpSearchSpec<<<blocks, threads, shared>>>(dDag, nItems2048, hdr[0], hdr[1],
        hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7], 0, COUNT, dOutB);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < ITERS; i++)
        kpSearchSpec<<<blocks, threads, shared>>>(dDag, nItems2048, hdr[0], hdr[1],
            hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7], (uint64_t)i * COUNT,
            COUNT, dOutB);
    nvmlDeviceGetPowerUsage(dev, &wB);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&msB, e0, e1);

    // The shipped interpreted kernel, same inputs, same nonce count.
    const uint64_t T[4] = {0, 0, 0, 0};   // never hits: we are timing, not mining
    om::kawpow::kpSearch<<<blocks, threads, shared>>>(dDag, nItems2048, prog, hdr[0], hdr[1],
        hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7], 0, COUNT,
        T[0], T[1], T[2], T[3], dSol, dFound, 4);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < ITERS; i++)
        om::kawpow::kpSearch<<<blocks, threads, shared>>>(dDag, nItems2048, prog, hdr[0],
            hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7],
            (uint64_t)i * COUNT, COUNT, T[0], T[1], T[2], T[3], dSol, dFound, 4);
    nvmlDeviceGetPowerUsage(dev, &wA);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&msA, e0, e1);

    // Correctness before speed: the specialised kernel's mix hash must equal
    // the host reference for the same nonce. The interpreter is what makes
    // generated code safe to trust, so this gates the timing below.
    {
        // A DEDICATED launch at nonceBase 0. The timing loop above runs with
        // nonceBase = i * COUNT, so dOutB otherwise holds hashes for nonce
        // (ITERS-1)*COUNT + n while the host computes nonce n - comparing
        // different nonces and failing for a reason that says nothing about
        // the kernel. The first version of this test did exactly that.
        kpSearchSpec<<<blocks, threads, shared>>>(dDag, nItems2048, hdr[0],
            hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7], 0, COUNT,
            dOutB);
        cudaDeviceSynchronize();

        std::vector<uint32_t> got(64 * 8);
        cudaMemcpy(got.data(), dOutB, got.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<uint32_t> l1(kL1CacheWords);
        cudaMemcpy(l1.data(), dDag, kL1CacheWords * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int n = 0; n < 8; n++) {
            uint32_t st[8], want[8];
            om::kawpow::initialKeccak(hdr, (uint64_t)n, st);
            const uint32_t hs[2] = {st[0], st[1]};
            hashMixLight(cache.data(), nCache, l1.data(), nItems2048, height,
                         om::kawpow::params(), hs, want);
            if (memcmp(want, &got[n * 8], 32) != 0) bad++;
        }
        printf("  correctness: %d/8 nonces match the host reference%s\n",
               8 - bad, bad ? "  <-- MISMATCH, ignore the timings" : "");
        if (bad) return 1;
    }

    printf("\n  interpreted:  %8.2f ms  %7.2f MH/s  %3u W  %.4f MH/W\n", msA,
           (double)ITERS * COUNT / (msA / 1000.0) / 1e6, wA / 1000,
           ((double)ITERS * COUNT / (msA / 1000.0) / 1e6) / (wA / 1000.0));
    printf("  specialised:  %8.2f ms  %7.2f MH/s  %3u W  %.4f MH/W\n",
           msB, (double)ITERS * COUNT / (msB / 1000.0) / 1e6, wB / 1000,
           ((double)ITERS * COUNT / (msB / 1000.0) / 1e6) / (wB / 1000.0));
    printf("\n  speedup: %.2fx throughput, %.2fx efficiency\n",
           msA / msB, (msA / msB) * ((double)wA / (double)wB));
    cudaFree(dDag);
    return 0;
}

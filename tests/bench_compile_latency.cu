// How long does a per-period kernel compile actually stall mining?
//
// This is the number that decides whether runtime compilation is usable rather
// than merely faster. KawPow's period is KAWPOW_PERIOD_LENGTH (3) blocks and
// Ravencoin targets one-minute blocks, so a compile lands roughly every 180
// seconds. A cost of 200 ms against that is 0.1% and invisible; a cost of 4 s
// is 2.2% and a stall users will notice and report.
//
// TWO NUMBERS, NOT ONE
//
// The first compile of a process carries one-time NVRTC and CUDA driver
// initialisation. Every compile after it does not. Averaging them together
// would overstate the steady-state cost and understate the startup cost, and
// the two are felt differently: startup happens once where a user is already
// waiting, and the per-period cost repeats forever while they are mining.
//
// WHAT IS AND IS NOT INCLUDED
//
// The per-period stall in algo.cu is compile + module load + verification, and
// all three are timed here. The verification runs against a SYNTHETIC DAG
// rather than the real 6.1 GB one, so its figure is a lower bound - though it
// is two launches of 512 nonces, which is dominated by launch overhead either
// way. The real miner prints its own measured figures at each period change, so
// the production number never has to be inferred from this one.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "algos/kawpow/mine.cuh"
#include "algos/kawpow/nvrtc_jit.h"

using namespace om::kawpow;
using om::progpow::Program;
using om::progpow::Solution32;

static double msSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t)
        .count();
}

int main(int argc, char **argv) {
    const int periods = argc > 1 ? atoi(argv[1]) : 12;
    const int baseHeight = argc > 2 ? atoi(argv[2]) : 4501000;

    cudaDeviceProp prop{};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }

    NvrtcJit jit;
    if (!jit.available()) {
        printf("SKIP: %s\n", jit.unavailableReason().c_str());
        return 0;
    }
    printf("device: %s (sm_%d%d), %d periods\n", prop.name, prop.major,
           prop.minor, periods);

    // A small synthetic DAG so the verification launches have something to
    // read. Random, not zeros - see tests/test_nvrtc_device.cu.
    const uint32_t numItems2048 = 4096;
    const size_t dagWords = (size_t)numItems2048 * 64;
    std::vector<uint32_t> hostDag(dagWords);
    uint32_t seed = 0x12345678u;
    for (size_t i = 0; i < dagWords; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        hostDag[i] = seed;
    }
    uint32_t *dDag = nullptr;
    if (cudaMalloc(&dDag, dagWords * 4) != cudaSuccess) {
        printf("SKIP: could not allocate the test DAG\n");
        return 0;
    }
    cudaMemcpy(dDag, hostDag.data(), dagWords * 4, cudaMemcpyHostToDevice);

    const uint64_t n = 512;  // what algo.cu verifies over
    Solution32 *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, sizeof(Solution32) * n);
    cudaMalloc(&dB, sizeof(Solution32) * n);

    const uint32_t hdr[8] = {0x9e3779b9u, 0x243f6a88u, 0x85a308d3u, 0x13198a2eu,
                             0x03707344u, 0xa4093822u, 0x299f31d0u, 0x082efa98u};
    const int threads = 512;
    const int groups = threads / (int)om::progpow::kNumLanes;
    const unsigned shared =
        (om::progpow::kL1CacheWords + groups * om::progpow::kNumLanes) * 4;
    const unsigned blocks = (unsigned)((n + groups - 1) / groups);

    std::vector<double> compileMs, verifyMs, loadMs;
    for (int i = 0; i < periods; i++) {
        const int height = baseHeight + i * kPeriodLength;
        const Program prog = om::progpow::buildProgram(height, params());

        auto t0 = std::chrono::steady_clock::now();
        const bool ok = jit.build(prog, prop.major, prop.minor);
        const double buildMs = msSince(t0);
        if (!ok) {
            printf("  period %d: build FAILED\n", i);
            continue;
        }
        // Split, because they are different costs with different fixes and
        // reporting them as one sends anyone optimising it at the wrong half.
        const double cms = jit.lastCompileMs();
        const double lms = jit.lastLoadMs();
        (void)buildMs;

        // The verification as algo.cu performs it: interpreted and generated
        // over the same nonces, both synchronised.
        auto t1 = std::chrono::steady_clock::now();
        kpHashBatch<<<blocks, threads, shared>>>(
            dDag, numItems2048, prog, hdr[0], hdr[1], hdr[2], hdr[3], hdr[4],
            hdr[5], hdr[6], hdr[7], 0x5EED0000ULL, n, dA);
        uint32_t nItems = numItems2048;
        uint64_t nb = 0x5EED0000ULL, cnt = n;
        uint32_t h[8];
        memcpy(h, hdr, sizeof(h));
        void *args[] = {&dDag, &nItems, &h[0], &h[1], &h[2], &h[3],
                        &h[4],  &h[5],   &h[6], &h[7], &nb,   &cnt, &dB};
        jit.launch(jit.hashKernel(), blocks, threads, shared, nullptr, args);
        cudaDeviceSynchronize();
        const double vms = msSince(t1);

        compileMs.push_back(cms);
        verifyMs.push_back(vms);
        printf("  period %2d: compile %7.1f   load %6.1f   verify %5.2f ms%s\n",
               i, cms, lms, vms, i == 0 ? "   <- first" : "");
        loadMs.push_back(lms);
    }

    cudaFree(dDag); cudaFree(dA); cudaFree(dB);
    if (compileMs.size() < 2) {
        printf("not enough successful builds to summarise\n");
        return 1;
    }

    auto summarise = [](std::vector<double> v, const char *label) {
        std::sort(v.begin(), v.end());
        const double med = v[v.size() / 2];
        double sum = 0;
        for (double d : v) sum += d;
        printf("%s: min %.1f  median %.1f  max %.1f  mean %.1f ms  (n=%zu)\n",
               label, v.front(), med, v.back(), sum / v.size(), v.size());
        return med;
    };

    const double firstCompile = compileMs.front();
    std::vector<double> steady(compileMs.begin() + 1, compileMs.end());
    std::vector<double> steadyVerify(verifyMs.begin() + 1, verifyMs.end());
    std::vector<double> steadyLoad(loadMs.begin() + 1, loadMs.end());

    printf("\nfirst compile (one-time init included): %.1f ms\n", firstCompile);
    const double medCompile = summarise(steady, "steady-state compile");
    const double medLoad = summarise(steadyLoad, "module load          ");
    const double medVerify = summarise(steadyVerify, "verification         ");

    // Against the period it amortises over. This is the whole point of the
    // measurement: a duration means nothing without the interval it repeats on.
    const double periodSeconds = (double)kPeriodLength * 60.0;  // 1-min blocks
    const double stallMs = medCompile + medLoad + medVerify;
    printf("\nper-period stall: %.1f ms every %.0f s (%d blocks at 60 s)\n",
           stallMs, periodSeconds, kPeriodLength);
    printf("that is %.3f%% of a period - ", 100.0 * stallMs / (periodSeconds * 1000.0));
    if (stallMs < 500.0)
        printf("invisible in practice\n");
    else if (stallMs < 2000.0)
        printf("noticeable in the hashrate readout, tolerable\n");
    else
        printf("a stall users will see and report\n");
    return 0;
}

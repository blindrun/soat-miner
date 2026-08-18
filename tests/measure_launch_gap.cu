// How much time is actually lost BETWEEN kernel launches?
//
// Codex's objection to the previous answer was fair twice over. First, my
// utilisation samples were filtered to those above 50%, which could have
// excluded exactly the idle intervals a CUDA graph would remove. Second, and
// worse, nvidia-smi's utilisation is a coarse sampled metric - it reports
// whether a kernel was resident over a sample period and cannot resolve a
// 3.6 us launch gap at all. It was the wrong instrument.
//
// This measures the gap directly and needs no profiler: time ONE launch, then
// time N back-to-back launches on the same stream. If launches were free,
// N launches would take exactly N times one. Whatever is left over IS the
// per-launch gap, which is the entire quantity a CUDA graph can recover.
#include <cstdio>
#include "algos/pearl-pow/noisy_gemm.cuh"
#include "algos/pearl-pow/prepare.cuh"

using namespace om::pearl;

int main(int argc, char **argv) {
    const int m = argc > 1 ? atoi(argv[1]) : 8192;
    const int n = argc > 2 ? atoi(argv[2]) : 16384;
    const int k = 2048, rank = 128;

    const size_t aB = (size_t)m * k, bB = (size_t)n * k;
    const size_t tiles = (size_t)(m / 16) * (n / 16);
    int8_t *dA, *dBn;
    uint32_t *dT;
    if (cudaMalloc(&dA, aB) || cudaMalloc(&dBn, bB) ||
        cudaMalloc(&dT, tiles * 16 * 4)) {
        printf("alloc failed\n");
        return 1;
    }
    genMatrix<<<(aB / 8 + 255) / 256, 256>>>(dA, aB, 7);
    genMatrix<<<(bB / 8 + 255) / 256, 256>>>(dBn, bB, 8);
    cudaDeviceSynchronize();

    constexpr int WM = 4, WN = 4, TM = 2, TN = 4, KKB = 64, ST = 3;
    constexpr int blockM = WM * TM * 16, blockN = WN * TN * 16;
    constexpr int smem = ST * (blockM * (KKB + 16) + blockN * (KKB + 16));
    cudaFuncSetAttribute(noisyGemmPtx<WM, WN, TM, TN, KKB, ST>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(n / blockN, m / blockM);
    const int threads = WM * WN * 32;

    auto fire = [&]() {
        noisyGemmPtx<WM, WN, TM, TN, KKB, ST>
            <<<grid, threads, smem>>>(dA, dBn, nullptr, dT, m, n, k, rank, false);
    };

    for (int i = 0; i < 5; i++) fire();          // warm up clocks and caches
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    double one = 1e9;
    for (int rep = 0; rep < 5; rep++) {          // min, not mean: interference
        cudaEventRecord(a);                      // only ever adds time
        fire();
        cudaEventRecord(b);
        cudaDeviceSynchronize();
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < one) one = ms;
    }

    const int N = 200;
    double many = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(a);
        for (int i = 0; i < N; i++) fire();
        cudaEventRecord(b);
        cudaDeviceSynchronize();
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < many) many = ms;
    }

    const double per = many / N;
    const double gap = per - one;
    printf("shape %dx%d  single launch %.4f ms\n", m, n, one);
    printf("%d back to back: %.3f ms total, %.4f ms each\n", N, many, per);
    printf("per-launch gap: %.4f ms (%.2f%% of a launch)\n", gap, 100.0 * gap / per);
    // Sweep shapes before drawing a conclusion. A launch gap is a CONSTANT;
    // a tail effect scales with the kernel. Measured on a 4090 the gap is
    // NEGATIVE at 1024x2048, 2048x4096 and 4096x8192 (-0.003 ms or so) and
    // only positive at 8192x16384, where it was +0.039 ms on one run and
    // +0.065 on the next. Negative means back-to-back launches are FASTER per
    // launch than an isolated one, which is pipelining - the driver queues the
    // next launch before the previous kernel drains. There is no gap to
    // recover, so a CUDA graph buys nothing here.
    //
    // Do not discard the negative readings as noise. They are the finding.
    if (gap < 0)
        printf("  -> negative: launches pipeline, no gap for a CUDA graph to "
               "recover\n");
    else
        printf("  -> sweep other shapes before believing this is a launch gap "
               "and not a tail effect\n");
    return 0;
}

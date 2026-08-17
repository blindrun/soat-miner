// The cp.async kernels against the double-buffered one, at the k, rank and
// SHAPE the miner actually mines at.
//
// This exists because the reference vectors did not catch a real bug. Their k
// is small, so the pipeline never drains the way it does over 64 k-slices and
// the copy for the final step has always landed by the time it is read. At
// k=2048 it has not: both cp.async kernels read a buffer whose copy was still
// in flight on the last step, because no new commit is issued once the loop
// starts draining and the steady-state wait_prior permits exactly the number
// of groups that are left. Only transcript word 15 ever differed, on about
// 10% of tiles, and only at the mining shape.
//
// noisyGemmMmaTiledDB is the oracle: it uses plain __syncthreads staging with
// no async pipeline at all, and has mined accepted blocks unchanged.
//
// Exit status is the gate. Any differing tile is a failure.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "algos/pearl-pow/noisy_gemm.cuh"
#include "algos/pearl-pow/prepare.cuh"

// The raw-mma kernel stages a k-block and pads both shared strides, which puts
// it past the 48 KB static limit - so its shared memory is dynamic and every
// launch site must size it and opt in. Getting this wrong is not subtle: the
// kernel reads a zero-length allocation.
template <int WM, int WN, int TM, int TN, int KKB, int ST = 3>
static inline int ptxSmem() {
    constexpr int kStages = ST;
    constexpr int blockM = WM * TM * 16, blockN = WN * TN * 16;
    constexpr int smem = kStages * (blockM * (KKB + 16) + blockN * (KKB + 16));
    cudaFuncSetAttribute(om::pearl::noisyGemmPtx<WM, WN, TM, TN, KKB, ST>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    return smem;
}

using namespace om::pearl;

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main(int argc, char **argv) {
    const int m = argc > 1 ? atoi(argv[1]) : 1024;
    const int n = argc > 2 ? atoi(argv[2]) : 1024;
    const int k = argc > 3 ? atoi(argv[3]) : 2048;
    const int rank = argc > 4 ? atoi(argv[4]) : 128;

    const size_t aB = (size_t)m * k, bB = (size_t)k * n;
    const size_t tiles = (size_t)(m / 16) * (n / 16);

    int8_t *dA, *dBk, *dBn;
    uint32_t *t1, *t2;
    CK(cudaMalloc(&dA, aB));
    CK(cudaMalloc(&dBk, bB));
    CK(cudaMalloc(&dBn, bB));
    CK(cudaMalloc(&t1, tiles * 16 * 4));
    CK(cudaMalloc(&t2, tiles * 16 * 4));

    genMatrix<<<(aB / 8 + 255) / 256, 256>>>(dA, aB, 11);
    genMatrix<<<(bB / 8 + 255) / 256, 256>>>(dBk, bB, 22);
    CK(cudaDeviceSynchronize());

    // n-major copy of the same bytes, so both kernels see identical data.
    {
        int8_t *hk = (int8_t *)malloc(bB), *hn = (int8_t *)malloc(bB);
        CK(cudaMemcpy(hk, dBk, bB, cudaMemcpyDeviceToHost));
        for (int j = 0; j < n; j++)
            for (int p = 0; p < k; p++) hn[(size_t)j * k + p] = hk[(size_t)p * n + j];
        CK(cudaMemcpy(dBn, hn, bB, cudaMemcpyHostToDevice));
        free(hk);
        free(hn);
    }

    CK(cudaMemset(t1, 0, tiles * 16 * 4));
    CK(cudaMemset(t2, 0, tiles * 16 * 4));

    // Oracle: dbuf 2x4/4x2, block 128x128.
    dim3 g1(n / 128, m / 128);
    noisyGemmMmaTiledDB<2, 4, 4, 2><<<g1, 256>>>(dA, dBk, nullptr, t1, m, n, k, rank, false);
    CK(cudaDeviceSynchronize());

    // Under test: selected by argv[5] - "ptx" (default) or "async".
    const char *which = argc > 5 ? argv[5] : "ptx";
    const int kkb = argc > 6 ? atoi(argv[6]) : 64;
    if (which[0] == 'a') {
        dim3 g2(n / 256, m / 128);
        noisyGemmMmaTiledAsync<2, 4, 4, 4><<<g2, 256>>>(dA, dBk, nullptr, t2, m, n, k, rank, false);
    } else {
        dim3 g2(n / 128, m / 256);
        noisyGemmPtx<4, 4, 4, 2, 32><<<g2, 512, ptxSmem<4, 4, 4, 2, 32>()>>>(dA, dBn, nullptr, t2, m, n, k, rank, false);
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    uint32_t *h1 = (uint32_t *)malloc(tiles * 16 * 4);
    uint32_t *h2 = (uint32_t *)malloc(tiles * 16 * 4);
    CK(cudaMemcpy(h1, t1, tiles * 16 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2, t2, tiles * 16 * 4, cudaMemcpyDeviceToHost));

    size_t badTiles = 0, badWords = 0, firstBad = (size_t)-1;
    int laneHist[16] = {};
    for (size_t t = 0; t < tiles; t++) {
        bool bad = false;
        for (int w = 0; w < 16; w++)
            if (h1[t * 16 + w] != h2[t * 16 + w]) {
                bad = true;
                badWords++;
                laneHist[w]++;
            }
        if (bad) {
            badTiles++;
            if (firstBad == (size_t)-1) firstBad = t;
        }
    }
    printf("  %-5s %dx%d k=%d rank=%d: %s (%zu/%zu tiles differ)\n", which, m, n,
           k, rank, badTiles ? "FAIL" : "ok", badTiles, tiles);
    if (badTiles) {
        printf("    per-word (lane) mismatch counts: ");
        for (int w = 0; w < 16; w++) printf("%d ", laneHist[w]);
        printf("\n");
    }
    if (firstBad != (size_t)-1) {
        printf("  first bad tile %zu:\n    dbuf:", firstBad);
        for (int w = 0; w < 16; w++) printf(" %08x", h1[firstBad * 16 + w]);
        printf("\n    ptx :");
        for (int w = 0; w < 16; w++) printf(" %08x", h2[firstBad * 16 + w]);
        printf("\n");
    }
    return badTiles != 0;
}

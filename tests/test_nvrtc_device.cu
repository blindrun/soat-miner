// Does the runtime-compiled kernel compute the same hash as the interpreted one?
//
// This is the check the whole runtime-compilation path rests on. Everything
// else - that it compiles, that the symbols resolve, that it is faster - is
// worthless if the answer here is no, because a miner that hashes wrong finds
// nothing and reports no error. It would look exactly like a working miner
// with unlucky variance.
//
// # Why kpHashSpec and not kpSearchSpec
//
// The search kernels only write nonces that come in under target, so comparing
// them compares two lists that are usually both empty. Empty equals empty, and
// the test passes while establishing nothing. kpHashSpec and kpHashBatch write
// EVERY nonce's result, so every nonce in the range is a comparison and there
// is no way for the test to pass by finding nothing.
//
// # Why the synthetic DAG is random and not zeros
//
// The mix indexes the DAG by a value derived from the mix state. If the DAG
// were zero-filled every element would be identical, so a kernel reading the
// WRONG index would read the same bytes as one reading the right index and the
// comparison would pass. Random contents make a misindex produce different
// output, which is the entire point. Same reason a fixture of all-identical
// rows proves nothing about a query's WHERE clause.
//
// A synthetic DAG is enough here because both kernels read the same buffer;
// agreement against the real algorithm is what tests/test_kawpow_device covers,
// on a real DAG against the host reference.

#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "algos/kawpow/mine.cuh"
#include "algos/kawpow/nvrtc_jit.h"

using namespace om::kawpow;
using om::progpow::Program;
using om::progpow::Solution32;

static int checks = 0, failures = 0;

static void check(bool ok, const char *what, const char *detail = nullptr) {
    checks++;
    if (ok) {
        printf("  ok    %s\n", what);
    } else {
        failures++;
        printf("  FAIL  %s\n", what);
        if (detail) printf("        %s\n", detail);
    }
}

int main(int argc, char **argv) {
    const int height = argc > 1 ? atoi(argv[1]) : 4501000;
    // Self-check mode: compile the generated kernel for a DIFFERENT period and
    // require the comparison to FAIL. An all-nonces-match result is only
    // evidence if a mismatch would have been caught, and a comparison nobody
    // has seen go red is indistinguishable from one that cannot. This is the
    // cheapest possible wrong kernel that is still a valid kernel.
    const bool mutate = argc > 2 && strcmp(argv[2], "--mutate") == 0;
    printf("nvrtc device: generated kernel vs interpreted kernel\n");

    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        printf("  SKIP  no CUDA device\n");
        return 0;
    }
    printf("  device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    NvrtcJit jit;
    if (!jit.available()) {
        // Not a failure: the whole point of dlopen'ing is that a machine
        // without the toolkit still mines. But say which it was, because
        // "SKIP" with no reason is how a permanently-skipping gate hides.
        printf("  SKIP  %s\n", jit.unavailableReason().c_str());
        return 0;
    }

    const Program prog = om::progpow::buildProgram(height, params());
    const Program jitProg = om::progpow::buildProgram(
        mutate ? height + kPeriodLength * 7 : height, params());
    if (mutate) printf("  MUTATED: generated kernel built for another period\n");
    if (!jit.build(jitProg, prop.major, prop.minor)) {
        check(false, "the specialised kernel compiles on this device");
        printf("%d check(s), %d failure(s)\n", checks, failures);
        return 1;
    }
    check(jit.loaded(), "the specialised kernel compiles and loads");

    // A small synthetic DAG: big enough that item indices spread, small enough
    // to allocate anywhere. RANDOM, for the reason in the file header.
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
        printf("  SKIP  could not allocate the test DAG\n");
        return 0;
    }
    cudaMemcpy(dDag, hostDag.data(), dagWords * 4, cudaMemcpyHostToDevice);

    const uint64_t count = 4096;
    Solution32 *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, sizeof(Solution32) * count);
    cudaMalloc(&dB, sizeof(Solution32) * count);
    cudaMemset(dA, 0, sizeof(Solution32) * count);
    cudaMemset(dB, 0, sizeof(Solution32) * count);

    const uint32_t hdr[8] = {0x9e3779b9u, 0x243f6a88u, 0x85a308d3u, 0x13198a2eu,
                             0x03707344u, 0xa4093822u, 0x299f31d0u, 0x082efa98u};
    const uint64_t nonceBase = 0xdeadbeefu;

    const int threads = 512;
    const int groups = threads / (int)om::progpow::kNumLanes;
    const unsigned shared =
        (om::progpow::kL1CacheWords + groups * om::progpow::kNumLanes) * 4;
    const unsigned blocks = (unsigned)((count + groups - 1) / groups);

    // Interpreted, via the ordinary runtime API.
    kpHashBatch<<<blocks, threads, shared>>>(dDag, numItems2048, prog, hdr[0],
                                             hdr[1], hdr[2], hdr[3], hdr[4],
                                             hdr[5], hdr[6], hdr[7], nonceBase,
                                             count, dA);
    check(cudaDeviceSynchronize() == cudaSuccess, "the interpreted kernel runs",
          cudaGetErrorString(cudaGetLastError()));

    // Generated, via the driver API. Same arguments minus the Program, which is
    // compiled in.
    uint32_t nItems = numItems2048;
    uint64_t nb = nonceBase, cnt = count;
    uint32_t h[8];
    memcpy(h, hdr, sizeof(h));
    void *args[] = {&dDag, &nItems, &h[0], &h[1], &h[2], &h[3],
                    &h[4],  &h[5],   &h[6], &h[7], &nb,   &cnt,
                    &dB};
    check(jit.launch(jit.hashKernel(), blocks, threads, shared, nullptr, args),
          "the generated kernel launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "the generated kernel runs",
          cudaGetErrorString(cudaGetLastError()));

    std::vector<Solution32> a(count), b(count);
    cudaMemcpy(a.data(), dA, sizeof(Solution32) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), dB, sizeof(Solution32) * count, cudaMemcpyDeviceToHost);

    // Guard against comparing two buffers of nothing. If the interpreted kernel
    // wrote all zeros the comparison below would pass on a generated kernel
    // that also did nothing - the same empty-equals-empty trap the search
    // kernels have, one level down.
    bool nonTrivial = false;
    for (uint64_t i = 0; i < count && !nonTrivial; i++)
        for (int w = 0; w < 8; w++)
            if (a[i].mix[w] != 0) { nonTrivial = true; break; }
    check(nonTrivial, "the interpreted kernel produced real output");

    uint64_t mismatches = 0;
    uint64_t firstBad = 0;
    for (uint64_t i = 0; i < count; i++) {
        bool same = a[i].nonce == b[i].nonce;
        for (int w = 0; w < 8 && same; w++)
            same = a[i].mix[w] == b[i].mix[w] && a[i].hit[w] == b[i].hit[w];
        if (!same) {
            if (mismatches == 0) firstBad = i;
            mismatches++;
        }
    }
    char detail[160];
    snprintf(detail, sizeof(detail),
             "%llu of %llu nonces differ; first at index %llu",
             (unsigned long long)mismatches, (unsigned long long)count,
             (unsigned long long)firstBad);
    if (mutate) {
        // Inverted on purpose: here a mismatch is the pass.
        check(mismatches > 0,
              "a kernel built for the wrong period IS caught", detail);
    } else {
        check(mismatches == 0,
              "every nonce hashes identically, generated vs interpreted",
              detail);
    }

    cudaFree(dDag); cudaFree(dA); cudaFree(dB);
    printf("%d check(s), %d failure(s)\n", checks, failures);
    if (checks == 0) { printf("FAIL: no checks ran\n"); return 1; }
    return failures == 0 ? 0 : 1;
}

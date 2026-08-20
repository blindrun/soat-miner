// KawPow: the CUDA kernels against the host reference.
//
// The host reference is already pinned to Ravencoin's published vectors and to
// a real mainnet block, so this test does not re-argue the algorithm. It asks
// three questions the host tests cannot:
//
//   1. Does the device DAG match what the host computes from the light cache?
//      Checked at pseudo-random indices spread across the whole dataset, not
//      the first few, because a DAG build that is wrong only past the first
//      launch slice would pass a check of item 0.
//   2. Does the device hash match the host, nonce by nonce? A kernel that is
//      wrong for one input in a million looks perfect against a single vector.
//   3. Does the search kernel actually report a winner under a real target,
//      with the mix hash a pool will check?
//
// Question 3 is the one that catches a miner that hashes correctly and never
// submits, which is a failure mode the first two are blind to.
//
//   ./tests/test_kawpow_device <headerHex> <height> <nonceHex> <mixHex>
//                              <finalHex> <bitsHex> [nonces]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "algos/kawpow/mine.cuh"

using namespace om::kawpow;

namespace {

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hexToWords(const char *hex, uint32_t *out, int words) {
    if (strlen(hex) != (size_t)words * 8) return false;
    for (int i = 0; i < words * 4; i++) {
        const int hi = hexVal(hex[i * 2]), lo = hexVal(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        if (i % 4 == 0) out[i / 4] = 0;
        out[i / 4] |= (uint32_t)((hi << 4) | lo) << ((i % 4) * 8);
    }
    return true;
}

std::string wordsToHex(const uint32_t *w, int words) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < words; i++)
        for (int b = 0; b < 4; b++) {
            const uint8_t v = (uint8_t)(w[i] >> (b * 8));
            s += d[v >> 4];
            s += d[v & 15];
        }
    return s;
}

void bitsToTarget(uint32_t bits, uint64_t target[4]) {
    const uint32_t exponent = bits >> 24;
    const uint64_t mantissa = bits & 0x007fffffu;
    for (int i = 0; i < 4; i++) target[i] = 0;
    if (exponent <= 3) {
        target[0] = mantissa >> (8 * (3 - exponent));
        return;
    }
    const int shift = 8 * ((int)exponent - 3);
    for (int b = 0; b < 3; b++) {
        const int bitPos = shift + 8 * b;
        if (bitPos >= 256) continue;
        target[bitPos / 64] |= ((mantissa >> (8 * b)) & 0xffu) << (bitPos % 64);
    }
}

bool cudaOk(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    fprintf(stderr, "  FAIL  %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: test_kawpow_device <headerHex> <height> <nonceHex> "
                "<mixHex> <finalHex> <bitsHex> [nonces]\n");
        return 2;
    }
    uint32_t header[8], wantMix[8], wantFinal[8];
    if (!hexToWords(argv[1], header, 8) || !hexToWords(argv[4], wantMix, 8) ||
        !hexToWords(argv[5], wantFinal, 8)) {
        fprintf(stderr, "a 64-hex-character argument was malformed\n");
        return 2;
    }
    const int height = atoi(argv[2]);
    const uint64_t nonce = strtoull(argv[3], nullptr, 16);
    const uint32_t bits = (uint32_t)strtoul(argv[6], nullptr, 16);
    const int nonces = argc > 7 ? atoi(argv[7]) : 8;

    const int epoch = epochOf(height);
    const int numCacheItems = lightCacheNumItems(epoch);
    const uint32_t numItems2048 = (uint32_t)(fullDatasetNumItems(epoch) / 2);
    const uint32_t numItems512 = (uint32_t)fullDatasetNumItems(epoch) * 2;
    const uint64_t dagBytes = fullDatasetBytes(epoch);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("  %s, block %d, epoch %d, DAG %.2f GB\n", prop.name, height, epoch,
           dagBytes / 1e9);
    if (dagBytes + 200u * 1024 * 1024 > prop.totalGlobalMem) {
        printf("  SKIP  %.1f GB of VRAM cannot hold a %.2f GB DAG\n",
               prop.totalGlobalMem / 1e9, dagBytes / 1e9);
        return 0;
    }

    // --- host side -------------------------------------------------------
    uint32_t seed[8];
    epochSeed(epoch, seed);
    std::vector<uint32_t> cache((size_t)numCacheItems * 16, 0);
    buildLightCache(cache.data(), numCacheItems, seed);

    std::vector<uint32_t> l1(kL1CacheWords, 0);
    for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
        datasetItem2048(cache.data(), numCacheItems, i, l1.data() + i * 64);

    // --- device DAG ------------------------------------------------------
    uint32_t *dCache = nullptr, *dDag = nullptr;
    cudaMalloc(&dCache, cache.size() * 4);
    cudaMalloc(&dDag, dagBytes);
    if (!cudaOk("allocating the DAG")) return 1;
    cudaMemcpy(dCache, cache.data(), cache.size() * 4, cudaMemcpyHostToDevice);

    // Sliced so no single launch owns the card for tens of seconds.
    const uint32_t sliceItems = 1u << 21;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (uint32_t start = 0; start < numItems512; start += sliceItems) {
        const uint32_t n =
            (numItems512 - start) < sliceItems ? (numItems512 - start) : sliceItems;
        om::progpow::ppBuildDag<<<(n + 255) / 256, 256>>>(
            dCache, numCacheItems, start, numItems512, dDag);
    }
    cudaEventRecord(t1);
    if (cudaDeviceSynchronize() != cudaSuccess || !cudaOk("building the DAG"))
        return 1;
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("  DAG built on device in %.1f s (%.2f GB/s written)\n", ms / 1000.0,
           dagBytes / 1e9 / (ms / 1000.0));

    int bad = 0;

    // --- 1. device DAG vs host dataset items -----------------------------
    // Spread across the whole dataset, including the last item, so a build
    // that stopped early or repeated a slice cannot pass.
    const uint32_t probes[] = {0u,
                               1u,
                               kL1CacheBytes / 256 - 1,
                               numItems2048 / 7,
                               numItems2048 / 3,
                               numItems2048 / 2,
                               numItems2048 - 12345,
                               numItems2048 - 1};
    for (uint32_t idx : probes) {
        uint32_t hostItem[64], devItem[64];
        datasetItem2048(cache.data(), numCacheItems, idx, hostItem);
        cudaMemcpy(devItem, dDag + (size_t)idx * 64, sizeof(devItem),
                   cudaMemcpyDeviceToHost);
        if (memcmp(hostItem, devItem, sizeof(devItem)) != 0) {
            printf("  FAIL  DAG item %u differs from the host\n", idx);
            printf("        host %s\n        dev  %s\n",
                   wordsToHex(hostItem, 8).c_str(), wordsToHex(devItem, 8).c_str());
            bad++;
        }
    }
    if (bad == 0)
        printf("  ok    %zu DAG items across the whole dataset match the host\n",
               sizeof(probes) / sizeof(probes[0]));

    // --- 2. device hash vs host reference, nonce by nonce -----------------
    const om::progpow::Program prog = om::progpow::buildProgram(height, params());
    const int threads = 256;                       // 16 hashes per block
    const int groups = threads / (int)kNumLanes;
    const size_t sharedBytes = (kL1CacheWords + groups * kNumLanes) * 4;

    om::progpow::Solution32 *dOut = nullptr;
    cudaMalloc(&dOut, sizeof(om::progpow::Solution32) * (size_t)nonces);
    const int blocks = (nonces + groups - 1) / groups;
    // Start a few before the winning nonce so the batch straddles it.
    const uint64_t base = nonce > 3 ? nonce - 3 : 0;
    kpHashBatch<<<blocks, threads, sharedBytes>>>(
        dDag, numItems2048, prog, header[0], header[1], header[2], header[3],
        header[4], header[5], header[6], header[7], base, (uint64_t)nonces, dOut);
    if (cudaDeviceSynchronize() != cudaSuccess || !cudaOk("hashing a batch"))
        return 1;

    std::vector<om::progpow::Solution32> got((size_t)nonces);
    cudaMemcpy(got.data(), dOut, sizeof(om::progpow::Solution32) * nonces,
               cudaMemcpyDeviceToHost);

    int mismatched = 0;
    for (int i = 0; i < nonces; i++) {
        uint32_t hFinal[8], hMix[8];
        hashLight(cache.data(), numCacheItems, l1.data(), numItems2048, height,
                  header, base + i, hFinal, hMix);
        if (memcmp(hFinal, got[i].hit, 32) != 0 ||
            memcmp(hMix, got[i].mix, 32) != 0) {
            if (mismatched < 3)
                printf("  FAIL  nonce %016llx\n        host %s\n        dev  %s\n",
                       (unsigned long long)(base + i), wordsToHex(hFinal, 8).c_str(),
                       wordsToHex(got[i].hit, 8).c_str());
            mismatched++;
        }
    }
    if (mismatched) {
        printf("  FAIL  %d of %d nonces differ from the host reference\n",
               mismatched, nonces);
        bad++;
    } else {
        printf("  ok    %d consecutive nonces match the host reference exactly\n",
               nonces);
    }

    // --- 3. the search kernel actually reports the winner ------------------
    uint64_t target[4];
    bitsToTarget(bits, target);
    om::progpow::Solution32 *dSol = nullptr;
    uint32_t *dCount = nullptr;
    cudaMalloc(&dSol, sizeof(om::progpow::Solution32) * 16);
    cudaMalloc(&dCount, sizeof(uint32_t));
    cudaMemset(dCount, 0, sizeof(uint32_t));
    kpSearch<<<blocks, threads, sharedBytes>>>(
        dDag, numItems2048, prog, header[0], header[1], header[2], header[3],
        header[4], header[5], header[6], header[7], base, (uint64_t)nonces,
        target[0], target[1], target[2], target[3], dSol, dCount, 16);
    if (cudaDeviceSynchronize() != cudaSuccess || !cudaOk("searching")) return 1;

    uint32_t found = 0;
    cudaMemcpy(&found, dCount, sizeof(found), cudaMemcpyDeviceToHost);
    if (found == 0) {
        printf("  FAIL  the search kernel found nothing, though nonce %016llx "
               "is under this block's own target\n",
               (unsigned long long)nonce);
        bad++;
    } else {
        std::vector<om::progpow::Solution32> sols(found > 16 ? 16 : found);
        cudaMemcpy(sols.data(), dSol,
                   sizeof(om::progpow::Solution32) * sols.size(),
                   cudaMemcpyDeviceToHost);
        bool matched = false;
        for (const auto &s : sols)
            if (s.nonce == nonce && memcmp(s.hit, wantFinal, 32) == 0 &&
                memcmp(s.mix, wantMix, 32) == 0)
                matched = true;
        if (!matched) {
            printf("  FAIL  the search found %u solution(s), none of them the "
                   "block's own nonce with its own mix\n", found);
            bad++;
        } else {
            printf("  ok    search reported nonce %016llx with the block's mix "
                   "hash\n", (unsigned long long)nonce);
        }
    }

    cudaFree(dSol);
    cudaFree(dCount);
    cudaFree(dOut);
    cudaFree(dDag);
    cudaFree(dCache);
    return bad == 0 ? 0 : 1;
}

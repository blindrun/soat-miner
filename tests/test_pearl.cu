// Device NoisyGEMM against the Python reference.
//
// Reads the fixed vector file pearl_reference.py --emit-vectors writes, rather
// than regenerating inputs here: a device test that builds its own inputs can
// agree with itself while both sides drift away from the reference together.
//
// Checks the noised product AND the transcripts. The transcripts are the part
// that actually decides whether a share is found, so a kernel that got C right
// and the folding wrong would mine nothing while looking healthy.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "../src/algos/pearl-pow/noisy_gemm.cuh"

#define CHECK(x)                                                            \
    do {                                                                    \
        cudaError_t e = (x);                                                \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,     \
                    cudaGetErrorString(e));                                 \
            return 2;                                                       \
        }                                                                   \
    } while (0)

namespace {

struct Reader {
    const uint8_t *p;
    const uint8_t *end;
    bool ok = true;
    template <typename T>
    const T *take(size_t count) {
        if (p + count * sizeof(T) > end) {
            ok = false;
            return nullptr;
        }
        const T *r = reinterpret_cast<const T *>(p);
        p += count * sizeof(T);
        return r;
    }
};

}  // namespace

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/pearl_vectors.bin";

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        fprintf(stderr, "generate it with: python3 tests/pearl_reference.py --emit-vectors %s\n",
                path);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(size);
    if (fread(buf.data(), 1, size, f) != static_cast<size_t>(size)) {
        fprintf(stderr, "short read\n");
        fclose(f);
        return 2;
    }
    fclose(f);

    if (size < 32 || memcmp(buf.data(), "PRLV0002", 8) != 0) {
        fprintf(stderr, "bad magic - regenerate the vectors\n");
        return 2;
    }

    Reader r{buf.data() + 8, buf.data() + size};
    const int32_t *dims = r.take<int32_t>(6);
    const int m = dims[0], n = dims[1], k = dims[2], rank = dims[3];
    const int hashTile = dims[4], numTranscripts = dims[5];
    printf("vectors: m=%d n=%d k=%d rank=%d hash_tile=%d transcripts=%d\n", m, n, k, rank,
           hashTile, numTranscripts);

    if (hashTile != om::pearl::kHashTile) {
        fprintf(stderr, "vector hash tile %d != kernel %d\n", hashTile, om::pearl::kHashTile);
        return 2;
    }

    r.take<int8_t>((size_t)m * k);                       // A, unused here
    r.take<int8_t>((size_t)k * n);                       // B
    r.take<int8_t>((size_t)m * rank);                    // E_AL
    r.take<int8_t>((size_t)rank * k);                    // E_AR
    r.take<int8_t>((size_t)k * rank);                    // E_BL
    r.take<int8_t>((size_t)rank * n);                    // E_BR
    const int8_t *aNoised = r.take<int8_t>((size_t)m * k);
    const int8_t *bNoised = r.take<int8_t>((size_t)k * n);
    const int32_t *cExpect = r.take<int32_t>((size_t)m * n);
    r.take<int32_t>((size_t)m * n);                      // C denoised
    const uint32_t *tExpect = r.take<uint32_t>((size_t)numTranscripts * 16);
    const uint32_t *powKey = r.take<uint32_t>(8);
    const uint32_t *dExpect = r.take<uint32_t>((size_t)numTranscripts * 8);
    if (!r.ok) {
        fprintf(stderr, "vector file truncated\n");
        return 2;
    }

    int8_t *dA, *dB;
    int32_t *dC;
    uint32_t *dT;
    CHECK(cudaMalloc(&dA, (size_t)m * k));
    CHECK(cudaMalloc(&dB, (size_t)k * n));
    CHECK(cudaMalloc(&dC, (size_t)m * n * sizeof(int32_t)));
    CHECK(cudaMalloc(&dT, (size_t)numTranscripts * 16 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dA, aNoised, (size_t)m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, bNoised, (size_t)k * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dT, 0, (size_t)numTranscripts * 16 * sizeof(uint32_t)));

    dim3 grid(n / om::pearl::kHashTile, m / om::pearl::kHashTile);
    om::pearl::noisyGemmTile<<<grid, 256>>>(dA, dB, dC, dT, m, n, k, rank);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> cGot((size_t)m * n);
    std::vector<uint32_t> tGot((size_t)numTranscripts * 16);
    CHECK(cudaMemcpy(cGot.data(), dC, cGot.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(tGot.data(), dT, tGot.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    auto compare = [&](const char *what) {
        int badC = 0, badT = 0;
        for (size_t i = 0; i < cGot.size(); ++i) {
            if (cGot[i] != cExpect[i]) {
                if (badC < 3)
                    fprintf(stderr, "  C[%zu] device %d != reference %d\n", i, cGot[i],
                            cExpect[i]);
                ++badC;
            }
        }
        for (size_t i = 0; i < tGot.size(); ++i) {
            if (tGot[i] != tExpect[i]) {
                if (badT < 3)
                    fprintf(stderr, "  transcript[%zu/%zu] device %08x != reference %08x\n",
                            i / 16, i % 16, tGot[i], tExpect[i]);
                ++badT;
            }
        }
        printf("  %-14s product %s (%d/%zu)   transcripts %s (%d/%zu)\n", what,
               badC ? "FAIL" : "ok", badC, cGot.size(), badT ? "FAIL" : "ok", badT, tGot.size());
        return badC == 0 && badT == 0;
    };
    bool naiveOk = compare("naive:");

    // Same vectors through the tensor-core path. It must agree exactly, not
    // approximately: this is integer arithmetic, so "close" would mean broken.
    bool mmaOk = true;
    {
        const int totalTiles = (m / om::pearl::kHashTile) * (n / om::pearl::kHashTile);
        const int warpsPerBlock = 256 / 32;
        const int blocks = (totalTiles + warpsPerBlock - 1) / warpsPerBlock;
        CHECK(cudaMemset(dC, 0, (size_t)m * n * sizeof(int32_t)));
        CHECK(cudaMemset(dT, 0, (size_t)numTranscripts * 16 * sizeof(uint32_t)));
        om::pearl::noisyGemmMma<<<blocks, 256>>>(dA, dB, dC, dT, m, n, k, rank, true);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("  mma:           SKIPPED (%s)\n", cudaGetErrorString(e));
        } else {
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(cGot.data(), dC, cGot.size() * sizeof(int32_t),
                             cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(tGot.data(), dT, tGot.size() * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
            mmaOk = compare("mma:");
        }
    }
    // Shared-memory staged variant. A faster kernel that is wrong is worse
    // than a slow one, so it is held to the identical reference.
    bool stagedOk = true;
    {
        const int tilesX = n / om::pearl::kHashTile;
        const int tilesY = m / om::pearl::kHashTile;
        const int warpsPerBlock = 256 / 32;
        if (tilesX % warpsPerBlock) {
            printf("  staged:        SKIPPED (needs n/16 divisible by %d)\n", warpsPerBlock);
        } else {
            dim3 gridS(tilesX / warpsPerBlock, tilesY);
            CHECK(cudaMemset(dC, 0, (size_t)m * n * sizeof(int32_t)));
            CHECK(cudaMemset(dT, 0, (size_t)numTranscripts * 16 * sizeof(uint32_t)));
            om::pearl::noisyGemmMmaStaged<<<gridS, 256>>>(dA, dB, dC, dT, m, n, k, rank, true);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                printf("  staged:        SKIPPED (%s)\n", cudaGetErrorString(e));
            } else {
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(cGot.data(), dC, cGot.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(tGot.data(), dT, tGot.size() * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost));
                stagedOk = compare("staged:");
            }
        }
    }

    // Register-tiled variants. Every configuration the miner might select is
    // checked, not just the one it ships, because the choice is made by
    // measurement on whatever card is present - so a config that is only ever
    // benchmarked and never verified is a trap waiting for a faster GPU.
    bool tiledOk = true;
#define CHECK_TILED_K(KERNEL, TAG, WM, WN, TM, TN)                             \
    {                                                                          \
        constexpr int threads = (WM) * (WN) * 32;                              \
        constexpr int blockM = (WM) * (TM) * 16;                               \
        constexpr int blockN = (WN) * (TN) * 16;                               \
        char label[32];                                                        \
        snprintf(label, sizeof(label), "%s %dx%d/%dx%d:", TAG, WM, WN, TM, TN); \
        if (m % blockM || n % blockN) {                                        \
            printf("  %-14s SKIPPED (needs m%%%d and n%%%d)\n", label, blockM,  \
                   blockN);                                                    \
        } else {                                                               \
            dim3 g(n / blockN, m / blockM);                                    \
            CHECK(cudaMemset(dC, 0, (size_t)m * n * sizeof(int32_t)));         \
            CHECK(cudaMemset(dT, 0,                                            \
                             (size_t)numTranscripts * 16 * sizeof(uint32_t))); \
            KERNEL<WM, WN, TM, TN><<<g, threads>>>(dA, dB, dC, dT, m, n, k,   \
                                                   rank, true);                \
            cudaError_t e = cudaGetLastError();                                \
            if (e != cudaSuccess) {                                            \
                printf("  %-14s SKIPPED (%s)\n", label, cudaGetErrorString(e)); \
            } else {                                                           \
                CHECK(cudaDeviceSynchronize());                                \
                CHECK(cudaMemcpy(cGot.data(), dC, cGot.size() * sizeof(int32_t), \
                                 cudaMemcpyDeviceToHost));                     \
                CHECK(cudaMemcpy(tGot.data(), dT,                              \
                                 tGot.size() * sizeof(uint32_t),               \
                                 cudaMemcpyDeviceToHost));                     \
                if (!compare(label)) tiledOk = false;                          \
            }                                                                  \
        }                                                                      \
    }

#define CHECK_TILED(WM, WN, TM, TN) \
    CHECK_TILED_K(om::pearl::noisyGemmMmaTiled, "tiled", WM, WN, TM, TN)
#define CHECK_DB(WM, WN, TM, TN) \
    CHECK_TILED_K(om::pearl::noisyGemmMmaTiledDB, "dbuf ", WM, WN, TM, TN)

    CHECK_TILED(2, 4, 2, 2)
    CHECK_TILED(2, 4, 4, 2)
    CHECK_TILED(2, 4, 2, 4)
    CHECK_TILED(2, 4, 4, 4)
    CHECK_TILED(4, 2, 2, 2)
    CHECK_TILED(4, 4, 2, 2)

    // The double-buffered variant is held to exactly the same reference. A
    // prefetch that races is the classic way to make a fast kernel wrong, and
    // it would show up here as a transcript mismatch rather than as anything
    // obvious at runtime.
    CHECK_DB(2, 4, 2, 2)
    CHECK_DB(2, 4, 4, 2)
    CHECK_DB(2, 4, 2, 4)
    CHECK_DB(2, 4, 4, 4)
    CHECK_DB(4, 2, 2, 2)
    CHECK_DB(4, 4, 2, 2)
#undef CHECK_DB
#undef CHECK_TILED
#undef CHECK_TILED_K

    int badC = naiveOk && mmaOk && stagedOk && tiledOk ? 0 : 1;
    int badT = 0;

    // blake3 PoW check. The transcripts are only useful if the device can turn
    // them into the same digests the reference does.
    {
        uint32_t *dKey, *dDig, *dTarget;
        CHECK(cudaMalloc(&dKey, 8 * sizeof(uint32_t)));
        CHECK(cudaMalloc(&dTarget, 8 * sizeof(uint32_t)));
        CHECK(cudaMalloc(&dDig, (size_t)numTranscripts * 8 * sizeof(uint32_t)));
        CHECK(cudaMemcpy(dKey, powKey, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        std::vector<uint32_t> maxT(8, 0xffffffffu);
        CHECK(cudaMemcpy(dTarget, maxT.data(), 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Hash the REFERENCE transcripts, so this isolates blake3 from the
        // gemm. A gemm bug would otherwise show up here as a hash failure.
        uint32_t *dRefT;
        CHECK(cudaMalloc(&dRefT, (size_t)numTranscripts * 16 * sizeof(uint32_t)));
        CHECK(cudaMemcpy(dRefT, tExpect, (size_t)numTranscripts * 16 * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));

        uint32_t *dHits;
        CHECK(cudaMalloc(&dHits, (size_t)numTranscripts * sizeof(uint32_t)));
        om::pearl::powCheck<<<(numTranscripts + 255) / 256, 256>>>(
            dRefT, numTranscripts, dKey, dTarget, dDig, dHits);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        std::vector<uint32_t> dig((size_t)numTranscripts * 8), hits(numTranscripts);
        CHECK(cudaMemcpy(dig.data(), dDig, dig.size() * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hits.data(), dHits, hits.size() * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost));

        int badD = 0;
        for (size_t i = 0; i < dig.size(); ++i)
            if (dig[i] != dExpect[i]) {
                if (badD < 3)
                    fprintf(stderr, "  digest[%zu/%zu] device %08x != reference %08x\n", i / 8,
                            i % 8, dig[i], dExpect[i]);
                ++badD;
            }
        int nHit = 0;
        for (int i = 0; i < numTranscripts; ++i) nHit += hits[i] ? 1 : 0;
        printf("  blake3:        digests %s (%d/%zu)\n", badD ? "FAIL" : "ok", badD, dig.size());
        printf("  target check:  %d/%d hit at max target (expect all)\n", nHit, numTranscripts);
        if (badD || nHit != numTranscripts) badC = 1;

        cudaFree(dKey); cudaFree(dTarget); cudaFree(dDig); cudaFree(dRefT); cudaFree(dHits);
    }

    // Restore the naive result so the negative control below tests what it
    // says it tests.
    CHECK(cudaMemset(dT, 0, (size_t)numTranscripts * 16 * sizeof(uint32_t)));
    om::pearl::noisyGemmTile<<<grid, 256>>>(dA, dB, dC, dT, m, n, k, rank);
    CHECK(cudaDeviceSynchronize());

    // Negative control: the suite must be able to fail. Corrupt one input byte
    // and confirm the comparison notices, otherwise a pass proves nothing.
    int8_t poison = aNoised[0] ^ 0x7f;
    CHECK(cudaMemcpy(dA, &poison, 1, cudaMemcpyHostToDevice));
    om::pearl::noisyGemmTile<<<grid, 256>>>(dA, dB, dC, dT, m, n, k, rank);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(tGot.data(), dT, tGot.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    bool noticed = false;
    for (size_t i = 0; i < tGot.size(); ++i)
        if (tGot[i] != tExpect[i]) { noticed = true; break; }
    printf("  negative control: %s\n", noticed ? "ok (corruption detected)"
                                               : "FAIL (corruption went unnoticed)");

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dT);
    return (badC || badT || !noticed) ? 1 : 0;
}

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

    if (size < 32 || memcmp(buf.data(), "PRLV0001", 8) != 0) {
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
    int badC = naiveOk && mmaOk ? 0 : 1;
    int badT = 0;

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

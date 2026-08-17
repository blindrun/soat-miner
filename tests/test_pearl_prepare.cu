// Pearl's prepare stage on the device, against the host reference.
//
// The kernels here are the part between a nonce and two noised matrices:
// generation, the two Merkle roots, the commitment chain, the noise draws and
// the noising. Everything they produce is checked against job.h, which is
// checked against the Python reference, which is checked against Pearl's own
// Rust. So a pass here reaches the network through three links, each pinned
// down separately.
//
// The Merkle root is the check worth watching. A blake3 tree over 1024-byte
// chunks reduced block-cooperatively in shared memory is easy to get subtly
// wrong - a flag, a counter, a pairing order - and every way of getting it
// wrong produces a perfectly plausible 32 bytes that no node will accept.
//
//   python3 tests/pearl_job.py --emit-vectors /tmp/pearl_job_vectors.bin
//   nvcc -O3 -std=c++17 -Isrc tests/test_pearl_prepare.cu -o tests/test_pearl_prepare
//   ./tests/test_pearl_prepare /tmp/pearl_job_vectors.bin

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/algos/pearl-pow/job.h"
#include "../src/algos/pearl-pow/noisy_gemm.cuh"
#include "../src/algos/pearl-pow/prepare.cuh"

using namespace om::pearl;

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

int gPass = 0, gFail = 0;

void check(const char *name, bool ok, const std::string &detail = std::string()) {
    if (ok) {
        gPass++;
        printf("  ok   %s\n", name);
    } else {
        gFail++;
        printf("  FAIL %s %s\n", name, detail.c_str());
    }
}

std::string hex(const uint8_t *p, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

std::vector<int8_t> synthMatrix(size_t rows, size_t cols, int64_t salt) {
    std::vector<int8_t> m(rows * cols);
    for (size_t i = 0; i < m.size(); i++)
        m[i] = (int8_t)((int64_t)(((int64_t)i * 37 + salt) & 0x7F) - 64);
    return m;
}

struct Reader {
    const uint8_t *p;
    const uint8_t *end;
    bool ok = true;
    const uint8_t *take(size_t n) {
        if (p + n > end) { ok = false; return nullptr; }
        const uint8_t *r = p;
        p += n;
        return r;
    }
};

/**
 * Drive the tree reduction: chunk CVs, then as many levels per launch as a
 * block can hold, then the root.
 *
 * The chunk count has to be a power of two for this to be blake3's own tree -
 * see prepare.cuh. `dScratch` needs room for the chunk CVs.
 */
int merkleRoot(const uint8_t *dData, uint32_t chunks, const uint32_t *dKey,
               uint8_t *dScratch, uint8_t *dRoot, cudaStream_t s) {
    chunkCvs<<<(chunks + 255) / 256, 256, 0, s>>>(dData, chunks, dKey, dScratch);

    uint32_t count = chunks;
    const uint8_t *in = dScratch;
    uint8_t *out = dScratch + (size_t)chunks * 32;
    while (count > 2) {
        uint32_t threads = 256;
        while (threads > 1 && 2u * threads > count / 2) threads >>= 1;
        const uint32_t blocks = count / (2 * threads);
        const size_t shared = (size_t)threads * 2 * 8 * sizeof(uint32_t);
        reduceTree<<<blocks, threads, shared, s>>>(in, count, dKey, out);
        count = blocks;
        const uint8_t *next = out;
        out = const_cast<uint8_t *>(in);
        in = next;
    }
    rootCv<<<1, 1, 0, s>>>(in, dKey, dRoot);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pearl_job_vectors.bin>\n", argv[0]);
        return 2;
    }
    FILE *fh = fopen(argv[1], "rb");
    if (!fh) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(fh, 0, SEEK_END);
    const long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    std::vector<uint8_t> blob((size_t)size);
    if (fread(blob.data(), 1, blob.size(), fh) != blob.size()) {
        fprintf(stderr, "short read\n");
        return 2;
    }
    fclose(fh);

    Reader r{blob.data(), blob.data() + blob.size()};
    const uint8_t *magic = r.take(8);
    if (!magic || memcmp(magic, "PRLJ0002", 8) != 0) {
        fprintf(stderr, "not a PRLJ0002 vector file\n");
        return 2;
    }
    const int32_t *dims = (const int32_t *)r.take(16);
    const uint32_t m = (uint32_t)dims[0], n = (uint32_t)dims[1];
    const uint32_t k = (uint32_t)dims[2], rank = (uint32_t)dims[3];

    const uint8_t *header = r.take(76);
    r.take(MiningConfig::kSerializedSize);
    const uint8_t *wantKey = r.take(32);
    const uint8_t *wantARoot = r.take(32);
    const uint8_t *wantBtRoot = r.take(32);
    const uint8_t *wantCommitA = r.take(32);
    const uint8_t *wantCommitB = r.take(32);
    r.take(32);                                       // penalised bound
    const int8_t *wantEAL = (const int8_t *)r.take((size_t)m * rank);
    const int8_t *wantEBR = (const int8_t *)r.take((size_t)rank * n);
    const uint8_t *wantArFirst = r.take((size_t)k * 2);
    const uint8_t *wantArSecond = r.take((size_t)k * 2);
    const uint8_t *wantBlFirst = r.take((size_t)k * 2);
    const uint8_t *wantBlSecond = r.take((size_t)k * 2);
    const uint8_t *wantANoised = r.take(32);
    const uint8_t *wantBNoised = r.take(32);
    if (!r.ok) { fprintf(stderr, "vector file truncated\n"); return 2; }

    printf("vectors: m=%u n=%u k=%u rank=%u\n", m, n, k, rank);
    MiningConfig cfg;
    cfg.commonDim = k;
    cfg.rank = (uint16_t)rank;

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    // ------------------------------------------------------------------
    printf("1. matrix generation agrees with the host\n");
    {
        const size_t count = 8192;
        std::vector<int8_t> want(count), got(count);
        const uint64_t seed = matrixSeed(0x0123456789ABCDEFULL, false);
        fillMatrix(want.data(), count, seed);

        int8_t *d = nullptr;
        CHECK(cudaMalloc(&d, count));
        genMatrix<<<(count / 8 + 255) / 256, 256, 0, s>>>(d, count, seed);
        CHECK(cudaMemcpyAsync(got.data(), d, count, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        check("device and host produce the same bytes", want == got);

        int8_t lo = 127, hi = -128;
        for (size_t i = 0; i < count; i++) {
            if (got[i] < lo) lo = got[i];
            if (got[i] > hi) hi = got[i];
        }
        check("values stay in [-64, 63], leaving room for the noise",
              lo >= -64 && hi <= 63,
              std::to_string(lo) + ".." + std::to_string(hi));

        std::vector<int8_t> other(count);
        fillMatrix(other.data(), count, matrixSeed(0x0123456789ABCDEFULL, true));
        check("A and B draw different streams from one nonce", other != want);
        CHECK(cudaFree(d));
    }

    // ------------------------------------------------------------------
    printf("2. merkle roots\n");
    const std::vector<int8_t> A = synthMatrix(m, k, 11);
    const std::vector<int8_t> Bt = synthMatrix(n, k, 91);
    const size_t aPad = ((A.size() + kChunkLen - 1) / kChunkLen) * kChunkLen;
    const size_t btPad = ((Bt.size() + kChunkLen - 1) / kChunkLen) * kChunkLen;
    const uint32_t aChunks = (uint32_t)(aPad / kChunkLen);
    const uint32_t btChunks = (uint32_t)(btPad / kChunkLen);

    uint8_t *dA = nullptr, *dBt = nullptr, *dScratch = nullptr, *dRoots = nullptr;
    uint32_t *dKey = nullptr;
    CHECK(cudaMalloc(&dA, aPad));
    CHECK(cudaMalloc(&dBt, btPad));
    CHECK(cudaMalloc(&dScratch, (size_t)(aChunks + btChunks) * 32 * 2));
    CHECK(cudaMalloc(&dRoots, 64));
    CHECK(cudaMalloc(&dKey, 32));
    CHECK(cudaMemsetAsync(dA, 0, aPad, s));
    CHECK(cudaMemsetAsync(dBt, 0, btPad, s));
    CHECK(cudaMemcpyAsync(dA, A.data(), A.size(), cudaMemcpyHostToDevice, s));
    CHECK(cudaMemcpyAsync(dBt, Bt.data(), Bt.size(), cudaMemcpyHostToDevice, s));
    CHECK(cudaMemcpyAsync(dKey, wantKey, 32, cudaMemcpyHostToDevice, s));
    {
        check("chunk counts are powers of two, as the reduction requires",
              (aChunks & (aChunks - 1)) == 0 && (btChunks & (btChunks - 1)) == 0,
              std::to_string(aChunks) + " and " + std::to_string(btChunks));

        merkleRoot(dA, aChunks, dKey, dScratch, dRoots, s);
        merkleRoot(dBt, btChunks, dKey, dScratch, dRoots + 32, s);
        uint8_t got[64];
        CHECK(cudaMemcpyAsync(got, dRoots, 64, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        check("A root matches", memcmp(got, wantARoot, 32) == 0,
              hex(got, 32) + " vs " + hex(wantARoot, 32));
        check("B^t root matches", memcmp(got + 32, wantBtRoot, 32) == 0,
              hex(got + 32, 32) + " vs " + hex(wantBtRoot, 32));
    }

    // ------------------------------------------------------------------
    printf("2b. the tree reduction at sizes the vectors do not reach\n");
    {
        // The vectors are 256 chunks, which the driver handles in one reduce
        // launch. Real dimensions are thousands, which needs several - and a
        // reduction that is right for one launch and wrong for two is exactly
        // the bug this catches. Checked against the host hash rather than a
        // fixed vector, which is fair here because that host hash is already
        // pinned to the blake3 library and to Pearl's Rust.
        static const uint32_t kChunkCounts[] = {2, 4, 8, 64, 1024, 4096, 8192};
        for (size_t c = 0; c < sizeof(kChunkCounts) / sizeof(kChunkCounts[0]); c++) {
            const uint32_t chunks = kChunkCounts[c];
            const size_t bytes = (size_t)chunks * kChunkLen;

            uint8_t *d = nullptr, *scratch = nullptr, *root = nullptr;
            CHECK(cudaMalloc(&d, bytes));
            CHECK(cudaMalloc(&scratch, (size_t)chunks * 32 * 2));
            CHECK(cudaMalloc(&root, 32));
            genMatrix<<<(bytes / 8 + 255) / 256, 256, 0, s>>>(
                (int8_t *)d, bytes, matrixSeed(chunks, false));
            merkleRoot(d, chunks, dKey, scratch, root, s);

            std::vector<uint8_t> host(bytes);
            uint8_t got[32], want[32];
            CHECK(cudaMemcpyAsync(host.data(), d, bytes, cudaMemcpyDeviceToHost, s));
            CHECK(cudaMemcpyAsync(got, root, 32, cudaMemcpyDeviceToHost, s));
            CHECK(cudaStreamSynchronize(s));
            b3::hash(wantKey, host.data(), host.size(), want);

            char nm[80];
            snprintf(nm, sizeof(nm), "%u-chunk root (%zu KB) matches the host",
                     chunks, bytes / 1024);
            check(nm, memcmp(got, want, 32) == 0,
                  hex(got, 32) + " vs " + hex(want, 32));
            CHECK(cudaFree(d));
            CHECK(cudaFree(scratch));
            CHECK(cudaFree(root));
        }
    }

    // ------------------------------------------------------------------
    printf("3. the commitment chain\n");
    uint8_t *dCommitA = nullptr, *dCommitB = nullptr, *dJobKey = nullptr;
    uint8_t *dSaltA = nullptr, *dSaltB = nullptr;
    CHECK(cudaMalloc(&dCommitA, 32));
    CHECK(cudaMalloc(&dCommitB, 32));
    CHECK(cudaMalloc(&dJobKey, 32));
    CHECK(cudaMalloc(&dSaltA, 32));
    CHECK(cudaMalloc(&dSaltB, 32));
    {
        uint8_t key[32];
        jobKey(header, cfg, key);
        check("the host derives the vectors' job key", memcmp(key, wantKey, 32) == 0);
        CHECK(cudaMemcpyAsync(dJobKey, key, 32, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(dSaltA, seedSaltA(), 32, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(dSaltB, seedSaltB(), 32, cudaMemcpyHostToDevice, s));

        deriveCommitments<<<1, 1, 0, s>>>(dRoots, dRoots + 32, dJobKey, dSaltA,
                                          dSaltB, m, n, 1, dCommitA, dCommitB);
        uint8_t ca[32], cb[32];
        CHECK(cudaMemcpyAsync(ca, dCommitA, 32, cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(cb, dCommitB, 32, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        check("commitment_A matches", memcmp(ca, wantCommitA, 32) == 0,
              hex(ca, 32) + " vs " + hex(wantCommitA, 32));
        check("commitment_B matches", memcmp(cb, wantCommitB, 32) == 0,
              hex(cb, 32) + " vs " + hex(wantCommitB, 32));

        // The unsalted V1/V2 path has to come out different, or the salted one
        // is not doing anything and a V3 block would be rejected.
        deriveCommitments<<<1, 1, 0, s>>>(dRoots, dRoots + 32, dJobKey, dSaltA,
                                          dSaltB, m, n, 0, dCommitA, dCommitB);
        uint8_t plain[32];
        CHECK(cudaMemcpyAsync(plain, dCommitA, 32, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        check("the unsalted derivation differs", memcmp(plain, ca, 32) != 0);

        // Put the salted values back for the noise below.
        deriveCommitments<<<1, 1, 0, s>>>(dRoots, dRoots + 32, dJobKey, dSaltA,
                                          dSaltB, m, n, 1, dCommitA, dCommitB);
    }

    // ------------------------------------------------------------------
    printf("4. noise draws\n");
    int8_t *dEAL = nullptr, *dEBRflat = nullptr, *dEBR = nullptr;
    uint16_t *dArF = nullptr, *dArS = nullptr, *dBlF = nullptr, *dBlS = nullptr;
    CHECK(cudaMalloc(&dEAL, (size_t)m * rank));
    CHECK(cudaMalloc(&dEBRflat, (size_t)n * rank));
    CHECK(cudaMalloc(&dEBR, (size_t)rank * n));
    CHECK(cudaMalloc(&dArF, (size_t)k * 2));
    CHECK(cudaMalloc(&dArS, (size_t)k * 2));
    CHECK(cudaMalloc(&dBlF, (size_t)k * 2));
    CHECK(cudaMalloc(&dBlS, (size_t)k * 2));
    uint8_t *dSeedA = nullptr, *dSeedB = nullptr;
    CHECK(cudaMalloc(&dSeedA, 32));
    CHECK(cudaMalloc(&dSeedB, 32));
    {
        uint8_t seedA[32], seedB[32];
        seedForA(seedA);
        seedForB(seedB);
        CHECK(cudaMemcpyAsync(dSeedA, seedA, 32, cudaMemcpyHostToDevice, s));
        CHECK(cudaMemcpyAsync(dSeedB, seedB, 32, cudaMemcpyHostToDevice, s));

        const int mask = 128 / 2 - 1, shift = (128 / 2) / 2;
        const uint32_t alCount = m * rank, brCount = n * rank;

        noiseUniform<<<((alCount + 31) / 32 + 255) / 256, 256, 0, s>>>(
            dEAL, alCount, dCommitA, dSeedA, mask, shift);
        noiseUniform<<<((brCount + 31) / 32 + 255) / 256, 256, 0, s>>>(
            dEBRflat, brCount, dCommitB, dSeedB, mask, shift);
        transposeNoiseB<<<(brCount + 255) / 256, 256, 0, s>>>(dEBRflat, dEBR, n,
                                                              (int)rank);
        noisePerm<<<((k + 7) / 8 + 255) / 256, 256, 0, s>>>(dArF, dArS, k, dCommitA,
                                                            dSeedA, (int)rank);
        noisePerm<<<((k + 7) / 8 + 255) / 256, 256, 0, s>>>(dBlF, dBlS, k, dCommitB,
                                                            dSeedB, (int)rank);

        std::vector<int8_t> eal((size_t)m * rank), ebr((size_t)rank * n);
        std::vector<uint16_t> arF(k), arS(k), blF(k), blS(k);
        CHECK(cudaMemcpyAsync(eal.data(), dEAL, eal.size(), cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(ebr.data(), dEBR, ebr.size(), cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(arF.data(), dArF, k * 2, cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(arS.data(), dArS, k * 2, cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(blF.data(), dBlF, k * 2, cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(blS.data(), dBlS, k * 2, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));

        check("E_AL matches", memcmp(eal.data(), wantEAL, eal.size()) == 0);
        check("E_BR matches, transposed to rank x n",
              memcmp(ebr.data(), wantEBR, ebr.size()) == 0);
        check("E_AR column indices match",
              memcmp(arF.data(), wantArFirst, k * 2) == 0 &&
                  memcmp(arS.data(), wantArSecond, k * 2) == 0);
        check("E_BL row indices match",
              memcmp(blF.data(), wantBlFirst, k * 2) == 0 &&
                  memcmp(blS.data(), wantBlSecond, k * 2) == 0);
    }

    // ------------------------------------------------------------------
    printf("5. the noising\n");
    int8_t *dANoised = nullptr, *dBNoised = nullptr;
    CHECK(cudaMalloc(&dANoised, (size_t)m * k));
    CHECK(cudaMalloc(&dBNoised, (size_t)k * n));
    {
        const uint64_t aCount = (uint64_t)m * k, bCount = (uint64_t)k * n;
        applyNoiseA<<<(aCount + 255) / 256, 256, 0, s>>>(
            (const int8_t *)dA, dEAL, dArF, dArS, dANoised, m, k, (int)rank);
        applyNoiseB<<<(bCount + 255) / 256, 256, 0, s>>>(
            (const int8_t *)dBt, dEBR, dBlF, dBlS, dBNoised, n, k);

        std::vector<int8_t> an(aCount), bn(bCount);
        CHECK(cudaMemcpyAsync(an.data(), dANoised, aCount, cudaMemcpyDeviceToHost, s));
        CHECK(cudaMemcpyAsync(bn.data(), dBNoised, bCount, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));

        uint8_t got[32];
        b3::hash(nullptr, (const uint8_t *)an.data(), an.size(), got);
        check("A_noised matches", memcmp(got, wantANoised, 32) == 0,
              hex(got, 32) + " vs " + hex(wantANoised, 32));
        b3::hash(nullptr, (const uint8_t *)bn.data(), bn.size(), got);
        check("B_noised matches, including the transpose into k x n",
              memcmp(got, wantBNoised, 32) == 0,
              hex(got, 32) + " vs " + hex(wantBNoised, 32));
    }

    // ------------------------------------------------------------------
    printf("6. only B is reused between attempts, and it changes the shape\n");
    {
        // Read the derivation and something falls out of it:
        //
        //   commitment_B = blake3(job_key || salted B^t root)
        //   commitment_A = blake3(commitment_B || salted A root)
        //
        // commitment_B does not depend on A at all. E_BL and E_BR are drawn
        // from commitment_B, so if only A varies between attempts then
        // B_noised is IDENTICAL every time - generating B, hashing it and
        // noising it are once per JOB, not once per attempt. Only A's chain
        // and the PoW key move.
        //
        // This is checked rather than asserted: run the whole prepare stage
        // twice with two different A and require B_noised to come out the same
        // and A_noised not to.
        const uint32_t tm = 128, tn = 128, tk = k, tr = rank;
        int8_t *xA = nullptr, *xAn = nullptr, *xBn = nullptr, *xBn2 = nullptr;
        CHECK(cudaMalloc(&xA, (size_t)tm * tk));
        CHECK(cudaMalloc(&xAn, (size_t)tm * tk));
        CHECK(cudaMalloc(&xBn, (size_t)tk * tn));
        CHECK(cudaMalloc(&xBn2, (size_t)tk * tn));

        std::vector<int8_t> firstA, firstB;
        for (int pass = 0; pass < 2; pass++) {
            genMatrix<<<((size_t)tm * tk / 8 + 255) / 256, 256, 0, s>>>(
                xA, (size_t)tm * tk, matrixSeed(pass, false));
            merkleRoot((const uint8_t *)xA, tm * tk / kChunkLen, dKey, dScratch,
                       dRoots, s);
            // B^t is untouched between passes - dBt and its root are whatever
            // section 2 left there.
            deriveCommitments<<<1, 1, 0, s>>>(dRoots, dRoots + 32, dJobKey, dSaltA,
                                              dSaltB, tm, tn, 1, dCommitA, dCommitB);
            noiseUniform<<<((tm * tr + 31) / 32 + 255) / 256, 256, 0, s>>>(
                dEAL, tm * tr, dCommitA, dSeedA, 63, 32);
            noiseUniform<<<((tn * tr + 31) / 32 + 255) / 256, 256, 0, s>>>(
                dEBRflat, tn * tr, dCommitB, dSeedB, 63, 32);
            transposeNoiseB<<<(tn * tr + 255) / 256, 256, 0, s>>>(dEBRflat, dEBR, tn,
                                                                  (int)tr);
            noisePerm<<<((tk + 7) / 8 + 255) / 256, 256, 0, s>>>(dArF, dArS, tk,
                                                                 dCommitA, dSeedA,
                                                                 (int)tr);
            noisePerm<<<((tk + 7) / 8 + 255) / 256, 256, 0, s>>>(dBlF, dBlS, tk,
                                                                 dCommitB, dSeedB,
                                                                 (int)tr);
            applyNoiseA<<<((size_t)tm * tk + 255) / 256, 256, 0, s>>>(
                xA, dEAL, dArF, dArS, xAn, tm, tk, (int)tr);
            applyNoiseB<<<((size_t)tk * tn + 255) / 256, 256, 0, s>>>(
                (const int8_t *)dBt, dEBR, dBlF, dBlS, pass ? xBn2 : xBn, tn, tk);

            std::vector<int8_t> an((size_t)tm * tk), bn((size_t)tk * tn);
            CHECK(cudaMemcpyAsync(an.data(), xAn, an.size(), cudaMemcpyDeviceToHost, s));
            CHECK(cudaMemcpyAsync(bn.data(), pass ? xBn2 : xBn, bn.size(),
                                  cudaMemcpyDeviceToHost, s));
            CHECK(cudaStreamSynchronize(s));
            if (pass == 0) { firstA = an; firstB = bn; }
            else {
                check("a new A changes A_noised", an != firstA);
                check("a new A leaves B_noised UNCHANGED, so B is per-job work",
                      bn == firstB);
            }
        }
        CHECK(cudaFree(xA)); CHECK(cudaFree(xAn));
        CHECK(cudaFree(xBn)); CHECK(cudaFree(xBn2));
    }

    // ------------------------------------------------------------------
    printf("7. what an attempt costs, and which shape to mine at\n");
    {
        // The header of prepare.cuh originally claimed the prepare stage was a
        // few percent. It is not, at a square shape - so this measures rather
        // than asserts, and measures the thing that actually recurs.
        //
        // Section 6 proved B_noised survives a new A, so a per-attempt cost is
        // A's chain only: generate A, hash A, the commitments, A's two noise
        // draws and the noising. B's whole side is amortised over the job.
        //
        // The shape then matters in a way that is not obvious. Hashing A grows
        // as m*k and the GEMM as m*n*k, so the per-attempt overhead falls as
        // 1/n. Holding m*n fixed keeps both the work and the transcript count
        // per attempt identical, and only moves that ratio.
        struct Shape { uint32_t m, n; };
        static const Shape kShapes[] = {
            {4096, 4096},  {1024, 16384}, {2048, 16384},
            {2048, 32768}, {4096, 32768}, {4096, 65536}};
        const uint32_t bk = 2048, br = 128;

        printf("    %-14s %8s %8s %8s %8s  %s\n", "shape", "attempt", "GEMM",
               "over%", "Mcand/s", "TOPS");
        // Rank by candidates per second, not by overhead. They disagree, and
        // overhead is the misleading one: the widest shape has the lowest
        // overhead and is not the fastest, because B stops fitting in L2 and
        // the GEMM itself slows down.
        double bestRate = 0, bestOver = 0;
        uint32_t bestM = 0, bestN = 0;
        for (size_t si = 0; si < sizeof(kShapes) / sizeof(kShapes[0]); si++) {
            const uint32_t bm = kShapes[si].m, bn = kShapes[si].n;
            const size_t aBytes = (size_t)bm * bk, bBytes = (size_t)bn * bk;
            const uint32_t aCh = (uint32_t)(aBytes / kChunkLen);
            const uint32_t bCh = (uint32_t)(bBytes / kChunkLen);
            const uint32_t tiles = (bm / 16) * (bn / 16);

            int8_t *gA, *gBt, *gAn, *gBn, *gEAL, *gEBRf, *gEBR;
            uint16_t *gArF, *gArS, *gBlF, *gBlS;
            uint8_t *gScratch, *gRoots, *gCA, *gCB;
            uint32_t *gTrans;
            CHECK(cudaMalloc(&gA, aBytes));
            CHECK(cudaMalloc(&gBt, bBytes));
            CHECK(cudaMalloc(&gAn, aBytes));
            CHECK(cudaMalloc(&gBn, (size_t)bk * bn));
            CHECK(cudaMalloc(&gEAL, (size_t)bm * br));
            CHECK(cudaMalloc(&gEBRf, (size_t)bn * br));
            CHECK(cudaMalloc(&gEBR, (size_t)br * bn));
            CHECK(cudaMalloc(&gArF, (size_t)bk * 2));
            CHECK(cudaMalloc(&gArS, (size_t)bk * 2));
            CHECK(cudaMalloc(&gBlF, (size_t)bk * 2));
            CHECK(cudaMalloc(&gBlS, (size_t)bk * 2));
            CHECK(cudaMalloc(&gScratch, (size_t)(aCh > bCh ? aCh : bCh) * 32 * 2));
            CHECK(cudaMalloc(&gRoots, 64));
            CHECK(cudaMalloc(&gCA, 32));
            CHECK(cudaMalloc(&gCB, 32));
            CHECK(cudaMalloc(&gTrans, (size_t)tiles * 16 * sizeof(uint32_t)));

            // Per-job work, done once and then left alone. All of it, not
            // just the hashing: leaving B_noised as untouched cudaMalloc'd
            // memory made the GEMM here disagree with section 8 by 8% on the
            // identical kernel, because a GEMM reading pages that were never
            // written is not measuring the same thing as one reading real
            // data. Measure the pipeline that runs, not a piece of it.
            genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, s>>>(gBt, bBytes,
                                                               matrixSeed(7, true));
            merkleRoot((const uint8_t *)gBt, bCh, dKey, gScratch, gRoots + 32, s);
            deriveCommitmentB<<<1, 1, 0, s>>>(gRoots + 32, dJobKey, dSaltB, bn, 1,
                                              gCB);
            noiseUniform<<<((bn * br + 31) / 32 + 255) / 256, 256, 0, s>>>(
                gEBRf, bn * br, gCB, dSeedB, 63, 32);
            transposeNoiseB<<<(bn * br + 255) / 256, 256, 0, s>>>(gEBRf, gEBR, bn,
                                                                  (int)br);
            noisePerm<<<((bk + 7) / 8 + 255) / 256, 256, 0, s>>>(gBlF, gBlS, bk, gCB,
                                                                 dSeedB, (int)br);
            applyNoiseB<<<(((size_t)bk * bn) + 255) / 256, 256, 0, s>>>(
                gBt, gEBR, gBlF, gBlS, gBn, bn, bk);
            CHECK(cudaStreamSynchronize(s));

            cudaEvent_t e0, e1, e2;
            CHECK(cudaEventCreate(&e0));
            CHECK(cudaEventCreate(&e1));
            CHECK(cudaEventCreate(&e2));

            double prepMs = 0, gemmMs = 0;
            const int kReps = 6;
            for (int rep = 0; rep < kReps; rep++) {
                CHECK(cudaEventRecord(e0, s));
                genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, s>>>(
                    gA, aBytes, matrixSeed(rep, false));
                merkleRoot((const uint8_t *)gA, aCh, dKey, gScratch, gRoots, s);
                deriveCommitmentA<<<1, 1, 0, s>>>(gRoots, gCB, dSaltA, bm, 1, gCA);
                noiseUniform<<<((bm * br + 31) / 32 + 255) / 256, 256, 0, s>>>(
                    gEAL, bm * br, gCA, dSeedA, 63, 32);
                noisePerm<<<((bk + 7) / 8 + 255) / 256, 256, 0, s>>>(
                    gArF, gArS, bk, gCA, dSeedA, (int)br);
                applyNoiseA<<<(aBytes + 255) / 256, 256, 0, s>>>(
                    gA, gEAL, gArF, gArS, gAn, bm, bk, (int)br);
                CHECK(cudaEventRecord(e1, s));

                // Only the kernel that ships. An earlier version timed the
                // staged one in the same rep loop and the tiled number came
                // out 8% slower than section 8 measured for the identical
                // kernel - 8.2 ms of heavy work immediately before it evicts
                // L2 and heats the card. Kernels get compared in section 8,
                // one at a time; this section is about the shape.
                dim3 gridT(bn / 128, bm / 128);
                noisyGemmMmaTiled<2, 4, 4, 2><<<gridT, 256, 0, s>>>(
                    gAn, gBn, nullptr, gTrans, (int)bm, (int)bn, (int)bk, (int)br,
                    false);
                CHECK(cudaEventRecord(e2, s));
                CHECK(cudaStreamSynchronize(s));

                float a = 0, b = 0;
                CHECK(cudaEventElapsedTime(&a, e0, e1));
                CHECK(cudaEventElapsedTime(&b, e1, e2));
                if (rep) { prepMs += a; gemmMs += b; }
            }
            prepMs /= (kReps - 1);
            gemmMs /= (kReps - 1);

            const double best = gemmMs;
            const double mac = (double)bm * bn * bk;
            const double over = 100.0 * prepMs / best;
            char shape[32];
            snprintf(shape, sizeof(shape), "%ux%u", bm, bn);
            printf("    %-14s %7.3f %8.3f %7.1f%% %8.1f  %.1f\n", shape, prepMs,
                   gemmMs, over, tiles / ((prepMs + best) * 1e-3) / 1e6,
                   2.0 * mac / (best * 1e-3) / 1e12);
            const double rate = tiles / ((prepMs + best) * 1e-3) / 1e6;
            if (rate > bestRate) {
                bestRate = rate;
                bestOver = over;
                bestM = bm;
                bestN = bn;
            }

            CHECK(cudaFree(gA)); CHECK(cudaFree(gBt)); CHECK(cudaFree(gAn));
            CHECK(cudaFree(gBn)); CHECK(cudaFree(gEAL)); CHECK(cudaFree(gEBRf));
            CHECK(cudaFree(gEBR)); CHECK(cudaFree(gArF)); CHECK(cudaFree(gArS));
            CHECK(cudaFree(gBlF)); CHECK(cudaFree(gBlS)); CHECK(cudaFree(gScratch));
            CHECK(cudaFree(gRoots)); CHECK(cudaFree(gCA)); CHECK(cudaFree(gCB));
            CHECK(cudaFree(gTrans));
            CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
            CHECK(cudaEventDestroy(e2));
        }
        printf("    fastest is %ux%u at %.1f M candidates/s (%.1f%% overhead)\n",
               bestM, bestN, bestRate, bestOver);

        // A gate, not just a number. If the best shape this card can find ever
        // drops below the square baseline, either a kernel or the shape choice
        // has regressed - and the failure mode is a quietly halved hashrate,
        // which nothing else here would notice.
        check("the best shape beats the square one", bestM != 4096 || bestN != 4096,
              std::to_string(bestM) + "x" + std::to_string(bestN));
    }


    // ------------------------------------------------------------------
    printf("8. GEMM tile configurations, at the shape that won\n");
    {
        // The tiled kernel trades arithmetic intensity against register
        // pressure, and where that balance lands is a property of the card
        // rather than of the algorithm - so it is swept, not guessed. Each
        // warp holds kTilesM*kTilesN accumulator fragments of eight int32, so
        // 4x4 is 128 registers of accumulator alone and may well spill.
        const uint32_t bm = 4096, bn = 32768, bk = 2048, br = 128;
        const size_t aBytes = (size_t)bm * bk, bBytes = (size_t)bn * bk;
        const uint32_t tiles = (bm / 16) * (bn / 16);

        int8_t *gAn, *gBn;
        uint32_t *gTrans;
        CHECK(cudaMalloc(&gAn, aBytes));
        CHECK(cudaMalloc(&gBn, bBytes));
        CHECK(cudaMalloc(&gTrans, (size_t)tiles * 16 * sizeof(uint32_t)));
        genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, s>>>((int8_t *)gAn, aBytes,
                                                           matrixSeed(8, false));
        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, s>>>((int8_t *)gBn, bBytes,
                                                           matrixSeed(8, true));

        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a));
        CHECK(cudaEventCreate(&b));

        double bestMs = 1e9;
        char bestName[32] = "";

#define SWEEP(WM, WN, TM, TN)                                                  \
        {                                                                      \
            constexpr int threads = (WM) * (WN) * 32;                          \
            constexpr int blockM = (WM) * (TM) * 16;                           \
            constexpr int blockN = (WN) * (TN) * 16;                           \
            if (bm % blockM == 0 && bn % blockN == 0) {                         \
                dim3 g(bn / blockN, bm / blockM);                              \
                double ms = 0;                                                 \
                for (int rep = 0; rep < 4; rep++) {                            \
                    CHECK(cudaEventRecord(a, s));                              \
                    noisyGemmMmaTiled<WM, WN, TM, TN><<<g, threads, 0, s>>>(   \
                        gAn, gBn, nullptr, gTrans, (int)bm, (int)bn, (int)bk,  \
                        (int)br, false);                                       \
                    CHECK(cudaEventRecord(b, s));                              \
                    CHECK(cudaStreamSynchronize(s));                           \
                    float d = 0;                                               \
                    CHECK(cudaEventElapsedTime(&d, a, b));                     \
                    if (rep) ms += d;                                          \
                }                                                              \
                ms /= 3;                                                       \
                const double mac = (double)bm * bn * bk;                       \
                printf("    warps %dx%d tiles %dx%d  block %3dx%3d  %2d acc  " \
                       "%7.3f ms  %6.1f TOPS\n", WM, WN, TM, TN, blockM,       \
                       blockN, (TM) * (TN), ms, 2.0 * mac / (ms * 1e-3) / 1e12);\
                if (ms < bestMs) {                                             \
                    bestMs = ms;                                               \
                    snprintf(bestName, sizeof(bestName), "%dx%d/%dx%d", WM, WN,\
                             TM, TN);                                          \
                }                                                              \
            }                                                                  \
        }

        {
            // The kernel this replaced, as a baseline - measured here rather
            // than in section 7 so it is timed the same way as everything it
            // is compared against: alone, not chained behind another GEMM.
            const int warps = 8;
            dim3 g((bn / 16 + warps - 1) / warps, bm / 16);
            double ms = 0;
            for (int rep = 0; rep < 4; rep++) {
                CHECK(cudaEventRecord(a, s));
                noisyGemmMmaStaged<<<g, 256, 0, s>>>(gAn, gBn, nullptr, gTrans,
                                                     (int)bm, (int)bn, (int)bk,
                                                     (int)br, false);
                CHECK(cudaEventRecord(b, s));
                CHECK(cudaStreamSynchronize(s));
                float d = 0;
                CHECK(cudaEventElapsedTime(&d, a, b));
                if (rep) ms += d;
            }
            ms /= 3;
            printf("    staged (one tile per warp, B from global)   "
                   "%7.3f ms  %6.1f TOPS\n", ms,
                   2.0 * (double)bm * bn * bk / (ms * 1e-3) / 1e12);
        }

        SWEEP(2, 4, 2, 2)
        SWEEP(4, 2, 2, 2)
        SWEEP(2, 4, 4, 2)
        SWEEP(2, 4, 2, 4)
        SWEEP(2, 4, 4, 4)
        SWEEP(4, 4, 2, 2)
        SWEEP(2, 8, 2, 2)
        SWEEP(4, 4, 2, 4)
#undef SWEEP

        printf("    fastest: %s at %.3f ms (%.1f TOPS), %.1f M candidates/s\n",
               bestName, bestMs,
               2.0 * (double)bm * bn * bk / (bestMs * 1e-3) / 1e12,
               tiles / ((bestMs + 0.22) * 1e-3) / 1e6);

        CHECK(cudaFree(gAn));
        CHECK(cudaFree(gBn));
        CHECK(cudaFree(gTrans));
        CHECK(cudaEventDestroy(a));
        CHECK(cudaEventDestroy(b));
    }

    // ------------------------------------------------------------------
    printf("9. does the kernel win survive the pipeline?\n");
    {
        // Section 8 says 2x4/4x2 beats 2x4/2x2 by 24% with nothing else
        // running. Section 7 says the whole attempt barely moved. Both cannot
        // be the number that matters, so this settles it the way this repo
        // learned to: A and B INTERLEAVED, not one after the other, because
        // sequential runs drift with clocks and cache state.
        //
        // Each rep runs a real attempt - A's chain, then the GEMM - and
        // alternates which kernel finishes it.
        const uint32_t bm = 4096, bn = 32768, bk = 2048, br = 128;
        const size_t aBytes = (size_t)bm * bk, bBytes = (size_t)bn * bk;
        const uint32_t aCh = (uint32_t)(aBytes / kChunkLen);
        const uint32_t bCh = (uint32_t)(bBytes / kChunkLen);
        const uint32_t tiles = (bm / 16) * (bn / 16);

        int8_t *gA, *gBt, *gAn, *gBn, *gEAL, *gEBRf, *gEBR;
        uint16_t *gArF, *gArS, *gBlF, *gBlS;
        uint8_t *gScratch, *gRoots, *gCA, *gCB;
        uint32_t *gTrans;
        CHECK(cudaMalloc(&gA, aBytes));       CHECK(cudaMalloc(&gBt, bBytes));
        CHECK(cudaMalloc(&gAn, aBytes));      CHECK(cudaMalloc(&gBn, bBytes));
        CHECK(cudaMalloc(&gEAL, (size_t)bm * br));
        CHECK(cudaMalloc(&gEBRf, (size_t)bn * br));
        CHECK(cudaMalloc(&gEBR, (size_t)bn * br));
        CHECK(cudaMalloc(&gArF, bk * 2));     CHECK(cudaMalloc(&gArS, bk * 2));
        CHECK(cudaMalloc(&gBlF, bk * 2));     CHECK(cudaMalloc(&gBlS, bk * 2));
        CHECK(cudaMalloc(&gScratch, (size_t)(aCh > bCh ? aCh : bCh) * 32 * 2));
        CHECK(cudaMalloc(&gRoots, 64));       CHECK(cudaMalloc(&gCA, 32));
        CHECK(cudaMalloc(&gCB, 32));
        CHECK(cudaMalloc(&gTrans, (size_t)tiles * 16 * sizeof(uint32_t)));

        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, s>>>(gBt, bBytes,
                                                           matrixSeed(9, true));
        merkleRoot((const uint8_t *)gBt, bCh, dKey, gScratch, gRoots + 32, s);
        deriveCommitmentB<<<1, 1, 0, s>>>(gRoots + 32, dJobKey, dSaltB, bn, 1, gCB);
        noiseUniform<<<((bn * br + 31) / 32 + 255) / 256, 256, 0, s>>>(
            gEBRf, bn * br, gCB, dSeedB, 63, 32);
        transposeNoiseB<<<(bn * br + 255) / 256, 256, 0, s>>>(gEBRf, gEBR, bn, (int)br);
        noisePerm<<<((bk + 7) / 8 + 255) / 256, 256, 0, s>>>(gBlF, gBlS, bk, gCB,
                                                             dSeedB, (int)br);
        applyNoiseB<<<(((size_t)bk * bn) + 255) / 256, 256, 0, s>>>(
            gBt, gEBR, gBlF, gBlS, gBn, bn, bk);
        CHECK(cudaStreamSynchronize(s));

        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a));
        CHECK(cudaEventCreate(&b));
        double ms[2] = {0, 0};
        int reps[2] = {0, 0};
        for (int rep = 0; rep < 12; rep++) {
            const int which = rep & 1;
            genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, s>>>(
                gA, aBytes, matrixSeed(rep, false));
            merkleRoot((const uint8_t *)gA, aCh, dKey, gScratch, gRoots, s);
            deriveCommitmentA<<<1, 1, 0, s>>>(gRoots, gCB, dSaltA, bm, 1, gCA);
            noiseUniform<<<((bm * br + 31) / 32 + 255) / 256, 256, 0, s>>>(
                gEAL, bm * br, gCA, dSeedA, 63, 32);
            noisePerm<<<((bk + 7) / 8 + 255) / 256, 256, 0, s>>>(gArF, gArS, bk, gCA,
                                                                 dSeedA, (int)br);
            applyNoiseA<<<(aBytes + 255) / 256, 256, 0, s>>>(gA, gEAL, gArF, gArS,
                                                             gAn, bm, bk, (int)br);
            CHECK(cudaEventRecord(a, s));
            if (which == 0) {
                dim3 g(bn / 128, bm / 64);
                noisyGemmMmaTiled<2, 4, 2, 2><<<g, 256, 0, s>>>(
                    gAn, gBn, nullptr, gTrans, (int)bm, (int)bn, (int)bk, (int)br,
                    false);
            } else {
                dim3 g(bn / 128, bm / 128);
                noisyGemmMmaTiled<2, 4, 4, 2><<<g, 256, 0, s>>>(
                    gAn, gBn, nullptr, gTrans, (int)bm, (int)bn, (int)bk, (int)br,
                    false);
            }
            CHECK(cudaEventRecord(b, s));
            CHECK(cudaStreamSynchronize(s));
            float d = 0;
            CHECK(cudaEventElapsedTime(&d, a, b));
            if (rep >= 2) { ms[which] += d; reps[which]++; }
        }
        const double mac = (double)bm * bn * bk;
        for (int i = 0; i < 2; i++) ms[i] /= reps[i];
        printf("    2x4/2x2 in a live attempt: %.3f ms  %.1f TOPS\n", ms[0],
               2.0 * mac / (ms[0] * 1e-3) / 1e12);
        printf("    2x4/4x2 in a live attempt: %.3f ms  %.1f TOPS\n", ms[1],
               2.0 * mac / (ms[1] * 1e-3) / 1e12);
        printf("    isolated, section 8 said 5.64 and 4.29 - a 24%% gap\n");
        printf("    here the gap is %.1f%%\n", 100.0 * (ms[0] - ms[1]) / ms[0]);

        check("the faster kernel is still faster inside a real attempt",
              ms[1] < ms[0],
              std::to_string(ms[0]) + " vs " + std::to_string(ms[1]));

        CHECK(cudaFree(gA)); CHECK(cudaFree(gBt)); CHECK(cudaFree(gAn));
        CHECK(cudaFree(gBn)); CHECK(cudaFree(gEAL)); CHECK(cudaFree(gEBRf));
        CHECK(cudaFree(gEBR)); CHECK(cudaFree(gArF)); CHECK(cudaFree(gArS));
        CHECK(cudaFree(gBlF)); CHECK(cudaFree(gBlS)); CHECK(cudaFree(gScratch));
        CHECK(cudaFree(gRoots)); CHECK(cudaFree(gCA)); CHECK(cudaFree(gCB));
        CHECK(cudaFree(gTrans));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    }

    printf("\n%d passed, %d failed\n", gPass, gFail);
    return gFail ? 1 : 0;
}

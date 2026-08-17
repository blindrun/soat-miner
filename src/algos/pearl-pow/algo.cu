// Pearl (pearl-pow) - the Algorithm implementation.
//
// Pearl does not fit the shape of every other PoW in this repo, and it is
// worth saying how before reading the code.
//
// There is no nonce. The miner picks the matrices A (m x k) and B (k x n), and
// their Merkle roots ARE the noise seeds and the PoW key, so choosing them is
// the whole of the search. `nonceBase` here indexes a choice of A rather than
// a value that gets hashed.
//
// One attempt is a whole GEMM and yields (m/16)*(n/16) transcripts, each an
// independent candidate. So an attempt is milliseconds, not nanoseconds, and
// `count` is read as "this many candidates" rather than "this many nonces" -
// which keeps the core's hashrate figure honest, since it is candidates per
// second either way.
//
// The one structural trick, measured rather than assumed (see the header of
// prepare.cuh):
//
//   commitment_B = blake3(job_key || salted B^t root) does not depend on A.
//   E_BL and E_BR come from it, so B_noised is bit-identical for every attempt
//   in a job. Generating, hashing and noising B are per-JOB work. Doing it
//   per attempt costs 48% overhead; doing it once costs 3%.
//
// A win is not a nonce either. It is a Merkle opening of the winning tile's
// sixteen rows of A and sixteen columns of B, tens of kilobytes, which travels
// in Solution::extra.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>

#include "../../core/algo.h"
#include "../../core/pearl_gateway.h"
#include "job.h"
#include "noisy_gemm.cuh"
#include "prepare.cuh"

namespace om {
namespace {

using namespace om::pearl;

/**
 * Shapes to mine at, largest first, chosen by measurement.
 *
 * Ranked on candidates per second, not on prepare overhead - those disagree,
 * and overhead is the misleading one. On a 4090, 4096x65536 has the lowest
 * overhead of anything measured and is the slowest, because B^t at 128 MB
 * stops being servable from L2 and the GEMM falls from 67 to 45 TOPS.
 *
 * m and n must be powers of two so the padded matrices have a power-of-two
 * chunk count and the device Merkle reduction is exactly blake3's tree, and
 * multiples of the rank so the transcript index inverts cleanly.
 */
struct Shape {
    uint32_t m, n;
};

const Shape kShapes[] = {
    // Ordered by measured candidates/s, largest-first only incidentally. The
    // ranking MOVED when the kernel got faster: 4096x32768 was the winner
    // while the GEMM was compute-bound and is well behind now that it is not,
    // because its working set no longer fits L2. Bigger is not better here.
    {2048, 32768},   // ~160 M candidates/s on a 4090, ~130 MB
    {2048, 16384},   // ~148
    {4096, 32768},   // ~94, was the old default
    {1024, 16384},   // ~129
    {512, 8192},     // small cards and integrated parts
    {256, 2048},
};

constexpr uint32_t kMaxHits = 64;
constexpr int kTileSide = 16;

/**
 * The tile configurations worth trying, and the launcher for each.
 *
 * This is a runtime choice, not a constant, because **the right answer is not
 * the same on two cards we own**. Measured at 4096x32768, k=2048:
 *
 *                      RTX 4090      RTX 5080
 *     2x4 / 2x2         97.3          58.0
 *     2x4 / 4x2        127.1 best     41.7 WORST
 *     2x4 / 2x4        110.1          68.5 best
 *     2x4 / 4x4        118.4          66.6
 *
 * Shipping the 4090's winner hardcoded would run a 5080 at 41.7 TOPS when that
 * card does 68.5 - giving away 64% for no reason a user could ever diagnose.
 * So prepare() times them on the actual device and picks.
 *
 * Every entry here is held to the reference in tests/test_pearl.cu. A
 * configuration that can be selected but was never verified is a correctness
 * bug waiting for someone else's GPU.
 */
struct TileConfig {
    const char *name;
    int blockM, blockN, threads;
    // The raw-mma kernel wants B stored n-major, because its B fragment reads
    // four consecutive k for one column. Both layouts are built once per job
    // (B is per-job work), so carrying a flag here costs nothing and keeps the
    // two kernel families in one table the tuner can sweep uniformly.
    bool nMajorB;
    // Minimum compute capability x10. The pre-Ampere fallbacks inside the
    // Ampere-only kernels are no-ops so the code still links for an sm_75
    // device pass - which means on such a card the tuner would time an empty
    // kernel, find it unbeatable, and select a configuration that mines
    // nothing at a spectacular rate. Skipping by capability is the fix; a
    // failed launch is not a reliable signal when the fallback succeeds.
    int minArch;
    void (*launch)(dim3, int, cudaStream_t, const int8_t *, const int8_t *,
                   uint32_t *, int, int, int, int);
};

template <int WM, int WN, int TM, int TN>
void launchTiled(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
                 const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
    noisyGemmMmaTiled<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
}

template <int WM, int WN, int TM, int TN>
void launchTiledDB(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
                   const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
    noisyGemmMmaTiledDB<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
}

template <int WM, int WN, int TM, int TN>
void launchTiledAsync(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
                      const int8_t *b, uint32_t *t, int m, int n, int k,
                      int rank) {
    noisyGemmMmaTiledAsync<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
}

template <int WM, int WN, int TM, int TN>
void launchPtx(dim3 grid, int threads, cudaStream_t s, const int8_t *a,
               const int8_t *b, uint32_t *t, int m, int n, int k, int rank) {
    noisyGemmPtx<WM, WN, TM, TN>
        <<<grid, threads, 0, s>>>(a, b, nullptr, t, m, n, k, rank, false);
}

const TileConfig kTileConfigs[] = {
    {"single 2x4/2x2", 64, 128, 256, false, 70, &launchTiled<2, 4, 2, 2>},
    {"single 2x4/4x2", 128, 128, 256, false, 70, &launchTiled<2, 4, 4, 2>},
    {"single 2x4/2x4", 64, 256, 256, false, 70, &launchTiled<2, 4, 2, 4>},
    {"single 2x4/4x4", 128, 256, 256, false, 70, &launchTiled<2, 4, 4, 4>},
    {"single 4x2/2x2", 128, 64, 256, false, 70, &launchTiled<4, 2, 2, 2>},
    {"single 4x4/2x2", 128, 128, 512, false, 70, &launchTiled<4, 4, 2, 2>},
    // Double-buffered: prefetch the next k-slice while the tensor cores chew
    // the current one. 1.6x the single-buffered kernel in isolation.
    {"dbuf 2x4/2x2", 64, 128, 256, false, 70, &launchTiledDB<2, 4, 2, 2>},
    {"dbuf 2x4/4x2", 128, 128, 256, false, 70, &launchTiledDB<2, 4, 4, 2>},
    {"dbuf 2x4/2x4", 64, 256, 256, false, 70, &launchTiledDB<2, 4, 2, 4>},
    {"dbuf 2x4/4x4", 128, 256, 256, false, 70, &launchTiledDB<2, 4, 4, 4>},
    {"dbuf 4x2/2x2", 128, 64, 256, false, 70, &launchTiledDB<4, 2, 2, 2>},
    {"dbuf 4x4/2x2", 128, 128, 512, false, 70, &launchTiledDB<4, 4, 2, 2>},
    // cp.async, three stages. Ampere and later only - on an older card the
    // launch fails with "invalid device function" and pickTileConfig() skips
    // it, which is exactly the behaviour wanted rather than a build-time split.
    {"async 2x4/2x2", 64, 128, 256, false, 80, &launchTiledAsync<2, 4, 2, 2>},
    {"async 2x4/4x2", 128, 128, 256, false, 80, &launchTiledAsync<2, 4, 4, 2>},
    {"async 2x4/2x4", 64, 256, 256, false, 80, &launchTiledAsync<2, 4, 2, 4>},
    {"async 2x4/4x4", 128, 256, 256, false, 80, &launchTiledAsync<2, 4, 4, 4>},
    {"async 4x2/2x2", 128, 64, 256, false, 80, &launchTiledAsync<4, 2, 2, 2>},
    {"async 4x4/2x2", 128, 128, 512, false, 80, &launchTiledAsync<4, 4, 2, 2>},
    // Raw mma.m16n8k32. WMMA's m16n16k16 needs two issues to cover the same
    // k that Ampere's native shape does in one, and the profiler had Tensor as
    // the busiest pipeline, so halving the issues is the point.
    {"ptx 2x4/2x2", 64, 128, 256, true, 80, &launchPtx<2, 4, 2, 2>},
    {"ptx 2x4/4x2", 128, 128, 256, true, 80, &launchPtx<2, 4, 4, 2>},
    {"ptx 2x4/2x4", 64, 256, 256, true, 80, &launchPtx<2, 4, 2, 4>},
    {"ptx 2x4/4x4", 128, 256, 256, true, 80, &launchPtx<2, 4, 4, 4>},
    {"ptx 4x2/2x2", 128, 64, 256, true, 80, &launchPtx<4, 2, 2, 2>},
    {"ptx 4x4/2x2", 128, 128, 512, true, 80, &launchPtx<4, 4, 2, 2>},
    // Sixteen-warp geometries. These lost while the fragment loads were four
    // scalar shared reads apiece - the extra warps had no issue slots to use.
    // With ldmatrix collapsing each fragment to one instruction, 4x4/4x2 went
    // from 323.5 to 363.2 TOPS and became the fastest thing on a 4090.
    {"ptx 4x4/4x2", 256, 128, 512, true, 80, &launchPtx<4, 4, 4, 2>},
    {"ptx 4x4/2x4", 128, 256, 512, true, 80, &launchPtx<4, 4, 2, 4>},
    {"ptx 8x2/2x2", 256, 64, 512, true, 80, &launchPtx<8, 2, 2, 2>},
};

class PearlPow : public Algorithm {
   public:
    PearlPow() { cudaStreamCreate(&stream_); }

    ~PearlPow() override {
        release();
        if (stream_) cudaStreamDestroy(stream_);
    }

    const char *name() const override { return "pearl-pow"; }

    size_t memoryBytes(const Job &job) const override {
        (void)job;
        return bytesFor(shape_);
    }

    /**
     * Pick a shape that fits and allocate for it.
     *
     * Called once, because Pearl has no epoch: there is no dataset to rebuild
     * when the height moves. What DOES have to be redone per job is the B
     * side, and that is keyed on the header rather than on the epoch, so it
     * lives in search().
     */
    bool prepare(const Job &job) override {
        (void)job;
        if (allocated_) return true;

        cudaDeviceProp prop{};
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            arch10_ = prop.major * 10 + prop.minor;

        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return false;

        // Leave the driver and whatever is on screen a real margin.
        const size_t headroom = 512ULL << 20;

        // Tune SHAPE and TILE CONFIGURATION together, because they interact.
        // Neither can be chosen first: the best shape moved once the kernel got
        // faster (a bigger working set stopped fitting L2), and the best
        // configuration differs per card - a 4090 wants 4x2 tiles per warp, a
        // 5080 wants 2x4, and each is near the other's worst. Choosing one and
        // then the other picks a local optimum on some card.
        //
        // A second or so at startup, once, against differences of 60%+.
        double bestRate = 0;
        bool found = false;
        for (const Shape &cand : kShapes) {
            if (bytesFor(cand) + headroom > freeB) continue;
            size_t cfg = 0;
            const double rate = tuneShape(cand, &cfg);
            if (rate > bestRate) {
                bestRate = rate;
                shape_ = cand;
                tile_ = cfg;
                found = true;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "[pearl-pow] no shape fits in %.0f MB of free device "
                    "memory\n", freeB / 1e6);
            return false;
        }

        if (!allocate()) {
            fprintf(stderr, "[pearl-pow] could not allocate %.0f MB\n",
                    bytesFor(shape_) / 1e6);
            return false;
        }
        allocated_ = true;
        fprintf(stderr,
                "[pearl-pow] %ux%u with %s: %.1f M candidates/s, %.0f MB\n",
                shape_.m, shape_.n, kTileConfigs[tile_].name, bestRate,
                bytesFor(shape_) / 1e6);
        return true;
    }

    /**
     * Time every eligible configuration at one shape; return the best rate in
     * millions of candidates per second and which configuration got it.
     *
     * Candidates per second, not milliseconds: shapes produce different numbers
     * of transcripts per GEMM, so raw time is not comparable across them.
     *
     * The buffers are filled before timing. A GEMM reading pages that were
     * never written is not measuring the same thing as one reading real data,
     * which already cost an hour once in this project's history.
     */
    double tuneShape(const Shape &s, size_t *bestCfg) {
        const size_t aBytes = (size_t)s.m * kK;
        const size_t bBytes = (size_t)kK * s.n;
        const size_t tiles = (size_t)(s.m / kTileSide) * (s.n / kTileSide);

        int8_t *tA = nullptr, *tB = nullptr;
        uint32_t *tT = nullptr;
        if (cudaMalloc(&tA, aBytes) != cudaSuccess) return 0;
        if (cudaMalloc(&tB, bBytes) != cudaSuccess) { cudaFree(tA); return 0; }
        if (cudaMalloc(&tT, tiles * 16 * sizeof(uint32_t)) != cudaSuccess) {
            cudaFree(tA);
            cudaFree(tB);
            return 0;
        }
        genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, stream_>>>(tA, aBytes, 1);
        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, stream_>>>(tB, bBytes, 2);
        cudaStreamSynchronize(stream_);

        cudaEvent_t evA, evB;
        cudaEventCreate(&evA);
        cudaEventCreate(&evB);

        double best = 0;
        *bestCfg = 0;
        for (size_t i = 0; i < sizeof(kTileConfigs) / sizeof(kTileConfigs[0]); i++) {
            const TileConfig &tc = kTileConfigs[i];
            if (arch10_ && arch10_ < tc.minArch) continue;
            // Escape hatch for A/B work: SOAT_PEARL_TILE=ptx restricts the
            // sweep to configurations whose name starts with that string, so
            // "is the new kernel actually faster end to end" can be answered
            // by measurement instead of by scaling one number by another.
            if (tileFilter_ && strncmp(tc.name, tileFilter_, strlen(tileFilter_)))
                continue;
            if (s.m % tc.blockM || s.n % tc.blockN) continue;
            dim3 grid(s.n / tc.blockN, s.m / tc.blockM);

            // Best of several, not the last of two. A single timed rep picked
            // a shape 20% off the best one here, because anything else on the
            // card perturbs one sample and the tuner then commits to it for
            // the whole run. Interference only ever makes a rep slower, so the
            // minimum is the honest estimate of what the kernel can do.
            float ms = 0.0f;
            bool ok = true;
            for (int rep = 0; rep < 4 && ok; rep++) {     // first pays warmup
                cudaEventRecord(evA, stream_);
                tc.launch(grid, tc.threads, stream_, tA, tB, tT, (int)s.m,
                          (int)s.n, (int)kK, kRank);
                cudaEventRecord(evB, stream_);
                if (cudaStreamSynchronize(stream_) != cudaSuccess ||
                    cudaGetLastError() != cudaSuccess) {
                    ok = false;
                    break;
                }
                float d = 0.0f;
                cudaEventElapsedTime(&d, evA, evB);
                if (rep && (ms == 0.0f || d < ms)) ms = d;
            }
            if (!ok || ms <= 0.0f) continue;
            const double rate = (double)tiles / (ms * 1e-3) / 1e6;
            if (rate > best) {
                best = rate;
                *bestCfg = i;
            }
        }
        cudaEventDestroy(evA);
        cudaEventDestroy(evB);
        cudaFree(tA);
        cudaFree(tB);
        cudaFree(tT);
        return best;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (!allocated_) return false;

        uint8_t header[76];
        int cert = 0;
        if (!unpackPearlExtra(job.extra, header, &cert)) {
            fprintf(stderr, "[pearl-pow] job carries no Pearl header\n");
            return false;
        }

        if (!beginJob(job, header, cert)) return false;

        // `count` is a candidate budget, and one attempt yields a whole tile
        // grid of them. Always at least one attempt, or a small batch setting
        // would spin without doing any work.
        const uint32_t perAttempt = tiles();
        uint64_t attempts = count / perAttempt;
        if (attempts == 0) attempts = 1;

        for (uint64_t i = 0; i < attempts; i++) {
            const uint64_t nonce = nonceBase + i;
            if (!runAttempt(nonce)) return false;

            uint32_t hits = 0;
            if (cudaMemcpyAsync(&hits, dHitCount_, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
                return false;
            if (cudaStreamSynchronize(stream_) != cudaSuccess) {
                fprintf(stderr, "[pearl-pow] attempt failed: %s\n",
                        cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            if (hits == 0) continue;
            if (hits > kMaxHits) hits = kMaxHits;

            // A and B are overwritten by the next attempt, so a win has to be
            // opened now. Reading them back beats regenerating them: 72 MB
            // over PCIe is milliseconds, and this happens once in a very long
            // time anyway.
            if (!openWin(job, nonce, hits, out)) return false;
        }
        return true;
    }

    /**
     * Recompute the winning tile on the host and re-check it against the bound.
     *
     * This is the same arithmetic the node does from the opened rows and
     * columns: sixteen by sixteen over k, half a million multiply-accumulates.
     * A card overclocked into instability produces wrong accumulators, and
     * there is no other way to tell that from a genuine win.
     */
    bool verify(const Job &job, const Solution &sol) const override {
        (void)job;
        for (const Win &w : wins_) {
            if (w.nonce != sol.nonce) continue;
            return w.verified;
        }
        return false;
    }

    void release() override {
        for (void *p : owned_) cudaFree(p);
        owned_.clear();
        allocated_ = false;
        haveJob_ = false;
    }

   private:
    struct Win {
        uint64_t nonce = 0;
        bool verified = false;
    };

    uint32_t tiles() const {
        return (shape_.m / kTileSide) * (shape_.n / kTileSide);
    }

    static size_t bytesFor(const Shape &s) {
        const size_t k = 2048, rank = 128;
        const size_t aBytes = (size_t)s.m * k, bBytes = (size_t)s.n * k;
        const size_t chunks = (aBytes > bBytes ? aBytes : bBytes) / kChunkLen;
        const size_t tileCount = (size_t)(s.m / kTileSide) * (s.n / kTileSide);
        return aBytes * 2                       // A, A_noised
               + bBytes * 2                     // B^t, B_noised
               + (size_t)s.m * rank             // E_AL
               + (size_t)s.n * rank * 2         // E_BR, flat and transposed
               + k * 8                          // permutation index pairs
               + chunks * 32 * 2                // merkle scratch
               + tileCount * 16 * sizeof(uint32_t)   // transcripts
               + kMaxHits * 36 + 4096;
    }

    template <typename T>
    bool alloc(T **p, size_t bytes) {
        if (cudaMalloc(p, bytes) != cudaSuccess) return false;
        owned_.push_back(*p);
        return true;
    }

    bool allocate() {
        const size_t k = kK, rank = kRank;
        const size_t aBytes = (size_t)shape_.m * k;
        const size_t bBytes = (size_t)shape_.n * k;
        const size_t chunks = (aBytes > bBytes ? aBytes : bBytes) / kChunkLen;
        return alloc(&dA_, aBytes) && alloc(&dAn_, aBytes) &&
               alloc(&dBt_, bBytes) && alloc(&dBn_, bBytes) && alloc(&dBnT_, bBytes) &&
               alloc(&dEAL_, (size_t)shape_.m * rank) &&
               alloc(&dEBRflat_, (size_t)shape_.n * rank) &&
               alloc(&dEBR_, (size_t)shape_.n * rank) &&
               alloc(&dArF_, k * 2) && alloc(&dArS_, k * 2) &&
               alloc(&dBlF_, k * 2) && alloc(&dBlS_, k * 2) &&
               alloc(&dScratch_, chunks * 32 * 2) && alloc(&dRoots_, 64) &&
               alloc(&dCommitA_, 32) && alloc(&dCommitB_, 32) &&
               alloc(&dJobKey_, 32) && alloc(&dSaltA_, 32) &&
               alloc(&dSaltB_, 32) && alloc(&dSeedA_, 32) &&
               alloc(&dSeedB_, 32) && alloc(&dTarget_, 32) &&
               alloc(&dTranscripts_, (size_t)tiles() * 16 * sizeof(uint32_t)) &&
               alloc(&dHitIndex_, kMaxHits * 4) &&
               alloc(&dHitDigest_, kMaxHits * 32) && alloc(&dHitCount_, 4);
    }

    /** Per-job work: everything that depends on the header but not on A. */
    bool beginJob(const Job &job, const uint8_t header[76], int cert) {
        if (haveJob_ && memcmp(header, header_, 76) == 0 && cert == cert_)
            return true;

        memcpy(header_, header, 76);
        cert_ = cert;
        cfg_.commonDim = kK;
        cfg_.rank = kRank;

        const std::string bad = cfg_.check(shape_.m, shape_.n);
        if (!bad.empty()) {
            fprintf(stderr, "[pearl-pow] illegal mining configuration: %s\n",
                    bad.c_str());
            return false;
        }
        if (!cfg_.penalizedTarget(U256::fromLimbs(job.target), &bound_)) {
            fprintf(stderr,
                    "[pearl-pow] this target does not scale into 256 bits for "
                    "the chosen configuration\n");
            return false;
        }

        jobKey(header_, cfg_, jobKey_);
        // A's stream is mixed with the job so two jobs never mine the same A.
        memcpy(&jobSalt_, jobKey_, sizeof(jobSalt_));

        uint8_t seedA[32], seedB[32], targetBytes[32];
        seedForA(seedA);
        seedForB(seedB);
        bound_.toBytesLE(targetBytes);

        cudaMemcpyAsync(dJobKey_, jobKey_, 32, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(dSaltA_, seedSaltA(), 32, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(dSaltB_, seedSaltB(), 32, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(dSeedA_, seedA, 32, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(dSeedB_, seedB, 32, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(dTarget_, targetBytes, 32, cudaMemcpyHostToDevice, stream_);

        // B, once. Roll a fresh one per job so a long run does not reuse the
        // same columns forever.
        btSeed_ = matrixSeed(jobSalt_, true);
        const size_t bBytes = (size_t)shape_.n * kK;
        genMatrix<<<(bBytes / 8 + 255) / 256, 256, 0, stream_>>>(dBt_, bBytes,
                                                                 btSeed_);
        merkleRoot((const uint8_t *)dBt_, (uint32_t)(bBytes / kChunkLen),
                   dRoots_ + 32);
        deriveCommitmentB<<<1, 1, 0, stream_>>>(dRoots_ + 32, dJobKey_, dSaltB_,
                                                shape_.n, cert_ >= 3, dCommitB_);

        const uint32_t brCount = shape_.n * kRank;
        noiseUniform<<<((brCount + 31) / 32 + 255) / 256, 256, 0, stream_>>>(
            dEBRflat_, brCount, dCommitB_, dSeedB_, kNoiseMask, kNoiseShift);
        transposeNoiseB<<<(brCount + 255) / 256, 256, 0, stream_>>>(
            dEBRflat_, dEBR_, shape_.n, kRank);
        noisePerm<<<((kK + 7) / 8 + 255) / 256, 256, 0, stream_>>>(
            dBlF_, dBlS_, kK, dCommitB_, dSeedB_, kRank);
        applyNoiseB<<<(((size_t)kK * shape_.n) + 255) / 256, 256, 0, stream_>>>(
            dBt_, dEBR_, dBlF_, dBlS_, dBn_, shape_.n, kK);
        applyNoiseBt<<<(((size_t)kK * shape_.n) + 255) / 256, 256, 0, stream_>>>(
            dBt_, dEBR_, dBlF_, dBlS_, dBnT_, shape_.n, kK);

        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            fprintf(stderr, "[pearl-pow] per-job setup failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        haveJob_ = true;
        wins_.clear();
        return true;
    }

    /** Per-attempt work: A's chain, the GEMM, and the PoW scan. */
    bool runAttempt(uint64_t nonce) {
        const size_t aBytes = (size_t)shape_.m * kK;
        cudaMemsetAsync(dHitCount_, 0, sizeof(uint32_t), stream_);

        genMatrix<<<(aBytes / 8 + 255) / 256, 256, 0, stream_>>>(
            dA_, aBytes, matrixSeed(nonce ^ jobSalt_, false));
        merkleRoot((const uint8_t *)dA_, (uint32_t)(aBytes / kChunkLen), dRoots_);
        deriveCommitmentA<<<1, 1, 0, stream_>>>(dRoots_, dCommitB_, dSaltA_,
                                                shape_.m, cert_ >= 3, dCommitA_);

        const uint32_t alCount = shape_.m * kRank;
        noiseUniform<<<((alCount + 31) / 32 + 255) / 256, 256, 0, stream_>>>(
            dEAL_, alCount, dCommitA_, dSeedA_, kNoiseMask, kNoiseShift);
        noisePerm<<<((kK + 7) / 8 + 255) / 256, 256, 0, stream_>>>(
            dArF_, dArS_, kK, dCommitA_, dSeedA_, kRank);
        applyNoiseA<<<(aBytes + 255) / 256, 256, 0, stream_>>>(
            dA_, dEAL_, dArF_, dArS_, dAn_, shape_.m, kK, kRank);

        // Whichever tile configuration prepare() measured fastest on this
        // card. See kTileConfigs for why that is not a constant.
        const TileConfig &tc = kTileConfigs[tile_];
        dim3 grid(shape_.n / tc.blockN, shape_.m / tc.blockM);
        tc.launch(grid, tc.threads, stream_, dAn_, tc.nMajorB ? dBnT_ : dBn_,
                  dTranscripts_,
                  (int)shape_.m, (int)shape_.n, (int)kK, kRank);

        powScan<<<(tiles() + 255) / 256, 256, 0, stream_>>>(
            dTranscripts_, tiles(), dCommitA_, (const uint32_t *)dTarget_,
            dHitIndex_, dHitDigest_, dHitCount_, kMaxHits);
        return cudaGetLastError() == cudaSuccess;
    }

    /**
     * Invert the transcript index, open the tile, and build the submission.
     *
     * The transcript layout is not row-major over tiles: noisy_gemm.cuh writes
     * them block-major over rank-sized blocks and then (hi, wi) inside each,
     * to match what the Python reference emits. Getting this inversion wrong
     * produces a proof that opens the wrong sixteen rows, which verifies
     * locally against nothing and is rejected by the node.
     */
    bool openWin(const Job &job, uint64_t nonce, uint32_t hits,
                 std::vector<Solution> *out) {
        std::vector<uint32_t> index(hits);
        std::vector<uint32_t> digest(hits * 8);
        cudaMemcpyAsync(index.data(), dHitIndex_, hits * 4, cudaMemcpyDeviceToHost,
                        stream_);
        cudaMemcpyAsync(digest.data(), dHitDigest_, hits * 32,
                        cudaMemcpyDeviceToHost, stream_);

        const size_t aBytes = (size_t)shape_.m * kK;
        const size_t bBytes = (size_t)shape_.n * kK;
        std::vector<int8_t> A(aBytes), Bt(bBytes);
        cudaMemcpyAsync(A.data(), dA_, aBytes, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(Bt.data(), dBt_, bBytes, cudaMemcpyDeviceToHost, stream_);
        uint8_t commitA[32], commitB[32];
        cudaMemcpyAsync(commitA, dCommitA_, 32, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(commitB, dCommitB_, 32, cudaMemcpyDeviceToHost, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return false;

        std::vector<uint8_t> aPad(aBytes, 0), btPad(bBytes, 0);
        memcpy(aPad.data(), A.data(), aBytes);
        memcpy(btPad.data(), Bt.data(), bBytes);

        // Only the first win is submitted. A second tile from the same attempt
        // is a second solution to the SAME block, so it can never be accepted
        // once the first is, and building its proof would be wasted work.
        const uint32_t flat = index[0];
        const uint32_t tilesPerSide = kRank / kTileSide;
        const uint32_t blocksPerRow = shape_.n / kRank;
        const uint32_t wi = flat % tilesPerSide;
        const uint32_t hi = (flat / tilesPerSide) % tilesPerSide;
        const uint32_t block = flat / (tilesPerSide * tilesPerSide);
        const uint32_t jIdx = block % blocksPerRow;
        const uint32_t iIdx = block / blocksPerRow;
        const uint32_t tRow = iIdx * kRank + hi * kTileSide;
        const uint32_t tCol = jIdx * kRank + wi * kTileSide;

        Win win;
        win.nonce = nonce;
        win.verified = recheck(A, Bt, commitA, commitB, tRow, tCol);
        wins_.push_back(win);
        if (wins_.size() > 32) wins_.erase(wins_.begin());
        if (!win.verified) {
            // Reported, not submitted: the core prints it as a host rejection.
            Solution sol;
            sol.nonce = nonce;
            out->push_back(sol);
            return true;
        }

        MerkleProof aProof, btProof;
        if (!buildProof(aPad.data(), aPad.size(), jobKey_,
                        leafIndicesFromRows(tRow, kTileSide, kK), &aProof) ||
            !buildProof(btPad.data(), btPad.size(), jobKey_,
                        leafIndicesFromRows(tCol, kTileSide, kK), &btProof))
            return false;

        Solution sol;
        sol.nonce = nonce;
        for (int i = 0; i < 4; i++)
            sol.hit[i] = (uint64_t)digest[i * 2] | ((uint64_t)digest[i * 2 + 1] << 32);
        sol.extra = base64(encodePlainProof(shape_.m, shape_.n, kK, kRank, aProof,
                                            tRow, kTileSide, btProof, tCol,
                                            kTileSide));
        out->push_back(sol);
        (void)job;
        return true;
    }

    /** The node's own check, on the host, from the two matrices. */
    bool recheck(const std::vector<int8_t> &A, const std::vector<int8_t> &Bt,
                 const uint8_t commitA[32], const uint8_t commitB[32],
                 uint32_t tRow, uint32_t tCol) const {
        Noise noise;
        noise.generate(commitA, commitB, shape_.m, shape_.n, kK, kRank);

        // Only the winning strips, not the whole product: sixteen rows of
        // A_noised and sixteen columns of B_noised.
        std::vector<int8_t> aStrip((size_t)kTileSide * kK);
        std::vector<int8_t> bStrip((size_t)kK * kTileSide);
        for (int i = 0; i < kTileSide; i++)
            for (size_t col = 0; col < kK; col++) {
                const int32_t e =
                    (int32_t)noise.eAL[(size_t)(tRow + i) * kRank + noise.ar.first[col]] -
                    (int32_t)noise.eAL[(size_t)(tRow + i) * kRank + noise.ar.second[col]];
                aStrip[(size_t)i * kK + col] =
                    (int8_t)(A[(size_t)(tRow + i) * kK + col] + e);
            }
        for (size_t p = 0; p < kK; p++)
            for (int j = 0; j < kTileSide; j++) {
                const int32_t e =
                    (int32_t)noise.eBR[(size_t)noise.bl.first[p] * shape_.n + tCol + j] -
                    (int32_t)noise.eBR[(size_t)noise.bl.second[p] * shape_.n + tCol + j];
                bStrip[p * kTileSide + j] =
                    (int8_t)(Bt[(size_t)(tCol + j) * kK + p] + e);
            }

        uint32_t transcript[kTranscriptWords];
        tileTranscript(aStrip.data(), bStrip.data(), kK, kTileSide, kRank, 0, 0,
                       kTileSide, kTileSide, transcript);
        uint8_t d[32];
        powDigest(transcript, commitA, d);
        return U256::fromBytesLE(d).le(bound_);
    }

    /** Chunk CVs, then as many tree levels per launch as a block can hold. */
    void merkleRoot(const uint8_t *data, uint32_t chunks, uint8_t *root) {
        chunkCvs<<<(chunks + 255) / 256, 256, 0, stream_>>>(
            data, chunks, (const uint32_t *)dJobKey_, dScratch_);
        uint32_t count = chunks;
        const uint8_t *in = dScratch_;
        uint8_t *outBuf = dScratch_ + (size_t)chunks * 32;
        while (count > 2) {
            uint32_t threads = 256;
            while (threads > 1 && 2u * threads > count / 2) threads >>= 1;
            const uint32_t blocks = count / (2 * threads);
            const size_t shared = (size_t)threads * 2 * 8 * sizeof(uint32_t);
            reduceTree<<<blocks, threads, shared, stream_>>>(
                in, count, (const uint32_t *)dJobKey_, outBuf);
            count = blocks;
            const uint8_t *next = outBuf;
            outBuf = const_cast<uint8_t *>(in);
            in = next;
        }
        rootCv<<<1, 1, 0, stream_>>>(in, (const uint32_t *)dJobKey_, root);
    }

    static constexpr uint32_t kK = 2048;
    static constexpr int kRank = 128;
    static constexpr int kNoiseMask = 63, kNoiseShift = 32;

    Shape shape_ = kShapes[sizeof(kShapes) / sizeof(kShapes[0]) - 1];
    int arch10_ = 0;           // compute capability x10, 0 until queried
    const char *tileFilter_ = getenv("SOAT_PEARL_TILE");
    size_t tile_ = 0;          // index into kTileConfigs, chosen by measurement
    bool allocated_ = false;
    bool haveJob_ = false;
    uint8_t header_[76] = {};
    int cert_ = 3;
    MiningConfig cfg_;
    U256 bound_;
    uint8_t jobKey_[32] = {};
    uint64_t jobSalt_ = 0, btSeed_ = 0;
    std::vector<Win> wins_;

    cudaStream_t stream_ = nullptr;
    std::vector<void *> owned_;

    int8_t *dA_ = nullptr, *dAn_ = nullptr, *dBt_ = nullptr, *dBn_ = nullptr;
    int8_t *dBnT_ = nullptr;   // the same noised B, stored n-major
    int8_t *dEAL_ = nullptr, *dEBRflat_ = nullptr, *dEBR_ = nullptr;
    uint16_t *dArF_ = nullptr, *dArS_ = nullptr, *dBlF_ = nullptr, *dBlS_ = nullptr;
    uint8_t *dScratch_ = nullptr, *dRoots_ = nullptr;
    uint8_t *dCommitA_ = nullptr, *dCommitB_ = nullptr, *dJobKey_ = nullptr;
    uint8_t *dSaltA_ = nullptr, *dSaltB_ = nullptr;
    uint8_t *dSeedA_ = nullptr, *dSeedB_ = nullptr, *dTarget_ = nullptr;
    uint32_t *dTranscripts_ = nullptr, *dHitIndex_ = nullptr;
    uint32_t *dHitDigest_ = nullptr, *dHitCount_ = nullptr;
};

}  // namespace

Algorithm *makePearlPow() { return new PearlPow(); }

}  // namespace om

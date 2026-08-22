// KawPow - the Algorithm implementation.
//
// The memory-hard case the algo.h interface anticipated: prepare() builds a
// 6.1 GB DAG on the device from a 95 MB light cache built on the host, and
// does it once per 7500-block epoch rather than once per job.
//
// Two clocks, which is the thing to keep straight in this file:
//
//   epoch   height / 7500, changes about every five days, costs a DAG rebuild
//   period  height / 3, changes about every three minutes, costs a table
//
// The core only knows about the first, because that is the expensive one, so
// the height travels in Job::extra and the period is handled here. See job.h.
//
// Verified against Ravencoin mainnet block 4501000 (tests/test_kawpow_device).

#include <cstdio>
#include <cstring>

#include <string>
#include <vector>

#include "../../core/algo.h"
#include "../../core/run.h"
#include "job.h"
#include "mine.cuh"
#include "nvrtc_jit.h"

namespace om {
namespace {

using namespace om::kawpow;

class KawPow : public Algorithm {
   public:
    KawPow() { cudaStreamCreateWithFlags(&mine_, cudaStreamNonBlocking); }

    const char *name() const override { return "kawpow"; }
    std::string launcherGlob() const override { return "mine_rvn_*"; }

    size_t memoryBytes(const Job &job) const override {
        const int height = jobHeight(job.extra, job.epoch);
        return (size_t)fullDatasetBytes(epochOf(height));
    }

    /**
     * Build the DAG for this job's epoch, if it is not already built.
     *
     * Guarded on the Ethash epoch rather than on being called, because the run
     * loop calls prepare() whenever Job::epoch changes and the benchmark path
     * changes it on a timer. Rebuilding 6 GB because a height moved by one
     * would be a self-inflicted stall every block.
     */
    bool prepare(const Job &job) override {
        const int height = jobHeight(job.extra, job.epoch);
        const int epoch = epochOf(height);
        if (epoch == builtEpoch_ && dDag_) return true;

        release();

        numCacheItems_ = lightCacheNumItems(epoch);
        numItems2048_ = (uint32_t)(fullDatasetNumItems(epoch) / 2);
        const uint32_t numItems512 = (uint32_t)fullDatasetNumItems(epoch) * 2;
        const uint64_t dagBytes = fullDatasetBytes(epoch);

        // Check the card is big enough BEFORE building a 95 MB light cache
        // that takes real time. The Vulkan path has always done this; CUDA
        // only found out at cudaMalloc, so a user with too small a card waited
        // through the whole build to be told it was never going to work.
        {
            cudaDeviceProp p{};
            int dev = 0;
            cudaGetDevice(&dev);
            const bool haveProp = cudaGetDeviceProperties(&p, dev) == cudaSuccess;
            const uint64_t total = haveProp ? (uint64_t)p.totalGlobalMem : 0;
            if (total && dagBytes + 256ull * 1024 * 1024 > total) {
                fprintf(stderr,
                    "[kawpow] this card cannot mine Ravencoin.\n"
                    "  card:  %s\n"
                    "  VRAM:  %.1f GB\n"
                    "  need:  %.2f GB for epoch %d, plus about 0.25 GB working space\n"
                    "  The DAG grows roughly 8 MB per epoch and never shrinks, so a\n"
                    "  card that is short today does not become able later.\n"
                    "  This card CAN mine the small-working-set algorithms on this\n"
                    "  same binary: --algo sha3-256t needs a few kB, and\n"
                    "  --algo autolykos2 needs about 7.5 GB.\n",
                    p.name, total / 1e9, dagBytes / 1e9, epoch);
                return false;
            }
        }

        // The host keeps its copy of the light cache for the lifetime of the
        // epoch. It is 95 MB, and it is what verify() re-derives a solution
        // from - asking the same possibly-unstable GPU to confirm its own
        // answer is how a bad hit gets confirmed twice and submitted.
        uint32_t seed[8];
        epochSeed(epoch, seed);
        cache_.assign((size_t)numCacheItems_ * 16, 0);
        buildLightCache(cache_.data(), numCacheItems_, seed);

        l1_.assign(kL1CacheWords, 0);
        for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
            datasetItem2048(cache_.data(), numCacheItems_, i, l1_.data() + i * 64);

        uint32_t *dCache = nullptr;
        if (cudaMalloc(&dCache, cache_.size() * 4) != cudaSuccess) {
            fprintf(stderr, "[kawpow] cannot allocate %.0f MB for the light cache\n",
                    cache_.size() * 4 / 1e6);
            return false;
        }
        if (cudaMalloc(&dDag_, dagBytes) != cudaSuccess) {
            // Name the card and say which of the two problems this is. They
            // need opposite actions: a card that is merely busy can mine
            // Ravencoin once something frees up, and a card that is too small
            // never will - and will only get further away, because the DAG
            // grows about 8 MB an epoch and never shrinks.
            cudaDeviceProp p{};
            int dev = 0;
            cudaGetDevice(&dev);
            const bool haveProp = cudaGetDeviceProperties(&p, dev) == cudaSuccess;
            size_t freeB = 0, totalB = 0;
            cudaMemGetInfo(&freeB, &totalB);

            fprintf(stderr,
                    "[kawpow] cannot allocate the %.2f GB DAG for epoch %d.\n"
                    "  card:     %s\n"
                    "  VRAM:     %.2f GB total, %.2f GB free\n"
                    "  need:     %.2f GB in one allocation\n",
                    dagBytes / 1e9, epoch,
                    haveProp ? p.name : "unknown",
                    totalB / 1e9, freeB / 1e9, dagBytes / 1e9);
            if (totalB && totalB < dagBytes) {
                fprintf(stderr,
                    "  This card is too small for Ravencoin and always will be -\n"
                    "  the DAG grows about 8 MB per epoch and never shrinks. It can\n"
                    "  still mine the algorithms with small working sets: try\n"
                    "  --algo sha3-256t, or --algo autolykos2 if it has 8 GB.\n");
            } else {
                fprintf(stderr,
                    "  The card is big enough but not enough is free right now.\n"
                    "  A display, a browser or another miner on the same card is\n"
                    "  the usual cause; close one and retry.\n");
            }
            cudaFree(dCache);
            dDag_ = nullptr;
            return false;
        }
        cudaMemcpy(dCache, cache_.data(), cache_.size() * 4, cudaMemcpyHostToDevice);

        // Sliced. One launch over 95 million items sits on the card for long
        // enough to trip a display driver's watchdog.
        const uint32_t slice = 1u << 21;
        for (uint32_t start = 0; start < numItems512; start += slice) {
            const uint32_t n =
                (numItems512 - start) < slice ? (numItems512 - start) : slice;
            om::progpow::ppBuildDag<<<(n + 255) / 256, 256, 0, mine_>>>(
                dCache, numCacheItems_, start, numItems512, dDag_);
        }
        const bool ok = cudaStreamSynchronize(mine_) == cudaSuccess;
        cudaFree(dCache);
        if (!ok) {
            fprintf(stderr, "[kawpow] DAG build failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            cudaFree(dDag_);
            dDag_ = nullptr;
            return false;
        }

        builtEpoch_ = epoch;
        ensureScratch();
        return dSol_ != nullptr && dCount_ != nullptr;
    }

    bool acceptsRuntimeCompilation() const override { return true; }

    void configure(const RunOptions &o) override {
        jitWanted_ = o.nvrtc;
        if (!jitWanted_) return;
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            ccMajor_ = prop.major;
            ccMinor_ = prop.minor;
        } else {
            // Without a compute capability there is nothing sensible to
            // compile for, so decline rather than guess an architecture.
            jitWanted_ = false;
        }
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (!dDag_) return false;
        const int height = jobHeight(job.extra, job.epoch);

        // The program is per period, so it is decoded when the period turns
        // and not once per launch.
        const int period = height / kPeriodLength;
        if (period != builtPeriod_) {
            prog_ = om::progpow::buildProgram(height, params());
            builtPeriod_ = period;
            if (jitWanted_ && !jitOff_) buildAndVerifyJit(job);
        }

        ensureScratch();
        if (!dSol_ || !dCount_) return false;

        cudaMemsetAsync(dCount_, 0, sizeof(uint32_t), mine_);

        const int threads = 512;                      // 32 hashes per block
        const int groups = threads / (int)kNumLanes;
        const size_t shared = (kL1CacheWords + groups * kNumLanes) * 4;
        uint64_t blocks = (count + groups - 1) / groups;
        if (blocks == 0) blocks = 1;

        uint32_t hdr[8];
        for (int i = 0; i < 8; i++) {
            memcpy(&hdr[i], job.msg + i * 4, 4);
        }

        // The specialised kernel is used only after it has reproduced the
        // interpreted kernel's output on this DAG, this period. jitReady_ is
        // false until then and false forever after a single disagreement.
        bool launched = false;
        if (jitReady_) {
            uint32_t nItems = numItems2048_;
            uint64_t nb = nonceBase, cnt = count;
            uint64_t t0 = job.target[0], t1 = job.target[1];
            uint64_t t2 = job.target[2], t3 = job.target[3];
            uint32_t cap = kMaxSolutions;
            void *args[] = {&dDag_, &nItems,  &hdr[0], &hdr[1], &hdr[2],
                            &hdr[3], &hdr[4], &hdr[5], &hdr[6], &hdr[7],
                            &nb,     &cnt,    &t0,     &t1,     &t2,
                            &t3,     &dSol_,  &dCount_, &cap};
            launched = jit_.launch(jit_.searchKernel(), (unsigned)blocks,
                                   threads, (unsigned)shared, mine_, args);
            if (!launched) {
                // Do not silently fall through to the interpreted kernel for
                // the rest of the run without saying so: a miner quietly at a
                // fifth of the speed it reported at startup is a support
                // question nobody can answer from the log.
                fprintf(stderr, "[kawpow] specialised kernel failed to launch; "
                                "reverting to the interpreted kernel\n");
                jitReady_ = false;
                jitOff_ = true;
            }
        }
        if (!launched)
        kpSearch<<<(unsigned)blocks, threads, shared, mine_>>>(
            dDag_, numItems2048_, prog_, hdr[0], hdr[1], hdr[2], hdr[3], hdr[4],
            hdr[5], hdr[6], hdr[7], nonceBase, count, job.target[0],
            job.target[1], job.target[2], job.target[3], dSol_, dCount_,
            kMaxSolutions);

        uint32_t found = 0;
        cudaMemcpyAsync(&found, dCount_, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                        mine_);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) {
            fprintf(stderr, "[kawpow] search failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (found == 0) return true;

        if (found > kMaxSolutions) {
            fprintf(stderr,
                    "[kawpow] warning: %u solutions in one batch exceeds the "
                    "%u-solution buffer; %u dropped. Difficulty may be too low.\n",
                    found, kMaxSolutions, found - kMaxSolutions);
            found = kMaxSolutions;
        }

        std::vector<om::progpow::Solution32> host(found);
        cudaMemcpyAsync(host.data(), dSol_,
                        sizeof(om::progpow::Solution32) * found,
                        cudaMemcpyDeviceToHost, mine_);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) return false;

        for (uint32_t i = 0; i < found; i++) {
            Solution s;
            s.nonce = host[i].nonce;
            hashToLimbs(host[i].hit, s.hit);
            // A ProgPoW share is not just a nonce: the pool checks the mix
            // hash too, and it is not derivable from the nonce without redoing
            // the whole hash. It rides out in Solution::extra.
            s.extra = toHex(host[i].mix);
            out->push_back(s);
        }
        return true;
    }

    /**
     * Re-derive the solution on the HOST, from the light cache.
     *
     * Slow - a fraction of a second - and that is the point. Shares are rare
     * and wrong ones are expensive, and a card overclocked into instability
     * will happily confirm its own bad answer if asked to check itself.
     */
    bool verify(const Job &job, const Solution &sol) const override {
        if (cache_.empty()) return false;
        const int height = jobHeight(job.extra, job.epoch);
        uint32_t hdr[8];
        for (int i = 0; i < 8; i++) memcpy(&hdr[i], job.msg + i * 4, 4);

        uint32_t final32[8], mix[8];
        hashLight(cache_.data(), numCacheItems_, l1_.data(), numItems2048_,
                  height, hdr, sol.nonce, final32, mix);

        uint64_t hit[4];
        hashToLimbs(final32, hit);
        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        if (!sol.extra.empty() && sol.extra != toHex(mix)) return false;
        return underTarget(hit, job.target);
    }

    void release() override {
        if (dDag_) cudaFree(dDag_);
        dDag_ = nullptr;
        builtEpoch_ = -1;
        builtPeriod_ = -1;
        // The compiled kernel is keyed to a period, and release() invalidates
        // the period. Leaving jitReady_ set would let a later search launch a
        // kernel built for a program that is no longer current - the exact
        // failure the per-period verification exists to prevent.
        jitReady_ = false;
        cache_.clear();
        cache_.shrink_to_fit();
        l1_.clear();
    }

    ~KawPow() override {
        release();
        if (dSol_) cudaFree(dSol_);
        if (dCount_) cudaFree(dCount_);
        if (dVerifyA_) cudaFree(dVerifyA_);
        if (dVerifyB_) cudaFree(dVerifyB_);
        if (mine_) cudaStreamDestroy(mine_);
    }

   private:
    /// Compile this period's kernel and refuse to use it unless it reproduces
    /// the interpreted kernel exactly.
    ///
    /// Run on the REAL DAG and this job's real header, every period - not once
    /// at startup. The program changes with the period, so a generator bug can
    /// be latent for hours and then produce a kernel that is wrong only for one
    /// program in a hundred. Checking at startup would miss exactly that.
    ///
    /// The cost is two extra launches of `kVerifyNonces` nonces per period.
    /// The compile itself is the larger cost and it is NOT MEASURED - stated
    /// rather than estimated, because an estimate written in a comment becomes
    /// a fact in a summary two hands later.
    void buildAndVerifyJit(const Job &job) {
        jitReady_ = false;
        if (!jit_.build(prog_, ccMajor_, ccMinor_)) {
            jitOff_ = true;
            return;
        }

        const uint64_t n = kVerifyNonces;
        if (!dVerifyA_ && cudaMalloc(&dVerifyA_, sizeof(Solution32) * n) !=
                              cudaSuccess) {
            jitOff_ = true;
            return;
        }
        if (!dVerifyB_ && cudaMalloc(&dVerifyB_, sizeof(Solution32) * n) !=
                              cudaSuccess) {
            jitOff_ = true;
            return;
        }
        cudaMemsetAsync(dVerifyA_, 0, sizeof(Solution32) * n, mine_);
        cudaMemsetAsync(dVerifyB_, 0, sizeof(Solution32) * n, mine_);

        uint32_t hdr[8];
        for (int i = 0; i < 8; i++) memcpy(&hdr[i], job.msg + i * 4, 4);
        const uint64_t base = 0x5EED0000ULL;

        const int threads = 512;
        const int groups = threads / (int)kNumLanes;
        const size_t shared = (kL1CacheWords + groups * kNumLanes) * 4;
        const unsigned blocks = (unsigned)((n + groups - 1) / groups);

        kpHashBatch<<<blocks, threads, shared, mine_>>>(
            dDag_, numItems2048_, prog_, hdr[0], hdr[1], hdr[2], hdr[3], hdr[4],
            hdr[5], hdr[6], hdr[7], base, n, dVerifyA_);

        uint32_t nItems = numItems2048_;
        uint64_t nb = base, cnt = n;
        void *args[] = {&dDag_, &nItems,  &hdr[0], &hdr[1], &hdr[2], &hdr[3],
                        &hdr[4], &hdr[5], &hdr[6], &hdr[7], &nb,     &cnt,
                        &dVerifyB_};
        const bool ok =
            jit_.launch(jit_.hashKernel(), blocks, threads, (unsigned)shared,
                        mine_, args) &&
            cudaStreamSynchronize(mine_) == cudaSuccess;

        std::vector<Solution32> a(n), b(n);
        if (ok) {
            cudaMemcpy(a.data(), dVerifyA_, sizeof(Solution32) * n,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), dVerifyB_, sizeof(Solution32) * n,
                       cudaMemcpyDeviceToHost);
        }

        // Two buffers of zeros compare equal, so an all-zero interpreted result
        // would pass a kernel that did nothing at all. Require real output
        // before the comparison is allowed to mean anything.
        bool real = false;
        for (uint64_t i = 0; i < n && !real; i++)
            for (int w = 0; w < 8; w++)
                if (a[i].mix[w] != 0) { real = true; break; }

        bool same = ok && real;
        for (uint64_t i = 0; i < n && same; i++) {
            if (a[i].nonce != b[i].nonce) { same = false; break; }
            for (int w = 0; w < 8; w++)
                if (a[i].mix[w] != b[i].mix[w] || a[i].hit[w] != b[i].hit[w]) {
                    same = false;
                    break;
                }
        }

        if (!same) {
            // Permanent, not per-period. A generator that disagrees once is not
            // trustworthy for the next period either, and a miner that hashes
            // wrong finds nothing and reports no error - so the failure has to
            // be loud here or it is never seen at all.
            fprintf(stderr,
                    "[kawpow] the runtime-compiled kernel disagreed with the "
                    "interpreted one; staying on the interpreted kernel for "
                    "the rest of this run\n");
            jitOff_ = true;
            return;
        }

        jitReady_ = true;
        if (!announced_) {
            fprintf(stderr, "[kawpow] runtime-compiled kernel verified against "
                            "the interpreted kernel; using it\n");
            announced_ = true;
        }
    }

    /// How many nonces the per-period verification compares.
    static const uint64_t kVerifyNonces = 512;

    om::kawpow::NvrtcJit jit_;
    Solution32 *dVerifyA_ = nullptr, *dVerifyB_ = nullptr;
    bool jitWanted_ = false;   ///< --nvrtc was passed
    bool jitReady_ = false;    ///< compiled AND verified for this period
    bool jitOff_ = false;      ///< failed once; do not try again this run
    bool announced_ = false;
    int ccMajor_ = 0, ccMinor_ = 0;

    // Same 256 as every other algorithm here, for the same reason: a
    // deliberately low test difficulty can find far more than a handful.
    static const uint32_t kMaxSolutions = 256;

    static std::string toHex(const uint32_t w[8]) {
        static const char *d = "0123456789abcdef";
        std::string s;
        s.reserve(64);
        for (int i = 0; i < 8; i++)
            for (int b = 0; b < 4; b++) {
                const uint8_t v = (uint8_t)(w[i] >> (b * 8));
                s += d[v >> 4];
                s += d[v & 15];
            }
        return s;
    }

    void ensureScratch() const {
        if (!dSol_)
            cudaMalloc(&dSol_, sizeof(om::progpow::Solution32) * kMaxSolutions);
        if (!dCount_) cudaMalloc(&dCount_, sizeof(uint32_t));
    }

    cudaStream_t mine_ = nullptr;
    uint32_t *dDag_ = nullptr;
    mutable om::progpow::Solution32 *dSol_ = nullptr;
    mutable uint32_t *dCount_ = nullptr;

    std::vector<uint32_t> cache_;
    std::vector<uint32_t> l1_;
    int numCacheItems_ = 0;
    uint32_t numItems2048_ = 0;
    int builtEpoch_ = -1;
    int builtPeriod_ = -1;
    om::progpow::Program prog_{};
};

}  // namespace

Algorithm *makeKawpow() { return new KawPow(); }

}  // namespace om

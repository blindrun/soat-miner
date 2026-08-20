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
#include "job.h"
#include "mine.cuh"

namespace om {
namespace {

using namespace om::kawpow;

class KawPow : public Algorithm {
   public:
    KawPow() { cudaStreamCreateWithFlags(&mine_, cudaStreamNonBlocking); }

    const char *name() const override { return "kawpow"; }

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
            fprintf(stderr,
                    "[kawpow] cannot allocate the %.2f GB DAG for epoch %d.\n"
                    "  KawPow needs that much VRAM free; a display and a browser "
                    "on the same card can be enough to prevent it.\n",
                    dagBytes / 1e9, epoch);
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
        cache_.clear();
        cache_.shrink_to_fit();
        l1_.clear();
    }

    ~KawPow() override {
        release();
        if (dSol_) cudaFree(dSol_);
        if (dCount_) cudaFree(dCount_);
        if (mine_) cudaStreamDestroy(mine_);
    }

   private:
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

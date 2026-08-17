// Autolykos v2 (Ergo) - the Algorithm implementation.
//
// Verified against the Ergo node's AutolykosPowScheme.scala and against real
// mainnet blocks: see tests/reference.py and tests/test_hit.cu.

#include <cstdio>
#include <cstring>

#include <string>
#include <utility>

#include "../../core/algo.h"
#include "mine.cuh"

namespace om {
namespace {

/** Table size N for a given height (consensus rule). */
uint32_t calcN(uint64_t height) {
    const uint32_t NBase = 1u << 26;
    const uint32_t Start = 600u * 1024u;
    const uint32_t Period = 50u * 1024u;
    const uint32_t MaxH = 4198400u;
    uint32_t h = (height < MaxH) ? (uint32_t)height : MaxH;
    if (h < Start) return NBase;
    uint32_t iters = (h - Start) / Period + 1;
    uint32_t N = NBase;
    for (uint32_t i = 0; i < iters; i++) N = N / 100 * 105;
    return N;
}

// Leave this much VRAM unclaimed when deciding whether a second dataset fits.
// The driver, the display and our own scratch all live in it.
//
// 1.5 GB rather than something tighter on purpose. A 16 GB card has about
// 8.0 GB left after the first 7.27 GB dataset, so a small headroom would let
// the second one squeeze in and leave ~0.7 GB for the driver and whatever is
// on screen. Build-ahead is worth well under 1%, and nowhere near enough to
// risk an out-of-memory on a card that was mining fine without it.
const size_t kPrefetchHeadroom = 1536ULL << 20;

class Autolykos2 : public Algorithm {
   public:
    Autolykos2() {
        // Non-blocking so neither stream is serialised against the legacy
        // default stream: the whole point is that a background build overlaps
        // mining rather than replacing it.
        //
        // Priorities matter here. The build kernel fills every SM, so without
        // them it takes the warp slots the mining kernel needs to keep enough
        // loads in flight, and the two just share the device proportionally.
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        cudaStreamCreateWithPriority(&mine_, cudaStreamNonBlocking, hi);
        cudaStreamCreateWithPriority(&build_, cudaStreamNonBlocking, lo);
    }

    const char *name() const override { return "autolykos2"; }

    size_t memoryBytes(const Job &job) const override {
        return (size_t)calcN(job.epoch) * 32ULL;
    }

    void setPrefetch(int mode) override { mode_ = mode; }

    bool servedFromPrefetch() const override { return servedFromPrefetch_; }

    std::string prefetchNote() const override { return note_; }

    bool prefetchReadyFor(const Job &job) const override {
        return nextValid_ && nextHeight_ == (uint32_t)job.epoch &&
               nextN_ == calcN(job.epoch);
    }

    bool prepare(const Job &job) override {
        const uint32_t height = (uint32_t)job.epoch;
        const uint32_t N = calcN(job.epoch);
        const size_t bytes = (size_t)N * 32ULL;
        servedFromPrefetch_ = false;

        // Fast path: the background build already covered this height.
        if (nextValid_ && nextHeight_ == height && nextN_ == N) {
            const cudaError_t e = cudaStreamSynchronize(build_);
            nextValid_ = false;
            if (e == cudaSuccess) {
                std::swap(dataset_, next_);
                std::swap(datasetCap_, nextCap_);
                n_ = N;
                height_ = height;
                servedFromPrefetch_ = true;
                ensureScratch();
                startPrefetch(height + 1);
                return true;
            }
            fprintf(stderr, "[autolykos2] prefetched build failed: %s\n",
                    cudaGetErrorString(e));
            // Fall through and build it the ordinary way.
        }

        // Any build still in flight is for a height we are not going to mine.
        // It cannot be cancelled, only left to drain - the next launch queues
        // behind it on the same stream.
        nextValid_ = false;

        if (!dataset_ || datasetCap_ < bytes) {
            releaseBuffer(&dataset_, &datasetCap_);
            if (cudaMalloc(&dataset_, bytes) != cudaSuccess) {
                fprintf(stderr, "[autolykos2] cudaMalloc %.2f GB failed\n",
                        bytes / 1e9);
                return false;
            }
            datasetCap_ = bytes;
        }
        n_ = N;

        // The dataset depends on the height, so it is rebuilt on every epoch
        // change even when N itself has not moved.
        al_build_dataset<<<(N + 255) / 256, 256, 0, mine_>>>(dataset_, N, height);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) {
            fprintf(stderr, "[autolykos2] dataset build failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        height_ = height;

        ensureScratch();
        startPrefetch(height + 1);
        return true;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (!dataset_) return false;
        ensureScratch();

        // Everything here stays on the mining stream. A plain cudaMemcpy or
        // cudaDeviceSynchronize would wait on the background build too, which
        // is exactly the stall this is meant to remove.
        cudaMemcpyAsync(dMsg_, job.msg, 32, cudaMemcpyHostToDevice, mine_);
        cudaMemcpyAsync(dTarget_, job.target, 32, cudaMemcpyHostToDevice, mine_);
        cudaMemsetAsync(dCount_, 0, sizeof(uint32_t), mine_);

        const int threads = 256;
        uint32_t blocks = (uint32_t)(count / threads);
        if (blocks == 0) blocks = 1;

        al_search<<<blocks, threads, 0, mine_>>>(dataset_, dMsg_, nonceBase, n_,
                                                 height_, dTarget_, dSol_,
                                                 dCount_, kMaxSolutions);

        uint32_t found = 0;
        cudaMemcpyAsync(&found, dCount_, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                        mine_);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) {
            fprintf(stderr, "[autolykos2] search failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }

        if (found == 0) return true;
        if (found > kMaxSolutions) {
            // The GPU found more solutions in one batch than the buffer holds,
            // so the excess were dropped. Harmless at real pool difficulty (a
            // batch almost never finds this many), but a very low difficulty
            // can trip it and silently lose valid shares - so make it visible
            // rather than have it masquerade as flaky hardware.
            fprintf(stderr,
                    "[autolykos2] warning: %u solutions in one batch exceeds the "
                    "%u-solution buffer; %u dropped. Difficulty may be too low.\n",
                    found, kMaxSolutions, found - kMaxSolutions);
            found = kMaxSolutions;
        }

        std::vector<AlSolution> host(found);
        cudaMemcpyAsync(host.data(), dSol_, sizeof(AlSolution) * found,
                        cudaMemcpyDeviceToHost, mine_);
        cudaStreamSynchronize(mine_);
        for (uint32_t i = 0; i < found; i++) {
            Solution s;
            s.nonce = host[i].nonce;
            memcpy(s.hit, host[i].hit, sizeof(s.hit));
            out->push_back(s);
        }
        return true;
    }

    bool verify(const Job &job, const Solution &sol) const override {
        // Recompute the hit on the device for exactly this nonce and require
        // both that it reproduces and that it is genuinely under target.
        if (!dataset_) return false;
        ensureScratch();
        cudaMemcpyAsync(dMsg_, job.msg, 32, cudaMemcpyHostToDevice, mine_);
        al_verify_one<<<1, 1, 0, mine_>>>(dataset_, dMsg_, sol.nonce, n_, height_,
                                          dHit_);
        uint64_t hit[4];
        cudaMemcpyAsync(hit, dHit_, 32, cudaMemcpyDeviceToHost, mine_);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) return false;

        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        for (int i = 3; i >= 0; i--) {
            if (hit[i] != job.target[i]) return hit[i] < job.target[i];
        }
        return false;
    }

    void release() override {
        // A queued build writes into next_, so it has to finish before the
        // memory under it goes away.
        cudaStreamSynchronize(build_);
        nextValid_ = false;
        releaseBuffer(&dataset_, &datasetCap_);
        releaseBuffer(&next_, &nextCap_);
        n_ = 0;
    }

    ~Autolykos2() override {
        release();
        if (dMsg_) cudaFree(dMsg_);
        if (dTarget_) cudaFree(dTarget_);
        if (dSol_) cudaFree(dSol_);
        if (dCount_) cudaFree(dCount_);
        if (dHit_) cudaFree(dHit_);
        if (mine_) cudaStreamDestroy(mine_);
        if (build_) cudaStreamDestroy(build_);
    }

   private:
    // 256, not 32: one batch at a deliberately low difficulty (e.g. a Lithos
    // test diff) can find well over 32 solutions, and anything past the buffer
    // is silently lost. 256 is still tiny (10 KB) and covers realistic bursts.
    static const uint32_t kMaxSolutions = 256;

    void releaseBuffer(uint4 **p, size_t *cap) {
        if (*p) cudaFree(*p);
        *p = nullptr;
        *cap = 0;
    }

    /**
     * Queue the build for `height` on the background stream.
     *
     * Two datasets have to be resident at once for this, so it is gated on
     * free VRAM and simply does not happen on a card where it would not fit.
     * A miner that was working before must not start failing to allocate.
     */
    void startPrefetch(uint32_t height) {
        if (mode_ == 0) return;

        const uint32_t N = calcN(height);
        const size_t bytes = (size_t)N * 32ULL;

        if (!next_ || nextCap_ < bytes) {
            if (next_) {
                cudaStreamSynchronize(build_);
                releaseBuffer(&next_, &nextCap_);
            }
            size_t freeB = 0, totalB = 0;
            if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return;
            if (mode_ != 1 && freeB < bytes + kPrefetchHeadroom) {
                note_ = "build-ahead off: needs a second " + gb(bytes) +
                        " GB dataset resident and only " + gb(freeB) +
                        " GB is free";
                return;
            }
            if (cudaMalloc(&next_, bytes) != cudaSuccess) {
                note_ = "build-ahead off: a second " + gb(bytes) +
                        " GB dataset would not allocate";
                return;
            }
            nextCap_ = bytes;
            note_ = "build-ahead on: holding two datasets, " +
                    gb(bytes * 2) + " GB total";
        }

        al_build_dataset<<<(N + 255) / 256, 256, 0, build_>>>(next_, N, height);
        if (cudaGetLastError() != cudaSuccess) return;
        nextValid_ = true;
        nextHeight_ = height;
        nextN_ = N;
    }

    static std::string gb(size_t bytes) {
        char b[32];
        snprintf(b, sizeof(b), "%.2f", bytes / 1e9);
        return b;
    }

    void ensureScratch() const {
        if (!dMsg_) cudaMalloc(&dMsg_, 32);
        if (!dTarget_) cudaMalloc(&dTarget_, 32);
        if (!dSol_) cudaMalloc(&dSol_, sizeof(AlSolution) * kMaxSolutions);
        if (!dCount_) cudaMalloc(&dCount_, sizeof(uint32_t));
        if (!dHit_) cudaMalloc(&dHit_, 32);
    }

    uint4 *dataset_ = nullptr;
    size_t datasetCap_ = 0;
    uint32_t n_ = 0;
    uint32_t height_ = 0;

    // The next height's dataset, built in the background.
    uint4 *next_ = nullptr;
    size_t nextCap_ = 0;
    bool nextValid_ = false;
    uint32_t nextHeight_ = 0;
    uint32_t nextN_ = 0;
    int mode_ = -1;
    bool servedFromPrefetch_ = false;
    std::string note_;

    cudaStream_t mine_ = nullptr;
    cudaStream_t build_ = nullptr;

    mutable uint8_t *dMsg_ = nullptr;
    mutable uint64_t *dTarget_ = nullptr;
    mutable AlSolution *dSol_ = nullptr;
    mutable uint32_t *dCount_ = nullptr;
    mutable uint64_t *dHit_ = nullptr;
};

}  // namespace

Algorithm *makeAutolykos2() { return new Autolykos2(); }

}  // namespace om

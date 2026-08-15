// Autolykos v2 (Ergo) - the Algorithm implementation.
//
// Verified against the Ergo node's AutolykosPowScheme.scala and against real
// mainnet blocks: see tests/reference.py and tests/test_hit.cu.

#include <cstdio>
#include <cstring>

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

class Autolykos2 : public Algorithm {
   public:
    const char *name() const override { return "autolykos2"; }

    size_t memoryBytes(const Job &job) const override {
        return (size_t)calcN(job.epoch) * 32ULL;
    }

    bool prepare(const Job &job) override {
        uint32_t N = calcN(job.epoch);
        size_t bytes = (size_t)N * 32ULL;

        if (N != n_ || !dataset_) {
            release();
            if (cudaMalloc(&dataset_, bytes) != cudaSuccess) {
                fprintf(stderr, "[autolykos2] cudaMalloc %.2f GB failed\n",
                        bytes / 1e9);
                return false;
            }
            n_ = N;
        }

        // The dataset depends on the height, so it is rebuilt on every epoch
        // change even when N itself has not moved.
        al_build_dataset<<<(N + 255) / 256, 256>>>(dataset_, N, (uint32_t)job.epoch);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "[autolykos2] dataset build failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        height_ = (uint32_t)job.epoch;

        ensureScratch();
        return true;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        if (!dataset_) return false;
        ensureScratch();

        cudaMemcpy(dMsg_, job.msg, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(dTarget_, job.target, 32, cudaMemcpyHostToDevice);
        cudaMemset(dCount_, 0, sizeof(uint32_t));

        const int threads = 256;
        uint32_t blocks = (uint32_t)(count / threads);
        if (blocks == 0) blocks = 1;

        al_search<<<blocks, threads>>>(dataset_, dMsg_, nonceBase, n_, height_,
                                       dTarget_, dSol_, dCount_, kMaxSolutions);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "[autolykos2] search failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }

        uint32_t found = 0;
        cudaMemcpy(&found, dCount_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (found == 0) return true;
        if (found > kMaxSolutions) found = kMaxSolutions;

        std::vector<AlSolution> host(found);
        cudaMemcpy(host.data(), dSol_, sizeof(AlSolution) * found,
                   cudaMemcpyDeviceToHost);
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
        uint64_t *dOut = nullptr;
        if (cudaMalloc(&dOut, 32) != cudaSuccess) return false;
        cudaMemcpy(dMsg_, job.msg, 32, cudaMemcpyHostToDevice);
        al_verify_one<<<1, 1>>>(dataset_, dMsg_, sol.nonce, n_, height_, dOut);
        cudaDeviceSynchronize();
        uint64_t hit[4];
        cudaMemcpy(hit, dOut, 32, cudaMemcpyDeviceToHost);
        cudaFree(dOut);

        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        for (int i = 3; i >= 0; i--) {
            if (hit[i] != job.target[i]) return hit[i] < job.target[i];
        }
        return false;
    }

    void release() override {
        if (dataset_) { cudaFree(dataset_); dataset_ = nullptr; }
        n_ = 0;
    }

    ~Autolykos2() override {
        release();
        if (dMsg_) cudaFree(dMsg_);
        if (dTarget_) cudaFree(dTarget_);
        if (dSol_) cudaFree(dSol_);
        if (dCount_) cudaFree(dCount_);
    }

   private:
    static const uint32_t kMaxSolutions = 32;

    void ensureScratch() {
        if (!dMsg_) cudaMalloc(&dMsg_, 32);
        if (!dTarget_) cudaMalloc(&dTarget_, 32);
        if (!dSol_) cudaMalloc(&dSol_, sizeof(AlSolution) * kMaxSolutions);
        if (!dCount_) cudaMalloc(&dCount_, sizeof(uint32_t));
    }

    uint4 *dataset_ = nullptr;
    uint32_t n_ = 0;
    uint32_t height_ = 0;
    mutable uint8_t *dMsg_ = nullptr;
    uint64_t *dTarget_ = nullptr;
    AlSolution *dSol_ = nullptr;
    uint32_t *dCount_ = nullptr;
};

}  // namespace

Algorithm *makeAutolykos2() { return new Autolykos2(); }

}  // namespace om

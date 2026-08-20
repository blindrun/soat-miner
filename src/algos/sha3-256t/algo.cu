// SHA3-256t (BitcoinIII / BC3) - the Algorithm implementation.
//
// The memoryless case the algo.h interface anticipated: prepare() has nothing
// to build, so it is a no-op and every epoch-change path in the run loop
// collapses to nothing. All the state is 80 bytes of header that arrives with
// the job.
//
// Verified against mainnet block 50204 (tests/test_sha3_hit).

#include <cstdio>
#include <cstring>

#include <string>
#include <vector>

#include "../../core/algo.h"
#include "../../core/btc_job.h"
#include "mine.cuh"

namespace om {
namespace {

class Sha3_256t : public Algorithm {
   public:
    Sha3_256t() { cudaStreamCreateWithFlags(&mine_, cudaStreamNonBlocking); }

    const char *name() const override { return "sha3-256t"; }

    size_t memoryBytes(const Job &) const override {
        return sizeof(S3Solution) * kMaxSolutions + sizeof(uint32_t);
    }

    /** Nothing to precompute. SHA3-256t has no dataset, no table and no epoch
     *  state - the entire input is the 80-byte header. */
    bool prepare(const Job &) override {
        ensureScratch();
        return dSol_ != nullptr && dCount_ != nullptr;
    }

    bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                std::vector<Solution> *out) override {
        s3::Header hdr;
        if (!loadJobHeader(job, &hdr)) {
            fprintf(stderr, "[sha3-256t] job carries no usable 80-byte header\n");
            return false;
        }
        ensureScratch();
        if (!dSol_ || !dCount_) return false;

        // A batch that would run past the 32-bit nonce space is clamped by the
        // run loop (nonceBitsOwned() == 32), so this only guards the bench
        // path, which owns the whole space and passes whatever it likes.
        uint64_t n = count;
        if (n > (1ULL << 32)) n = 1ULL << 32;
        const uint32_t base = (uint32_t)nonceBase;
        const uint32_t cnt = (uint32_t)n;

        cudaMemsetAsync(dCount_, 0, sizeof(uint32_t), mine_);

        const int threads = 256;
        uint32_t blocks = (uint32_t)((n + threads - 1) / threads);
        if (blocks == 0) blocks = 1;

        s3_search<<<blocks, threads, 0, mine_>>>(
            hdr, base, cnt, job.target[0], job.target[1], job.target[2],
            job.target[3], dSol_, dCount_, kMaxSolutions);

        uint32_t found = 0;
        cudaMemcpyAsync(&found, dCount_, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                        mine_);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) {
            fprintf(stderr, "[sha3-256t] search failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (found == 0) return true;

        if (found > kMaxSolutions) {
            fprintf(stderr,
                    "[sha3-256t] warning: %u solutions in one batch exceeds the "
                    "%u-solution buffer; %u dropped. Difficulty may be too low.\n",
                    found, kMaxSolutions, found - kMaxSolutions);
            found = kMaxSolutions;
        }

        std::vector<S3Solution> host(found);
        cudaMemcpyAsync(host.data(), dSol_, sizeof(S3Solution) * found,
                        cudaMemcpyDeviceToHost, mine_);
        if (cudaStreamSynchronize(mine_) != cudaSuccess) return false;

        for (uint32_t i = 0; i < found; i++) {
            Solution s;
            s.nonce = host[i].nonce;
            memcpy(s.hit, host[i].hit, sizeof(s.hit));
            out->push_back(s);
        }
        return true;
    }

    /**
     * Re-hash on the CPU, not on the device.
     *
     * Autolykos re-runs its verification kernel because recomputing a 7 GB
     * dataset lookup on the host would be absurd. Here the whole hash is three
     * Keccak permutations, so the host can do it outright - and a host recompute
     * is a genuinely independent check. Asking the same unstable GPU to confirm
     * its own answer is how a bad hit gets confirmed twice and submitted.
     */
    bool verify(const Job &job, const Solution &sol) const override {
        s3::Header hdr;
        if (!loadJobHeader(job, &hdr)) return false;
        uint64_t hit[4];
        s3::hash(hdr, (uint32_t)sol.nonce, hit);
        if (memcmp(hit, sol.hit, sizeof(hit)) != 0) return false;
        return s3::underTarget(hit, job.target);
    }

    void release() override {}

    ~Sha3_256t() override {
        if (dSol_) cudaFree(dSol_);
        if (dCount_) cudaFree(dCount_);
        if (mine_) cudaStreamDestroy(mine_);
    }

   private:
    // Same 256 as Autolykos, for the same reason: a deliberately low test
    // difficulty can find far more than a handful in one batch.
    static const uint32_t kMaxSolutions = 256;

    /**
     * The 80-byte header the job was built around.
     *
     * The benchmark path is the one case where a job arrives without one:
     * run.cpp fills in msg, epoch and a zero target and nothing else, because
     * Ergo needs nothing else. Rather than teach the shared loop about Bitcoin
     * headers, synthesise one here from msg. That is safe precisely because
     * bench sets an all-zero target, so no synthetic header can ever produce a
     * reportable share - and the hash rate is identical either way, since
     * Keccak's cost does not depend on its input.
     *
     * A job that has an extra but a malformed one is a real error and fails.
     */
    static bool loadJobHeader(const Job &job, s3::Header *hdr) {
        uint8_t bytes[kBtcHeaderBytes];
        if (btcJobHeader(job.extra, bytes)) {
            s3::loadHeader(bytes, hdr);
            return true;
        }
        if (!job.extra.empty()) return false;
        memset(bytes, 0, sizeof(bytes));
        bytes[0] = 0x00; bytes[1] = 0x10; bytes[2] = 0x00; bytes[3] = 0x20;
        memcpy(bytes + 36, job.msg, 32);
        s3::loadHeader(bytes, hdr);
        return true;
    }

    void ensureScratch() const {
        if (!dSol_) cudaMalloc(&dSol_, sizeof(S3Solution) * kMaxSolutions);
        if (!dCount_) cudaMalloc(&dCount_, sizeof(uint32_t));
    }

    cudaStream_t mine_ = nullptr;
    mutable S3Solution *dSol_ = nullptr;
    mutable uint32_t *dCount_ = nullptr;
};

}  // namespace

Algorithm *makeSha3_256t() { return new Sha3_256t(); }

}  // namespace om

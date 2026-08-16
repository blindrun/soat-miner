// The algorithm-agnostic run loop, shared by every backend binary.
//
// Both the CUDA build and the OpenCL build link this; the only difference
// between them is which Algorithm gets constructed and handed over.

#pragma once

#include <signal.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "algo.h"
#include "http.h"
#include "platform.h"
#include "telemetry.h"

namespace om {

/**
 * Where work comes from and where solutions go.
 *
 * This is the seam for pool support: a stratum client is another
 * implementation of this interface, and nothing else has to change.
 */
class JobSource {
   public:
    virtual ~JobSource() = default;
    virtual const char *describe() const = 0;
    virtual bool fetch(Job *job) = 0;
    virtual bool submit(const Job &job, const Solution &sol, std::string *err) = 0;

    /**
     * Share counts as confirmed by the far end, for sources that answer.
     * Returning false means "this source cannot tell you" - which is the
     * honest answer for solo, where the node's HTTP reply is synchronous.
     */
    virtual bool poolCounters(uint64_t *accepted, uint64_t *rejected,
                              uint64_t *pending, std::string *lastError) const {
        (void)accepted; (void)rejected; (void)pending; (void)lastError;
        return false;
    }
};

/** Solo mining against an Ergo node's /mining API. */
class ErgoNodeSource : public JobSource {
   public:
    explicit ErgoNodeSource(HttpTarget t) : t_(std::move(t)) {
        desc_ = "ergo node " + t_.host + ":" + std::to_string(t_.port) + " (solo)";
    }

    const char *describe() const override { return desc_.c_str(); }

    bool fetch(Job *job) override {
        bool ok = false;
        const std::string body =
            httpRequest(t_, "GET", "/mining/candidate", "", &ok);
        if (!ok) return false;

        std::string msgHex, bStr, hStr;
        if (!jsonString(body, "msg", &msgHex)) return false;
        if (!jsonNumber(body, "b", &bStr)) return false;
        if (!jsonNumber(body, "h", &hStr)) return false;
        jsonString(body, "pk", &job->extra);

        if (!hexToBytes(msgHex, job->msg, 32)) return false;
        decimalToLimbs(bStr, job->target);
        job->epoch = strtoull(hStr.c_str(), nullptr, 10);
        job->valid = true;
        return true;
    }

    bool submit(const Job &job, const Solution &sol, std::string *err) override {
        char nhex[17];
        snprintf(nhex, sizeof(nhex), "%016llx", (unsigned long long)sol.nonce);
        const std::string payload =
            "{\"pk\":\"" + job.extra + "\",\"n\":\"" + std::string(nhex) + "\"}";
        bool ok = false;
        const std::string r =
            httpRequest(t_, "POST", "/mining/solution", payload, &ok);
        if (!ok) *err = r;
        return ok;
    }

   private:
    HttpTarget t_;
    std::string desc_;
};

struct RunOptions {
    HttpTarget target;
    uint64_t batch = 1ULL << 22;
    bool bench = false;
    bool plain = false;
    uint64_t benchEpoch = 1851444;
    int reportSeconds = 5;
    std::string backendLabel;

    // Build the next height's dataset in the background: -1 auto, 0 off, 1 on.
    int prefetch = -1;

    // Memory clock offset in MHz of transfer rate. CUDA and Vulkan both force
    // performance state P2, which runs memory below its rated speed, and this
    // is what puts it back. 0 leaves the card alone.
    int memOffsetMhz = 0;

    // Benchmark only: pretend a new block arrives this often, so the cost of
    // an epoch change is included in the average. 0 means the height is fixed.
    int benchEpochSeconds = 0;

    // Pool mode. When poolHost is set, a StratumSource is used instead of the
    // node source.
    std::string poolHost;
    int poolPort = 0;
    std::string wallet;
    std::string worker = "soat";
    std::string password = "x";

    // Lithos mode. Lithos is not a coin or an algorithm: it is a decentralised
    // pool protocol whose reference client runs a local stratum server (default
    // 127.0.0.1:4444) speaking ordinary Ergo/Autolykos v2 stratum. The PoW,
    // the dataset and the kernels are unchanged.
    //
    // What does change is that rewards are settled on-chain from Non-Interactive
    // Share Proofs against the node the Lithos client is attached to, so the
    // stratum address is NOT a payout identifier the way it is on a
    // conventional pool. Demanding a valid Ergo address here would reject a
    // perfectly good Lithos setup.
    bool lithos = false;
};

/** Drives an already-constructed algorithm until *stop becomes non-zero. */
int runMiner(Algorithm *algo, const RunOptions &opt, const char *gpuName,
             double gpuMemGB, const char *archLabel, volatile sig_atomic_t *stop);

}  // namespace om

// The interface every algorithm implements.
//
// Design notes, since the shape here is what decides how painful the second
// algorithm is to add:
//
//  * Algorithms own their own memory. Autolykos v2 wants a 7 GB dataset
//    rebuilt when the height changes; Ethash-family algorithms want a DAG per
//    epoch; a memoryless algorithm wants nothing at all. Rather than model
//    "dataset" in the core, the core just calls prepare() when the job's epoch
//    key changes and lets the algorithm decide what that means.
//
//  * Jobs carry a 32-byte message and a 256-bit target because that covers
//    every PoW worth supporting. Anything algorithm-specific (Ergo's `pk`,
//    a pool's job id) rides along in `extra` and is opaque to the core.
//
//  * search() is synchronous and batched. The core owns the nonce counter, the
//    stop signal and the reporting, so an algorithm is only ever responsible
//    for "test these nonces, tell me which ones win".

#pragma once

#include <stdint.h>

#include <string>
#include <vector>

namespace om {

/** A unit of work handed out by a node or pool. */
struct Job {
    uint8_t msg[32] = {};      ///< header digest to hash against
    uint64_t target[4] = {};   ///< 256-bit target, limb[0] least significant
    uint64_t epoch = 0;        ///< height/epoch; a change triggers prepare()
    std::string extra;         ///< algorithm-specific payload (e.g. Ergo pk)
    bool valid = false;
};

/** A nonce whose hit fell below the job's target. */
struct Solution {
    uint64_t nonce = 0;
    uint64_t hit[4] = {};
};

class Algorithm {
   public:
    virtual ~Algorithm() = default;

    /** Short lowercase identifier, e.g. "autolykos2". */
    virtual const char *name() const = 0;

    /** Device memory this algorithm will allocate for `job`, for reporting. */
    virtual size_t memoryBytes(const Job &job) const = 0;

    /**
     * Called once per epoch change (and once at startup). Build whatever
     * precomputed state the algorithm needs. Returns false on failure.
     */
    virtual bool prepare(const Job &job) = 0;

    /**
     * Test `count` nonces starting at `nonceBase`. Appends any winners to
     * `out`. Returns false only on an unrecoverable device error.
     */
    virtual bool search(const Job &job, uint64_t nonceBase, uint64_t count,
                        std::vector<Solution> *out) = 0;

    /**
     * Host-side re-verification of a solution before it is submitted.
     * A GPU that is overclocked into instability produces wrong hits; this is
     * what stops those being sent upstream as invalid shares.
     */
    virtual bool verify(const Job &job, const Solution &sol) const = 0;

    /** Release device memory. */
    virtual void release() = 0;
};

/** Returns the algorithm registered under `name`, or nullptr. */
Algorithm *createAlgorithm(const std::string &name);

/** Names of every compiled-in algorithm. */
std::vector<std::string> availableAlgorithms();

}  // namespace om

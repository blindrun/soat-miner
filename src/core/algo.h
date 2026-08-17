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
    std::string jobId;         ///< pool's job id; travels with the job so a
                               ///< solution is submitted under the id it was
                               ///< computed against, not whatever arrived since
    bool valid = false;
};

/** A nonce whose hit fell below the job's target. */
struct Solution {
    uint64_t nonce = 0;
    uint64_t hit[4] = {};

    /**
     * Algorithm-specific payload for the submission, opaque to the core - the
     * mirror of Job::extra on the way back out.
     *
     * A nonce is enough to describe a win for every hash-and-compare PoW, and
     * is not enough for all of them. Pearl's win is a tile of a matrix product
     * the miner chose, and what gets submitted is a Merkle opening of sixteen
     * rows and sixteen columns - tens of kilobytes that no fixed field can
     * hold. Empty for algorithms whose nonce says everything.
     */
    std::string extra;
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

    /**
     * Opt in to precomputing the *next* epoch's state in the background, so a
     * new job does not stall mining while it is built.
     *
     * Mode: -1 decide automatically (build ahead only when the memory is
     * free), 0 never, 1 always. Default is a no-op, so a backend that has not
     * implemented it is unaffected.
     */
    virtual void setPrefetch(int mode) { (void)mode; }

    /** True when the last prepare() was served from a prefetched buffer. */
    virtual bool servedFromPrefetch() const { return false; }

    /**
     * True when prepare(job) will be instant because the state is already
     * built. Only used to decide whether to warn the user about a wait.
     */
    virtual bool prefetchReadyFor(const Job &job) const { (void)job; return false; }

    /** One line on what prefetch is doing, or empty if it is not running. */
    virtual std::string prefetchNote() const { return std::string(); }

    /** Release device memory. */
    virtual void release() = 0;
};

/** Returns the algorithm registered under `name`, or nullptr. */
Algorithm *createAlgorithm(const std::string &name);

/** Names of every compiled-in algorithm. */
std::vector<std::string> availableAlgorithms();

}  // namespace om

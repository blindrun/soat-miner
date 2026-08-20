// Bitcoin Stratum V1 client - the job source for BitcoinIII (BC3).
//
// A sibling of StratumSource, not a modification of it. run.h calls JobSource
// "the seam for pool support", and this is that seam used as intended: the two
// protocols share nothing but a socket idiom.
//
// Why a second client at all. Ergo stratum hands the miner a finished 32-byte
// message and asks it to roll a nonce. Bitcoin stratum hands out job PARTS and
// the miner assembles the block header itself:
//
//   -> {"id":1,"method":"mining.subscribe","params":["soat-miner/0.1"]}
//   <- {"id":1,"result":[[..subs..],"a6e60e12",8],"error":null}
//                                     ^extranonce1  ^extranonce2 size
//   -> {"id":2,"method":"mining.authorize","params":["<address>.<worker>","x"]}
//   <- {"id":null,"method":"mining.set_difficulty","params":[0.01]}
//   <- {"id":null,"method":"mining.notify","params":[
//         "<jobId>","<prevhash>","<coinb1>","<coinb2>",[<merkle branch>],
//         "<version>","<nbits>","<ntime>",<cleanJobs>]}
//   -> {"id":N,"method":"mining.submit","params":[
//         "<login>","<jobId>","<extranonce2 hex>","<ntime hex>","<nonce hex>"]}
//
//   coinbase    = coinb1 || extranonce1 || extranonce2 || coinb2
//   merkleRoot  = fold(SHA256d(coinbase), branch)      <- SHA256d, NOT SHA3
//   header      = version || prevhash || merkleRoot || ntime || nbits || nonce
//
// Four details that are easy to get wrong, all of them silent:
//
//  * prevhash arrives with each 4-BYTE WORD reversed, and nothing else about
//    it reversed. Every other hex scalar (version, nbits, ntime, and the nonce
//    on the way back) is plain big-endian hex of its numeric value and goes
//    into the header little-endian. merkle_branch entries are raw internal
//    order and go in untouched. Getting prevhash wrong produces a perfectly
//    well-formed header that hashes to nothing anybody wants.
//
//  * the merkle tree is still SHA-256d. BC3 changed the header hash to
//    SHA3-256t and changed nothing else, so reaching for SHA3 here builds a
//    header no other node agrees with.
//
//  * BC3's header nonce is 32 bits, and a 4090 exhausts 2^32 in about two
//    seconds. Rolling extranonce2 is therefore mandatory, not an optimisation -
//    without it the miner re-mines the same space forever and the pool rejects
//    everything after the first pass as duplicate. See rollExtranonce2().
//
//  * consensus requires version bit 12 (SHA3_VBIT) on every post-fork block,
//    and the standard BIP310 version-rolling mask 0x1fffe000 contains bit 12.
//    So this client never sends mining.configure and never rolls the version:
//    a pool that granted version rolling could hand out a mask that lets the
//    miner clear the one bit the chain requires.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo.h"
#include "http.h"
#include "platform.h"
#include "run.h"

namespace om {

class BitcoinStratumSource : public JobSource {
   public:
    BitcoinStratumSource(std::string host, int port, std::string wallet,
                         std::string worker, std::string password,
                         uint64_t batch)
        : host_(std::move(host)),
          port_(port),
          wallet_(std::move(wallet)),
          worker_(std::move(worker)),
          password_(std::move(password)),
          batch_(batch < 256 ? 256 : batch) {
        desc_ = host_ + ":" + std::to_string(port_) + " (pool)";
        login_ = wallet_ + (worker_.empty() ? "" : "." + worker_);
    }

    ~BitcoinStratumSource() override { stop(); }

    const char *describe() const override { return desc_.c_str(); }

    bool start(std::string *err);
    bool reconnect();
    void stop();

    bool fetch(Job *job) override;
    bool submit(const Job &job, const Solution &sol, std::string *err) override;
    bool poolCounters(uint64_t *accepted, uint64_t *rejected, uint64_t *pending,
                      std::string *lastError) const override;

    /** The whole 32-bit header nonce is ours; the pool's share of the search
     *  space is extranonce1, which lives in the coinbase, not in the nonce. */
    uint64_t noncePrefix() const { return 0; }
    int nonceBitsOwned() const { return 32; }

    bool loginRejected() const { return loginRejected_.load(); }
    std::string loginError() const {
        std::lock_guard<std::mutex> lk(mu_);
        return loginError_;
    }

    std::string takeJobWarning();
    std::string takeSubmitVerdict();

   private:
    friend struct BitcoinStratumTestAccess;

    void readerLoop();
    bool sendLine(const std::string &s);
    void handleLine(const std::string &line);
    void resetSession();

    /** Rebuilds coinbase, merkle root and header for the current extranonce2.
     *  Caller holds mu_. */
    bool buildHeader();

    /** Advances extranonce2 and rebuilds. Caller holds mu_. */
    void rollExtranonce2();

    /** Block height from the coinbase's BIP34 push. Caller holds mu_. */
    uint64_t coinbaseHeight() const;

    /**
     * Half the 32-bit nonce space per extranonce2.
     *
     * The run loop starts its counter at an arbitrary point rather than zero,
     * so a budget of the full 2^32 could wrap and re-issue nonces already
     * mined under this extranonce2. Half guarantees it cannot, and extranonce2
     * is eight bytes here, so throwing away half the nonce space costs
     * precisely nothing.
     */
    static const uint64_t kNonceBudget = 1ULL << 31;

    std::string host_, wallet_, worker_, password_, desc_, login_;
    int port_ = 0;
    uint64_t batch_ = 1ULL << 22;
    socket_t fd_ = OM_INVALID_SOCKET;

    std::thread reader_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};

    mutable std::mutex mu_;

    // Subscription state.
    std::vector<uint8_t> extranonce1_;
    size_t extranonce2Size_ = 4;
    bool subscribed_ = false;

    // The current template, exactly as mining.notify described it.
    std::string jobId_;
    uint8_t prevHash_[32] = {};   ///< already in header byte order
    std::vector<uint8_t> coinb1_, coinb2_;
    std::vector<std::string> branch_;  ///< 32 raw bytes each
    uint32_t version_ = 0, nbits_ = 0, ntime_ = 0;
    bool haveTemplate_ = false;

    // The job handed out, and what it was built from.
    Job current_;
    uint64_t extranonce2_ = 0;
    std::string extranonce2Bytes_;
    uint64_t fetchesSinceRoll_ = 0;
    bool haveJob_ = false;

    uint64_t shareTarget_[4] = {};
    double difficulty_ = 1.0;
    bool haveDifficulty_ = false;

    std::string jobWarning_;

    static const int kFirstSubmitId = 10;
    std::atomic<int> nextId_{kFirstSubmitId};
    std::atomic<bool> loginRejected_{false};
    std::string loginError_;

    mutable std::mutex submitMu_;
    uint64_t submitted_ = 0, accepted_ = 0, rejected_ = 0;
    std::string lastSubmitError_, verdict_;
};

}  // namespace om

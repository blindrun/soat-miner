// KawPow stratum client.
//
// A separate client from src/core/stratum.h rather than a mode inside it. The
// Ergo client is under active work in another branch, and more importantly the
// two protocols disagree about the thing that matters most: what a share is.
// An Ergo share is a nonce. A KawPow share is a nonce AND the mix hash, which
// is 32 bytes the algorithm produces and no other algorithm here has.
//
// Two flavours, both observed live
// --------------------------------
// tools/probe-stratum.py found that the flavour belongs to the ENDPOINT, not
// to the pool - ravencoin.herominers.com answers differently on 1140 and 1150,
// reproducibly:
//
//   kStratum    subscribe -> {"result":[null,"36c3"]}
//               submit    -> [login, jobId, "0x"+nonce, "0x"+header, "0x"+mix]
//               This is what Ravencoin pools document as stratum+tcp, and it
//               is the default here.
//
//   kNiceHash   subscribe -> {"result":[["mining.notify",id,
//                             "EthereumStratum/1.0.0"],"adb4"]}
//               submit    -> [login, jobId, nonce with the extranonce prefix
//                             REMOVED]
//
// The flavour is detected from the subscribe reply and can be forced with
// --stratum-mode, because a pool that answers in a shape nobody has seen
// should be servable without a rebuild.
//
// mining.notify carries, in order:
//
//   [jobId, headerHash, seedHash, shareTarget, cleanJobs, height, bits]
//
// The height is load-bearing and is why this client exists at all: KawPow's
// random program changes every 3 blocks, so a client that keeps only the epoch
// mines the wrong program for up to 7499 blocks while looking healthy. It goes
// into Job::extra - see src/algos/kawpow/job.h.
//
// The extranonce trap
// -------------------
// The extranonce is a PREFIX of the 8-byte nonce, and its width is counted in
// HEX CHARACTERS, not bytes. kawpowminer's own field is named
// extraNonceSizeBytes with a comment from its author saying the name is wrong.
// A 4-character extranonce leaves the miner 48 bits, not 32. Read it as bytes
// and every share is out of range, with a pool-side error that says nothing
// about why.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo.h"
#include "platform.h"
#include "run.h"

namespace om {

class KawpowStratumSource : public JobSource {
   public:
    enum Mode {
        kAuto = 0,     ///< decide from the subscribe reply
        kStratum,      ///< nonce + header + mix, Ravencoin's documented default
        kNiceHash,     ///< truncated nonce, EthereumStratum/1.0.0
    };

    KawpowStratumSource(std::string host, int port, std::string wallet,
                        std::string worker, std::string password,
                        Mode mode = kAuto)
        : host_(std::move(host)),
          port_(port),
          wallet_(std::move(wallet)),
          worker_(std::move(worker)),
          password_(std::move(password)),
          requestedMode_(mode),
          mode_(mode == kAuto ? kStratum : mode) {
        desc_ = host_ + ":" + std::to_string(port_) + " (pool)";
        login_ = wallet_ + (worker_.empty() ? "" : "." + worker_);
    }

    ~KawpowStratumSource() override { stop(); }

    const char *describe() const override { return desc_.c_str(); }

    bool start(std::string *err);
    bool reconnect();
    void stop();

    bool fetch(Job *job) override;
    bool submit(const Job &job, const Solution &sol, std::string *err) override;

    bool loginRejected() const { return loginRejected_.load(); }
    std::string loginError() const {
        std::lock_guard<std::mutex> lk(mu_);
        return loginError_;
    }

    /** Nonce prefix imposed by the extranonce, and how many bits are ours. */
    uint64_t noncePrefix() const {
        std::lock_guard<std::mutex> lk(mu_);
        return noncePrefix_;
    }
    int nonceBitsOwned() const {
        std::lock_guard<std::mutex> lk(mu_);
        return nonceBitsOwned_;
    }
    uint64_t extranonceGeneration() const { return extranonceGen_.load(); }

    Mode mode() const { return mode_; }

    /** Pops a one-off warning about an unusable job, so it is logged once. */
    std::string takeJobWarning() override;
    std::string takeRejectReason() override;

    bool poolCounters(uint64_t *accepted, uint64_t *rejected, uint64_t *pending,
                      std::string *lastError) const override;

    /**
     * Exposed for tests: hand the client one line as if it arrived from the
     * pool. The protocol is the part worth testing and it does not need a
     * socket to be tested, so the test drives this directly.
     */
    void feedLineForTest(const std::string &line) { handleLine(line); }

    /** Exposed for tests: the job as it stands, without reconnecting. */
    bool currentJobForTest(Job *job) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (!haveJob_) return false;
        *job = current_;
        return true;
    }

    /** Exposed for tests: the last line this client would have sent. */
    std::string lastSentForTest() const {
        std::lock_guard<std::mutex> lk(sendMu_);
        return lastSent_;
    }

   private:
    void readerLoop();
    bool sendLine(const std::string &s);
    void handleLine(const std::string &line);
    void handleSubscribeResult(const std::string &result);
    void handleNotify(const std::string &params);
    void applyExtranonce(const std::string &xnHex);

    std::string host_, wallet_, worker_, password_, desc_, login_;
    int port_ = 0;
    socket_t fd_ = OM_INVALID_SOCKET;

    std::thread reader_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};

    mutable std::mutex mu_;
    Job current_;
    bool haveJob_ = false;
    std::string jobWarning_;

    Mode requestedMode_ = kAuto;
    Mode mode_ = kStratum;

    uint64_t noncePrefix_ = 0;
    int nonceBitsOwned_ = 64;
    /** Hex characters, not bytes. See the note at the top of this file. */
    int extranonceNibbles_ = 0;
    std::atomic<uint64_t> extranonceGen_{0};

    static const int kFirstSubmitId = 10;
    std::atomic<int> nextId_{kFirstSubmitId};
    std::atomic<bool> loginRejected_{false};
    std::string loginError_;

    mutable std::mutex sendMu_;
    std::string lastSent_;
    uint64_t submitted_ = 0, accepted_ = 0, rejected_ = 0;
    std::string lastSubmitError_, verdict_;
    mutable std::mutex submitMu_;
};

}  // namespace om

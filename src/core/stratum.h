// Ergo stratum client.
//
// The protocol below was captured from live pools (herominers, woolypooly)
// rather than taken from documentation, because Ergo stratum is a de-facto
// standard with no spec:
//
//   -> {"id":1,"method":"mining.subscribe","params":["agent","EthereumStratum/1.0.0"]}
//   <- {"id":1,"error":null,"result":[<subs>,"6275",6]}
//   -> {"id":2,"method":"mining.authorize","params":["<address>.<worker>","x"]}
//   <- {"id":2,"error":null,"result":true}
//   <- {"id":null,"method":"mining.set_difficulty","params":[1]}
//   <- {"id":null,"method":"mining.notify","params":[
//         "<jobId>", <height>, "<msg 64 hex>", "", "", <n>,
//         "<target as DECIMAL string>", "", <cleanJobs>]}
//   -> {"id":N,"method":"mining.submit","params":["<address>.<worker>","<jobId>","<nonce 16 hex>"]}
//
// Two details that are easy to get wrong:
//
//  * params[6] of mining.notify is the target `b` in DECIMAL, not hex, and not
//    a difficulty. It goes straight into the same 256-bit compare the solo
//    path uses.
//  * the subscribe result's extranonce is a PREFIX of the 8-byte nonce. The
//    miner only owns the remaining bytes, so nonces must be generated inside
//    that subspace or every share is rejected as out of range.

#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#include "algo.h"
#include "http.h"
#include "platform.h"
#include "run.h"

namespace om {

class StratumSource : public JobSource {
   public:
    StratumSource(std::string host, int port, std::string wallet,
                  std::string worker, std::string password)
        : host_(std::move(host)),
          port_(port),
          wallet_(std::move(wallet)),
          worker_(std::move(worker)),
          password_(std::move(password)) {
        desc_ = host_ + ":" + std::to_string(port_) + " (pool)";
        login_ = wallet_ + (worker_.empty() ? "" : "." + worker_);
    }

    ~StratumSource() override { stop(); }

    const char *describe() const override { return desc_.c_str(); }

    /** Connects, subscribes, authorizes, and starts the reader thread. */
    bool start(std::string *err);
    bool reconnect();
    void stop();

    bool fetch(Job *job) override {
        if (!connected_) {
            // Reconnect in place rather than making the caller care.
            if (!reconnect()) return false;
            for (int i = 0; i < 50 && connected_; i++) {
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (haveJob_) break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (!connected_ || !haveJob_) return false;
        *job = current_;
        return true;
    }

    bool submit(const Job &job, const Solution &sol, std::string *err) override;

    /** Set if the pool explicitly refused mining.authorize. */
    bool loginRejected() const { return loginRejected_.load(); }
    std::string loginError() const {
        std::lock_guard<std::mutex> lk(mu_);
        return loginError_;
    }

    /** Nonce prefix imposed by the pool's extranonce, and how many bits we own. */
    uint64_t noncePrefix() const { return noncePrefix_; }
    int nonceBitsOwned() const { return nonceBitsOwned_; }

    /** What the POOL said about our shares, not what we hoped it said. */
    bool poolCounters(uint64_t *accepted, uint64_t *rejected, uint64_t *pending,
                      std::string *lastError) const override;

    /** Pops the most recent rejection reason, so it is logged exactly once. */
    std::string takeSubmitVerdict();

   private:
    void readerLoop();
    bool sendLine(const std::string &s);
    void handleLine(const std::string &line);

    std::string host_, wallet_, worker_, password_, desc_, login_;
    int port_ = 0;
    socket_t fd_ = OM_INVALID_SOCKET;

    std::thread reader_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};

    mutable std::mutex mu_;
    Job current_;
    std::string jobId_;
    bool haveJob_ = false;

    uint64_t noncePrefix_ = 0;
    int nonceBitsOwned_ = 64;

    /** Submit request ids start here, so a reply is identifiable by id alone. */
    static const int kFirstSubmitId = 10;
    /** params[3] is nTime. Ergo's mining.notify carries none, and pools only
     *  use it to de-duplicate submissions, so a constant is correct here. */
    static constexpr const char *kNTime = "00000000";

    std::atomic<int> nextId_{kFirstSubmitId};
    std::atomic<bool> loginRejected_{false};
    std::string loginError_;
    uint64_t submitted_ = 0, accepted_ = 0, rejected_ = 0;
    std::string lastSubmitError_, verdict_;
    std::mutex submitMu_;
};

}  // namespace om

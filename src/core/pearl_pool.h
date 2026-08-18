// Pearl pool mining.
//
// Pearl's own tooling has no pool path at all: pearl-gateway talks to a node
// you run yourself, holds the payout address, and turns a submitted PlainProof
// into the zero-knowledge certificate a block needs. That is fine if you run a
// node. Most people will not.
//
// Pools exist anyway, and they speak a stratum of their own. This is the wire,
// established by probing a live pool because nothing documents it:
//
//   -> {"id":1,"method":"mining.subscribe","params":{"agent":"soat-miner/x"}}
//   <- {"id":1,"error":null,"result":true}
//   -> {"id":2,"method":"mining.authorize","params":{"wallet":"prl1..."}}
//   <- {"id":2,"error":null,"result":true}
//   <- {"id":null,"method":"mining.notify","params":{
//         "job_id":"00000000_2097152",
//         "header":"<152 hex chars, the 76-byte incomplete header>",
//         "target":"<64 hex chars, big endian>",
//         "height":101009,"cert_version":3}}
//   -> {"id":9,"method":"mining.submit","params":{
//         "job_id":"...","plain_proof":"<base64 bincode PlainProof>"}}
//
// Three things are worth knowing before changing any of it:
//
//   * PARAMS ARE OBJECTS, not arrays. Sending stratum's usual positional array
//     gets {"code":20,"msg":"params must be an object"}.
//   * THE WALLET FIELD IS `wallet`. Not `login`, not `user` - both of those
//     authorize "successfully" and then fail with {"code":24,"Wallet is
//     missing"}, which is a confusing way to find out.
//   * The proof we already build for the gateway is EXACTLY what the pool
//     wants. `plain_proof` is the same base64 bincode PlainProof. Nothing
//     about the mining changes, only the transport.
//
// The pool PUSHES work rather than answering polls, so fetch() drains whatever
// has arrived and reports the newest job it has seen, instead of asking.

#pragma once

#include <stdio.h>
#include <string.h>

#include <chrono>
#include <string>

#include "algo.h"
#include "http.h"
#include "platform.h"
#include "pearl_gateway.h"   // packPearlExtra, unpackPearlExtra, base64/b3
#include "run.h"

namespace om {

/** Big-endian hex, as the pool sends targets, into the little-endian limbs
 *  Job::target carries. Getting this backwards yields a target that is either
 *  absurdly easy or unreachable, and both look like "the pool hates us". */
inline bool hexToLimbs(const std::string &hex, uint64_t out[4]) {
    uint8_t b[32];
    if (!hexToBytes(hex, b, 32)) return false;
    for (int limb = 0; limb < 4; limb++) {
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | b[(3 - limb) * 8 + i];
        out[limb] = v;
    }
    return true;
}

/** The job id rides in `extra` behind the header and cert version, so submit()
 *  can name the job the proof belongs to. unpackPearlExtra still parses the
 *  first two fields unchanged, so a gateway job and a pool job share a format. */
inline std::string packPoolExtra(const uint8_t header[76], int cert,
                                 const std::string &jobId) {
    return packPearlExtra(header, cert) + ":" + jobId;
}
inline std::string poolJobId(const std::string &extra) {
    const size_t first = extra.find(':');
    if (first == std::string::npos) return "";
    const size_t second = extra.find(':', first + 1);
    return second == std::string::npos ? "" : extra.substr(second + 1);
}

class PearlPoolSource : public JobSource {
   public:
    PearlPoolSource(std::string host, int port, std::string wallet,
                    std::string worker)
        : host_(std::move(host)), port_(port), wallet_(std::move(wallet)),
          worker_(std::move(worker)) {
        desc_ = host_ + ":" + std::to_string(port_) + " (pearl pool)";
    }

    ~PearlPoolSource() override { disconnect(); }

    const char *describe() const override { return desc_.c_str(); }

    bool fetch(Job *job) override {
        if (fd_ == OM_INVALID_SOCKET && !connectAndLogin()) return false;
        // ZERO, not 50 ms, once we have work. The core calls fetch() before
        // every search() batch, and a batch is about 10 ms of GPU time at
        // Pearl's rate - so a 50 ms wait here cost 83% of the hashrate and
        // showed up as 65 M candidates/s against 413 in the benchmark. Only
        // the FIRST job is worth blocking for.
        drain(haveJob_ ? 0 : 8000);
        if (!haveJob_) return false;

        pearl::b3::hash(nullptr, header_, 76, job->msg);
        for (int i = 0; i < 4; i++) job->target[i] = target_[i];
        job->extra = packPoolExtra(header_, cert_, jobId_);
        job->epoch = 0;
        job->valid = true;
        return true;
    }

    bool submit(const Job &job, const Solution &sol, std::string *err) override {
        if (sol.extra.empty()) {
            *err = "no proof attached to the solution";
            return false;
        }
        const std::string id = poolJobId(job.extra);
        if (id.empty()) {
            *err = "job is missing its pool job id";
            return false;
        }
        if (fd_ == OM_INVALID_SOCKET && !connectAndLogin()) {
            *err = "lost the pool connection";
            return false;
        }
        const std::string body =
            "{\"id\":" + std::to_string(++id_) +
            ",\"method\":\"mining.submit\",\"params\":{\"job_id\":\"" + id +
            "\",\"plain_proof\":\"" + sol.extra + "\"}}";
        if (!sendLine(body)) {
            *err = "could not send the share";
            disconnect();
            return false;
        }
        // The reply may arrive behind a pushed job, so read until our id shows.
        const std::string want = "\"id\":" + std::to_string(id_);
        for (int i = 0; i < 12; i++) {
            std::string line;
            if (!readLine(&line)) break;
            if (line.find("mining.notify") != std::string::npos) {
                takeNotify(line);
                continue;
            }
            if (line.find(want) == std::string::npos) continue;
            std::string msg;
            if (jsonString(line, "msg", &msg) && !msg.empty()) {
                *err = msg;          // the pool says exactly why, so pass it on
                return false;
            }
            return true;
        }
        *err = "no reply from the pool";
        return false;
    }

   private:
    bool connectAndLogin() {
        if (!openSocket()) return false;
        if (!sendLine("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":"
                      "{\"agent\":\"soat-miner\"}}")) return false;
        std::string line;
        if (!readLine(&line)) return false;

        std::string auth =
            "{\"id\":2,\"method\":\"mining.authorize\",\"params\":{\"wallet\":\"" +
            wallet_ + "\"";
        if (!worker_.empty()) auth += ",\"worker\":\"" + worker_ + "\"";
        auth += "}}";
        if (!sendLine(auth)) return false;
        if (!readLine(&line)) return false;
        std::string msg;
        if (jsonString(line, "msg", &msg) && !msg.empty()) {
            fprintf(stderr, "[pearl-pow] pool refused the login: %s\n", msg.c_str());
            disconnect();
            return false;
        }
        return true;
    }

    /** Read whatever has arrived, keeping the newest job. */
    void drain(int ms) {
        const long long deadline = nowMs() + ms;
        do {
            // Poll with no timeout when we already have work: take whatever
            // has arrived and get straight back to mining.
            if (!readable(fd_, haveJob_ ? 0 : 50)) {
                if (haveJob_) return;
                continue;
            }
            std::string line;
            if (!readLine(&line)) { disconnect(); return; }
            if (line.find("mining.notify") != std::string::npos) takeNotify(line);
        } while (nowMs() < deadline);
    }

    void takeNotify(const std::string &line) {
        std::string hdr, tgt, jid, certStr;
        if (!jsonString(line, "header", &hdr)) return;
        if (!jsonString(line, "target", &tgt)) return;
        if (!jsonString(line, "job_id", &jid)) return;
        if (hdr.size() != 152) return;
        uint8_t h[76];
        if (!hexToBytes(hdr, h, 76)) return;
        uint64_t t[4];
        if (!hexToLimbs(tgt, t)) return;
        memcpy(header_, h, 76);
        for (int i = 0; i < 4; i++) target_[i] = t[i];
        jobId_ = jid;
        cert_ = jsonNumber(line, "cert_version", &certStr) ? atoi(certStr.c_str()) : 3;
        if (cert_ <= 0) cert_ = 3;
        haveJob_ = true;
    }

    bool openSocket() {
        disconnect();
        struct addrinfo hints {};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        struct addrinfo *res = nullptr;
        const std::string port = std::to_string(port_);
        if (getaddrinfo(host_.c_str(), port.c_str(), &hints, &res) != 0) return false;
        fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd_ == OM_INVALID_SOCKET) { freeaddrinfo(res); return false; }
        socketTimeout(fd_, 20);
        if (::connect(fd_, res->ai_addr, (int)res->ai_addrlen) != 0) {
            OM_CLOSESOCKET(fd_);
            fd_ = OM_INVALID_SOCKET;
            freeaddrinfo(res);
            return false;
        }
        freeaddrinfo(res);
        pending_.clear();
        return true;
    }

    void disconnect() {
        if (fd_ != OM_INVALID_SOCKET) OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
        pending_.clear();
    }

    bool sendLine(const std::string &body) {
        const std::string line = body + "\n";
        return socketSend(fd_, line.data(), line.size()) >= 0;
    }

    bool readLine(std::string *out) {
        for (;;) {
            const size_t nl = pending_.find('\n');
            if (nl != std::string::npos) {
                *out = pending_.substr(0, nl);
                pending_.erase(0, nl + 1);
                return true;
            }
            char buf[4096];
            const int n = socketRecv(fd_, buf, sizeof(buf));
            if (n <= 0) return false;
            pending_.append(buf, n);
        }
    }

    static bool readable(socket_t fd, int ms) {
        if (fd == OM_INVALID_SOCKET) return false;
        fd_set r;
        FD_ZERO(&r);
        FD_SET(fd, &r);
        struct timeval tv;
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        return select((int)fd + 1, &r, nullptr, nullptr, &tv) > 0;
    }

    /** Wall clock, deliberately. clock() measures CPU time, and this thread
     *  spends its wait blocked in select() consuming none of it, so a
     *  clock()-based deadline would never expire. */
    static long long nowMs() {
        return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    std::string host_, wallet_, worker_, desc_, pending_, jobId_;
    int port_ = 0;
    int id_ = 100;
    int cert_ = 3;
    bool haveJob_ = false;
    uint8_t header_[76] = {};
    uint64_t target_[4] = {};
    socket_t fd_ = OM_INVALID_SOCKET;
};

}  // namespace om

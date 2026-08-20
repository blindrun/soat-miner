#include "stratum_btc.h"

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "btc_job.h"
#include "btc_protocol.h"
#include "json_lite.h"
#include "sha256.h"

namespace om {

bool BitcoinStratumSource::sendLine(const std::string &s) {
    const std::string line = s + "\n";
    return socketSend(fd_, line.data(), line.size()) == (int)line.size();
}

bool BitcoinStratumSource::start(std::string *err) {
    struct addrinfo hints {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = nullptr;
    const std::string port = std::to_string(port_);
    const int gai = getaddrinfo(host_.c_str(), port.c_str(), &hints, &res);
    if (gai != 0 || res == nullptr) {
        *err = "DNS lookup for '" + host_ + "' failed (getaddrinfo=" +
               std::to_string(gai) +
               ") - check the pool hostname and that this machine has working DNS";
        return false;
    }
    fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd_ == OM_INVALID_SOCKET) {
        freeaddrinfo(res);
        *err = "socket() failed";
        return false;
    }
    socketTimeout(fd_, 30);
    if (::connect(fd_, res->ai_addr, (int)res->ai_addrlen) != 0) {
#if defined(_WIN32)
        const int se = WSAGetLastError();
#else
        const int se = errno;
#endif
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
        freeaddrinfo(res);
        *err = "TCP connect to " + desc_ + " failed (error " +
               std::to_string(se) +
               ") - a firewall or the port being blocked is the usual cause";
        return false;
    }
    freeaddrinfo(res);

    // No mining.configure: see the header. Version rolling is refused by never
    // asking for it, which is the only way to be sure bit 12 survives.
    if (!sendLine(btc::subscribeRequest())) {
        *err = "subscribe send failed";
        return false;
    }
    if (!sendLine(btc::authorizeRequest(login_, password_))) {
        *err = "authorize send failed";
        return false;
    }

    connected_ = true;
    running_ = true;
    reader_ = std::thread(&BitcoinStratumSource::readerLoop, this);
    return true;
}

bool BitcoinStratumSource::reconnect() {
    running_ = false;
    if (fd_ != OM_INVALID_SOCKET) {
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
    }
    if (reader_.joinable()) reader_.join();
    resetSession();
    std::string err;
    return start(&err);
}

void BitcoinStratumSource::resetSession() {
    std::lock_guard<std::mutex> lk(mu_);
    extranonce1_.clear();
    extranonce2Size_ = 4;
    subscribed_ = false;
    jobId_.clear();
    memset(prevHash_, 0, sizeof(prevHash_));
    coinb1_.clear();
    coinb2_.clear();
    branch_.clear();
    version_ = nbits_ = ntime_ = 0;
    haveTemplate_ = false;
    extranonce2_ = 0;
    extranonce2Bytes_.clear();
    fetchesSinceRoll_ = 0;
    haveJob_ = false;
    memset(shareTarget_, 0, sizeof(shareTarget_));
    difficulty_ = 1.0;
    haveDifficulty_ = false;
    jobWarning_.clear();
    loginRejected_ = false;
    loginError_.clear();
}

void BitcoinStratumSource::stop() {
    running_ = false;
    connected_ = false;
    if (fd_ != OM_INVALID_SOCKET) {
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
    }
    if (reader_.joinable()) reader_.join();
}

void BitcoinStratumSource::readerLoop() {
    std::string buf;
    char chunk[8192];
    while (running_) {
        const int n = socketRecv(fd_, chunk, sizeof(chunk));
        if (n < 0) {
            // Pools go quiet between blocks; a receive timeout is not a
            // disconnect. BC3 aims at ten-minute blocks, so these are long.
#if defined(_WIN32)
            const int e = WSAGetLastError();
            if (e == WSAETIMEDOUT || e == WSAEWOULDBLOCK) continue;
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
#endif
            connected_ = false;
            break;
        }
        if (n == 0) {
            connected_ = false;
            break;
        }
        buf.append(chunk, (size_t)n);
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            const std::string line = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            if (!line.empty()) handleLine(line);
        }
    }
}

void BitcoinStratumSource::handleLine(const std::string &line) {
    std::string method;
    if (jsonRawValue(line, "method", &method)) {
        method = jsonUnquote(method);

        if (method == "mining.set_difficulty") {
            std::string params;
            if (!jsonRawValue(line, "params", &params)) return;
            const auto p = jsonSplitArray(params);
            if (p.empty()) return;
            const double d = atof(jsonUnquote(p[0]).c_str());
            if (!(d > 0.0)) return;
            std::lock_guard<std::mutex> lk(mu_);
            difficulty_ = d;
            haveDifficulty_ = true;
            btc::difficultyToTarget(d, shareTarget_);
            if (haveTemplate_) buildHeader();
            return;
        }

        if (method == "mining.set_extranonce") {
            std::string params;
            if (!jsonRawValue(line, "params", &params)) return;
            const auto p = jsonSplitArray(params);
            if (p.empty()) return;
            std::vector<uint8_t> xn1;
            if (!btc::hexToVec(jsonUnquote(p[0]), &xn1)) return;
            std::lock_guard<std::mutex> lk(mu_);
            extranonce1_ = xn1;
            if (p.size() >= 2) {
                const long sz = strtol(jsonUnquote(p[1]).c_str(), nullptr, 10);
                if (sz > 0 && sz <= 32) extranonce2Size_ = (size_t)sz;
            }
            if (haveTemplate_) rollExtranonce2();
            return;
        }

        if (method == "mining.notify") {
            std::string params;
            if (!jsonRawValue(line, "params", &params)) return;
            const auto p = jsonSplitArray(params);
            if (p.size() < 9) {
                std::lock_guard<std::mutex> lk(mu_);
                jobWarning_ = "pool sent a mining.notify with " +
                              std::to_string(p.size()) +
                              " parameters; Bitcoin stratum needs 9";
                return;
            }

            // prevhash: 32 bytes with each 4-byte word reversed.
            std::vector<uint8_t> ph;
            if (!btc::hexToVec(jsonUnquote(p[1]), &ph) || ph.size() != 32) return;
            uint8_t prev[32];
            for (int w = 0; w < 8; w++)
                for (int b = 0; b < 4; b++) prev[w * 4 + b] = ph[w * 4 + 3 - b];

            std::vector<uint8_t> cb1, cb2;
            if (!btc::hexToVec(jsonUnquote(p[2]), &cb1)) return;
            if (!btc::hexToVec(jsonUnquote(p[3]), &cb2)) return;

            std::vector<std::string> branch;
            for (const std::string &e : jsonSplitArray(p[4])) {
                std::vector<uint8_t> b;
                if (!btc::hexToVec(jsonUnquote(e), &b) || b.size() != 32) return;
                branch.push_back(std::string((const char *)b.data(), 32));
            }

            uint32_t ver = 0, bits = 0, tim = 0;
            if (!btc::hexScalar(jsonUnquote(p[5]), &ver)) return;
            if (!btc::hexScalar(jsonUnquote(p[6]), &bits)) return;
            if (!btc::hexScalar(jsonUnquote(p[7]), &tim)) return;

            std::lock_guard<std::mutex> lk(mu_);
            jobId_ = jsonUnquote(p[0]);
            memcpy(prevHash_, prev, 32);
            coinb1_ = cb1;
            coinb2_ = cb2;
            branch_ = branch;
            version_ = ver;
            nbits_ = bits;
            ntime_ = tim;
            haveTemplate_ = true;

            // A pool that has not sent set_difficulty yet is rare but legal;
            // diff 1 is the protocol default and beats hashing against a zero
            // target, which nothing can ever satisfy.
            if (!haveDifficulty_) btc::difficultyToTarget(1.0, shareTarget_);

            // Consensus requires version bit 12 on every post-fork block. If a
            // pool ever stops setting it the miner would hash a header the
            // chain rejects, and the shares would look fine right up until the
            // block was orphaned - so say so rather than mine it.
            if ((ver & 0x00001000u) == 0) {
                jobWarning_ =
                    "pool sent version " + jsonUnquote(p[5]) +
                    " without SHA3 bit 12 set. Post-fork BC3 blocks must have "
                    "it (validation.cpp:4227); this job would be rejected by "
                    "the network. Check the pool is on a current node.";
            }
            rollExtranonce2();
            return;
        }
        return;
    }

    // Replies carry an id. Subscribe is 1, authorize 2, submissions 10 and up.
    std::string idRaw;
    if (!jsonRawValue(line, "id", &idRaw)) return;
    const int id = atoi(idRaw.c_str());

    if (id == 1) {
        std::string result;
        if (!jsonRawValue(line, "result", &result)) return;
        const auto r = jsonSplitArray(result);
        if (r.size() < 3) return;
        std::vector<uint8_t> xn1;
        if (!btc::hexToVec(jsonUnquote(r[1]), &xn1)) return;
        const long sz = strtol(jsonUnquote(r[2]).c_str(), nullptr, 10);
        std::lock_guard<std::mutex> lk(mu_);
        extranonce1_ = xn1;
        extranonce2Size_ = (sz > 0 && sz <= 32) ? (size_t)sz : 4;
        subscribed_ = true;
        return;
    }

    if (id == 2) {
        std::string result;
        if (jsonRawValue(line, "result", &result) && result == "true") return;
        std::string e;
        jsonRawValue(line, "error", &e);
        loginRejected_ = true;
        std::lock_guard<std::mutex> lk(mu_);
        loginError_ = e.empty() ? "authorize returned false" : e;
        return;
    }

    if (id >= kFirstSubmitId) {
        std::string result;
        const bool ok = jsonRawValue(line, "result", &result) && result == "true";
        std::string e;
        jsonRawValue(line, "error", &e);
        std::lock_guard<std::mutex> lk(submitMu_);
        if (ok) {
            accepted_++;
        } else {
            rejected_++;
            lastSubmitError_ = (e.empty() || e == "null") ? "pool said false" : e;
            verdict_ = lastSubmitError_;
        }
    }
}

bool BitcoinStratumSource::buildHeader() {
    if (!haveTemplate_ || !subscribed_) return false;

    // coinbase = coinb1 || extranonce1 || extranonce2 || coinb2
    std::vector<uint8_t> cb;
    cb.reserve(coinb1_.size() + extranonce1_.size() + extranonce2Size_ +
               coinb2_.size());
    cb.insert(cb.end(), coinb1_.begin(), coinb1_.end());
    cb.insert(cb.end(), extranonce1_.begin(), extranonce1_.end());
    cb.insert(cb.end(), extranonce2Bytes_.begin(), extranonce2Bytes_.end());
    cb.insert(cb.end(), coinb2_.begin(), coinb2_.end());

    uint8_t root[32];
    merkleRootFromBranch(cb, branch_, root);

    uint8_t header[80];
    btc::putLE32(header + 0, version_);
    memcpy(header + 4, prevHash_, 32);
    memcpy(header + 36, root, 32);
    btc::putLE32(header + 68, ntime_);
    btc::putLE32(header + 72, nbits_);
    btc::putLE32(header + 76, 0);  // the nonce the search fills in

    Job j;
    // The merkle root doubles as the job's identity: it is 32 bytes and it is
    // the field that moves on every extranonce2 roll, so the run loop's
    // "msg changed => adopt" test does the right thing for free. See btc_job.h.
    memcpy(j.msg, root, 32);
    memcpy(j.target, shareTarget_, sizeof(j.target));
    j.epoch = coinbaseHeight();
    j.extra = encodeBtcJobExtra(header, extranonce2Bytes_);
    j.jobId = jobId_;
    j.valid = true;

    current_ = j;
    haveJob_ = true;
    return true;
}

void BitcoinStratumSource::rollExtranonce2() {
    extranonce2_++;
    extranonce2Bytes_.assign(extranonce2Size_, '\0');
    // Little-endian, which is only a convention: the pool treats extranonce2 as
    // opaque bytes and simply splices them back into the coinbase.
    for (size_t i = 0; i < extranonce2Size_ && i < 8; i++)
        extranonce2Bytes_[i] = (char)(uint8_t)(extranonce2_ >> (8 * i));
    fetchesSinceRoll_ = 0;
    buildHeader();
}

/**
 * Block height, read out of the coinbase's BIP34 push.
 *
 * Only ever used for the readout - prepare() is a no-op for this algorithm, so
 * a wrong answer costs nothing but a wrong number on screen. Hence the quiet
 * fallback rather than refusing the job.
 */
uint64_t BitcoinStratumSource::coinbaseHeight() const {
    // coinb1 = version(4) | inCount(1) | prevout(36) | scriptLen(1) | script...
    const size_t at = 4 + 1 + 36 + 1;
    if (coinb1_.size() < at + 1) return 0;
    const uint8_t push = coinb1_[at];
    if (push == 0 || push > 6 || coinb1_.size() < at + 1 + push) return 0;
    uint64_t h = 0;
    for (int i = push - 1; i >= 0; i--) h = (h << 8) | coinb1_[at + 1 + i];
    return h;
}

bool BitcoinStratumSource::fetch(Job *job) {
    if (!connected_) {
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

    // Hand out a fresh extranonce2 before the run loop can wrap its 32-bit
    // nonce counter and re-mine ground this extranonce2 already covered. The
    // loop consumes `batch` nonces between fetches, so counting fetches is an
    // exact measure of the space used.
    if (++fetchesSinceRoll_ * batch_ >= kNonceBudget) rollExtranonce2();

    *job = current_;
    return true;
}

bool BitcoinStratumSource::submit(const Job &job, const Solution &sol,
                                  std::string *err) {
    std::string xn2;
    if (!btcJobExtranonce2(job.extra, &xn2)) {
        *err = "job carries no extranonce2";
        return false;
    }
    uint8_t header[80];
    if (!btcJobHeader(job.extra, header)) {
        *err = "job carries no header";
        return false;
    }
    // ntime is submitted exactly as it was hashed, read back out of the header
    // rather than from the live template, which may already have moved on.
    const uint32_t ntime = (uint32_t)header[68] | ((uint32_t)header[69] << 8) |
                           ((uint32_t)header[70] << 16) |
                           ((uint32_t)header[71] << 24);

    const int id = nextId_++;
    const std::string msg = btc::submitRequest(id, login_, job.jobId, xn2,
                                               ntime, (uint32_t)sol.nonce);

    if (!sendLine(msg)) {
        *err = "submit send failed";
        connected_ = false;
        return false;
    }
    std::lock_guard<std::mutex> lk(submitMu_);
    submitted_++;
    return true;
}

bool BitcoinStratumSource::poolCounters(uint64_t *accepted, uint64_t *rejected,
                                        uint64_t *pending,
                                        std::string *lastError) const {
    std::lock_guard<std::mutex> lk(submitMu_);
    *accepted = accepted_;
    *rejected = rejected_;
    *pending = (submitted_ > accepted_ + rejected_)
                   ? submitted_ - accepted_ - rejected_
                   : 0;
    *lastError = lastSubmitError_;
    return true;
}

std::string BitcoinStratumSource::takeJobWarning() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string s;
    s.swap(jobWarning_);
    return s;
}

std::string BitcoinStratumSource::takeSubmitVerdict() {
    std::lock_guard<std::mutex> lk(submitMu_);
    std::string s;
    s.swap(verdict_);
    return s;
}

}  // namespace om

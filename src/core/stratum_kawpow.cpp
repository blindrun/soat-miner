#include "stratum_kawpow.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../algos/kawpow/job.h"
#include "../algos/kawpow/kawpow.h"
#include "../algos/progpow/ethash.h"
#include "json_lite.h"

namespace om {
namespace {

/** Strips an optional 0x and lowercases. Pools are inconsistent about both. */
std::string normHex(const std::string &s) {
    std::string h = s;
    if (h.size() > 1 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X'))
        h = h.substr(2);
    for (char &c : h) c = (char)tolower((unsigned char)c);
    return h;
}

bool isHex(const std::string &s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!isxdigit((unsigned char)c)) return false;
    return true;
}

/**
 * 64 hex characters, most significant first, into limbs with limb[0] least
 * significant - the convention Job::target uses.
 *
 * Short values are padded on the LEFT, because a pool that sends "1000" means
 * a tiny target, not a huge one. Padding on the right would turn a hard share
 * into a trivial one and every submission would be rejected.
 */
bool hexToTarget(const std::string &hexIn, uint64_t out[4]) {
    std::string h = normHex(hexIn);
    if (h.size() > 64 || !isHex(h)) return false;
    h.insert(h.begin(), 64 - h.size(), '0');
    for (int limb = 0; limb < 4; limb++) {
        // limb 3 is the most significant, so it takes the first 16 characters.
        const std::string part = h.substr((size_t)(3 - limb) * 16, 16);
        out[limb] = strtoull(part.c_str(), nullptr, 16);
    }
    return true;
}

/** 64 hex characters into 32 bytes, in order. */
bool hexToBytes32(const std::string &hexIn, uint8_t out[32]) {
    std::string h = normHex(hexIn);
    if (h.size() > 64 || !isHex(h)) return false;
    h.insert(h.begin(), 64 - h.size(), '0');
    for (int i = 0; i < 32; i++) {
        const std::string b = h.substr((size_t)i * 2, 2);
        out[i] = (uint8_t)strtoul(b.c_str(), nullptr, 16);
    }
    return true;
}

std::string bytes32ToHex(const uint8_t b[32]) {
    static const char *d = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (int i = 0; i < 32; i++) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
}

}  // namespace

bool KawpowStratumSource::sendLine(const std::string &s) {
    {
        std::lock_guard<std::mutex> lk(sendMu_);
        lastSent_ = s;
    }
    if (fd_ == OM_INVALID_SOCKET) return true;  // test-driven, no socket
    const std::string line = s + "\n";
    return socketSend(fd_, line.data(), line.size()) == (int)line.size();
}

bool KawpowStratumSource::start(std::string *err) {
    struct addrinfo hints {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = nullptr;
    const std::string port = std::to_string(port_);
    const int gai = getaddrinfo(host_.c_str(), port.c_str(), &hints, &res);
    if (gai != 0 || res == nullptr) {
        *err = "DNS lookup for '" + host_ + "' failed (getaddrinfo=" +
               std::to_string(gai) + ") - check the pool hostname and that "
               "this machine has working DNS";
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

    if (!sendLine("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":"
                  "[\"soat-miner/0.3.0\",\"EthereumStratum/1.0.0\"]}")) {
        *err = "subscribe send failed";
        return false;
    }
    if (!sendLine("{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" +
                  login_ + "\",\"" + password_ + "\"]}")) {
        *err = "authorize send failed";
        return false;
    }

    connected_ = true;
    running_ = true;
    reader_ = std::thread(&KawpowStratumSource::readerLoop, this);
    return true;
}

bool KawpowStratumSource::reconnect() {
    running_ = false;
    if (fd_ != OM_INVALID_SOCKET) {
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
    }
    if (reader_.joinable()) reader_.join();
    {
        std::lock_guard<std::mutex> lk(mu_);
        haveJob_ = false;
    }
    std::string err;
    return start(&err);
}

void KawpowStratumSource::stop() {
    running_ = false;
    connected_ = false;
    if (fd_ != OM_INVALID_SOCKET) {
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
    }
    if (reader_.joinable()) reader_.join();
}

void KawpowStratumSource::readerLoop() {
    std::string buf;
    char chunk[4096];
    while (running_) {
        const int n = socketRecv(fd_, chunk, sizeof(chunk));
        if (n < 0) {
            // A receive timeout is normal: pools go quiet between jobs.
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
    connected_ = false;
}

/**
 * The extranonce is a prefix of the 8-byte nonce and its width is the LENGTH
 * of the hex string, not its value in bytes.
 *
 * Refuses rather than guesses. Mining with an empty prefix when the pool
 * assigned one is not a safe fallback: it collides with every other miner on
 * that pool and the shares are rejected for a reason the miner cannot see.
 */
void KawpowStratumSource::applyExtranonce(const std::string &xnHexIn) {
    const std::string xn = normHex(xnHexIn);
    std::lock_guard<std::mutex> lk(mu_);

    if (xn.empty()) {
        // No extranonce at all is legitimate - the whole nonce is ours.
        noncePrefix_ = 0;
        nonceBitsOwned_ = 64;
        extranonceNibbles_ = 0;
        extranonceGen_++;
        return;
    }
    if (!isHex(xn) || xn.size() > 14 || (xn.size() & 1)) {
        // Odd lengths are ambiguous - "0xf" could mean 0x0f or 0xf0 - and a
        // prefix wider than 14 characters leaves under 8 bits of nonce, which
        // is exhausted faster than a job lives.
        noncePrefix_ = 0;
        nonceBitsOwned_ = 64;
        extranonceNibbles_ = 0;
        haveJob_ = false;
        current_.valid = false;
        jobWarning_ = "pool sent an unusable extranonce '" + xnHexIn +
                      "' (must be an even number of hex characters, at most "
                      "14); stopped mining until a valid one arrives.";
        extranonceGen_++;
        return;
    }

    // "adac" means the nonce starts 0xadac............ - left-justified into
    // the 16-character nonce, not right.
    std::string padded = xn;
    padded.resize(16, '0');
    noncePrefix_ = strtoull(padded.c_str(), nullptr, 16);
    extranonceNibbles_ = (int)xn.size();
    nonceBitsOwned_ = 64 - extranonceNibbles_ * 4;
    extranonceGen_++;
}

void KawpowStratumSource::handleSubscribeResult(const std::string &result) {
    const std::vector<std::string> parts = jsonSplitArray(result);
    if (parts.empty()) return;

    // Flavour detection. An announced subscription - a nested array naming
    // mining.notify - is the EthereumStratum form; a bare null first element
    // is what Ravencoin pools answer on their documented stratum ports.
    if (requestedMode_ == kAuto) {
        const bool announced = !parts[0].empty() && parts[0][0] == '[';
        mode_ = announced ? kNiceHash : kStratum;
    }

    // A reply of two bare strings matches no shape anyone has documented.
    // 2miners answers this way and the meaning of its second element is
    // unresolved - see tools/probe-stratum.py. Take the last string as the
    // extranonce, which is the likelier reading, but say so loudly: if the
    // guess is wrong every share is rejected as out of range, and nothing else
    // in the miner would ever point at the extranonce.
    if (parts.size() == 2 && parts[0][0] == '"' && parts[1][0] == '"') {
        std::lock_guard<std::mutex> lk(mu_);
        jobWarning_ =
            "this pool's subscribe reply is a shape soat-miner does not "
            "recognise (" + parts[0] + ", " + parts[1] +
            "). Guessing the second value is the extranonce. If shares are "
            "rejected as out of range, that guess is wrong - report the reply "
            "rather than changing the wallet.";
    }

    // The extranonce is the last string in the reply, whichever shape it took.
    for (size_t i = parts.size(); i-- > 0;) {
        const std::string v = jsonUnquote(parts[i]);
        if (!v.empty() && v != "null" && parts[i][0] == '"') {
            applyExtranonce(v);
            return;
        }
    }
}

void KawpowStratumSource::handleNotify(const std::string &params) {
    const std::vector<std::string> p = jsonSplitArray(params);
    // [jobId, headerHash, seedHash, target, cleanJobs, height, bits]
    if (p.size() < 4) {
        std::lock_guard<std::mutex> lk(mu_);
        jobWarning_ = "pool sent a mining.notify with " +
                      std::to_string(p.size()) +
                      " parameters; KawPow needs at least 4 (job, header, "
                      "seed, target).";
        return;
    }

    Job j;
    j.jobId = jsonUnquote(p[0]);
    if (!hexToBytes32(jsonUnquote(p[1]), j.msg)) {
        std::lock_guard<std::mutex> lk(mu_);
        jobWarning_ = "pool sent an unusable header hash";
        return;
    }
    if (!hexToTarget(jsonUnquote(p[3]), j.target)) {
        std::lock_guard<std::mutex> lk(mu_);
        jobWarning_ = "pool sent an unusable share target";
        return;
    }

    // The height decides the 3-block program, and a job without one is not
    // mineable: a wrong program produces perfect-looking hashes that the pool
    // rejects every time.
    int height = -1;
    if (p.size() >= 6) {
        const std::string h = jsonUnquote(p[5]);
        height = (int)strtol(h.c_str(), nullptr, h.rfind("0x", 0) == 0 ? 16 : 10);
    }
    if (height <= 0) {
        std::lock_guard<std::mutex> lk(mu_);
        jobWarning_ =
            "pool sent a mining.notify with no block height. KawPow's program "
            "changes every 3 blocks, so mining without it would produce "
            "rejected shares indefinitely; stopped until a job carries one.";
        haveJob_ = false;
        current_.valid = false;
        return;
    }

    // The pool tells us its seed hash. Ours is derived from the height. If
    // they disagree we have the wrong DAG, and every share would be rejected
    // for a reason nothing else would reveal - so check it here, once, rather
    // than discovering it from a reject counter.
    const int epoch =
        om::progpow::epochNumber(height, om::kawpow::kEpochLength);
    if (p.size() >= 3) {
        const std::string poolSeed = normHex(jsonUnquote(p[2]));
        uint32_t seed[8];
        om::progpow::epochSeed(epoch, seed);
        uint8_t sb[32];
        for (int i = 0; i < 8; i++)
            for (int b = 0; b < 4; b++) sb[i * 4 + b] = (uint8_t)(seed[i] >> (b * 8));
        const std::string ours = bytes32ToHex(sb);
        if (poolSeed.size() == 64 && poolSeed != ours) {
            std::lock_guard<std::mutex> lk(mu_);
            jobWarning_ =
                "pool's seed hash does not match the one this miner derives "
                "for height " + std::to_string(height) + " (epoch " +
                std::to_string(epoch) +
                "). That means a different epoch length, and every share would "
                "be rejected; stopped rather than mine a wrong DAG.";
            haveJob_ = false;
            current_.valid = false;
            return;
        }
    }

    j.epoch = (uint64_t)epoch;
    j.extra = om::kawpow::encodeJobExtra(height);
    j.valid = true;

    std::lock_guard<std::mutex> lk(mu_);
    current_ = j;
    haveJob_ = true;
}

void KawpowStratumSource::handleLine(const std::string &line) {
    std::string method;
    if (jsonRawValue(line, "method", &method)) {
        const std::string m = jsonUnquote(method);
        if (m == "mining.notify") {
            std::string params;
            if (jsonRawValue(line, "params", &params)) handleNotify(params);
            return;
        }
        if (m == "mining.set_extranonce") {
            std::string params;
            if (jsonRawValue(line, "params", &params)) {
                const std::vector<std::string> p = jsonSplitArray(params);
                if (!p.empty()) applyExtranonce(jsonUnquote(p[0]));
            }
            return;
        }
        // mining.set_difficulty and mining.set_target are deliberately
        // ignored: the share target arrives with every notify, and a
        // difficulty applied out of band would fight it.
        return;
    }

    std::string idRaw;
    if (!jsonRawValue(line, "id", &idRaw)) return;
    const int id = atoi(idRaw.c_str());

    std::string errRaw;
    const bool hasErr = jsonRawValue(line, "error", &errRaw);
    const bool isErr = hasErr && errRaw != "null";

    if (id == 1) {
        std::string result;
        if (!isErr && jsonRawValue(line, "result", &result))
            handleSubscribeResult(result);
        return;
    }
    if (id == 2) {
        if (isErr) {
            loginRejected_ = true;
            std::lock_guard<std::mutex> lk(mu_);
            loginError_ = errRaw;
        }
        return;
    }
    if (id >= kFirstSubmitId) {
        // Both verdicts are read out of the parsed `result` value, never
        // matched as bytes in the line: a pool is free to space and order its
        // JSON however it likes, and "result": true is the same reply as
        // "result":true.
        std::string warn;
        {
            std::lock_guard<std::mutex> lk(submitMu_);
            std::string result;
            const bool haveResult = jsonRawValue(line, "result", &result);
            if (isErr || (haveResult && result == "false")) {
                rejected_++;
                lastSubmitError_ =
                    isErr ? errRaw : std::string("pool said false");
                verdict_ = "share rejected: " + lastSubmitError_;
            } else if (haveResult && result == "true") {
                accepted_++;
                verdict_ = "share accepted";
            } else {
                // Neither. This pool answered a submission in a shape this
                // client cannot read, and there is no safe way to guess.
                //
                // Guessing "accepted" is the tempting default and the worst
                // one: it is a share count that disagrees with the payout,
                // and it disagrees in the direction that looks like nothing
                // is wrong. Leaving it uncounted parks it in `pending`, which
                // then climbs and stays climbing, and says so once - loudly -
                // rather than letting the operator find out from the payout.
                warn = "pool answered a share submission in a shape this "
                       "miner cannot read, so it counts as neither accepted "
                       "nor rejected: " + line;
            }
        }
        // Deliberately outside the submitMu_ scope. Nothing else in this file
        // holds both locks at once and this is not the place to start.
        if (!warn.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            jobWarning_ = warn;
        }
    }
}

bool KawpowStratumSource::fetch(Job *job) {
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
    if (!haveJob_) return false;
    *job = current_;
    return true;
}

bool KawpowStratumSource::submit(const Job &job, const Solution &sol,
                                 std::string *err) {
    const int id = nextId_++;
    const std::string nonce = hex64(sol.nonce);
    std::string params;

    if (mode_ == kNiceHash) {
        // The pool already knows the prefix it assigned, so only the part this
        // miner owns is sent. Cutting the wrong number of characters is the
        // classic way to have every share rejected as out of range.
        int cut;
        {
            std::lock_guard<std::mutex> lk(mu_);
            cut = extranonceNibbles_;
        }
        if (cut < 0 || cut > (int)nonce.size()) cut = 0;
        params = "[\"" + login_ + "\",\"" + job.jobId + "\",\"" +
                 nonce.substr((size_t)cut) + "\"]";
    } else {
        // Ravencoin's documented form. The mix hash is not derivable from the
        // nonce without redoing the whole hash, so the algorithm hands it out
        // in Solution::extra and it goes on the wire here.
        if (sol.extra.size() != 64) {
            *err = "solution carries no mix hash; a KawPow share cannot be "
                   "submitted without one";
            return false;
        }
        params = "[\"" + login_ + "\",\"" + job.jobId + "\",\"0x" + nonce +
                 "\",\"0x" + bytes32ToHex(job.msg) + "\",\"0x" + sol.extra + "\"]";
    }

    const std::string req = "{\"id\":" + std::to_string(id) +
                            ",\"method\":\"mining.submit\",\"params\":" +
                            params + "}";
    {
        std::lock_guard<std::mutex> lk(submitMu_);
        submitted_++;
    }
    if (!sendLine(req)) {
        *err = "submit send failed";
        return false;
    }
    return true;
}

bool KawpowStratumSource::poolCounters(uint64_t *accepted, uint64_t *rejected,
                                       uint64_t *pending,
                                       std::string *lastError) const {
    std::lock_guard<std::mutex> lk(submitMu_);
    *accepted = accepted_;
    *rejected = rejected_;
    *pending = submitted_ - accepted_ - rejected_;
    *lastError = lastSubmitError_;
    return true;
}

std::string KawpowStratumSource::takeJobWarning() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string w;
    w.swap(jobWarning_);
    return w;
}

std::string KawpowStratumSource::takeSubmitVerdict() {
    std::lock_guard<std::mutex> lk(submitMu_);
    std::string v;
    v.swap(verdict_);
    return v;
}

}  // namespace om

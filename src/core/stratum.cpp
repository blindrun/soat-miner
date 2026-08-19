#include "stratum.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace om {
namespace {

/** Extracts the raw text of a JSON value by key, without a JSON library. */
bool rawValue(const std::string &j, const char *key, std::string *out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < j.size() && isspace((unsigned char)j[p])) p++;
    if (p >= j.size()) return false;

    size_t start = p;
    if (j[p] == '[' || j[p] == '{') {
        const char open = j[p], close = (open == '[') ? ']' : '}';
        int depth = 0;
        bool inStr = false;
        for (; p < j.size(); p++) {
            const char c = j[p];
            if (inStr) {
                if (c == '\\') p++;
                else if (c == '"') inStr = false;
                continue;
            }
            if (c == '"') inStr = true;
            else if (c == open) depth++;
            else if (c == close) {
                depth--;
                if (depth == 0) { p++; break; }
            }
        }
    } else if (j[p] == '"') {
        p++;
        while (p < j.size() && j[p] != '"') {
            if (j[p] == '\\') p++;
            p++;
        }
        p++;
    } else {
        while (p < j.size() && j[p] != ',' && j[p] != '}' && j[p] != ']') p++;
        // Trim surrounding whitespace: `"id": 1 ,` is conforming JSON, and the
        // raw value must compare equal to "1"/"true", not "1 ". A quiet miss
        // here leaves a discarded subscribe reply on the default extranonce.
        size_t b = start, e = p;
        auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
        while (b < e && ws(j[b])) b++;
        while (e > b && ws(j[e - 1])) e--;
        *out = j.substr(b, e - b);
        return true;
    }
    *out = j.substr(start, p - start);
    return true;
}

/** Splits a JSON array's top-level elements into raw strings. */
std::vector<std::string> splitArray(const std::string &arr) {
    std::vector<std::string> out;
    if (arr.size() < 2) return out;
    size_t p = arr.find('[');
    if (p == std::string::npos) return out;
    p++;
    int depth = 0;
    bool inStr = false;
    size_t start = p;
    for (; p < arr.size(); p++) {
        const char c = arr[p];
        if (inStr) {
            if (c == '\\') p++;
            else if (c == '"') inStr = false;
            continue;
        }
        if (c == '"') inStr = true;
        else if (c == '[' || c == '{') depth++;
        else if (c == '}' || (c == ']' && depth > 0)) depth--;
        else if ((c == ',' && depth == 0) || (c == ']' && depth == 0)) {
            std::string e = arr.substr(start, p - start);
            while (!e.empty() && isspace((unsigned char)e.front())) e.erase(e.begin());
            while (!e.empty() && isspace((unsigned char)e.back())) e.pop_back();
            if (!e.empty()) out.push_back(e);
            start = p + 1;
            if (c == ']') break;
        }
    }
    return out;
}

std::string unquote(const std::string &s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

}  // namespace

bool StratumSource::sendLine(const std::string &s) {
    const std::string line = s + "\n";
    // socketSend now writes all bytes or returns -1, so require the full length.
    return socketSend(fd_, line.data(), line.size()) == (int)line.size();
}

bool StratumSource::reconnect() {
    // Tear down and re-run the handshake. Called by fetch() when the reader
    // thread has seen the connection go away.
    running_ = false;
    if (fd_ != OM_INVALID_SOCKET) { OM_CLOSESOCKET(fd_); fd_ = OM_INVALID_SOCKET; }
    if (reader_.joinable()) reader_.join();
    {
        std::lock_guard<std::mutex> lk(mu_);
        haveJob_ = false;
    }
    std::string err;
    return start(&err);
}

bool StratumSource::start(std::string *err) {
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
                  "[\"soat-miner/0.1\",\"EthereumStratum/1.0.0\"]}")) {
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
    reader_ = std::thread(&StratumSource::readerLoop, this);
    return true;
}

void StratumSource::stop() {
    running_ = false;
    connected_ = false;
    if (fd_ != OM_INVALID_SOCKET) {
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
    }
    if (reader_.joinable()) reader_.join();
}

void StratumSource::readerLoop() {
    std::string buf;
    char chunk[4096];
    while (running_) {
        const int n = socketRecv(fd_, chunk, sizeof(chunk));
        if (n < 0) {
            // A receive timeout is normal: pools go quiet between jobs. Only a
            // real error or a closed socket counts as a disconnect. Without
            // this the miner drops off the pool after one idle timeout.
#if defined(_WIN32)
            const int e = WSAGetLastError();
            if (e == WSAETIMEDOUT || e == WSAEWOULDBLOCK) continue;
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
#endif
            connected_ = false;
            break;
        }
        if (n == 0) {  // peer closed
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

void StratumSource::handleLine(const std::string &line) {
    std::string method;
    if (rawValue(line, "method", &method)) {
        method = unquote(method);

        if (method == "mining.notify") {
            std::string params;
            if (!rawValue(line, "params", &params)) return;
            const auto p = splitArray(params);
            // [jobId, height, msg, "", "", n, targetDecimal, "", cleanJobs]
            if (p.size() < 7) return;

            Job j;
            const std::string msgHex = unquote(p[2]);
            if (!hexToBytes(msgHex, j.msg, 32)) return;
            j.epoch = strtoull(unquote(p[1]).c_str(), nullptr, 10);
            decimalToLimbs(unquote(p[6]), j.target);

            // A zero target can never be beaten, so the miner would hash
            // forever, submit nothing and report no error - the worst failure
            // mode there is. Lithos reaches this legitimately: its
            // BlockTemplate has a constructor that leaves tau at 0, and tau is
            // what it publishes in params[6]. Refuse the job and say so.
            if ((j.target[0] | j.target[1] | j.target[2] | j.target[3]) == 0) {
                std::lock_guard<std::mutex> lk(mu_);
                jobWarning_ =
                    "pool sent job " + unquote(p[0]) +
                    " with a zero target - nothing can ever satisfy it. "
                    "On Lithos this means the client has no share target (tau) "
                    "set yet; wait for it to finish syncing.";
                // Drop any previous job. Without this, a zero-target notify
                // arriving mid-session leaves haveJob_ true, so fetch() keeps
                // handing out the stale job and the miner submits against work
                // the pool has moved on from. As a first job it also leaves the
                // startup path with nothing, which reads jobWarning_ to explain
                // the wait instead of a generic "no job in 20s".
                haveJob_ = false;
                current_.valid = false;
                return;
            }
            j.valid = true;
            j.jobId = unquote(p[0]);  // carry it with the job, not just in jobId_

            std::lock_guard<std::mutex> lk(mu_);
            current_ = j;
            jobId_ = j.jobId;
            haveJob_ = true;
            return;
        }

        // Lithos rotates the extranonce mid-session when two miners collide on
        // the same prefix (its StratumConnection answers a duplicate share by
        // issuing a fresh extraNonce1). Ignoring this leaves us mining in the
        // subspace it just took away from us.
        //   params: ["<extraNonce1 hex>", <extraNonce2Size>]
        if (method == "mining.set_extranonce") {
            std::string params;
            if (!rawValue(line, "params", &params)) return;
            const auto p = splitArray(params);
            if (p.size() < 2) return;
            applyExtranonce(unquote(p[0]), atoi(unquote(p[1]).c_str()));
            return;
        }

        if (method == "mining.set_difficulty" || method == "mining.set_target") {
            return;  // target arrives per-job in mining.notify
        }
        return;
    }

    // mining.authorize reply. A pool refusing the login looks exactly like a
    // dead connection from the outside - it just stops sending - so without
    // reading this the miner reports "cannot reach pool" when the real
    // problem is a bad wallet address.
    {
        std::string idRaw2, result, err;
        if (rawValue(line, "id", &idRaw2) && idRaw2 == "2") {
            rawValue(line, "result", &result);
            rawValue(line, "error", &err);
            const bool refused =
                (result == "false") || (!err.empty() && err != "null");
            if (refused) {
                std::lock_guard<std::mutex> lk(mu_);
                loginError_ = err.empty() || err == "null"
                                  ? "pool refused the login"
                                  : err;
                loginRejected_ = true;
            }
        }
    }

    // A submit reply. Reading this is not optional: a pool that refuses every
    // share looks exactly like a pool that accepts every share if you only
    // check that the send succeeded. That is how a 3-param mining.submit
    // shipped - the pool answered "Invalid params" to all of them and the
    // miner reported every one as accepted.
    {
        std::string idRaw3;
        if (rawValue(line, "id", &idRaw3)) {
            const int rid = atoi(idRaw3.c_str());
            if (rid >= kFirstSubmitId) {
                std::string result, errv;
                rawValue(line, "result", &result);
                rawValue(line, "error", &errv);
                const bool ok = (result == "true") &&
                                (errv.empty() || errv == "null");
                std::lock_guard<std::mutex> lk(mu_);
                if (ok) {
                    accepted_++;
                    verdict_ = "";
                } else {
                    rejected_++;
                    lastSubmitError_ = (errv.empty() || errv == "null")
                                           ? "pool rejected the share"
                                           : errv;
                    verdict_ = lastSubmitError_;
                }
                return;
            }
        }
    }

    // A response. The only one we care about is the subscribe reply, which
    // carries the extranonce prefix.
    std::string idRaw;
    if (rawValue(line, "id", &idRaw) && idRaw == "1") {
        std::string result;
        if (!rawValue(line, "result", &result)) return;
        const auto r = splitArray(result);
        // Read the last two elements positionally rather than assuming what
        // sits in front of them. A conventional Ergo pool answers
        // [<subscriptions>, extraNonce1, extraNonce2Size]; Lithos answers
        // [null, extraNonce1, extraNonce2Size] - same layout, but element 0 is
        // a bare null rather than an array.
        if (r.size() >= 2) {
            applyExtranonce(unquote(r[r.size() - 2]),
                            atoi(unquote(r[r.size() - 1]).c_str()));
        }
    }
}

// Adopt an extranonce assignment, from either the subscribe reply or a later
// mining.set_extranonce. `en2Size` is the number of nonce bytes left to us;
// the prefix occupies the bits above them.
void StratumSource::applyExtranonce(const std::string &xnHex, int en2Size) {
    // A malformed assignment is a protocol error. Mining with a guessed or
    // empty prefix is not a safe fallback: it collides with other miners and
    // gets shares rejected. Stop work until a valid job arrives.
    const bool badSize = (en2Size <= 0 || en2Size > 8);
    const int safeSize = badSize ? 8 : en2Size;

    uint64_t prefix = 0;
    size_t prefixNibbles = 0;
    bool badHex = false, tooLong = false;
    for (char c : xnHex) {
        if (!isxdigit((unsigned char)c)) { badHex = true; break; }
        // A uint64_t can carry at most 16 nibbles. Bound the accumulator even
        // if a malicious peer sends an arbitrarily long JSON string.
        if (prefixNibbles == 16) { tooLong = true; break; }
        const int v = (c <= '9') ? (c - '0') : ((c | 0x20) - 'a' + 10);
        prefix = (prefix << 4) | (uint64_t)v;
        prefixNibbles++;
    }
    const int ownedBits = safeSize * 8;
    std::lock_guard<std::mutex> lk(mu_);
    nonceBitsOwned_ = ownedBits >= 64 ? 64 : ownedBits;
    const int prefixBits = 64 - nonceBitsOwned_;
    const bool emptyPrefix = prefixNibbles == 0;
    const bool fits = emptyPrefix ||
        (nonceBitsOwned_ < 64 && prefixNibbles * 4 <= (size_t)prefixBits);
    if (!badSize && !badHex && !tooLong && fits) {
        // Guard the shift: nonceBitsOwned_ can be 64 (en2Size 8 with an empty
        // prefix reaches here via emptyPrefix), and even 0ULL << 64 is UB.
        // Owning all 64 bits means there is no prefix.
        noncePrefix_ = (nonceBitsOwned_ < 64) ? (prefix << nonceBitsOwned_) : 0ULL;
    } else {
        noncePrefix_ = 0ULL;
        haveJob_ = false;
        current_.valid = false;
        jobWarning_ = "pool sent a malformed extranonce assignment";
        if (badSize) jobWarning_ += " (invalid en2Size)";
        else if (badHex) jobWarning_ += " (non-hex prefix)";
        else if (tooLong || !fits) jobWarning_ += " (prefix does not fit nonce space)";
        jobWarning_ += "; stopped mining until a valid job arrives.";
    }
    extranonceGen_++;
}

std::string StratumSource::takeJobWarning() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string s;
    s.swap(jobWarning_);
    return s;
}

bool StratumSource::submit(const Job &job, const Solution &sol, std::string *err) {
    // The job id must be the one this solution was computed against - a
    // mining.notify arriving mid-batch updates jobId_, and submitting the old
    // solution under the new id gets it rejected. So use the id carried in the
    // job, falling back to the current one only for a job with none.
    std::string id = !job.jobId.empty() ? job.jobId : "";
    int ownedBits;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (id.empty()) id = jobId_;
        ownedBits = nonceBitsOwned_;
    }
    char nhex[17];
    snprintf(nhex, sizeof(nhex), "%016llx", (unsigned long long)sol.nonce);

    // params[2] is extraNonce2: the part of the nonce this miner owns, i.e.
    // the full nonce with the pool's extranonce prefix stripped off.
    const int prefixNibbles = (64 - ownedBits) / 4;
    const char *xn2 = nhex + (prefixNibbles < 16 ? prefixNibbles : 16);

    std::lock_guard<std::mutex> lk(submitMu_);
    const int rid = nextId_++;
    char payload[512];
    snprintf(payload, sizeof(payload),
             "{\"id\":%d,\"method\":\"mining.submit\",\"params\":"
             "[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}",
             rid, login_.c_str(), id.c_str(), xn2, kNTime, nhex);
    if (!sendLine(payload)) {
        *err = "submit send failed";
        return false;
    }
    // The pool's verdict arrives asynchronously on the reader thread, which
    // matches it back to this id. A send failure is the only thing this call
    // can report; poolCounters() carries what the pool actually said.
    {
        std::lock_guard<std::mutex> pk(mu_);
        submitted_++;
    }
    return connected_;
}

bool StratumSource::poolCounters(uint64_t *accepted, uint64_t *rejected,
                                 uint64_t *pending, std::string *lastError) const {
    std::lock_guard<std::mutex> lk(mu_);
    *accepted = accepted_;
    *rejected = rejected_;
    *pending = (submitted_ > accepted_ + rejected_) ? submitted_ - accepted_ - rejected_ : 0;
    *lastError = lastSubmitError_;
    return true;
}

std::string StratumSource::takeSubmitVerdict() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string v;
    v.swap(verdict_);
    return v;
}

}  // namespace om

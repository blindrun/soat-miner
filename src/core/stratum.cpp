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
    return socketSend(fd_, line.data(), line.size()) > 0;
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
    if (getaddrinfo(host_.c_str(), port.c_str(), &hints, &res) != 0) {
        *err = "cannot resolve " + host_;
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
        OM_CLOSESOCKET(fd_);
        fd_ = OM_INVALID_SOCKET;
        freeaddrinfo(res);
        *err = "cannot connect to " + desc_;
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
            j.valid = true;

            std::lock_guard<std::mutex> lk(mu_);
            current_ = j;
            jobId_ = unquote(p[0]);
            haveJob_ = true;
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

    // A response. The only one we care about is the subscribe reply, which
    // carries the extranonce prefix.
    std::string idRaw;
    if (rawValue(line, "id", &idRaw) && idRaw == "1") {
        std::string result;
        if (!rawValue(line, "result", &result)) return;
        const auto r = splitArray(result);
        if (r.size() >= 2) {
            const std::string xn = unquote(r[r.size() - 2]);
            int nonceBytes = atoi(unquote(r[r.size() - 1]).c_str());
            if (nonceBytes <= 0 || nonceBytes > 8) nonceBytes = 8;

            uint64_t prefix = 0;
            int prefixNibbles = 0;
            for (char c : xn) {
                if (!isxdigit((unsigned char)c)) continue;
                const int v = (c <= '9') ? (c - '0')
                                         : ((c | 0x20) - 'a' + 10);
                prefix = (prefix << 4) | (uint64_t)v;
                prefixNibbles++;
            }
            const int ownedBits = nonceBytes * 8;
            std::lock_guard<std::mutex> lk(mu_);
            nonceBitsOwned_ = ownedBits > 64 ? 64 : ownedBits;
            noncePrefix_ = (prefixNibbles > 0)
                               ? (prefix << nonceBitsOwned_)
                               : 0ULL;
        }
    }
}

bool StratumSource::submit(const Job &job, const Solution &sol, std::string *err) {
    (void)job;
    std::string id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        id = jobId_;
    }
    char nhex[17];
    snprintf(nhex, sizeof(nhex), "%016llx", (unsigned long long)sol.nonce);

    std::lock_guard<std::mutex> lk(submitMu_);
    const int rid = nextId_++;
    char payload[512];
    snprintf(payload, sizeof(payload),
             "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\"]}",
             rid, login_.c_str(), id.c_str(), nhex);
    if (!sendLine(payload)) {
        *err = "submit send failed";
        return false;
    }
    // Optimistic: the pool replies asynchronously and the reader thread would
    // have to correlate ids. Shares are cheap and the host already verified
    // the hit, so a send failure is the only thing treated as a rejection.
    return connected_;
}

}  // namespace om

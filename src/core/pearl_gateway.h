// Work source for Pearl: line-delimited JSON-RPC to pearl-gateway.
//
// This lives in core/ rather than under the algorithm because it is a protocol
// client, and core/ is where the other one already lives: stratum.h is just as
// algorithm-specific - it parses Ergo's own mining.notify field layout - and
// sits here for the same reason. What is under src/algos/pearl-pow/ is the
// arithmetic; what talks to a network peer belongs next to the code that owns
// the run loop.
//
// Pearl's miner does not talk to the node. It talks to `pearl-gateway`, which
// owns the node connection, builds the coinbase, assembles the block and runs
// the plonky2 prover that turns a submitted PlainProof into the certificate a
// block carries. That last step is why the split exists and why this cannot be
// a solo path against `getblocktemplate` instead: proving is not optional and
// not something a mining loop can do.
//
// The protocol is two methods, newline-framed over TCP (or a Unix socket,
// which we do not use):
//
//   -> {"jsonrpc":"2.0","method":"getMiningInfo","params":{},"id":1}
//   <- {"jsonrpc":"2.0","result":{"incomplete_header_bytes":"<base64 76>",
//                                 "target":<integer>,"cert_version":3},"id":1}
//   -> {"jsonrpc":"2.0","method":"submitPlainProof","params":{
//         "plain_proof":"<base64 bincode>",
//         "mining_job":{"incomplete_header_bytes":"<base64 76>",
//                       "target":<integer>,"cert_version":3}},"id":2}
//   <- {"jsonrpc":"2.0","result":"submitted","id":2}
//
// "submitted" is an acknowledgement, not an acceptance. The gateway proves and
// submits asynchronously, so the verdict never comes back on this connection -
// which is why poolCounters() is left unimplemented rather than reporting
// numbers it cannot know.
//
// Two details worth stating because they are easy to get wrong:
//
//  * `target` is the BLOCK target as a bare JSON integer, far too wide for a
//    double. It is read as a decimal string and converted to limbs, the same
//    way the Ergo pool path reads its own target. The bound a transcript is
//    measured against is this scaled by the mining configuration, and that
//    scaling belongs to the algorithm, not here.
//  * the gateway drops a submission whose header does not match its CURRENT
//    template, silently and with only a warning in its own log. So the header
//    a win was mined against has to travel with the win.

#pragma once

#include <stdio.h>
#include <string.h>

#include <string>

#include "algo.h"
#include "http.h"
#include "platform.h"
#include "run.h"
#include "../algos/pearl-pow/job.h"

namespace om {

/**
 * Encodes the pieces of a Pearl job that do not fit Job's fixed fields.
 *
 * Job carries a 32-byte message and a 256-bit target because that covers every
 * other PoW here; Pearl's header stub is 76 bytes and the certificate version
 * has nowhere to live at all. Both ride in `extra`, which the core treats as
 * opaque, as "<152 hex chars><colon><cert version>".
 */
inline std::string packPearlExtra(const uint8_t header[76], int certVersion) {
    static const char *d = "0123456789abcdef";
    std::string s;
    s.reserve(160);
    for (int i = 0; i < 76; i++) {
        s += d[header[i] >> 4];
        s += d[header[i] & 15];
    }
    s += ':';
    s += std::to_string(certVersion);
    return s;
}

inline bool unpackPearlExtra(const std::string &extra, uint8_t header[76],
                             int *certVersion) {
    const size_t colon = extra.find(':');
    if (colon != 152) return false;
    if (!hexToBytes(extra.substr(0, 152), header, 76)) return false;
    *certVersion = atoi(extra.c_str() + colon + 1);
    return *certVersion > 0;
}

class PearlGatewaySource : public JobSource {
   public:
    PearlGatewaySource(std::string host, int port)
        : host_(std::move(host)), port_(port) {
        desc_ = host_ + ":" + std::to_string(port_) + " (pearl-gateway)";
    }

    ~PearlGatewaySource() override { disconnect(); }

    const char *describe() const override { return desc_.c_str(); }

    bool fetch(Job *job) override {
        std::string resp;
        if (!call("{\"jsonrpc\":\"2.0\",\"method\":\"getMiningInfo\",\"params\":{},"
                  "\"id\":" + std::to_string(++id_) + "}",
                  &resp))
            return false;

        std::string headerB64, targetDec, certStr;
        if (!jsonString(resp, "incomplete_header_bytes", &headerB64)) return false;
        if (!jsonNumber(resp, "target", &targetDec)) return false;
        if (!jsonNumber(resp, "cert_version", &certStr)) return false;

        std::vector<uint8_t> header;
        if (!pearl::base64Decode(headerB64, &header) || header.size() != 76)
            return false;

        // Job::msg is what the run loop compares to notice a new job, and
        // Pearl has no 32-byte message of its own. A hash of the header is
        // exactly the right identity: it moves when the template does, for any
        // reason - new height, new transactions, new timestamp.
        pearl::b3::hash(nullptr, header.data(), header.size(), job->msg);
        decimalToLimbs(targetDec, job->target);
        job->extra = packPearlExtra(header.data(), atoi(certStr.c_str()));
        job->epoch = 0;      // nothing per-epoch to build; prepare() runs once
        job->valid = true;
        return true;
    }

    bool submit(const Job &job, const Solution &sol, std::string *err) override {
        if (sol.extra.empty()) {
            *err = "no proof attached to the solution";
            return false;
        }
        uint8_t header[76];
        int cert = 0;
        if (!unpackPearlExtra(job.extra, header, &cert)) {
            *err = "job is missing its header";
            return false;
        }
        std::vector<uint8_t> headerBytes(header, header + 76);

        const std::string body =
            "{\"jsonrpc\":\"2.0\",\"method\":\"submitPlainProof\",\"params\":{"
            "\"plain_proof\":\"" + sol.extra + "\","
            "\"mining_job\":{\"incomplete_header_bytes\":\"" +
            pearl::base64(headerBytes) + "\",\"target\":" + limbsToDecimal(job.target) +
            ",\"cert_version\":" + std::to_string(cert) + "}},\"id\":" +
            std::to_string(++id_) + "}";

        std::string resp;
        if (!call(body, &resp)) {
            *err = "gateway did not answer";
            return false;
        }
        if (resp.find("\"error\"") != std::string::npos &&
            resp.find("\"error\": null") == std::string::npos &&
            resp.find("\"error\":null") == std::string::npos) {
            *err = resp;
            return false;
        }
        return true;
    }

   private:
    /** 256-bit limbs back to the decimal integer the gateway's schema wants. */
    static std::string limbsToDecimal(const uint64_t limbs[4]) {
        // Repeated division by 10^19 would need 256-bit division; long division
        // over 32-bit limbs is shorter and this runs once per submission.
        uint32_t w[8];
        for (int i = 0; i < 4; i++) {
            w[2 * i] = (uint32_t)limbs[i];
            w[2 * i + 1] = (uint32_t)(limbs[i] >> 32);
        }
        std::string out;
        bool nonzero = true;
        while (nonzero) {
            uint64_t rem = 0;
            nonzero = false;
            for (int i = 7; i >= 0; i--) {
                const uint64_t cur = (rem << 32) | w[i];
                w[i] = (uint32_t)(cur / 10);
                rem = cur % 10;
                if (w[i]) nonzero = true;
            }
            out += (char)('0' + (int)rem);
        }
        if (out.empty()) return "0";
        return std::string(out.rbegin(), out.rend());
    }

    bool connect() {
        disconnect();
        struct addrinfo hints {};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        struct addrinfo *res = nullptr;
        const std::string port = std::to_string(port_);
        if (getaddrinfo(host_.c_str(), port.c_str(), &hints, &res) != 0) return false;

        fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd_ == OM_INVALID_SOCKET) {
            freeaddrinfo(res);
            return false;
        }
        socketTimeout(fd_, 15);
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

    /**
     * One request, one response line. Reconnects once on failure, because the
     * gateway closes idle connections and a miner must not stop over that.
     */
    bool call(const std::string &body, std::string *resp) {
        for (int attempt = 0; attempt < 2; attempt++) {
            if (fd_ == OM_INVALID_SOCKET && !connect()) return false;
            const std::string line = body + "\n";
            if (socketSend(fd_, line.data(), line.size()) < 0) {
                disconnect();
                continue;
            }
            if (readLine(resp)) return true;
            disconnect();
        }
        return false;
    }

    bool readLine(std::string *out) {
        for (;;) {
            const size_t nl = pending_.find('\n');
            if (nl != std::string::npos) {
                *out = pending_.substr(0, nl);
                pending_.erase(0, nl + 1);
                return true;
            }
            char buf[8192];
            const int n = socketRecv(fd_, buf, sizeof(buf));
            if (n <= 0) return false;
            pending_.append(buf, (size_t)n);
            // A proof is ~67 KB and its response is tiny, but a wedged peer
            // must not be allowed to grow this without bound.
            if (pending_.size() > (1u << 20)) return false;
        }
    }

    std::string host_;
    int port_;
    std::string desc_;
    socket_t fd_ = OM_INVALID_SOCKET;
    std::string pending_;
    uint64_t id_ = 0;
};

}  // namespace om

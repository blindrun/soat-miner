// Pure Bitcoin-Stratum V1 helpers shared by the client and offline tests.
//
// Nothing here opens a socket, reads a wallet, or touches a GPU. Keeping the
// byte-order, target and JSON rules here makes them testable from fixtures.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace om {
namespace btc {

inline bool hexToVec(const std::string &hex, std::vector<uint8_t> *out) {
    if (hex.size() % 2 != 0) return false;
    out->clear();
    out->reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        int hi = -1, lo = -1;
        for (int k = 0; k < 2; k++) {
            const char c = hex[i + k];
            int v;
            if (c >= '0' && c <= '9') v = c - '0';
            else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
            else return false;
            (k == 0 ? hi : lo) = v;
        }
        out->push_back((uint8_t)((hi << 4) | lo));
    }
    return true;
}

inline std::string vecToHex(const uint8_t *p, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s(n * 2, '0');
    for (size_t i = 0; i < n; i++) {
        s[2 * i] = d[p[i] >> 4];
        s[2 * i + 1] = d[p[i] & 15];
    }
    return s;
}

/** A big-endian hexadecimal scalar sent by Stratum. */
inline bool hexScalar(const std::string &hex, uint32_t *out) {
    if (hex.empty() || hex.size() > 8) return false;
    uint32_t v = 0;
    for (char c : hex) {
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return false;
        v = (v << 4) | (uint32_t)d;
    }
    *out = v;
    return true;
}

inline void putLE32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/** Compact Bitcoin target (nBits) expanded into little-endian 64-bit limbs. */
inline bool compactToTarget(uint32_t compact, uint64_t target[4]) {
    memset(target, 0, 4 * sizeof(*target));
    const uint32_t exponent = compact >> 24;
    uint32_t mantissa = compact & 0x007fffff;
    if (mantissa == 0 || (compact & 0x00800000) != 0 || exponent > 32) return false;
    std::vector<uint8_t> bytes(32, 0);
    if (exponent <= 3) {
        mantissa >>= 8 * (3 - exponent);
        for (uint32_t i = 0; i < exponent; i++) bytes[i] = (uint8_t)(mantissa >> (8 * i));
    } else {
        const uint32_t at = exponent - 3;
        if (at + 3 > bytes.size()) return false;
        bytes[at] = (uint8_t)mantissa;
        bytes[at + 1] = (uint8_t)(mantissa >> 8);
        bytes[at + 2] = (uint8_t)(mantissa >> 16);
    }
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 8; b++) target[i] |= (uint64_t)bytes[i * 8 + b] << (8 * b);
    return true;
}

/** Bitcoin Stratum difficulty to its 256-bit share target. */
inline void difficultyToTarget(double diff, uint64_t target[4]) {
    uint32_t t[8] = {};
    int k = 6;
    if (!(diff > 0.0)) diff = 1.0;
    for (; k > 0 && diff > 1.0; k--) diff /= 4294967296.0;
    const uint64_t m = (uint64_t)(4294901760.0 / diff);
    if (m == 0 && k == 6) {
        memset(t, 0xff, sizeof(t));
    } else {
        t[k] = (uint32_t)m;
        if (k + 1 < 8) t[k + 1] = (uint32_t)(m >> 32);
    }
    for (int i = 0; i < 4; i++)
        target[i] = (uint64_t)t[2 * i] | ((uint64_t)t[2 * i + 1] << 32);
}

inline std::string jsonQuote(const std::string &raw) {
    static const char hex[] = "0123456789abcdef";
    std::string quoted;
    quoted.reserve(raw.size() + 2);
    quoted.push_back('"');
    for (unsigned char c : raw) {
        switch (c) {
            case '"': quoted += "\\\""; break;
            case '\\': quoted += "\\\\"; break;
            case '\b': quoted += "\\b"; break;
            case '\f': quoted += "\\f"; break;
            case '\n': quoted += "\\n"; break;
            case '\r': quoted += "\\r"; break;
            case '\t': quoted += "\\t"; break;
            default:
                if (c < 0x20) {
                    quoted += "\\u00";
                    quoted.push_back(hex[c >> 4]);
                    quoted.push_back(hex[c & 15]);
                } else {
                    quoted.push_back((char)c);
                }
        }
    }
    quoted.push_back('"');
    return quoted;
}

inline std::string subscribeRequest() {
    return "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"soat-miner/0.1\"]}";
}

inline std::string authorizeRequest(const std::string &login,
                                    const std::string &password) {
    return "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[" +
           jsonQuote(login) + "," + jsonQuote(password) + "]}";
}

inline std::string submitRequest(int id, const std::string &login,
                                 const std::string &jobId,
                                 const std::string &extranonce2,
                                 uint32_t ntime, uint32_t nonce) {
    char ntimeHex[9], nonceHex[9];
    snprintf(ntimeHex, sizeof(ntimeHex), "%08x", ntime);
    snprintf(nonceHex, sizeof(nonceHex), "%08x", nonce);
    return "{\"id\":" + std::to_string(id) +
           ",\"method\":\"mining.submit\",\"params\":[" +
           jsonQuote(login) + "," + jsonQuote(jobId) + "," +
           jsonQuote(vecToHex((const uint8_t *)extranonce2.data(), extranonce2.size())) +
           ",\"" + ntimeHex + "\",\"" + nonceHex + "\"]}";
}

}  // namespace btc
}  // namespace om

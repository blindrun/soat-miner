// Minimal HTTP/1.1 client and JSON field extraction.
//
// Deliberately dependency-free. This talks to a node on localhost and needs
// one GET and one POST of small, well-formed JSON; pulling in libcurl and a
// JSON library on a project whose entire selling point is "you can read all of
// it" is a bad trade.

#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "platform.h"

namespace om {

struct HttpTarget {
    std::string host = "127.0.0.1";
    int port = 9053;
    std::string apiKey;
};

inline std::string httpRequest(const HttpTarget &t, const std::string &method,
                               const std::string &path, const std::string &body,
                               bool *ok) {
    *ok = false;
    struct addrinfo hints {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = nullptr;
    const std::string port = std::to_string(t.port);
    if (getaddrinfo(t.host.c_str(), port.c_str(), &hints, &res) != 0) return "";

    socket_t fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == OM_INVALID_SOCKET) { freeaddrinfo(res); return ""; }
    socketTimeout(fd, 10);

    if (::connect(fd, res->ai_addr, (int)res->ai_addrlen) != 0) {
        OM_CLOSESOCKET(fd);
        freeaddrinfo(res);
        return "";
    }
    freeaddrinfo(res);

    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + t.host +
                      "\r\nConnection: close\r\n";
    if (!t.apiKey.empty()) req += "api_key: " + t.apiKey + "\r\n";
    if (!body.empty()) {
        req += "Content-Type: application/json\r\n";
        req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    req += "\r\n" + body;

    if (socketSend(fd, req.data(), req.size()) < 0) { OM_CLOSESOCKET(fd); return ""; }

    std::string resp;
    char buf[4096];
    int n;
    while ((n = socketRecv(fd, buf, sizeof(buf))) > 0) resp.append(buf, (size_t)n);
    OM_CLOSESOCKET(fd);

    const size_t hdr = resp.find("\r\n\r\n");
    if (hdr == std::string::npos) return "";
    *ok = resp.compare(0, 12, "HTTP/1.1 200") == 0 ||
          resp.compare(0, 12, "HTTP/1.0 200") == 0;
    return resp.substr(hdr + 4);
}

/** Extracts a quoted string value for "key". */
inline bool jsonString(const std::string &j, const char *key, std::string *out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    const size_t a = j.find('"', p);
    if (a == std::string::npos) return false;
    const size_t b = j.find('"', a + 1);
    if (b == std::string::npos) return false;
    *out = j.substr(a + 1, b - a - 1);
    return true;
}

/** Extracts a bare integer value (possibly far wider than 64 bits) for "key". */
inline bool jsonNumber(const std::string &j, const char *key, std::string *out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '"')) p++;
    const size_t a = p;
    while (p < j.size() && isdigit((unsigned char)j[p])) p++;
    if (p == a) return false;
    *out = j.substr(a, p - a);
    return true;
}

/** Decimal string -> 256-bit little-endian limbs. */
inline void decimalToLimbs(const std::string &dec, uint64_t out[4]) {
    for (int i = 0; i < 4; i++) out[i] = 0;
    for (char ch : dec) {
        if (!isdigit((unsigned char)ch)) continue;
        uint64_t carry = (uint64_t)(ch - '0');
        for (int i = 0; i < 4; i++) {
            uint64_t hi = 0;
            out[i] = mulAdd64(out[i], 10ULL, carry, &hi);
            carry = hi;
        }
    }
}

inline bool hexToBytes(const std::string &hex, uint8_t *out, int n) {
    if ((int)hex.size() < n * 2) return false;
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(hex.c_str() + i * 2, "%2x", &v) != 1) return false;
        out[i] = (uint8_t)v;
    }
    return true;
}

}  // namespace om

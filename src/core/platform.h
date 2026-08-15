// Thin platform shim so the miner builds on both Linux and Windows.
//
// Only three things actually differ: sockets, sleeping, and the absence of
// __int128 on MSVC. Everything else is standard C++17.

#pragma once

#include <stdint.h>

#include <chrono>
#include <string>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
using socket_t = SOCKET;
#define OM_INVALID_SOCKET INVALID_SOCKET
#define OM_CLOSESOCKET closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#define OM_INVALID_SOCKET (-1)
#define OM_CLOSESOCKET ::close
#endif

namespace om {

/** Initialises Winsock on Windows; a no-op elsewhere. */
inline bool platformInit() {
#if defined(_WIN32)
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
    return true;
#endif
}

inline void platformShutdown() {
#if defined(_WIN32)
    WSACleanup();
#endif
}

inline void sleepSeconds(int s) {
    std::this_thread::sleep_for(std::chrono::seconds(s));
}

inline int socketSend(socket_t fd, const char *buf, size_t len) {
#if defined(_WIN32)
    return ::send(fd, buf, (int)len, 0);
#else
    return (int)::send(fd, buf, len, 0);
#endif
}

inline int socketRecv(socket_t fd, char *buf, size_t len) {
#if defined(_WIN32)
    return ::recv(fd, buf, (int)len, 0);
#else
    return (int)::recv(fd, buf, len, 0);
#endif
}

inline void socketTimeout(socket_t fd, int seconds) {
#if defined(_WIN32)
    DWORD ms = (DWORD)(seconds * 1000);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof(ms));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof(ms));
#else
    struct timeval tv {
        seconds, 0
    };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

/**
 * 64x64 -> 128 multiply-accumulate, used to parse the decimal target.
 *
 * MSVC has no __int128, so this is done with 32-bit halves rather than
 * #ifdef-ing two versions of the parser. It runs once per job; the clarity is
 * worth more than the cycles.
 */
inline uint64_t mulAdd64(uint64_t a, uint64_t b, uint64_t addend, uint64_t *carryOut) {
    const uint64_t aLo = a & 0xffffffffULL, aHi = a >> 32;
    const uint64_t bLo = b & 0xffffffffULL, bHi = b >> 32;

    uint64_t p0 = aLo * bLo;
    uint64_t p1 = aLo * bHi;
    uint64_t p2 = aHi * bLo;
    uint64_t p3 = aHi * bHi;

    uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffULL) + (p2 & 0xffffffffULL);
    uint64_t lo = (p0 & 0xffffffffULL) | (mid << 32);
    uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);

    uint64_t sum = lo + addend;
    if (sum < lo) hi++;

    *carryOut = hi;
    return sum;
}

}  // namespace om

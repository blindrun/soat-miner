// How a Bitcoin-stratum job travels through the core as an opaque blob.
//
// Job::msg is 32 bytes and a Bitcoin header is 80, so the header cannot ride
// there. It rides in Job::extra instead, which the core treats as opaque - the
// same seam Ergo uses for `pk` and Pearl uses for its openings.
//
// Job::msg carries the MERKLE ROOT, and that is load-bearing rather than
// arbitrary. run.cpp only adopts a new job when msg or epoch changes:
//
//     if (memcmp(fresh.msg, job.msg, 32) != 0 || fresh.epoch != job.epoch)
//
// The merkle root is exactly 32 bytes and is the one field that changes on
// every extranonce2 roll, so using it makes the core's existing change
// detection do the right thing with no modification. Put the header in extra
// and leave msg alone and every extranonce2 roll is silently dropped, which
// looks like a pool rejecting duplicate shares for no reason.

#pragma once

#include <stdint.h>

#include <cstring>
#include <string>

namespace om {

/** A serialised Bitcoin block header. */
static const size_t kBtcHeaderBytes = 80;

/** Byte offset of the nonce within the header. */
static const size_t kBtcNonceOffset = 76;

/** extra = 80-byte header, then one length byte, then the extranonce2. */
inline std::string encodeBtcJobExtra(const uint8_t header[80],
                                     const std::string &xn2) {
    std::string s;
    s.reserve(kBtcHeaderBytes + 1 + xn2.size());
    s.append(reinterpret_cast<const char *>(header), kBtcHeaderBytes);
    s.push_back((char)(uint8_t)xn2.size());
    s.append(xn2);
    return s;
}

inline bool btcJobHeader(const std::string &extra, uint8_t out[80]) {
    if (extra.size() < kBtcHeaderBytes) return false;
    memcpy(out, extra.data(), kBtcHeaderBytes);
    return true;
}

inline bool btcJobExtranonce2(const std::string &extra, std::string *out) {
    if (extra.size() < kBtcHeaderBytes + 1) return false;
    const size_t n = (uint8_t)extra[kBtcHeaderBytes];
    if (extra.size() < kBtcHeaderBytes + 1 + n) return false;
    out->assign(extra, kBtcHeaderBytes + 1, n);
    return true;
}

}  // namespace om

// Ravencoin payout address validation.
//
// A real Base58Check decode, not a shape test, because the mistake this
// catches is a typo in a 34-character string a user copied by hand. A shape
// test accepts a transposed pair of characters and the miner then works
// perfectly for hours while paying an address that does not exist. The
// checksum is four bytes of double SHA-256 and costs nothing to check once at
// startup.
//
// Ravencoin's version bytes:
//   60 (0x3C)  P2PKH, the ordinary payout address, renders as 'R...'
//  122 (0x7A)  P2SH, renders as 'r...'
//
// Both are accepted: a pool will pay either. Anything else - a Bitcoin
// address, an Ergo address, a testnet address - is refused by version byte
// rather than by prefix character, which is the same check the network does.
//
// This never sees a private key or a seed. A pool login is a public payout
// address; if something arriving here looks like recovery material it is
// refused outright rather than sent to a pool.

#pragma once

#include <stdint.h>

#include <cstring>
#include <string>
#include <vector>

#include "sha256.h"

namespace om {

/** Base58 decode. Returns false on any character outside the alphabet. */
inline bool base58Decode(const std::string &s, std::vector<uint8_t> *out) {
    static const char *kAlphabet =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    if (s.empty() || s.size() > 128) return false;

    std::vector<uint8_t> b256;
    b256.reserve(s.size());
    for (char c : s) {
        const char *p = strchr(kAlphabet, c);
        if (p == nullptr || c == '\0') return false;
        int carry = (int)(p - kAlphabet);
        for (size_t i = b256.size(); i-- > 0;) {
            const int x = b256[i] * 58 + carry;
            b256[i] = (uint8_t)(x & 0xff);
            carry = x >> 8;
        }
        while (carry) {
            b256.insert(b256.begin(), (uint8_t)(carry & 0xff));
            carry >>= 8;
        }
    }
    // Each leading '1' is a leading zero byte, which the arithmetic above
    // cannot represent.
    for (char c : s) {
        if (c != '1') break;
        b256.insert(b256.begin(), 0);
    }
    *out = b256;
    return true;
}

/** Why an address was refused, so the miner can say something useful. */
enum class RvnAddressVerdict {
    kOk,
    kEmpty,
    kNotBase58,
    kWrongLength,
    kBadChecksum,
    kWrongNetwork,
    kLooksLikeSecret,
};

inline RvnAddressVerdict rvnAddressCheck(const std::string &address) {
    if (address.empty()) return RvnAddressVerdict::kEmpty;

    // Recovery material must never reach a pool, and it is easy to paste by
    // accident. Space-separated input is a mnemonic, not an address.
    if (address.find(' ') != std::string::npos)
        return RvnAddressVerdict::kLooksLikeSecret;

    std::vector<uint8_t> raw;
    if (!base58Decode(address, &raw)) return RvnAddressVerdict::kNotBase58;
    if (raw.size() != 25) return RvnAddressVerdict::kWrongLength;

    uint8_t digest[32];
    sha256d(raw.data(), 21, digest);
    if (memcmp(digest, raw.data() + 21, 4) != 0)
        return RvnAddressVerdict::kBadChecksum;

    if (raw[0] != 60 && raw[0] != 122) return RvnAddressVerdict::kWrongNetwork;
    return RvnAddressVerdict::kOk;
}

inline bool rvnAddressValid(const std::string &address) {
    return rvnAddressCheck(address) == RvnAddressVerdict::kOk;
}

/** One line the user can act on. */
inline std::string rvnAddressReason(RvnAddressVerdict v) {
    switch (v) {
        case RvnAddressVerdict::kOk:
            return "";
        case RvnAddressVerdict::kEmpty:
            return "no --wallet given; pool mining needs your Ravencoin payout "
                   "address";
        case RvnAddressVerdict::kNotBase58:
            return "that is not a Ravencoin address: it contains characters "
                   "Base58 does not use (there is no 0, O, I or l)";
        case RvnAddressVerdict::kWrongLength:
            return "that is not a Ravencoin address: the wrong number of "
                   "characters, so something was truncated or pasted twice";
        case RvnAddressVerdict::kBadChecksum:
            return "the checksum does not match, so one or two characters are "
                   "wrong. Copy the address again rather than retyping it - "
                   "mining to it would pay an address that does not exist";
        case RvnAddressVerdict::kWrongNetwork:
            return "that is a valid Base58Check address but not a Ravencoin "
                   "one. Ravencoin addresses start with R (or r for a "
                   "multisig address)";
        case RvnAddressVerdict::kLooksLikeSecret:
            return "that looks like recovery material, not a payout address. "
                   "Never give a miner a seed phrase or a private key";
    }
    return "";
}

}  // namespace om

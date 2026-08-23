// Payout address validation for the KawPow coins.
//
// A real Base58Check decode, not a shape test, because the mistake this
// catches is a typo in a 34-character string a user copied by hand. A shape
// test accepts a transposed pair of characters and the miner then works
// perfectly for hours while paying an address that does not exist. The
// checksum is four bytes of double SHA-256 and costs nothing to check once at
// startup.
//
// One algorithm, several chains. Every version byte below was read out of that
// chain's own chainparams.cpp, inside its mainnet block, and not inferred from
// the letter an address happens to start with:
//
//   Ravencoin  60, 122   RavenProject/Ravencoin
//   Neurai     53, 117   NeuraiProject/Neurai
//   Neoxa      38, 122   NeoxaChain/Neoxa
//
// Note 122 is BOTH Ravencoin's and Neoxa's P2SH byte, so a version-122 address
// is genuinely ambiguous between the two chains and this says so rather than
// picking one.
//
// Kringle (KRGN) is deliberately absent: no source repository could be found
// to read a version byte out of, and guessing one would produce a miner that
// accepts an address it cannot be paid at.
//
// Naming the chain matters as much as accepting it. A user who pastes a Neurai
// address into a Ravencoin launcher gets told which chain it belongs to, which
// is a diagnosis; "not a Ravencoin address" is a dead end.
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
/** A chain this miner can be paid on, and the Base58 version it uses. */
struct KawpowChain {
    uint8_t version;
    const char *coin;
    const char *kind;
};

/** Read from each chain's own chainparams.cpp, mainnet block. See above. */
inline const KawpowChain *kawpowChains(int *count) {
    static const KawpowChain kChains[] = {
        {60, "Ravencoin", "P2PKH"},
        {122, "Ravencoin or Neoxa", "P2SH"},
        {53, "Neurai", "P2PKH"},
        {117, "Neurai", "P2SH"},
        {38, "Neoxa", "P2PKH"},
    };
    *count = (int)(sizeof(kChains) / sizeof(kChains[0]));
    return kChains;
}

/** Which chain an accepted address belongs to, or nullptr. */
inline const KawpowChain *kawpowChainFor(uint8_t version) {
    int n = 0;
    const KawpowChain *c = kawpowChains(&n);
    for (int i = 0; i < n; i++)
        if (c[i].version == version) return &c[i];
    return nullptr;
}

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

    if (kawpowChainFor(raw[0]) == nullptr)
        return RvnAddressVerdict::kWrongNetwork;
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
            return "that is a valid Base58Check address but not one of the "
                   "chains this miner knows how to be paid on. KawPow covers "
                   "Ravencoin (R...), Neurai (N...) and Neoxa";
        case RvnAddressVerdict::kLooksLikeSecret:
            return "that looks like recovery material, not a payout address. "
                   "Never give a miner a seed phrase or a private key";
    }
    return "";
}

}  // namespace om

namespace om {

/** The chain an address belongs to, for logging. Empty if it is not valid. */
inline std::string kawpowAddressChain(const std::string &address) {
    std::vector<uint8_t> raw;
    if (!base58Decode(address, &raw) || raw.size() != 25) return std::string();
    const KawpowChain *c = kawpowChainFor(raw[0]);
    return c ? std::string(c->coin) + " " + c->kind : std::string();
}

}  // namespace om

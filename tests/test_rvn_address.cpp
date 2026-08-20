// Ravencoin address validation.
//
// The case that matters is the near-miss: an address with two characters
// transposed is still 34 Base58 characters starting with R, so a shape test
// accepts it and the miner then runs perfectly while paying nowhere. Only the
// checksum catches that, so the test spends most of its effort there.
//
// The valid addresses below are constructed rather than taken from a wallet:
// each is a chosen 20-byte hash with Ravencoin's version byte and a real
// computed checksum. No live address belongs in a test.

#include <cstdio>
#include <string>
#include <vector>

#include "core/rvn_address.h"

using namespace om;

namespace {

int g_fail = 0;

void check(bool ok, const std::string &what) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) g_fail++;
}

/** Base58Check-encode a version byte plus a 20-byte hash. */
std::string encode(uint8_t version, const std::vector<uint8_t> &hash160) {
    static const char *kAlphabet =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> payload;
    payload.push_back(version);
    payload.insert(payload.end(), hash160.begin(), hash160.end());
    uint8_t digest[32];
    sha256d(payload.data(), payload.size(), digest);
    payload.insert(payload.end(), digest, digest + 4);

    std::vector<uint8_t> b58;
    for (uint8_t byte : payload) {
        int carry = byte;
        for (size_t i = b58.size(); i-- > 0;) {
            const int x = b58[i] * 256 + carry;
            b58[i] = (uint8_t)(x % 58);
            carry = x / 58;
        }
        while (carry) {
            b58.insert(b58.begin(), (uint8_t)(carry % 58));
            carry /= 58;
        }
    }
    std::string out;
    for (uint8_t byte : payload) {
        if (byte != 0) break;
        out += '1';
    }
    for (uint8_t v : b58) out += kAlphabet[v];
    return out;
}

std::vector<uint8_t> hash160(uint8_t fill) {
    return std::vector<uint8_t>(20, fill);
}

}  // namespace

int main() {
    const std::string p2pkh = encode(60, hash160(0x11));
    const std::string p2sh = encode(122, hash160(0x22));

    check(!p2pkh.empty() && p2pkh[0] == 'R',
          "a version-60 address renders as R... (" + p2pkh + ")");
    check(rvnAddressValid(p2pkh), "a well-formed P2PKH address is accepted");
    check(rvnAddressValid(p2sh),
          "a version-122 P2SH address is accepted too (" + p2sh + ")");

    // The whole reason this file exists.
    std::string transposed = p2pkh;
    for (size_t i = 1; i + 1 < transposed.size(); i++) {
        if (transposed[i] != transposed[i + 1]) {
            std::swap(transposed[i], transposed[i + 1]);
            break;
        }
    }
    check(transposed != p2pkh && !rvnAddressValid(transposed),
          "two transposed characters are caught by the checksum, not the shape");
    check(rvnAddressCheck(transposed) == RvnAddressVerdict::kBadChecksum,
          "and the reason given is the checksum, not something vaguer");

    check(!rvnAddressValid(p2pkh.substr(0, p2pkh.size() - 1)),
          "a truncated address is refused");
    check(!rvnAddressValid(p2pkh + "R"), "an over-long address is refused");
    check(rvnAddressCheck("") == RvnAddressVerdict::kEmpty,
          "an empty wallet is reported as empty, not as malformed");

    // A Bitcoin address is Base58Check-valid and must still be refused: the
    // version byte is the check, not the leading character.
    const std::string btc = encode(0, hash160(0x33));
    check(rvnAddressCheck(btc) == RvnAddressVerdict::kWrongNetwork,
          "a valid Bitcoin address is refused on its version byte (" + btc + ")");

    check(rvnAddressCheck("R0OIl" + p2pkh.substr(5)) ==
              RvnAddressVerdict::kNotBase58,
          "characters Base58 does not have are named as the problem");

    check(rvnAddressCheck("word1 word2 word3 word4 word5 word6") ==
              RvnAddressVerdict::kLooksLikeSecret,
          "space-separated recovery material never reaches a pool");

    // An Ergo address, the other thing this miner accepts elsewhere, must not
    // be mistaken for a Ravencoin one.
    check(!rvnAddressValid(
              "9fRAWhdxEsTcdb8PhGNrZfwqa65zfkuYHAMmkQLcic1gdLSV5vA"),
          "an Ergo address is not accepted as a Ravencoin one");

    if (g_fail == 0) printf("RVN address checks passed\n");
    return g_fail == 0 ? 0 : 1;
}

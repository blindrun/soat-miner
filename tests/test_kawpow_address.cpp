// Payout address validation for the KawPow chains.
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

#include "core/kawpow_address.h"

using namespace om;

namespace {

int g_fail = 0;

void check(bool ok, const std::string &what,
           const std::string &detail = std::string()) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        if (!detail.empty()) printf("        got %s\n", detail.c_str());
        g_fail++;
    }
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
          "an Ergo address is not accepted by any KawPow chain");

    // --- the other chains this algorithm serves ---------------------------
    //
    // Every version byte here was read out of that chain's own chainparams.cpp
    // inside its mainnet block. Without these checks the chain table could be
    // wrong in any entry and every test above would still pass, because they
    // only ever exercise Ravencoin.
    const std::string neurai = encode(53, hash160(0x44));
    const std::string neuraiP2SH = encode(117, hash160(0x55));
    const std::string neoxa = encode(38, hash160(0x66));

    check(rvnAddressValid(neurai),
          "a Neurai address is accepted (" + neurai + ")");
    check(kawpowAddressChain(neurai) == "Neurai P2PKH",
          "and is named as Neurai, not merely allowed",
          kawpowAddressChain(neurai));
    check(rvnAddressValid(neuraiP2SH) &&
              kawpowAddressChain(neuraiP2SH) == "Neurai P2SH",
          "a Neurai P2SH address is accepted and named");
    check(rvnAddressValid(neoxa) && kawpowAddressChain(neoxa) == "Neoxa P2PKH",
          "a Neoxa address is accepted and named (" + neoxa + ")");

    // 122 is Ravencoin's P2SH byte AND Neoxa's. An address carrying it is
    // genuinely ambiguous, and saying so beats picking one silently.
    check(kawpowAddressChain(p2sh) == "Ravencoin or Neoxa P2SH",
          "the shared 122 byte is reported as ambiguous, not resolved",
          kawpowAddressChain(p2sh));

    // A version byte outside the table must still be refused. Kringle is the
    // live example: no source repository could be found to read a byte out of,
    // so it is deliberately absent and its addresses will not validate.
    check(rvnAddressCheck(encode(45, hash160(0x77))) ==
              RvnAddressVerdict::kWrongNetwork,
          "an unknown chain's version byte is still refused");

    if (g_fail == 0) printf("KawPow address checks passed\n");
    return g_fail == 0 ? 0 : 1;
}

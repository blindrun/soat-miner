// KawPow end to end against a real Ravencoin mainnet block.
//
// The published vectors in test_kawpow_hash prove the algorithm. This proves
// the plumbing around it: that the 80-byte header prefix is hashed the way
// Ravencoin hashes it, that the epoch is derived from the right height, that
// the byte order surviving from a block explorer to a 256-bit compare is the
// one consensus uses, and that the target derived from nBits accepts the block
// its own network accepted.
//
// It is the same shape as the BC3 gate against mainnet block 50204, and for
// the same reason: a miner can pass every unit test and still submit garbage
// because one field was read in the wrong direction.
//
// Byte order, which is where this test earns its place. Ravencoin feeds
// ProgPoW the header hash as the uint256's *display* bytes, not its serialised
// bytes - hash.cpp:270 does to_hash256(nHeaderHash.GetHex()) - so the header
// hash is the double-SHA256 of the 80-byte prefix REVERSED. The mix hash and
// the final hash come back out the same way, which is why the final hash a
// block explorer shows for a KawPow block IS the ProgPoW output: for these
// blocks CBlockHeader::GetHash() returns the PoW hash itself, not a separate
// identifier.
//
// Regenerate the pinned vector with:
//   python3 tests/kawpow_block.py <height>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "algos/kawpow/kawpow.h"

using namespace om::kawpow;

namespace {

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hexToWords(const char *hex, uint32_t *out, int words) {
    if (strlen(hex) != (size_t)words * 8) return false;
    for (int i = 0; i < words * 4; i++) {
        const int hi = hexVal(hex[i * 2]), lo = hexVal(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        if (i % 4 == 0) out[i / 4] = 0;
        out[i / 4] |= (uint32_t)((hi << 4) | lo) << ((i % 4) * 8);
    }
    return true;
}

std::string wordsToHex(const uint32_t *w, int words) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < words; i++)
        for (int b = 0; b < 4; b++) {
            const uint8_t v = (uint8_t)(w[i] >> (b * 8));
            s += d[v >> 4];
            s += d[v & 15];
        }
    return s;
}

/** Bitcoin-style compact target: one exponent byte and a 24-bit mantissa. */
void bitsToTarget(uint32_t bits, uint64_t target[4]) {
    const uint32_t exponent = bits >> 24;
    const uint64_t mantissa = bits & 0x007fffffu;
    for (int i = 0; i < 4; i++) target[i] = 0;
    if (exponent <= 3) {
        target[0] = mantissa >> (8 * (3 - exponent));
        return;
    }
    const int shift = 8 * ((int)exponent - 3);   // bit position of the mantissa
    for (int b = 0; b < 3; b++) {
        const uint64_t byteVal = (mantissa >> (8 * b)) & 0xffu;
        const int bitPos = shift + 8 * b;
        if (bitPos >= 256) continue;
        target[bitPos / 64] |= byteVal << (bitPos % 64);
        // A byte can straddle a limb boundary only if bitPos % 64 > 56, which
        // cannot happen here: shift is a multiple of 8, so bitPos is too.
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: test_kawpow_block <headerHashHex> <height> <nonceHex> "
                "<mixHex> <finalHex> <bitsHex>\n");
        return 2;
    }

    uint32_t header[8], wantMix[8], wantFinal[8];
    if (!hexToWords(argv[1], header, 8) || !hexToWords(argv[4], wantMix, 8) ||
        !hexToWords(argv[5], wantFinal, 8)) {
        fprintf(stderr, "a 64-hex-character argument was malformed\n");
        return 2;
    }
    const int height = atoi(argv[2]);
    const uint64_t nonce = strtoull(argv[3], nullptr, 16);
    const uint32_t bits = (uint32_t)strtoul(argv[6], nullptr, 16);

    const int epoch = epochOf(height);
    const int numCacheItems = lightCacheNumItems(epoch);
    printf("  block %d, epoch %d, light cache %.1f MB, real DAG %.2f GB\n",
           height, epoch, numCacheItems * 64 / 1e6, fullDatasetBytes(epoch) / 1e9);

    uint32_t seed[8];
    epochSeed(epoch, seed);
    std::vector<uint32_t> cache((size_t)numCacheItems * 16, 0);
    buildLightCache(cache.data(), numCacheItems, seed);

    std::vector<uint32_t> l1(kL1CacheWords, 0);
    for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
        datasetItem2048(cache.data(), numCacheItems, i, l1.data() + i * 64);

    uint32_t gotFinal[8], gotMix[8];
    hashLight(cache.data(), numCacheItems, l1.data(),
                    (uint32_t)(fullDatasetNumItems(epoch) / 2), height, header,
                    nonce, gotFinal, gotMix);

    int bad = 0;
    if (memcmp(gotMix, wantMix, sizeof(gotMix)) != 0) {
        printf("  FAIL  mix   want %s\n              got  %s\n", argv[4],
               wordsToHex(gotMix, 8).c_str());
        bad++;
    }
    if (memcmp(gotFinal, wantFinal, sizeof(gotFinal)) != 0) {
        printf("  FAIL  final want %s\n              got  %s\n", argv[5],
               wordsToHex(gotFinal, 8).c_str());
        bad++;
    }

    // The block was accepted by its own network, so its hash must be under the
    // target its own nBits encodes. This is the only check that exercises the
    // digest-to-limb conversion the miner uses to decide it has found a share.
    uint64_t target[4], hit[4];
    bitsToTarget(bits, target);
    hashToLimbs(gotFinal, hit);
    if (!underTarget(hit, target)) {
        printf("  FAIL  hash is not under the block's own target %08x\n", bits);
        bad++;
    }

    if (bad == 0)
        printf("  ok    mix, PoW hash and target all reproduce block %d\n",
               height);
    return bad == 0 ? 0 : 1;
}

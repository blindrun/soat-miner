// The shared ProgPoW layer against stock ProgPoW's own published vectors.
//
// Why this exists as well as test_kawpow_hash
// -------------------------------------------
// The KawPow vectors prove the family layer through exactly one configuration:
// Ravencoin's period, Ravencoin's epoch length, Ravencoin's private keccak
// tail. If the layer were correct only for that one arrangement - a constant
// folded where it should be a parameter, say - the KawPow tests would still
// pass and the next coin would be quietly broken.
//
// So this drives the same code down a second, independent path: stock ProgPoW,
// with a different period, a different epoch length, and the plain keccak
// wrapper that absorbs the header twice instead of carrying state forward. The
// vectors come from the ProgPoW specification itself (ifdefelse/ProgPOW,
// test-vectors.md), not from Ravencoin.
//
// Which revision they are was determined by sweeping the parameter space
// against them rather than by reading a version number: period 10, 11 cache
// accesses, 18 math operations, 0.9.4 labelling. Period 3 and period 50 each
// reproduce only the two block-0 vectors, which every period length
// reproduces, and neither reproduces block 49 or 14999.
//
// What it does NOT cover, and this matters for the next coin: ProgPowZ is
// revision 0.9.2 - period 50, 12 cache accesses, 20 math operations, and the
// rotated random_math labelling. No published vector exercises that
// combination. Zano's own unit test for its proof of work is commented out and
// predates ProgPowZ entirely. Its anchor has to be real ZANO mainnet blocks.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "algos/progpow/progpow.h"

using namespace om::progpow;

namespace {

int g_fail = 0;

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

void hexToWords(const char *hex, uint32_t *out, int words) {
    for (int i = 0; i < words * 4; i++) {
        const uint8_t b = (uint8_t)((hexVal(hex[i * 2]) << 4) | hexVal(hex[i * 2 + 1]));
        if (i % 4 == 0) out[i / 4] = 0;
        out[i / 4] |= (uint32_t)b << ((i % 4) * 8);
    }
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

/**
 * Stock ProgPoW 0.9.4's two Keccak-f[800] wrappers.
 *
 * Determined by sweeping constructions against the vectors, not assumed, and
 * the answer is closer to KawPow than the version numbers suggest: the FINAL
 * permutation carries the first one's whole eight-word output forward and puts
 * the mix digest at words 8..15, exactly as Ravencoin does. The only
 * difference is what fills words 16..24 - zeros here, "rAVENCOINKAWPOW" there.
 *
 * That is worth stating plainly because it means KawPow's incompatibility with
 * stock ProgPoW rests on fifteen words and nothing else, which is why writing
 * one of them as 0x52 instead of 0x72 broke everything while every other layer
 * matched a reference exactly.
 *
 * ProgPowZ, being 0.9.2, uses the older construction that absorbs the header
 * again instead of carrying state forward. Three constructions, one family.
 */
void stockInitialKeccak(const uint32_t hdr[8], uint64_t nonce,
                        uint32_t state2[8]) {
    uint32_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i < 8; i++) st[i] = hdr[i];
    st[8] = (uint32_t)nonce;
    st[9] = (uint32_t)(nonce >> 32);
    keccakf800(st);
    for (int i = 0; i < 8; i++) state2[i] = st[i];
}

void stockFinalKeccak(const uint32_t state2[8], const uint32_t mix[8],
                      uint32_t out[8]) {
    uint32_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i < 8; i++) st[i] = state2[i];
    for (int i = 0; i < 8; i++) st[8 + i] = mix[i];
    keccakf800(st);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}

struct Vector {
    int block;
    const char *nonce;
    const char *mix;
    const char *final;
};

// ifdefelse/ProgPOW, test-vectors.md, "progPowHash". Every vector shares one
// header, which is what makes the block-number column the variable under test.
const char *kHeader =
    "ffeeddccbbaa9988776655443322110000112233445566778899aabbccddeeff";

const Vector kVectors[] = {
    {0, "123456789abcdef0",
     "c2e883b6876ec4cc514b9cea269f343095619faf9f2edcafb3fcf6928fa58141",
     "fa70fbf9979f80ec3db2c3f118a5e683fcf5f54ea7edc41b0b5d336508694cb8"},
    {0, "0000000000000001",
     "ab94222a9736b2c93282fbf6e7217f792f87504033a83eb5501beb24f0d235e7",
     "1260d102572f6ab9840556e8766ba670b511bcf768767b0ab05af45ea9fbad8d"},
    {49, "123456789abcdef0",
     "a0e00c15ccff10aefeeef6ca28260807fdd7f2daaff7948b15857e3a65908f09",
     "a66465873e0674e95ac58efba116458342b3252abeb47874adaf139843ef79bb"},
    {49, "0000000000000001",
     "1704a993e5a8603615b964990253896681da83ddd10c0e6e8fee2f273fa2a961",
     "528dff2f543825030a8e0943013de7bc6a4b7c203c7398607811176b03ce70f4"},
    {14999, "123456789abcdef0",
     "fbbed3db6316658244eef0a897a901fdb40956de9439cf15a74582427443d3bc",
     "caaa67746a4a26c102580851c4f8542f455cd97c6f2749de216c4425504d53c4"},
    {14999, "0000000000000001",
     "0a66b4d37962836650099ad914d2688ffb5dc8688424256cf177c3e7b3f85e88",
     "4bd39ef9155cfd42f0ebb486ee7097d08f793147a9d157027db3d188770ac29d"},
    {30000, "123456789abcdef0",
     "493c13e9807440571511b561132834bbd558dddaa3b70c09515080a6a1aff6d0",
     "46b72b75f238bea3fcfd227e0027dc173dceaa1fb71744bd3d5e030ed2fed053"},
    {30000, "0000000000000001",
     "67ed6cc54b89262d1ac9c3f24ad0d2362cb703c1e23881713d0350ca3035e2ae",
     "ad8fa791bfb2f474a3487d27075bf339d73e0e69d62ecab14166add94c8d0f92"},
};

/** Stock ProgPoW 0.9.4, as the sweep against these vectors established. */
Params stockParams() {
    Params p;
    p.periodLength = 10;
    p.epochLength = 30000;
    p.cacheAccesses = 11;
    p.mathOps = 18;
    p.mathVariant = kMath094;
    return p;
}

struct EpochState {
    int epoch = -1;
    int numCacheItems = 0;
    uint32_t numItems2048 = 0;
    std::vector<uint32_t> cache, l1;

    void build(int e) {
        if (epoch == e) return;
        epoch = e;
        numCacheItems = lightCacheNumItems(e);
        numItems2048 = (uint32_t)(fullDatasetNumItems(e) / 2);
        uint32_t seed[8];
        epochSeed(e, seed);
        cache.assign((size_t)numCacheItems * 16, 0);
        buildLightCache(cache.data(), numCacheItems, seed);
        l1.assign(kL1CacheWords, 0);
        for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
            datasetItem2048(cache.data(), numCacheItems, i, l1.data() + i * 64);
    }
};

}  // namespace

int main(int argc, char **argv) {
    bool expectFail = false;
    Params p = stockParams();
    // Overridable so the harness can prove these vectors DISCRIMINATE. A test
    // that passes for several parameter sets is not pinning the parameters.
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--expect-fail") expectFail = true;
        else if (a == "--period" && i + 1 < argc) p.periodLength = atoi(argv[++i]);
        else if (a == "--math092") p.mathVariant = kMath092;
        else if (a == "--ops" && i + 2 < argc) {
            p.cacheAccesses = atoi(argv[++i]);
            p.mathOps = atoi(argv[++i]);
        }
    }
    uint32_t hdr[8];
    hexToWords(kHeader, hdr, 8);

    EpochState st;
    int pass = 0, fail = 0;

    for (const Vector &v : kVectors) {
        const uint64_t nonce = strtoull(v.nonce, nullptr, 16);

        uint32_t init[8];
        stockInitialKeccak(hdr, nonce, init);
        // The seed is the first two words as they already sit. Byte-swapping
        // them was tried against these vectors and reproduces nothing, which
        // also confirms the spec document's "Mix seed" line - it prints
        // init[1]:init[0], and that is what these two words are.
        const uint32_t hashSeed[2] = {init[0], init[1]};

        st.build(epochNumber(v.block, p.epochLength));

        uint32_t mix[8];
        hashMixLight(st.cache.data(), st.numCacheItems, st.l1.data(),
                     st.numItems2048, v.block, p, hashSeed, mix);

        uint32_t fin[8];
        stockFinalKeccak(init, mix, fin);

        uint32_t wantMix[8], wantFinal[8];
        hexToWords(v.mix, wantMix, 8);
        hexToWords(v.final, wantFinal, 8);

        const bool ok = memcmp(mix, wantMix, 32) == 0 &&
                        memcmp(fin, wantFinal, 32) == 0;
        if (ok) {
            pass++;
            if (!expectFail)
                printf("  ok    block %-6d nonce %s\n", v.block, v.nonce);
        } else {
            fail++;
            if (!expectFail) {
                printf("  FAIL  block %d nonce %s\n", v.block, v.nonce);
                printf("        mix   want %s\n              got  %s\n", v.mix,
                       wordsToHex(mix, 8).c_str());
                printf("        final want %s\n              got  %s\n", v.final,
                       wordsToHex(fin, 8).c_str());
            }
        }
    }

    if (expectFail) {
        if (fail > 0) {
            printf("  ok    mutant rejected: %d/%d wrong\n", fail, pass + fail);
            return 0;
        }
        printf("  FAIL  mutant reproduced the vectors\n");
        return 1;
    }
    printf("%d/%zu stock ProgPoW vectors reproduced by the shared layer\n", pass,
           sizeof(kVectors) / sizeof(kVectors[0]));
    return fail == 0 ? 0 : 1;
}

// KawPow host reference against the published hash vectors.
//
// This is the anchor the rest of the algorithm is measured against: the CUDA
// and Vulkan kernels are checked against this host implementation, so if this
// is wrong, everything downstream is confidently wrong together.
//
// The vectors are the thirteen published in RavenProject/Ravencoin
// src/crypto/ethash/progpow_test_vectors.hpp. They are facts about the
// algorithm, not code, and are reproduced here with that origin recorded. Each
// one pins a block number, a header hash and a nonce to the mix hash and the
// final hash Ravencoin's own consensus code produces.
//
// The block numbers are chosen around period and epoch boundaries - 49/50,
// 29999/30000, 30049/30050 - so a wrong period length or a wrong epoch length
// fails here rather than in production.
//
// Run the mutants too. `make test-kawpow` builds four deliberately broken
// copies and requires each to fail:
//
//   period 50    ProgPoW's own default instead of Ravencoin's 3
//   epoch 30000  Ethereum's epoch length instead of Ravencoin's 7500
//   no raven     the "RAVENCOINKAWPOW" tail zeroed, i.e. stock ProgPoW
//   SHA3 pad     0x06 padding instead of original Keccak's 0x01
//
// A correctness test nobody has watched fail is decoration.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "algos/kawpow/kawpow.h"

using namespace om::kawpow;

namespace {

struct Vector {
    int blockNumber;
    const char *headerHash;
    const char *nonce;
    const char *mixHash;
    const char *finalHash;
};

// From RavenProject/Ravencoin, src/crypto/ethash/progpow_test_vectors.hpp.
const Vector kVectors[] = {
    {0, "0000000000000000000000000000000000000000000000000000000000000000",
     "0000000000000000",
     "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
     "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1"},
    {49, "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b",
     "0000000007073c07",
     "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
     "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463"},
    {50, "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922",
     "00000000076e482e",
     "d6dc634ae837e2785b347648ea515e25e5d8821ae0b95e1c2a9c2d497e0dcfbd",
     "ab0ad7ef8d8ee317dd12d10310aceed7321d34fb263791c2de5776a6658d177e"},
    {99, "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d",
     "000000003917afab",
     "fa706860e5e0e830d5d1d7157e5bea7f5f8a350c7c8612ac1d1fcf2974d64244",
     "aa85340690f2e907054324a5021937910e15edfd1ef1577231843e7d32ec3a61"},
    {29950, "ac7b55e801511b77e11d52e9599206101550144525b5679f2dab19386f23dcce",
     "005d409dbc23a62a",
     "5359807b77a74878269c3a3044df8618a576ce8dc52e1c48d927d4a60e7c6b79",
     "022019e5408683f7f8326b4e46b42864a3a069f17b6151e434fcaedecaadd918"},
    {29999, "e43d7e0bdc8a4a3f6e291a5ed790b9fa1a0948a2b9e33c844888690847de19f5",
     "005db5fa4c2a3d03",
     "d15de3f9bfedd9b6d0f498273eb3b437115bdc8326c96c6457ac06deb5c9f389",
     "4e93630b81198752f876b24380999189b7b9366c08222ac05e4237b87114f305"},
    {30000, "d34519f72c97cae8892c277776259db3320820cb5279a299d0ef1e155e5c6454",
     "005db8607994ff30",
     "de0348b69bf91dfe2c3d3dba6f0132e9048a5284e57b8d9d20adc5f3dc0d3236",
     "c7953d848cda6e304f77b4c6d735645c8e8508a5e74c9e9814ef37b19087cd6c"},
    {30049, "8b6ce5da0b06d18db7bd8492d9e5717f8b53e7e098d9fef7886d58a6e913ef64",
     "005e2e215a8ca2e7",
     "975c6a9decc89cba7ace69338d4de8510d9619aef42b1d35d0bef7e0ce0614a9",
     "c262d8055e288d04b951a844bfca8ba529f5b4d652b408e3942727d7dd90957a"},
    {30050, "c2c46173481b9ced61123d2e293b42ede5a1b323210eb2a684df0874ffe09047",
     "005e30899481055e",
     "362f2fabdb9699d3634b6499703f939f378ee4eac803396c2b0ed0fe1d154972",
     "4cd7e6e79e0b63d42b2b06716a919ccc7834077ec727a9ea94edcdaff2fefab8"},
    {30099, "ea42197eb2ba79c63cb5e655b8b1f612c5f08aae1a49ff236795a3516d87bc71",
     "005ea6aef136f88b",
     "b1196457261bd05ccb387a8ff3fd02687bf496bd7943d89419465289669e27aa",
     "39d1ebfa783b61a6fa8e9747d0f9f134efae5cfba284a2c80e8deabae6b98676"},
    {59950, "49e15ba4bf501ce8fe8876101c808e24c69a859be15de554bf85dbc095491bd6",
     "02ebe0503bd7b1da",
     "df3dbb1669fd35dbb0ae96bbea2d498f0c6992cbddd092aeace42dd933505f95",
     "b8984cf4021c4433f753654848d721f33a0792b4417241f0cf7c7c2db011a54a"},
    {59999, "f5c50ba5c0d6210ddb16250ec3efda178de857b2b1703d8d5403bd0f848e19cf",
     "02edb6275bd221e3",
     "5017df70e97ca35638cf439cdbe54f30383d335e18eb4a74d6e166736f1038fa",
     "4cf1fa62f25b577ac822a6a28d55f8b7e3ae7fe983abd868ae00927e68c41016"},
    {170915, "5b3e8dfa1aafd3924a51f33e2d672d8dae32fa528d8b1d378d6e4db0ec5d665d",
     "0000000044975727",
     "efb29147484c434f1cc59629da90fd0343e3b047407ecd36e9ad973bd51bbac5",
     "e7e6bb3b2f9acd3864bc86f72f87237eaf475633ef650c726ac80eb0adf116b6"},
};

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/** 64 hex characters -> 8 little-endian words, the convention in keccak.h. */
bool hexToWords(const char *hex, uint32_t *out, int words) {
    for (int i = 0; i < words * 4; i++) {
        const int hi = hexVal(hex[i * 2]), lo = hexVal(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        const uint8_t b = (uint8_t)((hi << 4) | lo);
        if (i % 4 == 0) out[i / 4] = 0;
        out[i / 4] |= (uint32_t)b << ((i % 4) * 8);
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

/** Light cache plus the 16 KB L1 cache, built once per epoch and reused. */
struct EpochState {
    int epoch = -1;
    int numCacheItems = 0;
    uint32_t numItems2048 = 0;
    std::vector<uint32_t> cache;
    std::vector<uint32_t> l1;

    void build(int e) {
        if (epoch == e) return;
        epoch = e;
        numCacheItems = lightCacheNumItems(e);
        numItems2048 = (uint32_t)(fullDatasetNumItems(e) / 2);
        uint32_t seed[8];
        epochSeed(e, seed);
        cache.assign((size_t)numCacheItems * 16, 0);
        buildLightCache(cache.data(), numCacheItems, seed);

        // The L1 cache is literally the first 16 KB of the dataset.
        l1.assign(kL1CacheWords, 0);
        for (uint32_t i = 0; i < kL1CacheBytes / 256; i++)
            datasetItem2048(cache.data(), numCacheItems, i, l1.data() + i * 64);
    }
};

}  // namespace

int main(int argc, char **argv) {
    // The mutants are expected to fail, so the harness inverts the exit code.
    const bool expectFail = argc > 1 && std::string(argv[1]) == "--expect-fail";

    EpochState st;
    int pass = 0, fail = 0;

    for (const Vector &v : kVectors) {
        uint32_t header[8], wantMix[8], wantFinal[8];
        if (!hexToWords(v.headerHash, header, 8) ||
            !hexToWords(v.mixHash, wantMix, 8) ||
            !hexToWords(v.finalHash, wantFinal, 8)) {
            fprintf(stderr, "bad vector literal at block %d\n", v.blockNumber);
            return 2;
        }
        const uint64_t nonce = strtoull(v.nonce, nullptr, 16);

        st.build(epochOf(v.blockNumber));

        uint32_t gotFinal[8], gotMix[8];
        hashLight(st.cache.data(), st.numCacheItems, st.l1.data(),
                        st.numItems2048, v.blockNumber, header, nonce, gotFinal,
                        gotMix);

        const bool mixOk = memcmp(gotMix, wantMix, sizeof(gotMix)) == 0;
        const bool finalOk = memcmp(gotFinal, wantFinal, sizeof(gotFinal)) == 0;
        if (mixOk && finalOk) {
            pass++;
            if (!expectFail)
                printf("  ok    block %-7d epoch %-3d %s\n", v.blockNumber,
                       st.epoch, v.finalHash);
        } else {
            fail++;
            if (!expectFail) {
                printf("  FAIL  block %d\n", v.blockNumber);
                if (!mixOk)
                    printf("        mix   want %s\n              got  %s\n",
                           v.mixHash, wordsToHex(gotMix, 8).c_str());
                if (!finalOk)
                    printf("        final want %s\n              got  %s\n",
                           v.finalHash, wordsToHex(gotFinal, 8).c_str());
            }
        }
    }

    const int total = pass + fail;
    if (expectFail) {
        if (fail > 0) {
            printf("  ok    mutant rejected: %d/%d vectors wrong\n", fail, total);
            return 0;
        }
        printf("  FAIL  mutant produced correct hashes - the mutation seam is "
               "not reaching the code under test\n");
        return 1;
    }

    printf("%d/%d KawPow vectors reproduced\n", pass, total);
    return fail == 0 ? 0 : 1;
}

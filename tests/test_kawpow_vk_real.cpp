// The Vulkan backend at its REAL geometry, against a real mainnet block.
//
// Why this exists as well as tests/test_kawpow_vulkan
// ---------------------------------------------------
// That test runs the shaders over a 16 MB DAG in four 4 MB buffers, because
// that makes the split arithmetic testable in seconds on any card. It is a
// good test and it has never once exercised the configuration the miner
// actually runs: a 6.11 GB DAG in three 2 GiB device-local buffers.
//
// That gap matters more than it looks. wF:p1 pointed out that a 32-bit BYTE
// offset tops out at 4 GB, which 6.11 GB straddles and 4.6 GB does not - and
// this backend has a measured performance cliff at exactly that boundary. If
// any index between the descriptor and the load is 32-bit, the symptom would
// be a WRAPPED READ, which is a correctness bug hiding behind a slow number.
// A miner that hashes wrong at full DAG size and right in the test harness is
// the worst failure this project can ship.
//
// So this drives the real Algorithm object - the same makeKawpowVK() the miner
// uses, the same power-of-two split, the same device-local memory - at
// Ravencoin block 4501000, and requires it to find that block's own nonce with
// that block's own mix hash under that block's own target.
//
// Needs ~7 GB of VRAM and about half a minute. It skips politely without them.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include "algos/kawpow/job.h"
#include "algos/kawpow/kawpow.h"
#include "core/algo.h"

namespace om {
Algorithm *makeKawpowVK(int deviceIndex);
}

namespace {

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hexToBytes(const char *hex, uint8_t *out, int n) {
    if ((int)strlen(hex) != n * 2) return false;
    for (int i = 0; i < n; i++) {
        const int hi = hexVal(hex[i * 2]), lo = hexVal(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

void bitsToTarget(uint32_t bits, uint64_t target[4]) {
    const uint32_t exponent = bits >> 24;
    const uint64_t mantissa = bits & 0x007fffffu;
    for (int i = 0; i < 4; i++) target[i] = 0;
    if (exponent <= 3) {
        target[0] = mantissa >> (8 * (3 - exponent));
        return;
    }
    const int shift = 8 * ((int)exponent - 3);
    for (int b = 0; b < 3; b++) {
        const int bitPos = shift + 8 * b;
        if (bitPos >= 256) continue;
        target[bitPos / 64] |= ((mantissa >> (8 * b)) & 0xffu) << (bitPos % 64);
    }
}

}  // namespace

int main(int argc, char **argv) {
    // Defaults are the pinned block from the Makefile.
    const char *headerHex =
        argc > 1 ? argv[1]
                 : "8dc2a5086848cb872bb610d1a06294e3c41196bb04b041c8fe84da0153ba5db7";
    const int height = argc > 2 ? atoi(argv[2]) : 4501000;
    const uint64_t nonce =
        strtoull(argc > 3 ? argv[3] : "329dd53ca526c2cd", nullptr, 16);
    const char *mixHex =
        argc > 4 ? argv[4]
                 : "7b1e003a71d7bf9f81d95c0c68126f72876de28b69b175c84e4c11234ad5d7c7";
    const uint32_t bits =
        (uint32_t)strtoul(argc > 5 ? argv[5] : "1b06292b", nullptr, 16);

    std::unique_ptr<om::Algorithm> algo(om::makeKawpowVK(-1));
    if (!algo) {
        printf("  SKIP  no usable Vulkan device\n");
        return 0;
    }

    om::Job job;
    if (!hexToBytes(headerHex, job.msg, 32)) {
        fprintf(stderr, "bad header hash\n");
        return 2;
    }
    job.epoch = (uint64_t)om::kawpow::epochOf(height);
    job.extra = om::kawpow::encodeJobExtra(height);
    job.jobId = "real";
    bitsToTarget(bits, job.target);
    job.valid = true;

    const double gb = algo->memoryBytes(job) / 1e9;
    printf("  block %d, epoch %llu, DAG %.2f GB - the geometry the miner uses\n",
           height, (unsigned long long)job.epoch, gb);

    if (!algo->prepare(job)) {
        printf("  SKIP  could not build the DAG (needs about %.1f GB free)\n",
               gb + 0.5);
        return 0;
    }

    // Straddle the winning nonce so the batch is not a single lucky lane.
    const uint64_t base = nonce - 64;
    std::vector<om::Solution> sols;
    if (!algo->search(job, base, 256, &sols)) {
        printf("  FAIL  search() failed on the real DAG\n");
        return 1;
    }

    uint8_t wantMix[32];
    if (!hexToBytes(mixHex, wantMix, 32)) return 2;

    int fail = 0;
    bool found = false;
    for (const om::Solution &s : sols) {
        // Every solution must survive the host re-derivation, whatever it is.
        // A wrapped index would produce a hit the host cannot reproduce, which
        // is exactly the failure this test exists to catch.
        if (!algo->verify(job, s)) {
            printf("  FAIL  device reported nonce %016llx that the host cannot "
                   "reproduce - the DAG read is wrong at this size\n",
                   (unsigned long long)s.nonce);
            fail++;
            continue;
        }
        if (s.nonce == nonce) {
            found = true;
            uint8_t got[32];
            if (!hexToBytes(s.extra.c_str(), got, 32) ||
                memcmp(got, wantMix, 32) != 0) {
                printf("  FAIL  right nonce, wrong mix hash\n        want %s\n"
                       "        got  %s\n", mixHex, s.extra.c_str());
                fail++;
            }
        }
    }

    if (!found) {
        printf("  FAIL  the block's own nonce %016llx was not reported, though "
               "it is under the block's own target (%zu other solution(s) "
               "found)\n", (unsigned long long)nonce, sols.size());
        fail++;
    } else if (fail == 0) {
        printf("  ok    Vulkan found block %d's own nonce with its own mix "
               "hash, on a %.2f GB DAG\n", height, gb);
    }
    return fail == 0 ? 0 : 1;
}

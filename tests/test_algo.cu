// Drives the real Autolykos2 Algorithm object - prepare(), search(), verify()
// and the build-ahead buffer swap - against a real mainnet block.
//
// test_hit covers the kernels. This covers the class around them: the streams,
// the async copies, the solution readback and the swap that build-ahead does.
// A kernel can be perfectly correct and the miner still find nothing if the
// count comes back before the kernel wrote it, or if the swap hands mining a
// stale pointer, and neither of those shows up as an error anywhere.
//
// usage: test_algo <msg_hex64> <height> <nonce_hex16> <expected_hit_hex64>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <chrono>
#include <vector>

#include "../src/core/algo.h"

namespace om {
Algorithm *makeAutolykos2();
}

static void hex2bin(const char *hex, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v = 0;
        sscanf(hex + i * 2, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

/** 64 hex chars, most significant first, into limbs with limb[0] least. */
static void hexToLimbs(const char *hex, uint64_t limb[4]) {
    for (int l = 0; l < 4; l++) {
        uint64_t v = 0;
        for (int j = 0; j < 16; j++) {
            unsigned d = 0;
            sscanf(hex + (3 - l) * 16 + j, "%1x", &d);
            v = (v << 4) | d;
        }
        limb[l] = v;
    }
}

static void limbsToHex(const uint64_t limb[4], char out[65]) {
    for (int l = 3; l >= 0; l--)
        snprintf(out + (3 - l) * 16, 17, "%016llx", (unsigned long long)limb[l]);
}

static int failures = 0;
static void check(bool ok, const char *what) {
    printf("  %-58s %s\n", what, ok ? "YES" : "NO");
    if (!ok) failures++;
}

int main(int argc, char **argv) {
    using namespace om;
    if (argc < 5) {
        printf("usage: %s <msg_hex64> <height> <nonce_hex16> <hit_hex64>\n",
               argv[0]);
        return 1;
    }

    Job job;
    hex2bin(argv[1], job.msg, 32);
    job.epoch = strtoull(argv[2], nullptr, 10);
    const uint64_t wantNonce = strtoull(argv[3], nullptr, 16);
    uint64_t wantHit[4];
    hexToLimbs(argv[4], wantHit);
    job.valid = true;

    // Target = the known hit + 1, so that nonce is a solution and essentially
    // nothing else in the range is.
    memcpy(job.target, wantHit, sizeof(wantHit));
    for (int i = 0; i < 4; i++) {
        if (++job.target[i] != 0) break;  // carry only while a limb wrapped
    }

    Algorithm *algo = makeAutolykos2();
    // Forced on: this test is here to exercise build-ahead, not to re-decide
    // whether it fits.
    algo->setPrefetch(1);

    printf("height=%llu  dataset=%.2f GB\n", (unsigned long long)job.epoch,
           algo->memoryBytes(job) / 1e9);

    if (!algo->prepare(job)) {
        printf("FAIL: prepare() failed\n");
        return 1;
    }

    // --- the real block's nonce must come back out of search() --------------
    const uint64_t base = wantNonce - 2048;
    std::vector<Solution> sols;
    if (!algo->search(job, base, 4096, &sols)) {
        printf("FAIL: search() failed\n");
        return 1;
    }

    bool found = false;
    for (const auto &s : sols) {
        if (s.nonce != wantNonce) continue;
        found = true;
        char got[65];
        limbsToHex(s.hit, got);
        printf("  nonce=%016llx\n  hit  =%s\n", (unsigned long long)s.nonce, got);
        check(memcmp(s.hit, wantHit, sizeof(wantHit)) == 0,
              "search() returns the block's own hit");
        check(algo->verify(job, s), "verify() accepts it");

        Solution bad = s;
        bad.hit[0] ^= 1ULL;
        check(!algo->verify(job, bad), "verify() rejects a tampered hit");
    }
    check(found, "search() finds the block's winning nonce");

    // --- build-ahead: the next height must already be resident --------------
    Job next = job;
    next.epoch = job.epoch + 1;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = algo->prepare(next);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    check(ok, "prepare() of the next height succeeds");
    check(algo->servedFromPrefetch(), "next height was already built ahead");
    printf("  swapped in in %.3f s\n", secs);
    const std::string note = algo->prefetchNote();
    if (!note.empty()) printf("  %s\n", note.c_str());

    // The dataset is the only thing that differs between two heights, so if the
    // swap handed mining a stale pointer the old solution would still be there.
    // It must not be.
    sols.clear();
    if (!algo->search(next, base, 4096, &sols)) {
        printf("FAIL: search() after the swap failed\n");
        return 1;
    }
    bool stale = false;
    for (const auto &s : sols)
        if (s.nonce == wantNonce) stale = true;
    check(!stale, "the swapped-in dataset is the next height's, not the old one");

    algo->release();
    delete algo;

    if (failures) {
        printf("FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    printf("PASS: Algorithm object reproduces block %llu and builds ahead\n",
           (unsigned long long)job.epoch);
    return 0;
}

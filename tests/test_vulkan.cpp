// Verifies the Vulkan backend against the same real mainnet block used for
// the CUDA backend. All backends must produce a byte-identical hit.
//
// usage: test_vulkan <msg_hex64> <height> <nonce_hex16> <expect_hex64>
//
// The expected hit is passed in rather than hardcoded so this shares the one
// pinned vector in the Makefile with test_hit. It used to carry its own
// hardcoded hit for a different block than the Makefile's, which meant the two
// backends were not actually being held to the same reference.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/core/algo.h"

namespace om {
Algorithm *makeAutolykos2VK(int deviceIndex);
const char *vkDeviceName();
}  // namespace om

static void hex2bin(const char *hex, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v;
        sscanf(hex + i * 2, "%2x", &v);
        out[i] = (uint8_t)v;
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("usage: %s <msg_hex64> <height> <nonce_hex16> <expect_hex64>\n",
               argv[0]);
        return 1;
    }
    using namespace om;

    Job job;
    hex2bin(argv[1], job.msg, 32);
    job.epoch = strtoull(argv[2], nullptr, 10);
    const uint64_t nonce = strtoull(argv[3], nullptr, 16);
    // Max target so verify() only checks the hit reproduces, not that it wins.
    for (int i = 0; i < 4; i++) job.target[i] = ~0ULL;
    job.valid = true;

    // The expected hit arrives as 32 big-endian bytes; limb 3 is the most
    // significant, matching HIT_EXPECT in the Makefile.
    uint8_t eb[32];
    hex2bin(argv[4], eb, 32);
    uint64_t expect[4];
    for (int l = 0; l < 4; l++) {
        uint64_t x = 0;
        for (int j = 0; j < 8; j++) x = (x << 8) | (uint64_t)eb[(3 - l) * 8 + j];
        expect[l] = x;
    }

    Algorithm *algo = makeAutolykos2VK(-1);
    if (!algo) {
        printf("Vulkan init failed\n");
        return 1;
    }
    printf("device: %s\n", vkDeviceName());

    if (!algo->prepare(job)) {
        printf("prepare failed\n");
        return 1;
    }
    printf("dataset ready (%.2f GB)\n", algo->memoryBytes(job) / 1e9);

    // verify() recomputes the hit for exactly this nonce on the device and
    // memcmp's it against the one supplied, so feeding it the known-good hit
    // from the CUDA/python reference is a direct byte-for-byte equality test.
    Solution sol;
    sol.nonce = nonce;
    for (int l = 0; l < 4; l++) sol.hit[l] = expect[l];
    const bool ok = algo->verify(job, sol);

    // Negative control. Without it a verify() that returned true
    // unconditionally - or a kernel that never ran at all - would pass this
    // test and report a green build while mining nothing.
    Solution bad = sol;
    bad.hit[0] ^= 1ULL;
    const bool rejectsWrong = !algo->verify(job, bad);

    printf("hit matches CUDA/python reference: %s\n", ok ? "YES" : "NO");
    printf("rejects a deliberately wrong hit:  %s\n", rejectsWrong ? "YES" : "NO");

    algo->release();

    if (ok && rejectsWrong) {
        printf("PASS: Vulkan backend reproduces block %llu byte-for-byte\n",
               (unsigned long long)job.epoch);
        return 0;
    }
    printf("FAIL\n");
    return 1;
}

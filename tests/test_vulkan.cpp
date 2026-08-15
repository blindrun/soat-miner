// Verifies the Vulkan backend against the same real mainnet block used for
// the CUDA backend. All backends must produce a byte-identical hit.
//
// usage: test_vulkan <msg_hex64> <height> <nonce_hex16>

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
    if (argc < 4) {
        printf("usage: %s <msg_hex64> <height> <nonce_hex16>\n", argv[0]);
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

    // verify() recomputes the hit on device and compares against sol.hit, so
    // seeding sol.hit with zeros and reading back is not possible through the
    // public interface. Instead ask it to verify a deliberately wrong hit and
    // then a correct one is impossible here - so we use the search path with a
    // max target over a single nonce, which returns the real hit.
    // verify() recomputes the hit for exactly this nonce on the device and
    // memcmp's it against the one supplied, so feeding it the known-good hit
    // from the CUDA/Python reference is a direct byte-for-byte equality test.
    Solution sol;
    sol.nonce = nonce;
    sol.hit[3] = 0x000000000001429dULL;
    sol.hit[2] = 0x07faffa88ebae0deULL;
    sol.hit[1] = 0xe7800749c59482ccULL;
    sol.hit[0] = 0xb094940ec5ca96d9ULL;

    const bool ok = algo->verify(job, sol);
    printf("hit matches CUDA/python reference: %s\n", ok ? "YES" : "NO");
    const bool found = ok;

    algo->release();
    return found ? 0 : 1;
}

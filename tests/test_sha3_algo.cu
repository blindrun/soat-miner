// SHA3-256t end-to-end on the device, against real BC3 mainnet blocks.
//
// The gate that matters for this algorithm. Each vector is a real block: its
// 76-byte header prefix, the nonce that actually won it, and the hash that
// nonce produces. The test hands the Algorithm a job whose target is that
// exact hash, searches a window containing the winning nonce, and requires
// that the device finds it and that verify() accepts it.
//
// Setting the target to the block's own hash is deliberate. It is the tightest
// target the block satisfies, so a kernel that is subtly wrong - a Keccak-256
// pad instead of SHA3's 0x06, two passes instead of three, a byte-swapped
// comparison - cannot pass by finding some other nonce that happens to be
// under a loose target.
//
// The vectors live in tests/sha3_vectors.h, shared with the Vulkan gate so
// the two backends are held to the same reference rather than to two
// separately-maintained tables.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/algo.h"
#include "core/btc_job.h"
#include "sha3_vectors.h"

namespace om {
Algorithm *makeSha3_256t();
}


static bool hexToBytes(const char *hex, std::vector<uint8_t> *out) {
    const size_t n = strlen(hex);
    if (n % 2) return false;
    out->clear();
    for (size_t i = 0; i < n; i += 2) {
        char b[3] = {hex[i], hex[i + 1], 0};
        char *end = nullptr;
        const long v = strtol(b, &end, 16);
        if (end != b + 2) return false;
        out->push_back((uint8_t)v);
    }
    return true;
}

int main() {
    om::Algorithm *algo = om::makeSha3_256t();
    if (!algo) {
        printf("FAIL: could not construct sha3-256t\n");
        return 1;
    }
    int failures = 0;

    for (const Sha3Vector &v : kSha3Vectors) {
        std::vector<uint8_t> hdr, hit;
        if (!hexToBytes(v.header, &hdr) || hdr.size() != 80 ||
            !hexToBytes(v.hitLE, &hit) || hit.size() != 32) {
            printf("FAIL %s: bad vector\n", v.name);
            failures++;
            continue;
        }

        const uint32_t winner = (uint32_t)hdr[76] | ((uint32_t)hdr[77] << 8) |
                                ((uint32_t)hdr[78] << 16) |
                                ((uint32_t)hdr[79] << 24);

        om::Job job;
        // The job's header must carry a zero nonce: the whole point is that the
        // search supplies it. Leaving the winner in place would still pass on a
        // kernel that ignored `base` entirely.
        uint8_t tmpl[80];
        memcpy(tmpl, hdr.data(), 80);
        memset(tmpl + 76, 0, 4);
        job.extra = om::encodeBtcJobExtra(tmpl, std::string(8, '\0'));
        memcpy(job.msg, tmpl + 36, 32);
        for (int i = 0; i < 4; i++) {
            uint64_t limb = 0;
            for (int b = 7; b >= 0; b--) limb = (limb << 8) | hit[i * 8 + b];
            job.target[i] = limb;
        }
        job.epoch = 0;
        job.valid = true;

        if (!algo->prepare(job)) {
            printf("FAIL %s: prepare\n", v.name);
            failures++;
            continue;
        }

        // A window that starts below the winner, so finding it proves the base
        // offset is applied rather than the thread index being used raw.
        const uint32_t base = winner - 4096;
        std::vector<om::Solution> sols;
        if (!algo->search(job, base, 8192, &sols)) {
            printf("FAIL %s: search returned false\n", v.name);
            failures++;
            continue;
        }

        bool found = false;
        for (const om::Solution &s : sols) {
            if ((uint32_t)s.nonce != winner) continue;
            found = true;
            if (memcmp(s.hit, job.target, 32) != 0) {
                printf("FAIL %s: hit does not match the block's hash\n", v.name);
                failures++;
            } else if (!algo->verify(job, s)) {
                printf("FAIL %s: verify() rejected the real winning nonce\n",
                       v.name);
                failures++;
            } else {
                printf("ok   %s: found nonce %08x, hit matches, verify passed\n",
                       v.name, winner);
            }
            break;
        }
        if (!found) {
            printf("FAIL %s: winning nonce %08x not found in %zu solutions\n",
                   v.name, winner, sols.size());
            failures++;
            continue;
        }

        // A tampered hit must not survive verification: this is the guard that
        // stops an unstable GPU's wrong answer reaching the pool.
        om::Solution bad;
        bad.nonce = winner;
        memcpy(bad.hit, job.target, 32);
        bad.hit[0] ^= 1;
        if (algo->verify(job, bad)) {
            printf("FAIL %s: verify() accepted a corrupted hit\n", v.name);
            failures++;
        } else {
            printf("ok   %s: verify() rejects a corrupted hit\n", v.name);
        }

        // A nonce that did not win must not verify either.
        om::Solution wrong;
        wrong.nonce = winner + 1;
        memcpy(wrong.hit, job.target, 32);
        if (algo->verify(job, wrong)) {
            printf("FAIL %s: verify() accepted the wrong nonce\n", v.name);
            failures++;
        } else {
            printf("ok   %s: verify() rejects the wrong nonce\n", v.name);
        }
    }

    algo->release();
    delete algo;
    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all sha3-256t device checks passed\n");
    return 0;
}

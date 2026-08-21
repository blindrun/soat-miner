// Do the CUDA and Vulkan backends produce the SAME proof?
//
// This is the test that makes a long mining run unnecessary. What actually
// needs establishing about the new Vulkan backend is that a real gateway would
// accept its proofs - and mining a block to find out costs hours on a shared
// card, at 3.8% of CUDA's rate, and covers exactly one nonce when it lands.
//
// Byte-identity is stronger and cheaper. Both backends are handed the same job
// and the same nonce; if the proofs match byte for byte, then wherever CUDA's
// proofs are accepted Vulkan's are too, because nothing downstream can tell
// them apart. That is the same equivalence argument every one of the nine
// shader gates was proven with, applied once at the top.
//
// The shape has to be pinned. CUDA tunes across several shapes at startup and
// Vulkan has one fixed 4096x16384, and a proof encodes m and n - so without
// SOAT_PEARL_SHAPE the two would differ for a legitimate reason and the test
// would be measuring the tuner.
//
// usage: test_pearl_backend_parity [nonce]
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "../src/core/algo.h"
#include "../src/core/pearl_gateway.h"
#include "../src/algos/pearl-pow/job.h"

namespace om {
Algorithm *makePearlPow();
Algorithm *makePearlPowVK(int deviceIndex);
}  // namespace om

using namespace om;

static int failures = 0, checks = 0;
static void expect(const char *what, bool ok) {
    checks++;
    if (!ok) { failures++; printf("  FAIL: %s\n", what); }
    else printf("  ok: %s\n", what);
}

/** A fixed synthetic job. Never a captured header: it would tie this to a
 *  chain height and to somebody's payout address. */
static Job makeJob() {
    uint8_t header[76];
    for (int i = 0; i < 76; i++) header[i] = (uint8_t)(i * 7 + 3);
    Job job;
    // HARD ENOUGH THAT ONE TILE WINS, not merely hard enough to win.
    //
    // The first version of this used a target so easy that the penalised bound
    // was 2^255 - half of all 262144 tiles in an attempt were winners. Both
    // backends then found a solution, both were valid, and they disagreed,
    // because powScan records at most 64 hits through an atomic and each
    // backend caught a different subset of the flood. That is not a defect and
    // the test was measuring hardware scheduling.
    //
    // 2^214 puts the penalised bound at 2^233, so the expected winners per
    // attempt is 262144 * 2^-23 = 0.031: about one attempt in 32 wins, and a
    // winning attempt has a second winner about 1.5% of the time. Rare enough
    // to compare, common enough to find in a second.
    memset(job.target, 0, sizeof(job.target));
    job.target[3] = 0x400000ULL;
    job.extra = packPearlExtra(header, 3);
    pearl::b3::hash(nullptr, header, 76, job.msg);
    return job;
}

static std::string hex(const uint8_t *p, size_t n) {
    static const char *h = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s += h[p[i] >> 4]; s += h[p[i] & 15]; }
    return s;
}

int main(int argc, char **argv) {
    // Both backends must mine the SAME shape or the proofs differ for a reason
    // that has nothing to do with correctness.
    setenv("SOAT_PEARL_SHAPE", "4096x16384", 1);

    const uint64_t nonce = argc > 1 ? strtoull(argv[1], nullptr, 10) : 20260821;
    const Job job = makeJob();

    printf("pearl backend parity: nonce %llu, shape pinned 4096x16384\n",
           (unsigned long long)nonce);

    std::unique_ptr<Algorithm> cuda(makePearlPow());
    std::unique_ptr<Algorithm> vk(makePearlPowVK(0));
    if (!cuda || !vk) { printf("could not create both backends\n"); return 2; }

    if (!cuda->prepare(job)) { printf("CUDA prepare failed\n"); return 2; }
    std::vector<Solution> cudaSols;
    // 64 attempts, so a winning nonce turns up. Both backends walk nonces
    // from the same base in the same order, so they meet at the same one.
    const uint64_t budget = 64ULL * 256ULL * 1024ULL;
    if (!cuda->search(job, nonce, budget, &cudaSols)) { printf("CUDA search failed\n"); return 2; }

    if (!vk->prepare(job)) { printf("Vulkan prepare failed\n"); return 2; }
    std::vector<Solution> vkSols;
    if (!vk->search(job, nonce, budget, &vkSols)) { printf("Vulkan search failed\n"); return 2; }

    // A run that finds nothing proves nothing, and would otherwise "pass" by
    // comparing two empty lists.
    expect("CUDA found a solution at this nonce", !cudaSols.empty());
    expect("Vulkan found a solution at this nonce", !vkSols.empty());
    if (cudaSols.empty() || vkSols.empty()) {
        printf("\nNO SOLUTION in 64 attempts - unlucky, or the target moved.\n");
        return 1;
    }

    printf("  (cuda %zu solution(s), vulkan %zu)\n", cudaSols.size(), vkSols.size());
    const Solution &c = cudaSols[0], &v = vkSols[0];
    expect("same nonce", c.nonce == v.nonce);
    expect("same digest", memcmp(c.hit, v.hit, sizeof(c.hit)) == 0);
    if (memcmp(c.hit, v.hit, sizeof(c.hit)) != 0) {
        printf("    cuda %s\n", hex((const uint8_t *)c.hit, 32).c_str());
        printf("    vk   %s\n", hex((const uint8_t *)v.hit, 32).c_str());
    }
    expect("same proof length", c.extra.size() == v.extra.size());
    expect("BYTE-IDENTICAL PROOF", c.extra == v.extra);
    if (c.extra != v.extra) {
        printf("    cuda %zu bytes, vk %zu bytes\n", c.extra.size(), v.extra.size());
        for (size_t i = 0; i < c.extra.size() && i < v.extra.size(); i++)
            if (c.extra[i] != v.extra[i]) {
                printf("    first difference at byte %zu\n", i);
                break;
            }
    }

    expect("CUDA verifies its own", cuda->verify(job, c));
    expect("Vulkan verifies its own", vk->verify(job, v));

    // DETERMINISM WITHIN A BACKEND, which is the property that makes the
    // comparison above possible and which nothing else tests. Taking the
    // lowest tile index instead of whichever the atomic recorded first was
    // supposed to make a share a function of the job and the nonce; this is
    // the assertion that it actually is. A second run on the same card, same
    // job, same nonce must produce the same bytes.
    std::vector<Solution> again;
    if (vk->search(job, nonce, budget, &again) && !again.empty()) {
        expect("Vulkan is deterministic across runs",
               again[0].nonce == v.nonce && again[0].extra == v.extra &&
                   memcmp(again[0].hit, v.hit, sizeof(v.hit)) == 0);
    } else {
        expect("Vulkan found the same solution on a second run", false);
    }
    std::vector<Solution> againC;
    if (cuda->search(job, nonce, budget, &againC) && !againC.empty()) {
        expect("CUDA is deterministic across runs",
               againC[0].nonce == c.nonce && againC[0].extra == c.extra &&
                   memcmp(againC[0].hit, c.hit, sizeof(c.hit)) == 0);
    } else {
        expect("CUDA found the same solution on a second run", false);
    }

    printf("\n%d check(s), %d failure(s)\n", checks, failures);
    return failures ? 1 : 0;
}

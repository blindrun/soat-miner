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
// Vectors regenerate from any explorer:
//   curl https://bc3mempool.codefalcon.dev/api/block/<hash>/header

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/algo.h"
#include "core/btc_job.h"

namespace om {
Algorithm *makeSha3_256t();
}

struct Vector {
    const char *name;
    const char *header;  // 80 bytes; the last 4 are the winning nonce
    const char *hitLE;   // the SHA3-256t digest, little-endian, as hashed
};

// Post-fork blocks: 30240 is the fork block itself, the rest are spread over
// the months since. All have version bit 12 set.
static const Vector kVectors[] = {
    // The fork block itself: the first SHA3-256t block on the chain, mined at
    // the difficulty-1 reset chainparams.cpp:256 mandates (nbits ffff001d).
    {"30240 (fork block)",
     "0010002030d2767637fc55d1f70edc5b20a41fddd17d8a12854ba7150400000000000000"
     "18dcbe866321fcf914b8e4f187158d285d7a1e6966e14e2b57e7071ded479309a067106a"
     "ffff001d05417434",
     "ab75656190cd4814d075c09f1a73bae25983d3f5ea884876875e537c00000000"},
    {"38337",
     "0010002093b06b66e411dacc7117dc3f79c2d801e388e202f1d5b0c15d8c410000000000"
     "c0b8ce9df398623782eb0eab5fe75ce7523a0e94cddcedd30017e69162241a3463bb136a"
     "ffff001c8e8df150",
     "0cd925995b5792bbfaabb3419b87ee0a480eb7fd2ad413fae8a6910000000000"},
    {"43713",
     "001000205c3c2d23b5c04686450247e171d468fa2efd423f3033c0af9dd3060000000000"
     "c9bc4904d71e0455f5ce43d9799dc1b75eaca5b47fd4f398338f08a4220fc269cc9a256a"
     "1ea4121b3673bcc6",
     "991083632107a9096d34bbe49ce043992f949088607e8b30f0e7060000000000"},
    {"50204",
     "00100020fefeebfabe324c205c624d70ba649c488e46f73b32712940284e0100000000008d"
     "ba9c56417ab4c217aef6ecee23c5a998d0c673ac7743ab376301a514c557579492616a0556"
     "131b4636619e",
     "8ee77b21a567d28d0cf0a7775e836f29e443a13e18c3fa860d00000000000000"},
    {"58000",
     "00100020624d55b7e520e8c6f534bbba0becad5747ba16162c7555d154c0000000000000a"
     "badc54d73961d26b0ba40ba431116f7ae91d399fda1a2651f4b93d26644a981879c826a36"
     "a9011b6af38dcb",
     "ec7dc8a7c6bd521f15532958649f5307e210ba01152c658a4c11000000000000"},
    // Current-epoch difficulty: nbits 1a6a4d80, the same value the live pools
    // were handing out when this was written.
    {"58600",
     "00100020f5b0415715052f9fcf731e4fc8ff597df7884750a3005634ae2c000000000000"
     "d31a91a3cd4a579d68966d884f50f306db72b2af27a69a50a260d7987f149d3576d8836a"
     "804d6a1a439f3af3",
     "3d35a40fb46d0c3ca517e07d8b2e1589360c20baaf8c1f92a666000000000000"},
};

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

    for (const Vector &v : kVectors) {
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

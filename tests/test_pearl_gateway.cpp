// The Pearl work source, and the whole host path, against a live gateway.
//
// Everything else in the Pearl test set is offline: kernels against vectors,
// host code against the Python reference, the Python reference against Pearl's
// own Rust. This is the one that closes the loop - it fetches real work, mines
// it on the CPU, opens the winning tile and submits, and a node either takes
// the block or does not.
//
// The CPU mining here is not a fallback miner and is not meant to be fast. It
// exists so the transport, the job encoding and the submission format can be
// proven against a real node before any of the device work depends on them.
//
//   # a pearl-gateway pointed at a regtest node, on 127.0.0.1:8455
//   c++ -O2 -std=c++17 tests/test_pearl_gateway.cpp -o tests/test_pearl_gateway
//   ./tests/test_pearl_gateway 127.0.0.1 8455
//
// Regtest is the right target: its difficulty makes roughly one transcript in
// thirty-two a winner, so one 256x256 GEMM produces several. On mainnet this
// same program would run for a very long time and find nothing, which is the
// correct behaviour rather than a bug.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/algos/pearl-pow/gateway.h"
#include "../src/algos/pearl-pow/job.h"

using namespace om;
using namespace om::pearl;

namespace {

int gPass = 0, gFail = 0;

void check(const char *name, bool ok, const std::string &detail = std::string()) {
    if (ok) {
        gPass++;
        printf("  ok   %s\n", name);
    } else {
        gFail++;
        printf("  FAIL %s %s\n", name, detail.c_str());
    }
}

/** xorshift64*, so the matrices are reproducible from a seed on any platform. */
struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    /** int8 in [-64, 63], the range A and B are required to stay inside. */
    void fill(std::vector<int8_t> *out) {
        for (size_t i = 0; i < out->size(); i += 8) {
            uint64_t v = next();
            for (size_t j = 0; j < 8 && i + j < out->size(); j++, v >>= 8)
                (*out)[i + j] = (int8_t)((int)(v & 0x7F) - 64);
        }
    }
};

}  // namespace

int main(int argc, char **argv) {
    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const int port = argc > 2 ? atoi(argv[2]) : 8455;
    const uint32_t m = 256, n = 256;

    platformInit();

    printf("1. packing a job's header into Job::extra\n");
    {
        uint8_t header[76];
        for (int i = 0; i < 76; i++) header[i] = (uint8_t)(i * 7 + 3);
        const std::string packed = packPearlExtra(header, 3);
        check("packs to 152 hex characters plus the version",
              packed.size() == 154 && packed[152] == ':', packed.substr(0, 8));
        uint8_t back[76];
        int cert = 0;
        check("unpacks to the same header",
              unpackPearlExtra(packed, back, &cert) &&
                  memcmp(back, header, 76) == 0 && cert == 3);
        check("a truncated payload is refused",
              !unpackPearlExtra(packed.substr(0, 100), back, &cert));
    }

    printf("2. talking to the gateway\n");
    PearlGatewaySource source(host, port);
    Job job;
    if (!source.fetch(&job)) {
        printf("  FAIL no answer from %s:%d\n", host.c_str(), port);
        printf("\nStart a pearl-gateway against a regtest node first.\n");
        return 1;
    }
    gPass++;
    printf("  ok   fetched a job from %s\n", source.describe());

    uint8_t header[76];
    int cert = 0;
    check("the job carries a 76-byte header and a certificate version",
          unpackPearlExtra(job.extra, header, &cert) && cert >= 1,
          job.extra.substr(0, 16));

    U256 target = U256::fromLimbs(job.target);
    bool anyTarget = false;
    for (int i = 0; i < 4; i++) anyTarget = anyTarget || job.target[i];
    check("the target is non-zero", anyTarget);

    MiningConfig cfg;
    cfg.commonDim = 2048;
    cfg.rank = 128;
    check("our mining configuration is legal for these dimensions",
          cfg.check(m, n).empty(), cfg.check(m, n));

    U256 bound;
    check("the target scales into a usable bound",
          cfg.penalizedTarget(target, &bound));

    printf("3. mining it\n");
    uint8_t key[32];
    jobKey(header, cfg, key);

    const size_t k = cfg.commonDim;
    std::vector<int8_t> A((size_t)m * k), Bt((size_t)n * k);
    Rng rng{0x9E3779B97F4A7C15ULL};
    rng.fill(&A);
    rng.fill(&Bt);

    std::vector<uint8_t> Apad(((A.size() + kChunkLen - 1) / kChunkLen) * kChunkLen, 0);
    std::vector<uint8_t> Btpad(((Bt.size() + kChunkLen - 1) / kChunkLen) * kChunkLen, 0);
    memcpy(Apad.data(), A.data(), A.size());
    memcpy(Btpad.data(), Bt.data(), Bt.size());

    uint8_t aRoot[32], btRoot[32], commitA[32], commitB[32];
    b3::hash(key, Apad.data(), Apad.size(), aRoot);
    b3::hash(key, Btpad.data(), Btpad.size(), btRoot);
    commitments(aRoot, btRoot, key, m, n, cert >= 3, commitA, commitB);

    Noise noise;
    noise.generate(commitA, commitB, m, n, k, cfg.rank);

    std::vector<int8_t> aNoised((size_t)m * k), bNoised((size_t)k * n);
    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < k; col++) {
            const int32_t e =
                (int32_t)noise.eAL[row * cfg.rank + noise.ar.first[col]] -
                (int32_t)noise.eAL[row * cfg.rank + noise.ar.second[col]];
            aNoised[row * k + col] = (int8_t)(A[row * k + col] + e);
        }
    for (size_t row = 0; row < k; row++)
        for (size_t col = 0; col < n; col++) {
            const int32_t e =
                (int32_t)noise.eBR[(size_t)noise.bl.first[row] * n + col] -
                (int32_t)noise.eBR[(size_t)noise.bl.second[row] * n + col];
            bNoised[row * n + col] = (int8_t)(Bt[col * k + row] + e);
        }

    size_t winRow = 0, winCol = 0;
    bool found = false;
    uint8_t digest[32];
    for (size_t tr = 0; tr + cfg.tileH <= m && !found; tr += cfg.tileH) {
        for (size_t tc = 0; tc + cfg.tileW <= n && !found; tc += cfg.tileW) {
            uint32_t transcript[kTranscriptWords];
            tileTranscript(aNoised.data(), bNoised.data(), k, n, cfg.rank, tr, tc,
                           cfg.tileH, cfg.tileW, transcript);
            powDigest(transcript, commitA, digest);
            if (U256::fromBytesLE(digest).le(bound)) {
                winRow = tr;
                winCol = tc;
                found = true;
            }
        }
    }
    check("found a winning tile", found,
          "regtest difficulty should make this near-certain in one GEMM");
    if (!found) {
        printf("\n%d passed, %d failed\n", gPass, gFail);
        return 1;
    }
    printf("  ok   tile at rows %zu, cols %zu, digest %02x%02x%02x...\n", winRow,
           winCol, digest[0], digest[1], digest[2]);
    gPass++;

    printf("4. opening it\n");
    MerkleProof aProof, btProof;
    check("A opens",
          buildProof(Apad.data(), Apad.size(), key,
                     leafIndicesFromRows(winRow, cfg.tileH, k), &aProof));
    check("B^t opens",
          buildProof(Btpad.data(), Btpad.size(), key,
                     leafIndicesFromRows(winCol, cfg.tileW, k), &btProof));
    check("the openings carry the roots the commitments were built from",
          memcmp(aProof.root, aRoot, 32) == 0 &&
              memcmp(btProof.root, btRoot, 32) == 0);

    const std::vector<uint8_t> proof =
        encodePlainProof(m, n, k, cfg.rank, aProof, winRow, cfg.tileH, btProof,
                         winCol, cfg.tileW);
    printf("  ok   proof is %zu bytes\n", proof.size());
    gPass++;

    printf("5. submitting it\n");
    Solution sol;
    sol.extra = base64(proof);
    memcpy(sol.hit, digest, 32);
    std::string err;
    check("the gateway accepted the submission", source.submit(job, sol, &err), err);

    printf("\n%d passed, %d failed\n", gPass, gFail);
    printf("The gateway's answer is an acknowledgement, not an acceptance - it\n"
           "proves and submits asynchronously. Check the node's block height.\n");
    return gFail ? 1 : 0;
}

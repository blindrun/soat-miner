// The Pearl host side against fixed vectors from the Python reference.
//
// job.h is a port of tests/pearl_job.py, which is itself checked against
// Pearl's own Rust in tests/pearl_oracle.py - including a proof their verifier
// accepts. So agreeing with these vectors means agreeing with the network,
// through two links that were each pinned down separately.
//
// The vectors carry expected outputs only; A and B_t are regenerated here from
// the same three-integer rule, which keeps the file at a few hundred bytes
// instead of megabytes of int8 without weakening anything - the outputs are
// still fixed, not recomputed by both sides.
//
//   python3 tests/pearl_job.py --emit-vectors /tmp/pearl_job_vectors.bin
//   c++ -O2 -std=c++17 tests/test_pearl_job.cpp -o tests/test_pearl_job
//   ./tests/test_pearl_job /tmp/pearl_job_vectors.bin

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/algos/pearl-pow/job.h"

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

std::string hex(const uint8_t *p, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

/** The same rule pearl_job.py's synth_matrix uses. */
std::vector<int8_t> synthMatrix(size_t rows, size_t cols, int64_t salt) {
    std::vector<int8_t> m(rows * cols);
    for (size_t i = 0; i < m.size(); i++)
        m[i] = (int8_t)((int64_t)(((int64_t)i * 37 + salt) & 0x7F) - 64);
    return m;
}

std::vector<uint8_t> padToChunk(const std::vector<int8_t> &in) {
    const size_t padded = ((in.size() + kChunkLen - 1) / kChunkLen) * kChunkLen;
    std::vector<uint8_t> out(padded, 0);
    memcpy(out.data(), in.data(), in.size());
    return out;
}

struct Reader {
    const uint8_t *p;
    const uint8_t *end;
    bool ok = true;
    const uint8_t *take(size_t n) {
        if (p + n > end) {
            ok = false;
            return nullptr;
        }
        const uint8_t *r = p;
        p += n;
        return r;
    }
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pearl_job_vectors.bin>\n", argv[0]);
        return 2;
    }

    FILE *fh = fopen(argv[1], "rb");
    if (!fh) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    fseek(fh, 0, SEEK_END);
    const long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    std::vector<uint8_t> blob((size_t)size);
    if (fread(blob.data(), 1, blob.size(), fh) != blob.size()) {
        fprintf(stderr, "short read\n");
        return 2;
    }
    fclose(fh);

    Reader r{blob.data(), blob.data() + blob.size()};
    const uint8_t *magic = r.take(8);
    if (!magic || memcmp(magic, "PRLJ0001", 8) != 0) {
        fprintf(stderr, "not a PRLJ0001 vector file\n");
        return 2;
    }
    const int32_t *dims = (const int32_t *)r.take(16);
    const uint32_t m = (uint32_t)dims[0], n = (uint32_t)dims[1];
    const uint32_t k = (uint32_t)dims[2], rank = (uint32_t)dims[3];

    const uint8_t *header = r.take(76);
    const uint8_t *wantCfg = r.take(MiningConfig::kSerializedSize);
    const uint8_t *wantKey = r.take(32);
    const uint8_t *wantARoot = r.take(32);
    const uint8_t *wantBtRoot = r.take(32);
    const uint8_t *wantCommitA = r.take(32);
    const uint8_t *wantCommitB = r.take(32);
    const uint8_t *wantBound = r.take(32);
    const uint8_t *proofLenP = r.take(4);
    if (!r.ok) {
        fprintf(stderr, "vector file truncated\n");
        return 2;
    }
    uint32_t proofLen = 0;
    memcpy(&proofLen, proofLenP, 4);
    const uint8_t *wantProof = r.take(proofLen);
    if (!r.ok) {
        fprintf(stderr, "vector file truncated in the proof\n");
        return 2;
    }

    printf("vectors: m=%u n=%u k=%u rank=%u, proof %u bytes\n", m, n, k, rank,
           proofLen);

    MiningConfig cfg;
    cfg.commonDim = k;
    cfg.rank = (uint16_t)rank;

    printf("1. blake3\n");
    {
        // Sizes that straddle every boundary the implementation has: block,
        // chunk, and the odd-node carry in the tree.
        static const size_t kSizes[] = {0, 1, 63, 64, 65, 1023, 1024, 1025,
                                        2048, 3000, 4096, 7 * 1024, 11 * 1024};
        // Expected digests come from the vectors indirectly: the roots below
        // are keyed hashes of multi-chunk data, so a broken tree fails there.
        // Here the only claim is self-consistency of the two entry points.
        for (size_t i = 0; i < sizeof(kSizes) / sizeof(kSizes[0]); i++) {
            std::vector<uint8_t> data(kSizes[i]);
            for (size_t j = 0; j < data.size(); j++)
                data[j] = (uint8_t)((j * 37 + kSizes[i]) & 0xFF);
            uint8_t a[32], b[32];
            b3::hash(wantKey, data.data(), data.size(), a);
            data.push_back(1);
            b3::hash(wantKey, data.data(), data.size(), b);
            char nm[64];
            snprintf(nm, sizeof(nm), "appending a byte changes the %zu-byte digest",
                     kSizes[i]);
            check(nm, memcmp(a, b, 32) != 0);
        }
    }

    printf("2. mining configuration\n");
    {
        uint8_t got[MiningConfig::kSerializedSize];
        cfg.toBytes(got);
        check("52 bytes match the reference",
              memcmp(got, wantCfg, sizeof(got)) == 0,
              hex(got, sizeof(got)) + " vs " + hex(wantCfg, sizeof(got)));
        check("this configuration is legal", cfg.check(m, n).empty(), cfg.check(m, n));

        MiningConfig bad = cfg;
        bad.rank = 64;
        check("rank 64 is refused", !bad.check(m, n).empty());
        bad = cfg;
        bad.commonDim = 1024;
        check("k below 16*rank is refused", !bad.check(m, n).empty());
    }

    printf("3. job key\n");
    {
        uint8_t got[32];
        jobKey(header, cfg, got);
        check("job_key matches", memcmp(got, wantKey, 32) == 0,
              hex(got, 32) + " vs " + hex(wantKey, 32));
    }

    printf("4. matrix roots and the commitment chain\n");
    const std::vector<int8_t> A = synthMatrix(m, k, 11);
    const std::vector<int8_t> Bt = synthMatrix(n, k, 91);
    const std::vector<uint8_t> Apad = padToChunk(A);
    const std::vector<uint8_t> Btpad = padToChunk(Bt);
    {
        uint8_t aRoot[32], btRoot[32];
        b3::hash(wantKey, Apad.data(), Apad.size(), aRoot);
        b3::hash(wantKey, Btpad.data(), Btpad.size(), btRoot);
        check("A root matches", memcmp(aRoot, wantARoot, 32) == 0,
              hex(aRoot, 32) + " vs " + hex(wantARoot, 32));
        check("B^t root matches", memcmp(btRoot, wantBtRoot, 32) == 0,
              hex(btRoot, 32) + " vs " + hex(wantBtRoot, 32));

        uint8_t ca[32], cb[32];
        commitments(aRoot, btRoot, wantKey, m, n, true, ca, cb);
        check("commitment_A matches", memcmp(ca, wantCommitA, 32) == 0,
              hex(ca, 32) + " vs " + hex(wantCommitA, 32));
        check("commitment_B matches", memcmp(cb, wantCommitB, 32) == 0,
              hex(cb, 32) + " vs " + hex(wantCommitB, 32));

        // The V3 salting is what a wrong cert version gets wrong, and it is
        // invisible without a check like this: an unsalted chain produces a
        // perfectly well-formed proof that the node rejects.
        uint8_t ca2[32], cb2[32];
        commitments(aRoot, btRoot, wantKey, m, n, false, ca2, cb2);
        check("the unsalted V1/V2 chain differs", memcmp(ca, ca2, 32) != 0);

        commitments(aRoot, btRoot, wantKey, m + 1, n, true, ca2, cb2);
        check("V3 salting binds m", memcmp(ca, ca2, 32) != 0);
    }

    printf("5. rank-penalised target\n");
    {
        // 2^232, which is regtest's target and what the vectors used.
        U256 target;
        target.v[3] = 1ULL << (232 - 192);
        U256 bound;
        check("scaling succeeds", cfg.penalizedTarget(target, &bound));
        uint8_t got[32];
        bound.toBytesLE(got);
        check("bound matches", memcmp(got, wantBound, 32) == 0,
              hex(got, 32) + " vs " + hex(wantBound, 32));

        U256 huge;
        for (int i = 0; i < 4; i++) huge.v[i] = ~0ULL;
        U256 ignored;
        check("an unusably easy target reports overflow rather than accepting all",
              !cfg.penalizedTarget(huge, &ignored));

        U256 small;
        small.v[0] = 5;
        check("comparison is little-endian by limb", small.le(target));
        check("and not the other way round", !target.le(small));
    }

    printf("6. leaf indices, Merkle proofs and the wire encoding\n");
    {
        const std::vector<uint64_t> aLeaves = leafIndicesFromRows(16, 16, k);
        const std::vector<uint64_t> bLeaves = leafIndicesFromRows(16, 16, k);
        check("16 rows of k=2048 need 32 leaves", aLeaves.size() == 32,
              std::to_string(aLeaves.size()));
        check("and they start at row 16 * 2048 / 1024", aLeaves[0] == 32,
              std::to_string(aLeaves[0]));

        MerkleProof aProof, btProof;
        check("A proof builds",
              buildProof(Apad.data(), Apad.size(), wantKey, aLeaves, &aProof));
        check("B^t proof builds",
              buildProof(Btpad.data(), Btpad.size(), wantKey, bLeaves, &btProof));
        check("the proof's root is the matrix root",
              memcmp(aProof.root, wantARoot, 32) == 0);

        const std::vector<uint8_t> blob2 =
            encodePlainProof(m, n, k, rank, aProof, 16, 16, btProof, 16, 16);
        check("the whole PlainProof is byte-identical to the reference",
              blob2.size() == proofLen &&
                  memcmp(blob2.data(), wantProof, proofLen) == 0,
              std::to_string(blob2.size()) + " vs " + std::to_string(proofLen));
    }

    printf("7. base64 round-trips\n");
    {
        for (size_t len = 0; len < 8; len++) {
            std::vector<uint8_t> in(len);
            for (size_t i = 0; i < len; i++) in[i] = (uint8_t)(i * 61 + 7);
            std::vector<uint8_t> back;
            check("round-trip", base64Decode(base64(in), &back) && back == in);
        }
        const std::vector<uint8_t> proof(wantProof, wantProof + proofLen);
        std::vector<uint8_t> back;
        check("the real proof round-trips",
              base64Decode(base64(proof), &back) && back == proof);
    }

    printf("\n%d passed, %d failed\n", gPass, gFail);
    return gFail ? 1 : 0;
}
